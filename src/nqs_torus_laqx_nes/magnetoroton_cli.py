"""Run the Monte Carlo GMP estimator from a saved NQS ground state."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import NetworkConfig, SystemConfig
from .magnetoroton import estimate_sma_samples, summarize_sma
from .network import wavefunction_matrix
from .physics import TorusCoulombHamiltonian, build_sector_basis
from .sampling import MetropolisSamplerJAX
from .spectral import momentum_magnitude_lB


class CheckpointAmplitudes:
    def __init__(self, params, config: NetworkConfig, batch_size: int):
        self.params = jax.device_put(params)
        self.config = config
        self.batch_size = int(batch_size)
        self.shifts = jnp.arange(config.num_sites, dtype=jnp.uint64)
        self.forward = jax.jit(
            lambda occupations: wavefunction_matrix(self.params, occupations, self.config)
        )

    def __call__(self, bitstrings: np.ndarray) -> np.ndarray:
        states = np.asarray(bitstrings, dtype=np.uint64).reshape(-1)
        values = np.empty(states.size, dtype=np.complex128)
        for start in range(0, states.size, self.batch_size):
            batch = states[start : start + self.batch_size]
            packed = np.zeros(self.batch_size, dtype=np.uint64)
            packed[: batch.size] = batch
            packed_device = jnp.asarray(packed, dtype=jnp.uint64)
            occupations = ((packed_device[:, None] >> self.shifts) & 1).astype(
                jnp.float32
            )
            values[start : start + batch.size] = np.asarray(
                self.forward(occupations)[: batch.size, 0]
            )
        return values


def _ground_energy_from_history(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 100:
        raise ValueError("ground-state history needs at least 100 steps")
    return float(np.mean([float(row["energy"]) for row in rows[-100:]]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--q", nargs=2, type=int, action="append", required=True,
                        metavar=("S", "T"))
    parser.add_argument("--ground-energy", type=float)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--source-tk", type=int)
    parser.add_argument("--n-chains", type=int, default=512)
    parser.add_argument("--n-burn", type=int, default=2000)
    parser.add_argument("--n-between", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--density-draws", type=int, default=2)
    parser.add_argument("--hamiltonian-draws", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--sample-cache", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_chains <= 0 or args.n_samples <= 0:
        raise ValueError("n_chains and n_samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    sectors = payload["sectors"]
    source_tk = (
        int(args.source_tk)
        if args.source_tk is not None
        else int(next(iter(sectors)))
    )
    params = sectors[source_tk]
    system = replace(SystemConfig(**payload["config"]["system"]), tk=source_tk)
    network = replace(
        NetworkConfig(**payload["config"]["network"]),
        num_sites=system.nphi,
        forward_batch_size=args.batch_size,
    )
    if network.k != 1:
        raise ValueError("the current GMP estimator requires a single NQS state")
    if args.ground_energy is None:
        history_path = args.history or args.checkpoint.with_name("history.csv")
        ground_energy = _ground_energy_from_history(history_path)
    else:
        ground_energy = args.ground_energy

    amplitudes = CheckpointAmplitudes(params, network, args.batch_size)
    sample_cache = args.sample_cache or args.output_dir / "source_samples.npz"
    if sample_cache.exists():
        cached = np.load(sample_cache)
        source_states = np.asarray(cached["bitstrings"], dtype=np.uint64)
        if int(cached["source_tk"]) != source_tk:
            raise ValueError("sample cache belongs to a different sector")
        acceptance = float(cached["acceptance"])
        print(f"loaded {source_states.size} cached source samples", flush=True)
    else:
        basis = build_sector_basis(system, source_tk)
        sampler = MetropolisSamplerJAX(
            params,
            basis.bitstrings,
            args.n_chains,
            network,
            seed=args.seed,
            proposal_mode="sector_uniform",
            sector_states_host=basis.bitstrings,
            sector_nphi=system.nphi,
            sector_n_particles=system.ne,
            sector_on_host=True,
        )
        start = time.perf_counter()
        positions, acceptance = sampler.sample(
            args.n_burn, args.n_between, args.n_samples
        )
        source_states = np.asarray(basis.bitstrings[positions.reshape(-1)], dtype=np.uint64)
        sample_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sample_cache,
            bitstrings=source_states,
            source_tk=source_tk,
            acceptance=acceptance,
        )
        print(
            f"sampled {source_states.size} states in {time.perf_counter()-start:.1f}s; "
            f"acceptance={acceptance:.5f}; unique={np.unique(source_states).size}",
            flush=True,
        )
    hamiltonian = TorusCoulombHamiltonian(system)
    hamiltonian.generate_coulomb_matrix()
    print(
        f"full Hamiltonian: {len(hamiltonian.jlist)} interaction terms; "
        f"source tk={source_tk}, E0={ground_energy:.9f}",
        flush=True,
    )
    results = []
    for s, t in args.q:
        if s % system.nphi == 0 and t % system.nphi == 0:
            continue
        started = time.perf_counter()
        samples = estimate_sma_samples(
            source_states,
            amplitudes,
            hamiltonian.build_sparse_hamiltonian_bitstring_rows,
            nphi=system.nphi,
            qx_index=s,
            qy_index=t,
            ground_energy=ground_energy,
            rng=np.random.default_rng(args.seed + s * 104729 + t * 1009),
            density_draws=args.density_draws,
            hamiltonian_draws=args.hamiltonian_draws,
            evaluation_chunk_size=args.batch_size,
        )
        summary = summarize_sma(samples, block_size=args.n_chains)
        q_l_b = momentum_magnitude_lB(system.nphi, system.ratio_rect, s, t)
        result = {
            "qx_index": s,
            "qy_index": t,
            "target_tk": (source_tk + t) % system.nphi,
            "q_lB": q_l_b,
            "ground_energy": ground_energy,
            "sample_acceptance": acceptance,
            "unique_source_states": int(np.unique(source_states).size),
            "physical_static_structure_factor": (
                np.exp(-0.5 * q_l_b * q_l_b)
                * summary["guiding_center_norm"]
                / system.ne
            ),
            "elapsed_seconds": time.perf_counter() - started,
            **summary,
        }
        np.savez_compressed(
            args.output_dir / f"q_s{s}_t{t}_samples.npz",
            numerator=samples.numerator,
            denominator=samples.denominator,
        )
        results.append(result)
        (args.output_dir / "results.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "sample_cache": str(sample_cache),
                    "source_tk": source_tk,
                    "density_draws": args.density_draws,
                    "hamiltonian_draws": args.hamiltonian_draws,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"q=({s},{t}) |q|lB={q_l_b:.4f} E_SMA={summary['energy']:.7f} "
            f"+/- {summary['stderr_block_jackknife']:.7f} "
            f"D={summary['guiding_center_norm']:.7f} "
            f"time={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
