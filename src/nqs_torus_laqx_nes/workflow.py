"""Top-level orchestration for torus NQS calculations."""

from __future__ import annotations

import pickle
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import jax
import numpy as np
from scipy.special import binom

from .config import RunConfig
from .continuous import build_continuous_evaluation_plan
from .diagnostics import (
    plot_energy_curves,
    plot_probability_distribution,
    plot_sector_spectrum,
    write_training_history_csv,
)
from .network import (
    init_params,
    projected_ritz_states_sparse,
)
from .physics import (
    BasisData,
    SparseHamiltonianRows,
    TorusCoulombHamiltonian,
    bitstrings_to_occupations,
    build_basis,
    build_sector_basis,
    sector_basis_bitstrings,
)
from .sampling import build_momentum_scattering_table
from .training import (
    TrainingResult,
    train_continuous_vmc,
    train_projected,
    train_vmc,
)


@dataclass
class SectorResult:
    tk: int
    sector_size: int
    params: object
    training: TrainingResult
    nqs_eigenvalues: np.ndarray
    spectral_eigenvalues: np.ndarray
    overlap_eigenvalues: np.ndarray
    bitstrings: np.ndarray
    nqs_wavefunctions: np.ndarray
    exact_eigenvalues: np.ndarray | None = None
    probability_plot: str | None = None


@dataclass
class RunResult:
    config: RunConfig
    sector_results: list[SectorResult] = field(default_factory=list)
    energy_curve_output: str | None = None
    spectrum_output: str | None = None
    history_output: str | None = None
    state_archive_output: str | None = None
    params_output: str | None = None
    elapsed_seconds: float = 0.0


def save_parameter_checkpoint(
    results: list[SectorResult], config: RunConfig, output: str
) -> str:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "config": asdict(config),
        "sectors": {
            int(result.tk): jax.device_get(result.params) for result in results
        },
    }
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return str(path)


def run_sector(
    tk: int,
    basis: BasisData,
    hamiltonian: TorusCoulombHamiltonian,
    config: RunConfig,
) -> SectorResult | None:
    sector_size = int(basis.number_tk[tk])
    if sector_size <= 0:
        print(f"skip tk={tk}: empty sector")
        return None
    if sector_size < config.network.k:
        raise ValueError(
            f"sector_size={sector_size} is smaller than k={config.network.k}."
        )
    print(
        f"\nsector tk={tk}: sector_size={sector_size}, "
        f"k={config.network.k}, network={config.network.architecture}"
    )

    stream_sampled_vmc_rows = (
        config.training.optimize_method == "vmc"
        and config.training.vmc_row_batch_size is not None
        and not config.training.vmc_prebuild_sparse_rows
    )
    if (
        config.sampler.proposal_mode == "laqx_hamiltonian"
        and stream_sampled_vmc_rows
    ):
        raise ValueError(
            "laqx_hamiltonian sampling requires full prebuilt sparse rows; "
            "enable --vmc_prebuild_sparse_rows or omit --vmc_row_batch_size."
        )
    sparse_rows_cpu: SparseHamiltonianRows | None = None
    if not stream_sampled_vmc_rows:
        sparse_rows_cpu = hamiltonian.build_sparse_hamiltonian_rows(
            tk, basis
        )
    continuous_plan = None
    if config.training.optimize_method == "continuous_vmc":
        if sparse_rows_cpu is None:
            raise RuntimeError(
                "continuous_vmc requires prebuilt sparse Hamiltonian rows."
            )
        continuous_plan = build_continuous_evaluation_plan(
            sparse_rows_cpu
        ).to_jax()
    sparse_rows = (
        None if sparse_rows_cpu is None else sparse_rows_cpu.to_jax()
    )
    # Exact ED is sparse for large sectors.  Dense H is retained only for
    # probability diagnostics, whose eigvec path currently requires it.
    needs_dense_hamiltonian = config.output.prob_plot_output is not None
    h_exact = (
        hamiltonian.build_exact_hamiltonian(tk, basis)
        if needs_dense_hamiltonian
        else None
    )
    bitstrings = sector_basis_bitstrings(basis, tk)
    packed_training_only = (
        config.training.optimize_method == "vmc"
        and config.sampler.proposal_mode
        in {
            "sector_uniform",
            "conserved_pair",
            "hamiltonian_weighted",
            "hamiltonian_pair",
            "hamiltonian_hpsi_mtm",
        }
        and config.output.skip_final_ritz
        and config.sampler.prob_n_samples == 0
        and config.output.prob_plot_output is None
        and not config.output.prob_snapshot_steps
    )
    states = (
        None
        if packed_training_only
        else bitstrings_to_occupations(bitstrings, config.system.num_sites)
    )
    if packed_training_only:
        print("  CPU sector occupations: packed bitstrings only")
    params = init_params(
        config.training.seed + tk,
        config.system.num_sites,
        config.network,
    )
    training = TrainingResult(params, [])
    if not config.training.skip_train:
        if config.training.optimize_method == "vmc":
            training = train_vmc(
                params,
                states,
                sparse_rows,
                tk,
                config,
                h_exact=h_exact,
                hamiltonian=hamiltonian,
                basis=basis,
                sector_bitstrings_np=bitstrings,
            )
        elif config.training.optimize_method == "continuous_vmc":
            training = train_continuous_vmc(
                params,
                states,
                sparse_rows,
                continuous_plan,
                tk,
                config,
            )
        else:
            training = train_projected(
                params,
                states,
                sparse_rows,
                config,
            )
    params = training.params

    if config.output.skip_final_ritz:
        evals_nqs = np.empty(0, dtype=np.float64)
        spectral_evals = np.empty(0, dtype=np.float64)
        overlap_evals = np.empty(0, dtype=np.float64)
        nqs_wavefunctions = np.empty((0, 0), dtype=np.complex128)
        print("skipping final full-sector Ritz evaluation")
    else:
        if sparse_rows_cpu is None:
            sparse_rows_cpu = hamiltonian.build_sparse_hamiltonian_rows(
                tk, basis
            )
            sparse_rows = sparse_rows_cpu.to_jax()
        evals_nqs, spectral_evals, nqs_wavefunctions, overlap_evals = (
            projected_ritz_states_sparse(
                params,
                states,
                sparse_rows,
                config.network,
                config.training.overlap_shift,
                config.training.projection_rcond,
            )
        )
        print(
            "JAX NQS projected eigenvalues:",
            evals_nqs[: min(8, len(evals_nqs))],
        )
        print("overlap eigenvalues:", overlap_evals)

    evals_exact = None
    if config.output.build_exact and not config.output.skip_final_ritz:
        evals_exact = hamiltonian.sparse_exact_eigenvalues(
            tk, basis, count=max(8, config.network.k)
        )
        print("exact eigenvalues (sparse ED):", evals_exact[:8])
    elif config.output.prob_plot_output is not None:
        if h_exact is None:
            raise RuntimeError("Dense Hamiltonian is required for probability diagnostics.")
        evals_exact = np.linalg.eigvalsh(h_exact)
        if config.output.build_exact:
            print(
                "exact eigenvalues:",
                evals_exact[: min(8, len(evals_exact))],
            )

    probability_plot = None
    if config.output.prob_plot_output is not None:
        probability_plot = plot_probability_distribution(
            params,
            states,
            h_exact,
            tk,
            config.output.prob_plot_output,
            config,
            sparse_rows=sparse_rows,
        )
    return SectorResult(
        tk=tk,
        sector_size=sector_size,
        params=params,
        training=training,
        nqs_eigenvalues=evals_nqs,
        spectral_eigenvalues=spectral_evals,
        overlap_eigenvalues=overlap_evals,
        bitstrings=bitstrings,
        nqs_wavefunctions=nqs_wavefunctions,
        exact_eigenvalues=evals_exact,
        probability_plot=probability_plot,
    )


def run(config: RunConfig) -> RunResult:
    """Run one or all momentum sectors and return structured results."""
    jax.config.update("jax_enable_x64", True)
    config = replace(
        config,
        network=replace(config.network, num_sites=config.system.num_sites),
    )
    config.validate()
    start = time.perf_counter()
    if config.sampler.proposal_mode == "momentum_scattering":
        scattering_table = build_momentum_scattering_table(
            config.system.nphi
        )
        pair_count = int(np.sum(scattering_table.counts))
        scattering_count = int(
            np.sum(scattering_table.counts * scattering_table.counts)
        )
        print(
            "momentum-scattering table ready: "
            f"pairs={pair_count}, scatterings={scattering_count}"
        )
    print(
        f"Nphi={config.system.nphi}, Ne={config.system.ne}, "
        f"nu={config.system.p}/{config.system.q}, "
        f"landau_level={config.system.landau_level}, "
        f"ratio={config.system.ratio_rect}, "
        f"jax_backend={jax.default_backend()}, "
        f"hilbert={int(binom(config.system.nphi, config.system.ne))}"
    )

    basis = (
        build_basis(config.system)
        if config.system.all_sectors
        else build_sector_basis(config.system, config.system.tk)
    )
    print("sector sizes:", basis.number_tk.tolist())
    print("generating ED_torus_debug Coulomb matrix ...")
    hamiltonian = TorusCoulombHamiltonian(config.system)
    _, method, cutoff = hamiltonian.generate_coulomb_matrix()
    print(f"  Coulomb precompute method: {method}, cutoff={cutoff}")
    print(
        f"Coulomb matrix ready; jlist size={len(hamiltonian.jlist)}; "
        f"elapsed={time.perf_counter() - start:.2f}s"
    )

    sectors = (
        range(config.system.nphi)
        if config.system.all_sectors
        else [config.system.tk]
    )
    results = []
    for tk in sectors:
        result = run_sector(tk, basis, hamiltonian, config)
        if result is not None:
            results.append(result)
    if config.system.all_sectors:
        print(f"completed {len(results)} sectors")

    spectrum_output = (
        plot_sector_spectrum(results, config)
        if config.system.all_sectors
        else None
    )
    energy_curve_output = plot_energy_curves(results, config)
    history_output = (
        write_training_history_csv(results, config.output.history_output)
        if config.output.history_output is not None
        else None
    )
    state_archive_output = None
    if config.output.state_archive_output is not None:
        from .spectral import save_run_state_archive

        state_archive_output = save_run_state_archive(
            results, config, config.output.state_archive_output
        )
        print(f"saved NQS Ritz-state archive: {state_archive_output}")
    params_output = None
    if config.output.params_output is not None:
        params_output = save_parameter_checkpoint(
            results, config, config.output.params_output
        )
        print(f"saved parameter checkpoint: {params_output}")
    elapsed = time.perf_counter() - start
    print(f"total time: {elapsed:.2f}s")
    return RunResult(
        config=config,
        sector_results=results,
        energy_curve_output=energy_curve_output,
        spectrum_output=spectrum_output,
        history_output=history_output,
        state_archive_output=state_archive_output,
        params_output=params_output,
        elapsed_seconds=elapsed,
    )
