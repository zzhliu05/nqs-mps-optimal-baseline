"""Density spectral weights for all-sector torus NQS Ritz states."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class SectorStates:
    tk: int
    bitstrings: np.ndarray
    energies: np.ndarray
    wavefunctions: np.ndarray

    def validate(self, nphi: int) -> None:
        bitstrings = np.asarray(self.bitstrings)
        energies = np.asarray(self.energies)
        wavefunctions = np.asarray(self.wavefunctions)
        if bitstrings.ndim != 1:
            raise ValueError(f"sector {self.tk}: bitstrings must be one-dimensional")
        if wavefunctions.shape != (bitstrings.size, energies.size):
            raise ValueError(
                f"sector {self.tk}: wavefunctions shape {wavefunctions.shape} "
                f"does not match ({bitstrings.size}, {energies.size})"
            )
        if not 0 <= self.tk < nphi:
            raise ValueError(f"invalid sector tk={self.tk} for Nphi={nphi}")
        norms = np.sum(np.abs(wavefunctions) ** 2, axis=0)
        np.testing.assert_allclose(norms, 1.0, rtol=2e-6, atol=2e-7)
        gram = wavefunctions.conj().T @ wavefunctions
        np.testing.assert_allclose(
            gram,
            np.eye(energies.size),
            rtol=2e-6,
            atol=2e-7,
        )


@dataclass(frozen=True)
class StateArchive:
    nphi: int
    ne: int
    ratio_rect: float
    sectors: Mapping[int, SectorStates]
    metadata: Mapping[str, object]

    def validate(self) -> None:
        if self.nphi <= 0 or not 0 < self.ne <= self.nphi:
            raise ValueError("invalid Nphi or Ne in state archive")
        for sector in self.sectors.values():
            sector.validate(self.nphi)


@dataclass(frozen=True)
class SpectralPoint:
    qx_index: int
    qy_index: int
    q_lB: float
    source_tk: int
    source_level: int
    target_tk: int
    target_level: int
    source_energy: float
    target_energy: float
    omega: float
    matrix_element_real: float
    matrix_element_imag: float
    raw_weight: float
    guiding_center_weight: float
    projected_density_weight: float


@dataclass(frozen=True)
class SpectralSummary:
    qx_index: int
    qy_index: int
    q_lB: float
    target_tk: int
    captured_guiding_center_weight: float
    static_guiding_center_weight: float
    captured_fraction: float
    captured_projected_density_weight: float
    static_projected_density_weight: float


def save_run_state_archive(results, config, output: str) -> str:
    """Save normalized Ritz vectors returned by ``workflow.run``."""
    path = Path(output)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "nqs_torus_ritz_states_v2",
        "nphi": config.system.nphi,
        "ne": config.system.ne,
        "ratio_rect": config.system.ratio_rect,
        "p": config.system.p,
        "q": config.system.q,
        "network_architecture": config.network.architecture,
        "k": config.network.k,
        "optimize_method": config.training.optimize_method,
        "seed": config.training.seed,
        "sectors": [int(result.tk) for result in results],
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))
    }
    for result in results:
        prefix = f"sector_{int(result.tk)}"
        arrays[f"{prefix}_bitstrings"] = np.asarray(
            result.bitstrings, dtype=np.int64
        )
        arrays[f"{prefix}_energies"] = np.asarray(
            result.spectral_eigenvalues, dtype=np.float64
        )
        arrays[f"{prefix}_wavefunctions"] = np.asarray(
            result.nqs_wavefunctions, dtype=np.complex128
        )
        if (
            config.network.architecture == "mps"
            and config.network.mps_tensor_generation == "backflow"
        ):
            import importlib

            import jax

            from . import network as network_module
            from .physics import bitstrings_to_occupations

            if not hasattr(network_module, "mps_backflow_matrix_variation"):
                network_module = importlib.reload(network_module)

            tensor_mlp = result.params["mps"]["tensor_mlp"]
            for layer_index, layer in enumerate(tensor_mlp):
                for name, value in layer.items():
                    arrays[
                        f"{prefix}_mps_tensor_mlp_{layer_index}_{name}"
                    ] = np.asarray(jax.device_get(value))

            occupations = bitstrings_to_occupations(
                result.bitstrings, config.system.nphi
            )
            variation = network_module.mps_backflow_matrix_variation(
                result.params, occupations, config.network
            )
            for name, value in variation.items():
                arrays[f"{prefix}_backflow_matrix_{name}"] = np.asarray(
                    value
                )
            metadata["backflow_matrix_relative_rms_definition"] = (
                "sqrt(E[||A-mean(A)||_F^2] / E[||A||_F^2]) at fixed "
                "(site, local_occupation, state_index)"
            )
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    return str(path)


def load_state_archive(filename: str) -> StateArchive:
    with np.load(filename, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        sectors = {}
        for tk in metadata["sectors"]:
            prefix = f"sector_{int(tk)}"
            sectors[int(tk)] = SectorStates(
                tk=int(tk),
                bitstrings=np.asarray(data[f"{prefix}_bitstrings"], dtype=np.int64),
                energies=np.asarray(data[f"{prefix}_energies"], dtype=np.float64),
                wavefunctions=np.asarray(
                    data[f"{prefix}_wavefunctions"], dtype=np.complex128
                ),
            )
    archive = StateArchive(
        nphi=int(metadata["nphi"]),
        ne=int(metadata["ne"]),
        ratio_rect=float(metadata["ratio_rect"]),
        sectors=sectors,
        metadata=metadata,
    )
    archive.validate()
    return archive


def momentum_magnitude_lB(
    nphi: int, ratio_rect: float, qx_index: int, qy_index: int
) -> float:
    prefactor = 2.0 * math.pi / (nphi * ratio_rect)
    return math.sqrt(
        prefactor
        * (qx_index * qx_index + (ratio_rect * qy_index) ** 2)
    )


def _hop_sign(conf: int, source: int, destination: int) -> int:
    if source == destination:
        return 1
    sign = -1 if (conf >> (source + 1)).bit_count() % 2 else 1
    removed = conf & ~(1 << source)
    if (removed >> (destination + 1)).bit_count() % 2:
        sign = -sign
    return sign


def apply_guiding_center_density(
    source_bitstrings: np.ndarray,
    target_bitstrings: np.ndarray,
    source_vector: np.ndarray,
    nphi: int,
    qx_index: int,
    qy_index: int,
) -> np.ndarray:
    r"""Apply the torus guiding-center density ``rho(q)``.

    In the Landau-gauge convention used by the Hamiltonian,

    .. math::
       \bar\rho_{s,t}=\sum_j e^{-2\pi i s(j+t/2)/N_\phi}
       c^\dagger_{j+t}c_j.

    ``qx_index=s`` controls the orbital phase and ``qy_index=t`` shifts the
    orbital momentum, hence the target sector is ``tk+t`` modulo ``Nphi``.
    """
    source_bitstrings = np.asarray(source_bitstrings, dtype=np.int64)
    target_bitstrings = np.asarray(target_bitstrings, dtype=np.int64)
    source_vector = np.asarray(source_vector, dtype=np.complex128)
    if source_vector.shape != (source_bitstrings.size,):
        raise ValueError("source_vector shape does not match source basis")
    target_lookup = {int(conf): row for row, conf in enumerate(target_bitstrings)}
    output = np.zeros(target_bitstrings.size, dtype=np.complex128)
    for column, (conf_value, coefficient) in enumerate(
        zip(source_bitstrings, source_vector)
    ):
        del column
        if coefficient == 0:
            continue
        conf = int(conf_value)
        for source in range(nphi):
            if not ((conf >> source) & 1):
                continue
            destination = (source + qy_index) % nphi
            if destination != source and ((conf >> destination) & 1):
                continue
            new_conf = conf
            if destination != source:
                new_conf &= ~(1 << source)
                new_conf |= 1 << destination
            row = target_lookup.get(new_conf)
            if row is None:
                raise ValueError(
                    "density operator produced a configuration outside the "
                    "expected target sector"
                )
            phase = np.exp(
                -2j
                * np.pi
                * qx_index
                * (source + 0.5 * qy_index)
                / nphi
            )
            output[row] += coefficient * _hop_sign(
                conf, source, destination
            ) * phase
    return output


def choose_source_state(
    archive: StateArchive,
    source_tk: int | None,
    source_level: int,
) -> tuple[int, SectorStates]:
    if source_tk is None:
        candidates = [
            (float(sector.energies[source_level]), tk, sector)
            for tk, sector in archive.sectors.items()
            if sector.energies.size > source_level
        ]
        if not candidates:
            raise ValueError(f"no sector contains source level {source_level}")
        _, source_tk, source_sector = min(candidates, key=lambda item: item[0])
        return source_tk, source_sector
    if source_tk not in archive.sectors:
        raise ValueError(f"source sector {source_tk} is absent from the archive")
    source_sector = archive.sectors[source_tk]
    if not 0 <= source_level < source_sector.energies.size:
        raise ValueError("source_level is outside the archived Ritz levels")
    return source_tk, source_sector


def calculate_spectral_weights(
    archive: StateArchive,
    qx_indices: Iterable[int],
    qy_indices: Iterable[int],
    source_tk: int | None = None,
    source_level: int = 0,
) -> tuple[list[SpectralPoint], list[SpectralSummary]]:
    archive.validate()
    source_tk, source_sector = choose_source_state(
        archive, source_tk, source_level
    )
    source_energy = float(source_sector.energies[source_level])
    source_vector = source_sector.wavefunctions[:, source_level]
    points = []
    summaries = []
    for qy_index in qy_indices:
        target_tk = (source_tk + int(qy_index)) % archive.nphi
        if target_tk not in archive.sectors:
            raise ValueError(
                f"target sector {target_tk} required by qy={qy_index} "
                "is absent from the archive"
            )
        target_sector = archive.sectors[target_tk]
        for qx_index in qx_indices:
            q_lB = momentum_magnitude_lB(
                archive.nphi,
                archive.ratio_rect,
                int(qx_index),
                int(qy_index),
            )
            rho_psi = apply_guiding_center_density(
                source_sector.bitstrings,
                target_sector.bitstrings,
                source_vector,
                archive.nphi,
                int(qx_index),
                int(qy_index),
            )
            amplitudes = target_sector.wavefunctions.conj().T @ rho_psi
            raw_weights = np.abs(amplitudes) ** 2
            guiding_weights = raw_weights / archive.ne
            form_factor_sq = math.exp(-0.5 * q_lB * q_lB)
            projected_weights = guiding_weights * form_factor_sq
            static_guiding = float(np.vdot(rho_psi, rho_psi).real / archive.ne)
            captured_guiding = float(np.sum(guiding_weights))
            captured_fraction = (
                captured_guiding / static_guiding if static_guiding > 1e-15 else 0.0
            )
            summaries.append(
                SpectralSummary(
                    qx_index=int(qx_index),
                    qy_index=int(qy_index),
                    q_lB=q_lB,
                    target_tk=target_tk,
                    captured_guiding_center_weight=captured_guiding,
                    static_guiding_center_weight=static_guiding,
                    captured_fraction=captured_fraction,
                    captured_projected_density_weight=(
                        captured_guiding * form_factor_sq
                    ),
                    static_projected_density_weight=(
                        static_guiding * form_factor_sq
                    ),
                )
            )
            for level, (target_energy, amplitude) in enumerate(
                zip(target_sector.energies, amplitudes)
            ):
                points.append(
                    SpectralPoint(
                        qx_index=int(qx_index),
                        qy_index=int(qy_index),
                        q_lB=q_lB,
                        source_tk=source_tk,
                        source_level=source_level,
                        target_tk=target_tk,
                        target_level=level,
                        source_energy=source_energy,
                        target_energy=float(target_energy),
                        omega=float(target_energy - source_energy),
                        matrix_element_real=float(amplitude.real),
                        matrix_element_imag=float(amplitude.imag),
                        raw_weight=float(raw_weights[level]),
                        guiding_center_weight=float(guiding_weights[level]),
                        projected_density_weight=float(projected_weights[level]),
                    )
                )
    return points, summaries


def _write_dataclass_csv(rows, filename: Path) -> str:
    filename.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty spectral table")
    fieldnames = list(rows[0].__dataclass_fields__)
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in fieldnames})
    return str(filename)


def plot_spectral_weights(
    points: list[SpectralPoint],
    output: Path,
    weight_name: str,
    broadening: float,
) -> tuple[str, str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional plotting.
        raise RuntimeError("matplotlib is required to plot spectral weights") from exc
    if weight_name not in {
        "guiding_center_weight",
        "projected_density_weight",
    }:
        raise ValueError("unsupported plot weight")
    output.parent.mkdir(parents=True, exist_ok=True)
    q_labels = []
    q_positions = {}
    for point in points:
        key = (point.qx_index, point.qy_index)
        if key not in q_positions:
            q_positions[key] = len(q_labels)
            q_labels.append(key)
    x = np.asarray([q_positions[(p.qx_index, p.qy_index)] for p in points])
    omega = np.asarray([p.omega for p in points])
    weights = np.asarray([getattr(p, weight_name) for p in points])
    positive = omega >= -1e-10

    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(q_labels)), 5.4))
    scale = 20.0 + 800.0 * weights / max(float(weights.max()), 1e-15)
    scatter = ax.scatter(
        x[positive],
        omega[positive],
        c=weights[positive],
        s=scale[positive],
        cmap="viridis",
        edgecolors="none",
    )
    ax.set_xticks(range(len(q_labels)))
    ax.set_xticklabels([f"({qx},{qy})" for qx, qy in q_labels], rotation=60)
    ax.set_xlabel(r"$(q_x,q_y)$ reciprocal-lattice indices")
    ax.set_ylabel(r"$\omega=E_n-E_0$")
    ax.set_title("NQS density spectral weights")
    fig.colorbar(scatter, ax=ax, label=weight_name.replace("_", " "))
    fig.tight_layout()
    scatter_path = output.with_name(output.name + "_levels.png")
    fig.savefig(scatter_path, dpi=220)
    plt.close(fig)

    positive_omega = omega[positive]
    max_omega = max(float(positive_omega.max(initial=0.0)), broadening * 5.0)
    omega_grid = np.linspace(0.0, max_omega + 4.0 * broadening, 600)
    intensity = np.zeros((omega_grid.size, len(q_labels)), dtype=np.float64)
    normalization = 1.0 / (math.sqrt(2.0 * math.pi) * broadening)
    for point in points:
        if point.omega < -1e-10:
            continue
        column = q_positions[(point.qx_index, point.qy_index)]
        weight = getattr(point, weight_name)
        intensity[:, column] += (
            weight
            * normalization
            * np.exp(-0.5 * ((omega_grid - point.omega) / broadening) ** 2)
        )
    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(q_labels)), 5.4))
    image = ax.imshow(
        intensity,
        origin="lower",
        aspect="auto",
        extent=(-0.5, len(q_labels) - 0.5, omega_grid[0], omega_grid[-1]),
        cmap="magma",
    )
    ax.set_xticks(range(len(q_labels)))
    ax.set_xticklabels([f"({qx},{qy})" for qx, qy in q_labels], rotation=60)
    ax.set_xlabel(r"$(q_x,q_y)$ reciprocal-lattice indices")
    ax.set_ylabel(r"$\omega$")
    ax.set_title(rf"Broadened $S(q,\omega)$, $\sigma={broadening:g}$")
    fig.colorbar(image, ax=ax, label=r"$S(q,\omega)$")
    fig.tight_layout()
    heatmap_path = output.with_name(output.name + "_dynamic_structure_factor.png")
    fig.savefig(heatmap_path, dpi=220)
    plt.close(fig)
    return str(scatter_path), str(heatmap_path)


def centered_indices(nphi: int) -> list[int]:
    return list(range(-(nphi // 2), (nphi - 1) // 2 + 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute torus NQS density spectral weights from a Ritz-state archive."
    )
    parser.add_argument("--state_archive", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--source_tk", type=int, default=None)
    parser.add_argument("--source_level", type=int, default=0)
    parser.add_argument(
        "--qx_indices",
        type=int,
        nargs="+",
        default=[0],
        help="Density phase indices s; default is the qx=0 cut.",
    )
    parser.add_argument(
        "--qy_indices",
        type=int,
        nargs="+",
        default=None,
        help="Orbital-shift indices t; default is one centered Brillouin zone.",
    )
    parser.add_argument("--broadening", type=float, default=0.005)
    parser.add_argument(
        "--plot_weight",
        choices=["guiding_center_weight", "projected_density_weight"],
        default="guiding_center_weight",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.broadening <= 0:
        raise ValueError("broadening must be positive")
    archive = load_state_archive(args.state_archive)
    qy_indices = (
        centered_indices(archive.nphi)
        if args.qy_indices is None
        else args.qy_indices
    )
    points, summaries = calculate_spectral_weights(
        archive,
        args.qx_indices,
        qy_indices,
        source_tk=args.source_tk,
        source_level=args.source_level,
    )
    prefix = Path(args.output_prefix)
    points_path = _write_dataclass_csv(
        points, prefix.with_name(prefix.name + "_spectral_weights.csv")
    )
    summary_path = _write_dataclass_csv(
        summaries, prefix.with_name(prefix.name + "_sum_rule.csv")
    )
    plot_paths = plot_spectral_weights(
        points, prefix, args.plot_weight, args.broadening
    )
    source = points[0]
    print(
        f"source: tk={source.source_tk}, level={source.source_level}, "
        f"energy={source.source_energy:.12g}"
    )
    print(f"spectral points: {len(points)}")
    print(
        "captured fraction range: "
        f"{min(item.captured_fraction for item in summaries):.6g} .. "
        f"{max(item.captured_fraction for item in summaries):.6g}"
    )
    print(f"wrote: {points_path}")
    print(f"wrote: {summary_path}")
    for path in plot_paths:
        print(f"wrote: {path}")


if __name__ == "__main__":
    main()
