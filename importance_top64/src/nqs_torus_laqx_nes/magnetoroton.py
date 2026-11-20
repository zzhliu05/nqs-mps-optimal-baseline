"""Monte Carlo single-mode (GMP) energy from a fixed torus ground state.

The sampler supplies source-sector states distributed as |psi|^2.  Density
and Hamiltonian actions are evaluated in occupation space; no target-sector
eigenstate or full Hamiltonian matrix is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class BitstringRows(Protocol):
    diag: object
    neighbor_bitstrings: object
    vals: object
    mask: object


AmplitudeFn = Callable[[np.ndarray], np.ndarray]
RowsFn = Callable[[np.ndarray], BitstringRows]


@dataclass(frozen=True)
class SMASamples:
    numerator: np.ndarray
    denominator: np.ndarray
    density_paths: int
    hamiltonian_paths: int


def density_transitions(
    state: int, nphi: int, qx_index: int, qy_index: int
) -> dict[int, complex]:
    """Return <target|rho_bar(q)|state> for every allowed target bitstring."""
    if not 0 < nphi <= 63:
        raise ValueError("packed uint64 states require 1 <= nphi <= 63")
    transitions: dict[int, complex] = {}
    t = int(qy_index)
    for source in range(nphi):
        if not (state >> source) & 1:
            continue
        destination = (source + t) % nphi
        if destination != source and (state >> destination) & 1:
            continue
        target = state
        sign = 1
        if destination != source:
            lower, upper = sorted((source, destination))
            between = (state >> (lower + 1)) & ((1 << (upper - lower - 1)) - 1)
            sign = -1 if between.bit_count() & 1 else 1
            target = (state & ~(1 << source)) | (1 << destination)
        phase = np.exp(
            -2j * np.pi * int(qx_index) * (source + 0.5 * t) / nphi
        )
        transitions[target] = transitions.get(target, 0j) + sign * phase
    return transitions


def _evaluate_density_action(
    target_states: np.ndarray,
    amplitudes: AmplitudeFn,
    nphi: int,
    qx_index: int,
    qy_index: int,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate (rho_bar(q) psi)(target) by reverse one-body hops."""
    result = np.zeros(len(target_states), dtype=np.complex128)
    for start in range(0, len(target_states), chunk_size):
        batch = target_states[start : start + chunk_size]
        predecessors: list[int] = []
        coefficients: list[complex] = []
        row_ids: list[int] = []
        for row, state in enumerate(batch):
            for source, reverse_coeff in density_transitions(
                int(state), nphi, -qx_index, -qy_index
            ).items():
                predecessors.append(source)
                coefficients.append(np.conjugate(reverse_coeff))
                row_ids.append(row)
        if not predecessors:
            continue
        unique, inverse = np.unique(
            np.asarray(predecessors, dtype=np.uint64), return_inverse=True
        )
        psi = np.asarray(amplitudes(unique), dtype=np.complex128).reshape(-1)
        if psi.size != unique.size:
            raise ValueError("amplitude callback must return one value per state")
        np.add.at(
            result[start : start + len(batch)],
            np.asarray(row_ids, dtype=np.int64),
            np.asarray(coefficients) * psi[inverse],
        )
    return result


def _selected_hamiltonian_edges(
    rows: BitstringRows,
    rng: np.random.Generator,
    hamiltonian_draws: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbors = np.asarray(rows.neighbor_bitstrings, dtype=np.uint64)
    values = np.asarray(rows.vals, dtype=np.complex128)
    mask = np.asarray(rows.mask, dtype=bool)
    if neighbors.shape != values.shape or mask.shape != values.shape:
        raise ValueError("Hamiltonian row arrays must have matching shapes")
    edge_rows: list[np.ndarray] = []
    edge_states: list[np.ndarray] = []
    edge_coefficients: list[np.ndarray] = []
    for row in range(neighbors.shape[0]):
        valid = np.flatnonzero(mask[row] & (np.abs(values[row]) > 0))
        if valid.size == 0:
            continue
        if hamiltonian_draws is None:
            chosen = valid
            coeff = values[row, chosen]
        else:
            if hamiltonian_draws <= 0:
                raise ValueError("hamiltonian_draws must be positive")
            weights = np.abs(values[row, valid])
            probabilities = weights / weights.sum()
            picks = rng.choice(valid.size, size=hamiltonian_draws, p=probabilities)
            chosen = valid[picks]
            coeff = values[row, chosen] / (
                hamiltonian_draws * probabilities[picks]
            )
        edge_rows.append(np.full(chosen.size, row, dtype=np.int64))
        edge_states.append(neighbors[row, chosen])
        edge_coefficients.append(coeff)
    if not edge_rows:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.complex128),
        )
    return (
        np.concatenate(edge_rows),
        np.concatenate(edge_states),
        np.concatenate(edge_coefficients),
    )


def estimate_sma_samples(
    source_states: np.ndarray,
    amplitudes: AmplitudeFn,
    hamiltonian_rows: RowsFn,
    *,
    nphi: int,
    qx_index: int,
    qy_index: int,
    ground_energy: float,
    rng: np.random.Generator,
    density_draws: int | None = 1,
    hamiltonian_draws: int | None = 16,
    evaluation_chunk_size: int = 4096,
) -> SMASamples:
    """Estimate both SMA numerator and norm on paired |psi|^2 samples.

    ``density_draws=None`` and ``hamiltonian_draws=None`` exactly sum all
    operator paths for each source state.  Positive integer values instead
    sample paths with replacement and apply inverse-probability weights.
    The Hamiltonian callback must return its full target-sector row, including
    every allowed interaction momentum transfer.
    """
    source_states = np.asarray(source_states, dtype=np.uint64).reshape(-1)
    if source_states.size == 0:
        raise ValueError("source_states must not be empty")
    if evaluation_chunk_size <= 0:
        raise ValueError("evaluation_chunk_size must be positive")
    if density_draws is not None and density_draws <= 0:
        raise ValueError("density_draws must be positive")
    if qx_index % nphi == 0 and qy_index % nphi == 0:
        raise ValueError("q=0 creates the ground state, not a neutral mode")
    unique_source, source_inverse = np.unique(source_states, return_inverse=True)
    psi_unique = np.asarray(amplitudes(unique_source), dtype=np.complex128).reshape(-1)
    if psi_unique.size != unique_source.size:
        raise ValueError("amplitude callback must return one value per state")
    psi_source = psi_unique[source_inverse]
    if np.any(~np.isfinite(psi_source)) or np.any(np.abs(psi_source) == 0):
        raise ValueError("sampled ground-state amplitude is zero or nonfinite")

    target_states: list[int] = []
    source_ids: list[int] = []
    bra_coefficients: list[complex] = []
    path_weights: list[float] = []
    for sample, state in enumerate(source_states):
        moves = density_transitions(int(state), nphi, qx_index, qy_index)
        if not moves:
            continue
        targets = list(moves)
        count = len(targets)
        chosen = (
            np.arange(count)
            if density_draws is None
            else rng.integers(count, size=density_draws)
        )
        factor = 1.0 if density_draws is None else count / density_draws
        for index in chosen:
            y = targets[int(index)]
            target_states.append(y)
            source_ids.append(sample)
            bra_coefficients.append(np.conjugate(moves[y]))
            path_weights.append(factor)

    numerator = np.zeros(source_states.size, dtype=np.complex128)
    denominator = np.zeros(source_states.size, dtype=np.complex128)
    if not target_states:
        return SMASamples(numerator, denominator, 0, 0)

    unique_target, target_inverse = np.unique(
        np.asarray(target_states, dtype=np.uint64), return_inverse=True
    )
    rows = hamiltonian_rows(unique_target)
    diagonal = np.asarray(rows.diag, dtype=np.complex128).reshape(-1)
    if diagonal.size != unique_target.size:
        raise ValueError("Hamiltonian callback returned the wrong row count")
    edge_rows, edge_states, edge_coeff = _selected_hamiltonian_edges(
        rows, rng, hamiltonian_draws
    )
    needed_states, needed_inverse = np.unique(
        np.concatenate((unique_target, edge_states)), return_inverse=True
    )
    phi = _evaluate_density_action(
        needed_states,
        amplitudes,
        nphi,
        qx_index,
        qy_index,
        evaluation_chunk_size,
    )
    phi_target = phi[needed_inverse[: unique_target.size]]
    h_phi = (diagonal - ground_energy) * phi_target
    if edge_rows.size:
        np.add.at(
            h_phi,
            edge_rows,
            edge_coeff * phi[needed_inverse[unique_target.size :]],
        )
    sample_ids = np.asarray(source_ids, dtype=np.int64)
    prefactor = (
        np.asarray(path_weights)
        * np.asarray(bra_coefficients)
        / psi_source[sample_ids]
    )
    np.add.at(denominator, sample_ids, prefactor * phi_target[target_inverse])
    np.add.at(numerator, sample_ids, prefactor * h_phi[target_inverse])
    return SMASamples(
        numerator,
        denominator,
        len(target_states),
        int(edge_rows.size),
    )


def summarize_sma(samples: SMASamples, block_size: int = 256) -> dict[str, float]:
    """Return a paired ratio and delete-one-block jackknife error."""
    numerator = np.asarray(samples.numerator, dtype=np.complex128)
    denominator = np.asarray(samples.denominator, dtype=np.complex128)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("numerator and denominator must be paired 1D arrays")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    num_mean = np.mean(numerator)
    den_mean = np.mean(denominator)
    if abs(den_mean) <= 1e-14:
        raise ValueError("density norm is zero; SMA energy is undefined")
    energy = num_mean / den_mean
    blocks = [
        np.arange(start, min(start + block_size, numerator.size))
        for start in range(0, numerator.size, block_size)
    ]
    jackknife = []
    if len(blocks) >= 2:
        num_sum = np.sum(numerator)
        den_sum = np.sum(denominator)
        for block in blocks:
            denominator_without = den_sum - np.sum(denominator[block])
            if abs(denominator_without) <= 1e-14:
                continue
            jackknife.append(
                ((num_sum - np.sum(numerator[block])) / denominator_without).real
            )
    jackknife = np.asarray(jackknife)
    stderr = (
        float(
            np.sqrt(
                (len(jackknife) - 1)
                * np.mean((jackknife - jackknife.mean()) ** 2)
            )
        )
        if len(jackknife) >= 2
        else float("nan")
    )
    return {
        "energy": float(energy.real),
        "energy_imag": float(energy.imag),
        "stderr_block_jackknife": stderr,
        "guiding_center_norm": float(den_mean.real),
        "guiding_center_norm_imag": float(den_mean.imag),
        "numerator": float(num_mean.real),
        "numerator_imag": float(num_mean.imag),
        "samples": int(numerator.size),
        "blocks": int(len(jackknife)),
        "density_paths": int(samples.density_paths),
        "hamiltonian_paths": int(samples.hamiltonian_paths),
    }
