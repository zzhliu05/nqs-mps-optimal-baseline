"""Fixed-sector Metropolis sampling for determinant NQS states."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from functools import lru_cache
import os

import jax
import jax.numpy as jnp
import numpy as np

from .config import NetworkConfig
from .network import (
    deduplicated_wavefunction_matrix_from_positions,
    wavefunction_matrix,
    occupation_states_from_positions,
)
from .physics import antisym_coulomb

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration.
    njit = None
    prange = range
    NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class MomentumScatteringTable:
    pairs: np.ndarray
    masks: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class AllowedOutgoingPairsCPU:
    """CPU-only CSR table keyed by (sector basis state, total momentum)."""

    nphi: int
    bitstrings: np.ndarray
    sorted_bitstrings: np.ndarray
    sorted_positions: np.ndarray
    occupied_orbitals: np.ndarray
    occupied_pair_slots: np.ndarray
    offsets: np.ndarray
    outgoing_masks: np.ndarray
    max_row_nnz: int

    def bounds(self, position: int, total_momentum: int) -> tuple[int, int]:
        row = int(position) * self.nphi + int(total_momentum)
        return int(self.offsets[row]), int(self.offsets[row + 1])

    @property
    def nbytes(self) -> int:
        return int(
            self.bitstrings.nbytes
            + self.sorted_bitstrings.nbytes
            + self.sorted_positions.nbytes
            + self.occupied_orbitals.nbytes
            + self.occupied_pair_slots.nbytes
            + self.offsets.nbytes
            + self.outgoing_masks.nbytes
        )


@dataclass(frozen=True)
class LaQXHamiltonianProposalTable:
    """CPU proposal table mirroring LaQX's Hamiltonian-weighted MCMC."""

    cols: np.ndarray
    weights: np.ndarray
    cumulative_weights: np.ndarray
    weight_sums: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(
            self.cols.nbytes
            + self.weights.nbytes
            + self.cumulative_weights.nbytes
            + self.weight_sums.nbytes
        )


@dataclass(frozen=True)
class HamiltonianPairProposalTable:
    """State-independent two-body event table for cheap H-guided proposals."""

    pairs: np.ndarray
    pair_index: np.ndarray
    outgoing_indices: np.ndarray
    outgoing_counts: np.ndarray
    cumulative_weights: np.ndarray
    weight_sums: np.ndarray
    pair_weights: np.ndarray


def build_hamiltonian_pair_proposal_table(
    nphi: int,
    coulomb_matrix: np.ndarray,
) -> HamiltonianPairProposalTable:
    pairs = np.asarray(
        [(left, right) for left in range(nphi) for right in range(left + 1, nphi)],
        dtype=np.int64,
    )
    npairs = int(pairs.shape[0])
    pair_index = np.full((nphi, nphi), -1, dtype=np.int64)
    totals = np.empty(npairs, dtype=np.int64)
    for index, (left, right) in enumerate(pairs):
        pair_index[left, right] = index
        pair_index[right, left] = index
        totals[index] = (int(left) + int(right)) % nphi
    groups = [np.flatnonzero(totals == total) for total in range(nphi)]
    max_group = max(len(group) for group in groups)
    outgoing = np.zeros((npairs, max_group), dtype=np.int64)
    outgoing_counts = np.zeros(npairs, dtype=np.int64)
    cumulative = np.zeros((npairs, max_group), dtype=np.float64)
    sums = np.zeros(npairs, dtype=np.float64)
    dense_weights = np.zeros((npairs, npairs), dtype=np.float64)
    for incoming, (j3, j4) in enumerate(pairs):
        group = groups[int(totals[incoming])]
        outgoing_counts[incoming] = len(group)
        weights = np.asarray(
            [
                abs(
                    antisym_coulomb(
                        coulomb_matrix,
                        int(pairs[out_idx, 0]),
                        int(pairs[out_idx, 1]),
                        int(j3),
                        int(j4),
                    )
                )
                for out_idx in group
            ],
            dtype=np.float64,
        )
        outgoing[incoming, : len(group)] = group
        cumulative[incoming, : len(group)] = np.cumsum(weights)
        if len(group) < max_group:
            cumulative[incoming, len(group) :] = np.sum(weights)
        sums[incoming] = np.sum(weights)
        dense_weights[incoming, group] = weights
    return HamiltonianPairProposalTable(
        pairs=pairs,
        pair_index=pair_index,
        outgoing_indices=outgoing,
        outgoing_counts=outgoing_counts,
        cumulative_weights=cumulative,
        weight_sums=sums,
        pair_weights=dense_weights,
    )


def build_laqx_hamiltonian_proposal_table(
    sparse_rows,
    weight_floor: float = 0.05,
) -> LaQXHamiltonianProposalTable:
    """Build LaQX-style proposal weights from full sparse Hamiltonian rows."""
    if sparse_rows is None:
        raise ValueError(
            "laqx_hamiltonian sampling requires prebuilt full sparse rows."
        )
    cols = np.asarray(sparse_rows.cols, dtype=np.int64)
    vals = np.asarray(sparse_rows.vals, dtype=np.complex128)
    mask = np.asarray(sparse_rows.mask, dtype=np.bool_)
    if cols.ndim != 2 or vals.shape != cols.shape or mask.shape != cols.shape:
        raise ValueError("sparse Hamiltonian rows must share a rank-2 shape.")
    weights = np.where(mask, np.maximum(np.abs(vals), weight_floor), 0.0)
    cumulative = np.cumsum(weights, axis=1)
    return LaQXHamiltonianProposalTable(
        cols=cols,
        weights=weights,
        cumulative_weights=cumulative,
        weight_sums=np.sum(weights, axis=1),
    )


@lru_cache(maxsize=None)
def build_momentum_scattering_table(
    nphi: int,
) -> MomentumScatteringTable:
    """Group unordered orbital pairs by total momentum modulo Nphi."""
    if nphi < 2:
        raise ValueError("momentum scattering requires Nphi >= 2.")
    grouped = [[] for _ in range(nphi)]
    for first in range(nphi):
        for second in range(first + 1, nphi):
            grouped[(first + second) % nphi].append((first, second))
    counts = np.asarray([len(group) for group in grouped], dtype=np.int64)
    max_count = int(np.max(counts))
    pairs = np.zeros((nphi, max_count, 2), dtype=np.int64)
    masks = np.zeros((nphi, max_count), dtype=np.int64)
    for total, group in enumerate(grouped):
        pairs[total, : len(group)] = np.asarray(group, dtype=np.int64)
        for slot, (first, second) in enumerate(group):
            masks[total, slot] = (
                np.left_shift(np.int64(1), first)
                | np.left_shift(np.int64(1), second)
            )
    return MomentumScatteringTable(
        pairs=pairs,
        masks=masks,
        counts=counts,
    )


@lru_cache(maxsize=None)
def build_pair_slots(count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("momentum scattering requires at least two particles.")
    return np.asarray(
        [
            (first, second)
            for first in range(count)
            for second in range(first + 1, count)
        ],
        dtype=np.int64,
    )


def build_allowed_outgoing_pairs_cpu(
    sector_states: np.ndarray,
) -> AllowedOutgoingPairsCPU:
    """Precompute all Pauli-allowed outgoing pairs in host memory."""
    states = np.asarray(sector_states, dtype=np.int8)
    if states.ndim != 2:
        raise ValueError("sector_states must be a rank-2 occupation array.")
    sector_size, nphi = states.shape
    particle_counts = np.sum(states, axis=1).astype(np.int64)
    if sector_size == 0 or not np.all(particle_counts == particle_counts[0]):
        raise ValueError("sector states must have a fixed particle count.")
    n_particles = int(particle_counts[0])
    pair_slots = build_pair_slots(n_particles)
    occupied_orbitals = np.argsort(-states, axis=1)[
        :, :n_particles
    ].astype(np.int64)
    powers = np.left_shift(
        np.int64(1), np.arange(nphi, dtype=np.int64)
    )
    bitstrings = np.sum(
        states.astype(np.int64) * powers[None, :],
        axis=1,
        dtype=np.int64,
    )
    order = np.argsort(bitstrings)
    scattering = build_momentum_scattering_table(nphi)
    offsets = np.zeros(sector_size * nphi + 1, dtype=np.int64)
    outgoing_masks = []
    cursor = 0
    for position, bits in enumerate(bitstrings):
        for total_momentum in range(nphi):
            count = int(scattering.counts[total_momentum])
            masks = scattering.masks[total_momentum, :count]
            allowed = masks[(bits & masks) == 0]
            outgoing_masks.extend(int(mask) for mask in allowed)
            cursor += int(allowed.shape[0])
            offsets[position * nphi + total_momentum + 1] = cursor
    if NUMBA_AVAILABLE:
        row_nnz = _allowed_row_nnz_numba(
            occupied_orbitals, pair_slots, offsets, nphi
        )
    else:  # pragma: no cover - production uses Numba.
        row_nnz = np.zeros(sector_size, dtype=np.int64)
        for position in range(sector_size):
            for left, right in pair_slots:
                total_momentum = int(
                    occupied_orbitals[position, left]
                    + occupied_orbitals[position, right]
                ) % nphi
                row = position * nphi + total_momentum
                row_nnz[position] += offsets[row + 1] - offsets[row]
    return AllowedOutgoingPairsCPU(
        nphi=nphi,
        bitstrings=bitstrings,
        sorted_bitstrings=bitstrings[order],
        sorted_positions=order.astype(np.int64),
        occupied_orbitals=occupied_orbitals,
        occupied_pair_slots=pair_slots,
        offsets=offsets,
        outgoing_masks=np.asarray(outgoing_masks, dtype=np.int64),
        max_row_nnz=int(np.max(row_nnz, initial=0)),
    )


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _allowed_row_nnz_numba(
        occupied_orbitals, pair_slots, offsets, nphi
    ):
        sector_size = occupied_orbitals.shape[0]
        row_nnz = np.zeros(sector_size, dtype=np.int64)
        for position in prange(sector_size):
            count = 0
            for pair_index in range(pair_slots.shape[0]):
                left = pair_slots[pair_index, 0]
                right = pair_slots[pair_index, 1]
                total_momentum = (
                    occupied_orbitals[position, left]
                    + occupied_orbitals[position, right]
                ) % nphi
                row = position * nphi + total_momentum
                count += offsets[row + 1] - offsets[row]
            row_nnz[position] = count
        return row_nnz

    @njit(cache=True, parallel=True)
    def _momentum_scattering_proposals_numba(
        positions,
        rows,
        incoming_slots,
        outgoing_uniform,
        nphi,
        bitstrings,
        sorted_bitstrings,
        sorted_positions,
        occupied_orbitals,
        occupied_pair_slots,
        offsets,
        outgoing_masks,
    ):
        n_chains, k = positions.shape
        proposal = positions.copy()
        valid = np.zeros(n_chains, dtype=np.bool_)
        log_hastings = np.zeros(n_chains, dtype=np.float64)
        sorted_size = sorted_bitstrings.shape[0]
        outgoing_size = outgoing_masks.shape[0]

        for chain in prange(n_chains):
            row = rows[chain]
            current_position = positions[chain, row]
            pair_slot = occupied_pair_slots[incoming_slots[chain]]
            incoming_first = occupied_orbitals[
                current_position, pair_slot[0]
            ]
            incoming_second = occupied_orbitals[
                current_position, pair_slot[1]
            ]
            total_momentum = (
                incoming_first + incoming_second
            ) % nphi
            csr_row = current_position * nphi + total_momentum
            start = offsets[csr_row]
            stop = offsets[csr_row + 1]
            forward_count = stop - start
            if forward_count <= 0 or outgoing_size == 0:
                continue

            outgoing_slot = start + int(
                outgoing_uniform[chain] * forward_count
            )
            if outgoing_slot >= stop:
                outgoing_slot = stop - 1
            outgoing_mask = outgoing_masks[outgoing_slot]
            incoming_mask = (
                (np.int64(1) << incoming_first)
                | (np.int64(1) << incoming_second)
            )
            proposed_bits = (
                bitstrings[current_position] & ~incoming_mask
            ) | outgoing_mask

            left = 0
            right = sorted_size
            while left < right:
                middle = (left + right) // 2
                if sorted_bitstrings[middle] < proposed_bits:
                    left = middle + 1
                else:
                    right = middle
            if (
                left >= sorted_size
                or sorted_bitstrings[left] != proposed_bits
            ):
                continue
            candidate = sorted_positions[left]
            reverse_row = candidate * nphi + total_momentum
            reverse_count = (
                offsets[reverse_row + 1] - offsets[reverse_row]
            )
            if reverse_count <= 0:
                continue

            duplicate = False
            for column in range(k):
                if column != row and positions[chain, column] == candidate:
                    duplicate = True
                    break
            if duplicate:
                continue

            proposal[chain, row] = candidate
            valid[chain] = True
            log_hastings[chain] = (
                np.log(forward_count) - np.log(reverse_count)
            )
        return proposal, valid, log_hastings

    @njit(cache=True, parallel=True)
    def _hamiltonian_pair_proposals_numba(
        positions,
        rows,
        incoming_slots,
        outgoing_uniform,
        nphi,
        n_particles,
        bitstrings,
        sorted_bitstrings,
        sorted_positions,
        occupied_pair_slots,
        pairs,
        pair_index,
        outgoing_indices,
        cumulative_weights,
        weight_sums,
        pair_weights,
    ):
        n_chains, k = positions.shape
        proposal = positions.copy()
        valid = np.zeros(n_chains, dtype=np.bool_)
        log_hastings = np.zeros(n_chains, dtype=np.float64)
        sorted_size = sorted_bitstrings.shape[0]

        for chain in prange(n_chains):
            row = rows[chain]
            current_position = positions[chain, row]
            current_bits = bitstrings[current_position]
            occupied = np.empty(n_particles, dtype=np.int64)
            write = 0
            for orbital in range(nphi):
                if (current_bits >> orbital) & 1:
                    occupied[write] = orbital
                    write += 1
            pair_slot = occupied_pair_slots[incoming_slots[chain]]
            incoming_first = occupied[pair_slot[0]]
            incoming_second = occupied[pair_slot[1]]
            incoming_index = pair_index[incoming_first, incoming_second]
            forward_sum = weight_sums[incoming_index]
            if incoming_index < 0 or forward_sum <= 0.0:
                continue

            draw = outgoing_uniform[chain] * forward_sum
            slot = 0
            width = cumulative_weights.shape[1]
            while slot < width - 1 and cumulative_weights[incoming_index, slot] < draw:
                slot += 1
            outgoing_index = outgoing_indices[incoming_index, slot]
            forward_weight = pair_weights[incoming_index, outgoing_index]
            reverse_weight = pair_weights[outgoing_index, incoming_index]
            reverse_sum = weight_sums[outgoing_index]
            if forward_weight <= 0.0 or reverse_weight <= 0.0 or reverse_sum <= 0.0:
                continue

            incoming_mask = (
                (np.int64(1) << incoming_first)
                | (np.int64(1) << incoming_second)
            )
            outgoing_first = pairs[outgoing_index, 0]
            outgoing_second = pairs[outgoing_index, 1]
            outgoing_mask = (
                (np.int64(1) << outgoing_first)
                | (np.int64(1) << outgoing_second)
            )
            remaining = current_bits & ~incoming_mask
            if (remaining & outgoing_mask) != 0:
                continue
            proposed_bits = remaining | outgoing_mask
            if proposed_bits == current_bits:
                continue

            left = 0
            right = sorted_size
            while left < right:
                middle = (left + right) // 2
                if sorted_bitstrings[middle] < proposed_bits:
                    left = middle + 1
                else:
                    right = middle
            if left >= sorted_size or sorted_bitstrings[left] != proposed_bits:
                continue
            candidate = sorted_positions[left]
            duplicate = False
            for column in range(k):
                if column != row and positions[chain, column] == candidate:
                    duplicate = True
                    break
            if duplicate:
                continue

            proposal[chain, row] = candidate
            valid[chain] = True
            log_hastings[chain] = (
                np.log(reverse_weight)
                - np.log(reverse_sum)
                - np.log(forward_weight)
                + np.log(forward_sum)
            )
        return proposal, valid, log_hastings

    @njit(cache=True, parallel=True)
    def _hamiltonian_hpsi_candidates_numba(
        positions,
        rows,
        incoming_slots,
        outgoing_uniform,
        nphi,
        n_particles,
        bitstrings,
        sorted_bitstrings,
        sorted_positions,
        occupied_pair_slots,
        pairs,
        pair_index,
        outgoing_indices,
        outgoing_counts,
        pair_weights,
    ):
        """Draw symmetric uniform two-body events for exact MTM."""
        n_chains, k = positions.shape
        n_trials = incoming_slots.shape[1]
        candidates = np.empty((n_chains, n_trials), dtype=np.int64)
        weights = np.zeros((n_chains, n_trials), dtype=np.float64)
        valid = np.zeros((n_chains, n_trials), dtype=np.bool_)
        sorted_size = sorted_bitstrings.shape[0]

        for chain in prange(n_chains):
            row = rows[chain]
            current_position = positions[chain, row]
            current_bits = bitstrings[current_position]
            occupied = np.empty(n_particles, dtype=np.int64)
            write = 0
            for orbital in range(nphi):
                if (current_bits >> orbital) & 1:
                    occupied[write] = orbital
                    write += 1

            for trial in range(n_trials):
                candidates[chain, trial] = current_position
                pair_slot = occupied_pair_slots[incoming_slots[chain, trial]]
                incoming_first = occupied[pair_slot[0]]
                incoming_second = occupied[pair_slot[1]]
                incoming_index = pair_index[incoming_first, incoming_second]
                if incoming_index < 0:
                    continue
                count = outgoing_counts[incoming_index]
                if count <= 0:
                    continue
                slot = int(outgoing_uniform[chain, trial] * count)
                if slot >= count:
                    slot = count - 1
                outgoing_index = outgoing_indices[incoming_index, slot]
                matrix_weight = pair_weights[incoming_index, outgoing_index]
                if matrix_weight <= 0.0:
                    continue

                incoming_mask = (
                    (np.int64(1) << incoming_first)
                    | (np.int64(1) << incoming_second)
                )
                outgoing_first = pairs[outgoing_index, 0]
                outgoing_second = pairs[outgoing_index, 1]
                outgoing_mask = (
                    (np.int64(1) << outgoing_first)
                    | (np.int64(1) << outgoing_second)
                )
                remaining = current_bits & ~incoming_mask
                if (remaining & outgoing_mask) != 0:
                    continue
                proposed_bits = remaining | outgoing_mask
                if proposed_bits == current_bits:
                    continue

                left = 0
                right = sorted_size
                while left < right:
                    middle = (left + right) // 2
                    if sorted_bitstrings[middle] < proposed_bits:
                        left = middle + 1
                    else:
                        right = middle
                if left >= sorted_size or sorted_bitstrings[left] != proposed_bits:
                    continue
                candidate = sorted_positions[left]
                duplicate = False
                for column in range(k):
                    if column != row and positions[chain, column] == candidate:
                        duplicate = True
                        break
                if duplicate:
                    continue

                candidates[chain, trial] = candidate
                weights[chain, trial] = matrix_weight
                valid[chain, trial] = True
        return candidates, weights, valid

    @njit(parallel=True, cache=True)
    def _sample_initial_bitstrings_numba(
        n_chains, nphi, ne, tk, rng_ints
    ):
        """Initialize fixed-Ne, fixed-tk chains without a sector table."""
        out = np.zeros(n_chains, dtype=np.uint64)
        n_attempts = rng_ints.shape[1]
        for chain in prange(n_chains):
            for attempt in range(n_attempts):
                orbitals = np.arange(nphi, dtype=np.int64)
                for particle in range(ne - 1):
                    draw = int(rng_ints[chain, attempt, particle]) % (
                        nphi - particle
                    )
                    swap = orbitals[particle]
                    orbitals[particle] = orbitals[particle + draw]
                    orbitals[particle + draw] = swap
                momentum = 0
                for particle in range(ne - 1):
                    momentum += orbitals[particle]
                last = (tk - momentum) % nphi
                duplicate = False
                for particle in range(ne - 1):
                    if orbitals[particle] == last:
                        duplicate = True
                        break
                if duplicate:
                    continue
                packed = 0
                for particle in range(ne - 1):
                    packed |= 1 << int(orbitals[particle])
                packed |= 1 << int(last)
                out[chain] = np.uint64(packed)
                break
        return out

    @njit(parallel=True, cache=True)
    def _scatter_bitstring_proposals_numba(
        confs, nphi, ne, tk, rng_ints
    ):
        """Symmetric two-particle scattering on packed fixed-sector states."""
        n_chains = confs.shape[0]
        n_attempts = rng_ints.shape[1]
        proposal = confs.copy()
        valid = np.zeros(n_chains, dtype=np.bool_)
        n_empty = nphi - ne
        for chain in prange(n_chains):
            conf = int(confs[chain])
            occupied = np.empty(ne, dtype=np.int64)
            empty = np.empty(n_empty, dtype=np.int64)
            occupied_count = 0
            empty_count = 0
            for orbital in range(nphi):
                if (conf >> orbital) & 1:
                    occupied[occupied_count] = orbital
                    occupied_count += 1
                else:
                    empty[empty_count] = orbital
                    empty_count += 1
            for attempt in range(n_attempts):
                first_slot = int(rng_ints[chain, attempt, 0]) % ne
                second_slot = int(rng_ints[chain, attempt, 1]) % ne
                if first_slot == second_slot:
                    continue
                first = occupied[first_slot]
                second = occupied[second_slot]
                target_momentum = (first + second) % nphi
                incoming = empty[
                    int(rng_ints[chain, attempt, 2]) % n_empty
                ]
                partner = (target_momentum - incoming) % nphi
                if partner == incoming or ((conf >> partner) & 1):
                    continue
                updated = conf
                updated &= ~(1 << int(first))
                updated &= ~(1 << int(second))
                updated |= 1 << int(incoming)
                updated |= 1 << int(partner)
                proposal[chain] = np.uint64(updated)
                valid[chain] = True
                break
        return proposal, valid
else:  # pragma: no cover - import compatibility without Numba.
    _momentum_scattering_proposals_numba = None
    _hamiltonian_pair_proposals_numba = None
    _hamiltonian_hpsi_candidates_numba = None
    _sample_initial_bitstrings_numba = None
    _scatter_bitstring_proposals_numba = None


def logprob_from_positions(
    params, sector_states, positions, net_cfg: NetworkConfig
):
    positions = jnp.atleast_2d(jnp.asarray(positions, dtype=jnp.int64))
    if (
        net_cfg.architecture == "mps"
        and net_cfg.mps_tensor_generation
        in {"backflow", "site_head_backflow"}
    ):
        phi_flat = deduplicated_wavefunction_matrix_from_positions(
            params, sector_states, positions, net_cfg
        )
    else:
        states = occupation_states_from_positions(
            sector_states, positions.reshape((-1,)), net_cfg
        )
        phi_flat = wavefunction_matrix(params, states, net_cfg)
    psi_mats = phi_flat.reshape(
        (positions.shape[0], net_cfg.k, net_cfg.k)
    )
    _, logabsdet = jnp.linalg.slogdet(psi_mats)
    return 2.0 * logabsdet


@partial(
    jax.jit,
    static_argnames=("net_cfg", "n_chains", "n_target_chains"),
)
def _sector_uniform_step_fn(
    params,
    sector_states,
    key,
    positions,
    logprob,
    chain_betas,
    net_cfg: NetworkConfig,
    n_chains: int,
    n_target_chains: int,
):
    row_key, cand_key, u_key = jax.random.split(key, 3)
    row = jax.random.randint(
        row_key,
        shape=(n_chains,),
        minval=0,
        maxval=net_cfg.k,
        dtype=jnp.int64,
    )
    candidate = jax.random.randint(
        cand_key,
        shape=(n_chains,),
        minval=0,
        maxval=sector_states.shape[0],
        dtype=jnp.int64,
    )

    chain_ids = jnp.arange(n_chains)
    proposal = positions.at[chain_ids, row].set(candidate)
    columns = jnp.arange(net_cfg.k)[None, :]
    duplicate_other_row = jnp.any(
        (positions == candidate[:, None]) & (columns != row[:, None]), axis=1
    )
    unchanged = candidate == positions[chain_ids, row]
    valid = (~duplicate_other_row) & (~unchanged)

    prop_logprob = logprob_from_positions(
        params, sector_states, proposal, net_cfg
    )
    logu = jnp.log(
        jax.random.uniform(u_key, shape=(n_chains,), dtype=jnp.float64)
    )
    accept = (logu < chain_betas * (prop_logprob - logprob)) & valid
    new_positions = jnp.where(accept[:, None], proposal, positions)
    new_logprob = jnp.where(accept, prop_logprob, logprob)
    accept_rate = jnp.sum(accept[:n_target_chains]) / jnp.maximum(
        1, n_target_chains
    )
    return new_positions, new_logprob, accept_rate


@partial(
    jax.jit,
    static_argnames=(
        "net_cfg",
        "n_chains",
        "n_target_chains",
        "n_burn",
        "n_between",
        "n_samples",
    ),
)
def _sector_uniform_sample_scan_fn(
    params,
    sector_states,
    key,
    positions,
    logprob,
    chain_betas,
    net_cfg: NetworkConfig,
    n_chains: int,
    n_target_chains: int,
    n_burn: int,
    n_between: int,
    n_samples: int,
):
    """Run a complete persistent-chain sample without host synchronization.

    The scalar ``float(rate)`` conversion in ``MetropolisSamplerJAX.step``
    synchronizes the host after every transition.  Keeping the same transition
    kernel inside one scan removes those round trips while preserving the PRNG
    split sequence and all proposal/acceptance rules.
    """

    def transition(carry, _):
        current_key, current_positions, current_logprob, accepted = carry
        current_key, subkey = jax.random.split(current_key, 2)
        current_positions, current_logprob, rate = _sector_uniform_step_fn(
            params,
            sector_states,
            subkey,
            current_positions,
            current_logprob,
            chain_betas,
            net_cfg,
            n_chains,
            n_target_chains,
        )
        accepted = accepted + rate * n_target_chains
        return (
            current_key,
            current_positions,
            current_logprob,
            accepted,
        ), rate

    initial = (
        key,
        positions,
        logprob,
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    burned, _ = jax.lax.scan(transition, initial, None, length=n_burn)

    def collect_sample(carry, _):
        next_carry, rates = jax.lax.scan(
            transition, carry, None, length=n_between
        )
        return next_carry, (
            next_carry[1][:n_target_chains],
            jnp.sum(rates),
        )

    final, (sampled_positions, sample_rate_sum) = jax.lax.scan(
        collect_sample, burned, None, length=n_samples
    )
    sample_rate = jnp.sum(sample_rate_sum) / jnp.maximum(
        1, n_samples * n_between
    )
    return (
        final[0],
        final[1],
        final[2],
        sampled_positions,
        sample_rate,
        final[3],
    )


@partial(jax.jit, static_argnames=("net_cfg",))
def _logprob_from_packed_batches(
    params, packed_batches, net_cfg: NetworkConfig, site_tensor_bank=None
):
    """Evaluate log probabilities after a CPU-side sector-table gather."""
    packed_batches = jnp.asarray(packed_batches, dtype=jnp.uint64)
    if packed_batches.ndim == 1:
        packed_batches = packed_batches[:, None]
    m, k = packed_batches.shape
    shifts = jnp.arange(net_cfg.num_sites, dtype=jnp.uint64)
    occupations = (
        (packed_batches[..., None] >> shifts) & jnp.uint64(1)
    ).astype(jnp.float32)
    decoded = occupations.reshape((m * k, net_cfg.num_sites))
    if site_tensor_bank is None:
        phi_flat = wavefunction_matrix(params, decoded, net_cfg)
    else:
        from .network import local_conv_site_wavefunction_from_bank
        phi_flat = local_conv_site_wavefunction_from_bank(
            site_tensor_bank, decoded, net_cfg
        )
    psi_mats = phi_flat.reshape((m, k, net_cfg.k))
    _, logabsdet = jnp.linalg.slogdet(psi_mats)
    return 2.0 * logabsdet


@partial(
    jax.jit,
    static_argnames=(
        "net_cfg", "n_chains", "n_target_chains", "nphi", "ne",
        "n_burn", "n_between", "n_samples",
    ),
)
def _conserved_pair_sample_scan_fn(
    params,
    key,
    positions,
    logprob,
    chain_betas,
    net_cfg: NetworkConfig,
    n_chains: int,
    n_target_chains: int,
    nphi: int,
    ne: int,
    n_burn: int,
    n_between: int,
    n_samples: int,
):
    """Run symmetric single-attempt pair scattering entirely on the GPU."""

    orbitals = jnp.arange(nphi, dtype=jnp.uint64)
    bit_masks = jnp.left_shift(jnp.uint64(1), orbitals)
    chain_ids = jnp.arange(n_chains)

    def select_slot(mask, slot):
        ranks = jnp.cumsum(mask.astype(jnp.int32), axis=1) - 1
        return jnp.argmax(
            mask & (ranks == slot[:, None]), axis=1
        ).astype(jnp.int32)

    def transition(carry, _):
        current_key, current_positions, current_logprob, accepted = carry
        current_key, first_key, second_key, empty_key, uniform_key = (
            jax.random.split(current_key, 5)
        )
        bits = current_positions.reshape((n_chains,))
        occupied_mask = (bits[:, None] & bit_masks[None, :]) != 0
        empty_mask = ~occupied_mask
        first_slot = jax.random.randint(
            first_key, (n_chains,), 0, ne, dtype=jnp.int32
        )
        second_slot = jax.random.randint(
            second_key, (n_chains,), 0, ne, dtype=jnp.int32
        )
        empty_slot = jax.random.randint(
            empty_key, (n_chains,), 0, nphi - ne, dtype=jnp.int32
        )
        first = select_slot(occupied_mask, first_slot)
        second = select_slot(occupied_mask, second_slot)
        incoming = select_slot(empty_mask, empty_slot)
        partner = (first + second - incoming) % nphi
        partner_mask = jnp.left_shift(
            jnp.uint64(1), partner.astype(jnp.uint64)
        )
        valid = (
            (first_slot != second_slot)
            & (partner != incoming)
            & ((bits & partner_mask) == 0)
        )
        outgoing = (
            jnp.left_shift(jnp.uint64(1), first.astype(jnp.uint64))
            | jnp.left_shift(jnp.uint64(1), second.astype(jnp.uint64))
        )
        incoming_pair = (
            jnp.left_shift(jnp.uint64(1), incoming.astype(jnp.uint64))
            | partner_mask
        )
        proposed_bits = (bits & ~outgoing) | incoming_pair
        proposed_bits = jnp.where(valid, proposed_bits, bits)
        proposal = proposed_bits[:, None]
        proposed_logprob = _logprob_from_packed_batches(
            params, proposal, net_cfg
        )
        logu = jnp.log(
            jax.random.uniform(
                uniform_key, (n_chains,), dtype=jnp.float64
            )
        )
        accept = (
            logu < chain_betas * (proposed_logprob - current_logprob)
        ) & valid
        next_positions = jnp.where(
            accept[:, None], proposal, current_positions
        )
        next_logprob = jnp.where(
            accept, proposed_logprob, current_logprob
        )
        rate = jnp.sum(accept[:n_target_chains]) / jnp.maximum(
            1, n_target_chains
        )
        return (
            current_key,
            next_positions,
            next_logprob,
            accepted + rate * n_target_chains,
        ), rate

    initial = (
        key,
        positions,
        logprob,
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    burned, _ = jax.lax.scan(transition, initial, None, length=n_burn)

    def collect_sample(carry, _):
        next_carry, rates = jax.lax.scan(
            transition, carry, None, length=n_between
        )
        return next_carry, (
            next_carry[1][:n_target_chains], jnp.sum(rates)
        )

    final, (sampled_positions, sample_rate_sum) = jax.lax.scan(
        collect_sample, burned, None, length=n_samples
    )
    sample_rate = jnp.sum(sample_rate_sum) / jnp.maximum(
        1, n_samples * n_between
    )
    return (
        final[0], final[1], final[2], sampled_positions,
        sample_rate, final[3],
    )


@partial(
    jax.jit,
    static_argnames=("net_cfg", "n_chains", "n_target_chains"),
)
def _accept_host_proposals_fn(
    params,
    sector_states,
    key,
    positions,
    logprob,
    proposal,
    valid,
    log_hastings,
    chain_betas,
    net_cfg: NetworkConfig,
    n_chains: int,
    n_target_chains: int,
):
    prop_logprob = logprob_from_positions(
        params, sector_states, proposal, net_cfg
    )
    logu = jnp.log(
        jax.random.uniform(key, shape=(n_chains,), dtype=jnp.float64)
    )
    accept = (
        logu < (chain_betas * (prop_logprob - logprob) + log_hastings)
    ) & valid
    new_positions = jnp.where(accept[:, None], proposal, positions)
    new_logprob = jnp.where(accept, prop_logprob, logprob)
    accept_rate = jnp.sum(accept[:n_target_chains]) / jnp.maximum(
        1, n_target_chains
    )
    return new_positions, new_logprob, accept_rate


def sample_initial_positions_cpu(
    rng: np.random.Generator,
    n_chains: int,
    sector_size: int,
    k: int,
) -> np.ndarray:
    """Sample distinct sector indices using O(n_chains * k) storage."""
    if n_chains <= 0:
        raise ValueError("n_chains must be positive.")
    if k <= 0 or k > sector_size:
        raise ValueError("k must lie in [1, sector_size].")
    if k == 1:
        return rng.integers(
            0, sector_size, size=(n_chains, 1), dtype=np.int64
        )

    positions = np.empty((n_chains, k), dtype=np.int64)
    for chain in range(n_chains):
        positions[chain] = rng.choice(sector_size, size=k, replace=False)
    return positions


class MetropolisSamplerJAX:
    """Metropolis sampler inside one momentum sector."""

    def __init__(
        self,
        params,
        sector_states,
        n_chains: int,
        net_cfg: NetworkConfig,
        seed: int = 1234,
        proposal_mode: str = "sector_uniform",
        sparse_rows=None,
        tempering_betas: tuple[float, ...] | None = None,
        tempering_swap_interval: int = 1,
        sector_states_host=None,
        sector_nphi: int | None = None,
        sector_n_particles: int | None = None,
        sector_on_host: bool = False,
        streaming_hamiltonian=None,
        streaming_basis=None,
        streaming_tk: int | None = None,
        streaming_row_nnz: int | None = None,
        hpsi_candidates: int = 4,
        sector_tk: int | None = None,
    ):
        self.params = params
        self.net_cfg = net_cfg
        # Occupation states are binary; keep the full basis table compact on
        # the accelerator and cast selected batches at the network boundary.
        # Reuse a caller-owned device array when available so training and the
        # sampler do not retain two complete copies of a large sector table.
        self.sector_states_host = None
        self.sector_on_host = bool(sector_on_host)
        self.sector_tk = sector_tk
        if self.sector_tk is not None:
            if proposal_mode != "conserved_pair":
                raise ValueError(
                    "sector-table-free sampling requires conserved_pair proposals."
                )
            if not self.sector_on_host or net_cfg.k != 1:
                raise ValueError(
                    "conserved_pair currently requires host-managed packed chains "
                    "and network.k == 1."
                )
            if not NUMBA_AVAILABLE:
                raise RuntimeError("conserved_pair requires Numba.")
        if self.sector_on_host:
            if proposal_mode not in {
                "sector_uniform",
                "conserved_pair",
                "hamiltonian_weighted",
                "hamiltonian_pair",
                "hamiltonian_hpsi_mtm",
            }:
                raise ValueError(
                    "CPU-resident sector states currently require "
                    "sector_uniform or hamiltonian_weighted proposals."
                )
            host_source = (
                sector_states_host
                if sector_states_host is not None
                else sector_states
            )
            host_packed = np.asarray(host_source, dtype=np.uint64)
            if host_packed.ndim != 1:
                raise ValueError(
                    "CPU-resident sector states must be packed uint64 values."
                )
            if sector_nphi is None or sector_n_particles is None:
                raise ValueError(
                    "CPU-resident sector states require sector_nphi and "
                    "sector_n_particles."
                )
            self.sector_states = None
            self.sector_states_host = host_packed
            sector_states_np = None
        elif isinstance(sector_states, jax.Array):
            packed_sector = sector_states.ndim == 1
            expected_dtype = jnp.uint64 if packed_sector else jnp.int8
            if sector_states.dtype != expected_dtype:
                raise ValueError(
                    "device sector_states must use uint64 packed storage or "
                    "int8 occupation storage."
                )
            self.sector_states = sector_states
            if sector_states_host is None:
                if packed_sector:
                    if sector_nphi is None or sector_n_particles is None:
                        raise ValueError(
                            "packed states without a host occupation table "
                            "require sector_nphi and sector_n_particles."
                        )
                    sector_states_np = None
                else:
                    sector_states_np = np.asarray(jax.device_get(sector_states))
            else:
                sector_states_np = np.asarray(sector_states_host)
                if sector_states_np.shape[0] != sector_states.shape[0]:
                    raise ValueError(
                        "sector_states_host length must match sector_states."
                    )
        else:
            sector_states_np = np.asarray(sector_states, dtype=np.int8)
            self.sector_states = jax.device_put(sector_states_np)
        if self.sector_tk is not None:
            self.sector_size = max(1, int(net_cfg.k))
        else:
            self.sector_size = int(
                self.sector_states_host.shape[0]
                if self.sector_on_host
                else sector_states.shape[0]
            )
            if self.sector_size < net_cfg.k:
                raise ValueError(
                    f"sector_size={self.sector_size} is smaller than k={net_cfg.k}."
                )
        self.n_target_chains = int(n_chains)
        betas = np.asarray(
            (1.0,) if tempering_betas is None else tempering_betas,
            dtype=np.float64,
        )
        if betas.ndim != 1 or betas.size == 0:
            raise ValueError("tempering_betas must be a non-empty rank-1 sequence.")
        if not np.isclose(betas[0], 1.0):
            raise ValueError("the first tempering beta must equal 1.")
        if np.any(betas <= 0.0) or np.any(betas > 1.0):
            raise ValueError("tempering betas must lie in (0, 1].")
        if np.any(np.diff(betas) >= 0.0):
            raise ValueError("tempering betas must be strictly decreasing.")
        if tempering_swap_interval <= 0:
            raise ValueError("tempering_swap_interval must be positive.")
        self.tempering_betas = betas
        self.n_replicas = int(betas.size)
        self.tempering_swap_interval = int(tempering_swap_interval)
        self.n_chains = self.n_target_chains * self.n_replicas
        self.chain_betas = jnp.asarray(
            np.repeat(betas, self.n_target_chains), dtype=jnp.float64
        )
        self.proposal_mode = proposal_mode
        if proposal_mode not in {
            "sector_uniform",
            "conserved_pair",
            "momentum_scattering",
            "laqx_hamiltonian",
            "hamiltonian_weighted",
            "hamiltonian_pair",
            "hamiltonian_hpsi_mtm",
        }:
            raise ValueError(
                f"Unsupported sampler proposal mode: {proposal_mode}"
            )
        if sector_states_np is None:
            self.nphi = int(sector_nphi)
            self.n_particles = int(sector_n_particles)
        else:
            self.nphi = int(sector_states_np.shape[1])
            particle_counts = np.sum(sector_states_np, axis=1).astype(np.int64)
            if not np.all(particle_counts == particle_counts[0]):
                raise ValueError("sector states must have a fixed particle count.")
            self.n_particles = int(particle_counts[0])
        self.allowed_outgoing_cpu = None
        self.laqx_proposal_cpu = None
        self.hamiltonian_pair_cpu = None
        self.hamiltonian_pair_sorted_bitstrings = None
        self.hamiltonian_pair_sorted_positions = None
        self.hpsi_candidates = int(hpsi_candidates)
        if self.hpsi_candidates < 2:
            raise ValueError("hpsi_candidates must be at least 2.")
        self.streaming_hamiltonian = streaming_hamiltonian
        self.streaming_basis = streaming_basis
        self.streaming_tk = streaming_tk
        self.streaming_row_nnz = streaming_row_nnz
        self.positions_host = None
        self.host_rng = np.random.default_rng(seed)
        self.cpu_proposal_backend = (
            "numba_parallel" if NUMBA_AVAILABLE else "numpy"
        )
        if proposal_mode == "momentum_scattering":
            if sector_states_np is None:
                raise ValueError(
                    "momentum_scattering requires a host occupation table."
                )
            if self.n_particles < 2:
                raise ValueError(
                    "momentum scattering requires at least two particles."
                )
            self.allowed_outgoing_cpu = build_allowed_outgoing_pairs_cpu(
                sector_states_np
            )
        elif proposal_mode == "laqx_hamiltonian":
            self.laqx_proposal_cpu = build_laqx_hamiltonian_proposal_table(
                sparse_rows
            )
            if self.laqx_proposal_cpu.cols.shape[0] != self.sector_size:
                raise ValueError(
                    "sparse Hamiltonian row count must match sector size."
                )
        elif proposal_mode == "hamiltonian_weighted":
            if streaming_hamiltonian is None or streaming_basis is None:
                raise ValueError(
                    "hamiltonian_weighted sampling requires a streaming "
                    "Hamiltonian and basis."
                )
            if streaming_tk is None:
                raise ValueError(
                    "hamiltonian_weighted sampling requires streaming_tk."
                )
        elif proposal_mode in {"hamiltonian_pair", "hamiltonian_hpsi_mtm"}:
            if streaming_hamiltonian is None or streaming_basis is None:
                raise ValueError(
                    "hamiltonian_pair sampling requires a streaming "
                    "Hamiltonian and basis."
                )
            if streaming_tk is None:
                raise ValueError(
                    "hamiltonian_pair sampling requires streaming_tk."
                )
            self.hamiltonian_pair_cpu = build_hamiltonian_pair_proposal_table(
                self.nphi,
                streaming_hamiltonian._require_coulomb_matrix(),
            )
            (
                self.hamiltonian_pair_sorted_bitstrings,
                self.hamiltonian_pair_sorted_positions,
            ) = streaming_hamiltonian._sector_sorted_arrays(
                int(streaming_tk), streaming_basis
            )
        self.key = jax.random.PRNGKey(seed)
        self.positions = None
        self.logprob = None
        self.n_accepted = 0
        self.n_proposed = 0
        self.n_swap_accepted = 0
        self.n_swap_proposed = 0
        self._local_step = 0
        self._swap_phase = 0
        self.reinitialize()

    def set_params(self, params, site_tensor_bank=None) -> None:
        self.params = params
        self.site_tensor_bank = site_tensor_bank
        self.refresh_logprob()

    def _split(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def refresh_logprob(self) -> None:
        if self.sector_on_host:
            packed = (
                np.asarray(self.positions_host, dtype=np.uint64)
                if self.sector_tk is not None
                else self.sector_states_host[self.positions_host]
            )
            self.logprob = _logprob_from_packed_batches(
                self.params, jnp.asarray(packed, dtype=jnp.uint64), self.net_cfg,
                self.site_tensor_bank,
            )
        else:
            self.logprob = logprob_from_positions(
                self.params, self.sector_states, self.positions, self.net_cfg
            )

    def reinitialize(self) -> None:
        self.site_tensor_bank = None
        if self.sector_tk is not None:
            rng_ints = self.host_rng.integers(
                0,
                2**31,
                size=(self.n_chains, 64, max(1, self.n_particles - 1)),
                dtype=np.int64,
            )
            bitstrings = _sample_initial_bitstrings_numba(
                self.n_chains,
                self.nphi,
                self.n_particles,
                self.sector_tk,
                rng_ints,
            )
            if np.any(bitstrings == 0):
                raise RuntimeError("Failed to initialize conserved-pair chains.")
            self.positions_host = bitstrings.reshape((-1, self.net_cfg.k))
            self.positions = jnp.asarray(self.positions_host, dtype=jnp.uint64)
        else:
            initial_positions = sample_initial_positions_cpu(
                self.host_rng,
                self.n_chains,
                self.sector_size,
                self.net_cfg.k,
            )
            self.positions = jnp.asarray(initial_positions, dtype=jnp.int64)
            if self.sector_on_host or self.proposal_mode in {
                "momentum_scattering",
                "laqx_hamiltonian",
                "hamiltonian_weighted",
                "hamiltonian_pair",
                "hamiltonian_hpsi_mtm",
            }:
                self.positions_host = initial_positions.copy()
            else:
                self.positions_host = None
        self.refresh_logprob()
        self.n_accepted = 0
        self.n_proposed = 0
        self.n_swap_accepted = 0
        self.n_swap_proposed = 0
        self._local_step = 0
        self._swap_phase = 0

    def step(self) -> float:
        if self.sector_tk is not None:
            current = np.asarray(self.positions_host, dtype=np.uint64).reshape(-1)
            # Draw exactly one scattering tuple per chain. Invalid tuples are
            # self-loops, which keeps q(x -> y) = q(y -> x) without a
            # state-dependent retry normalization or Hastings correction.
            rng_ints = self.host_rng.integers(
                0, 2**31, size=(current.size, 1, 3), dtype=np.int64
            )
            proposal_bits, valid = _scatter_bitstring_proposals_numba(
                current,
                self.nphi,
                self.n_particles,
                self.sector_tk,
                rng_ints,
            )
            prop_logprob = np.asarray(
                _logprob_from_packed_batches(
                    self.params,
                    jnp.asarray(proposal_bits, dtype=jnp.uint64),
                    self.net_cfg,
                    self.site_tensor_bank,
                )
            )
            logprob = np.array(self.logprob, dtype=np.float64, copy=True)
            logu = np.log(self.host_rng.random(self.n_chains))
            betas = np.repeat(self.tempering_betas, self.n_target_chains)
            accept = (logu < betas * (prop_logprob - logprob)) & valid
            reshaped = proposal_bits.reshape((-1, self.net_cfg.k))
            self.positions_host[accept] = reshaped[accept]
            logprob[accept] = prop_logprob[accept]
            self.positions = jnp.asarray(self.positions_host, dtype=jnp.uint64)
            self.logprob = jnp.asarray(logprob, dtype=jnp.float64)
            rate = float(np.mean(accept[: self.n_target_chains]))
            self._local_step += 1
            if (
                self.n_replicas > 1
                and self._local_step % self.tempering_swap_interval == 0
            ):
                self._attempt_replica_swaps()
            self.n_accepted += int(np.sum(accept[: self.n_target_chains]))
            self.n_proposed += self.n_target_chains
            return rate
        if (
            self.sector_on_host
            and self.proposal_mode == "sector_uniform"
        ):
            row = self.host_rng.integers(
                0, self.net_cfg.k, size=self.n_chains, dtype=np.int64
            )
            candidate = self.host_rng.integers(
                0, self.sector_size, size=self.n_chains, dtype=np.int64
            )
            chain_ids = np.arange(self.n_chains)
            proposal = self.positions_host.copy()
            duplicate_other_row = np.any(
                (self.positions_host == candidate[:, None])
                & (np.arange(self.net_cfg.k)[None, :] != row[:, None]),
                axis=1,
            )
            unchanged = candidate == self.positions_host[chain_ids, row]
            valid = (~duplicate_other_row) & (~unchanged)
            proposal[chain_ids, row] = candidate
            proposal_packed = self.sector_states_host[proposal]
            prop_logprob = np.asarray(
                _logprob_from_packed_batches(
                    self.params,
                    jnp.asarray(proposal_packed, dtype=jnp.uint64),
                    self.net_cfg,
                    self.site_tensor_bank,
                )
            )
            # JAX device buffers expose read-only NumPy views.  The accepted
            # chains are updated in place below, so materialize writable host
            # storage rather than relying on np.asarray.
            logprob = np.array(self.logprob, dtype=np.float64, copy=True)
            logu = np.log(self.host_rng.random(self.n_chains))
            betas = np.repeat(self.tempering_betas, self.n_target_chains)
            accept = (logu < betas * (prop_logprob - logprob)) & valid
            self.positions_host[accept] = proposal[accept]
            logprob[accept] = prop_logprob[accept]
            self.positions = jnp.asarray(self.positions_host, dtype=jnp.int64)
            self.logprob = jnp.asarray(logprob, dtype=jnp.float64)
            rate = float(np.mean(accept[: self.n_target_chains]))
            self._local_step += 1
            accepted = int(rate * self.n_target_chains)
            self.n_accepted += accepted
            self.n_proposed += self.n_target_chains
            return rate
        if self.proposal_mode in {
            "hamiltonian_weighted",
            "hamiltonian_pair",
            "hamiltonian_hpsi_mtm",
        }:
            if self.proposal_mode == "hamiltonian_hpsi_mtm":
                return self._hamiltonian_hpsi_mtm_step()
            if self.proposal_mode == "hamiltonian_pair":
                proposal, valid, log_hastings = (
                    self._hamiltonian_pair_proposals_cpu()
                )
            else:
                proposal, valid, log_hastings = (
                    self._hamiltonian_weighted_proposals_cpu()
                )
            proposal_packed = self.sector_states_host[proposal]
            prop_logprob = np.asarray(
                _logprob_from_packed_batches(
                    self.params,
                    jnp.asarray(proposal_packed, dtype=jnp.uint64),
                    self.net_cfg,
                )
            )
            logprob = np.array(self.logprob, dtype=np.float64, copy=True)
            logu = np.log(self.host_rng.random(self.n_chains))
            betas = np.repeat(self.tempering_betas, self.n_target_chains)
            accept = (
                logu
                < betas * (prop_logprob - logprob) + log_hastings
            ) & valid
            self.positions_host[accept] = proposal[accept]
            logprob[accept] = prop_logprob[accept]
            self.positions = jnp.asarray(self.positions_host, dtype=jnp.int64)
            self.logprob = jnp.asarray(logprob, dtype=jnp.float64)
            rate = float(np.mean(accept[: self.n_target_chains]))
            self._local_step += 1
            accepted = int(np.sum(accept[: self.n_target_chains]))
            self.n_accepted += accepted
            self.n_proposed += self.n_target_chains
            return rate
        self.key, subkey = jax.random.split(self.key, 2)
        if self.proposal_mode in {
            "momentum_scattering",
            "laqx_hamiltonian",
        }:
            if self.proposal_mode == "momentum_scattering":
                proposal, valid, log_hastings = (
                    self._momentum_scattering_proposals_cpu()
                )
            else:
                proposal, valid, log_hastings = (
                    self._laqx_hamiltonian_proposals_cpu()
                )
            self.positions, self.logprob, rate = _accept_host_proposals_fn(
                self.params,
                self.sector_states,
                subkey,
                self.positions,
                self.logprob,
                jnp.asarray(proposal, dtype=jnp.int64),
                jnp.asarray(valid, dtype=jnp.bool_),
                jnp.asarray(log_hastings, dtype=jnp.float64),
                self.chain_betas,
                self.net_cfg,
                self.n_chains,
                self.n_target_chains,
            )
            self.positions_host = np.asarray(self.positions).copy()
        else:
            self.positions, self.logprob, rate = _sector_uniform_step_fn(
                self.params,
                self.sector_states,
                subkey,
                self.positions,
                self.logprob,
                self.chain_betas,
                self.net_cfg,
                self.n_chains,
                self.n_target_chains,
            )
        self._local_step += 1
        if (
            self.n_replicas > 1
            and self._local_step % self.tempering_swap_interval == 0
        ):
            self._attempt_replica_swaps()
        accepted = int(float(rate) * self.n_target_chains)
        self.n_accepted += accepted
        self.n_proposed += self.n_target_chains
        return float(rate)

    def _attempt_replica_swaps(self) -> float:
        positions = np.asarray(self.positions).reshape(
            self.n_replicas,
            self.n_target_chains,
            self.net_cfg.k,
        ).copy()
        logprob = np.asarray(self.logprob).reshape(
            self.n_replicas, self.n_target_chains
        ).copy()
        accepted_total = 0
        proposed_total = 0
        for left in range(self._swap_phase, self.n_replicas - 1, 2):
            right = left + 1
            log_alpha = (
                (self.tempering_betas[left] - self.tempering_betas[right])
                * (logprob[right] - logprob[left])
            )
            accept = np.log(self.host_rng.random(self.n_target_chains)) < np.minimum(
                0.0, log_alpha
            )
            if np.any(accept):
                left_positions = positions[left, accept].copy()
                positions[left, accept] = positions[right, accept]
                positions[right, accept] = left_positions
                left_logprob = logprob[left, accept].copy()
                logprob[left, accept] = logprob[right, accept]
                logprob[right, accept] = left_logprob
            accepted_total += int(np.sum(accept))
            proposed_total += self.n_target_chains
        self._swap_phase = 1 - self._swap_phase
        self.n_swap_accepted += accepted_total
        self.n_swap_proposed += proposed_total
        flat_positions = positions.reshape(self.n_chains, self.net_cfg.k)
        position_dtype = jnp.uint64 if self.sector_tk is not None else jnp.int64
        self.positions = jnp.asarray(flat_positions, dtype=position_dtype)
        self.logprob = jnp.asarray(logprob.reshape(self.n_chains), dtype=jnp.float64)
        if self.proposal_mode in {
            "conserved_pair",
            "momentum_scattering",
            "laqx_hamiltonian",
            "hamiltonian_weighted",
            "hamiltonian_pair",
            "hamiltonian_hpsi_mtm",
        }:
            self.positions_host = flat_positions.copy()
        return accepted_total / max(1, proposed_total)

    def _momentum_scattering_proposals_cpu(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = self.allowed_outgoing_cpu
        positions = self.positions_host
        rows = self.host_rng.integers(
            0,
            self.net_cfg.k,
            size=self.n_chains,
            dtype=np.int64,
        )
        incoming_slots = self.host_rng.integers(
            0,
            table.occupied_pair_slots.shape[0],
            size=self.n_chains,
            dtype=np.int64,
        )
        outgoing_uniform = self.host_rng.random(self.n_chains)
        if NUMBA_AVAILABLE:
            return _momentum_scattering_proposals_numba(
                positions,
                rows,
                incoming_slots,
                outgoing_uniform,
                table.nphi,
                table.bitstrings,
                table.sorted_bitstrings,
                table.sorted_positions,
                table.occupied_orbitals,
                table.occupied_pair_slots,
                table.offsets,
                table.outgoing_masks,
            )
        return self._momentum_scattering_proposals_numpy(
            rows, incoming_slots, outgoing_uniform
        )

    def _laqx_hamiltonian_proposals_cpu(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw Hamiltonian-connected proposals with LaQX operator weights."""
        table = self.laqx_proposal_cpu
        positions = self.positions_host
        chain_ids = np.arange(self.n_chains)
        rows = self.host_rng.integers(
            0,
            self.net_cfg.k,
            size=self.n_chains,
            dtype=np.int64,
        )
        current = positions[chain_ids, rows]
        forward_sum = table.weight_sums[current]
        draws = self.host_rng.random(self.n_chains) * forward_sum
        slots = np.sum(
            table.cumulative_weights[current] < draws[:, None], axis=1
        )
        safe_slots = np.minimum(slots, table.cols.shape[1] - 1)
        candidate = table.cols[current, safe_slots]
        forward_weight = table.weights[current, safe_slots]

        reverse_sum = table.weight_sums[candidate]
        reverse_matches = table.cols[candidate] == current[:, None]
        reverse_weight = np.sum(
            table.weights[candidate] * reverse_matches, axis=1
        )
        columns = np.arange(self.net_cfg.k)[None, :]
        duplicate_other_row = np.any(
            (positions == candidate[:, None]) & (columns != rows[:, None]),
            axis=1,
        )
        valid = (
            (forward_sum > 0.0)
            & (forward_weight > 0.0)
            & (reverse_sum > 0.0)
            & (reverse_weight > 0.0)
            & (~duplicate_other_row)
        )
        proposal = positions.copy()
        proposal[chain_ids[valid], rows[valid]] = candidate[valid]
        log_hastings = np.zeros(self.n_chains, dtype=np.float64)
        log_hastings[valid] = (
            np.log(reverse_weight[valid])
            - np.log(reverse_sum[valid])
            - np.log(forward_weight[valid])
            + np.log(forward_sum[valid])
        )
        return proposal, valid, log_hastings

    def _hamiltonian_weighted_proposals_cpu(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stream |H_xy|-weighted proposals with an exact Hastings ratio."""
        positions = self.positions_host
        chain_ids = np.arange(self.n_chains)
        rows = self.host_rng.integers(
            0,
            self.net_cfg.k,
            size=self.n_chains,
            dtype=np.int64,
        )
        current = positions[chain_ids, rows]
        forward = self.streaming_hamiltonian.build_sparse_hamiltonian_rows_subset(
            int(self.streaming_tk),
            self.streaming_basis,
            current,
            pad_to_nnz=self.streaming_row_nnz,
        )
        forward_mask = np.asarray(forward.mask, dtype=np.bool_)
        forward_weights = np.where(
            forward_mask,
            np.abs(np.asarray(forward.vals, dtype=np.complex128)),
            0.0,
        )
        forward_sums = np.sum(forward_weights, axis=1)
        cumulative = np.cumsum(forward_weights, axis=1)
        draws = self.host_rng.random(self.n_chains) * forward_sums
        slots = np.sum(cumulative < draws[:, None], axis=1)
        safe_slots = np.minimum(slots, forward_weights.shape[1] - 1)
        candidate = np.asarray(forward.cols, dtype=np.int64)[
            chain_ids, safe_slots
        ]
        forward_weight = forward_weights[chain_ids, safe_slots]

        reverse = self.streaming_hamiltonian.build_sparse_hamiltonian_rows_subset(
            int(self.streaming_tk),
            self.streaming_basis,
            candidate,
            pad_to_nnz=self.streaming_row_nnz,
        )
        reverse_mask = np.asarray(reverse.mask, dtype=np.bool_)
        reverse_weights = np.where(
            reverse_mask,
            np.abs(np.asarray(reverse.vals, dtype=np.complex128)),
            0.0,
        )
        reverse_sums = np.sum(reverse_weights, axis=1)
        reverse_matches = (
            np.asarray(reverse.cols, dtype=np.int64) == current[:, None]
        ) & reverse_mask
        reverse_weight = np.sum(reverse_weights * reverse_matches, axis=1)

        columns = np.arange(self.net_cfg.k)[None, :]
        duplicate_other_row = np.any(
            (positions == candidate[:, None]) & (columns != rows[:, None]),
            axis=1,
        )
        valid = (
            (forward_sums > 0.0)
            & (forward_weight > 0.0)
            & (reverse_sums > 0.0)
            & (reverse_weight > 0.0)
            & (~duplicate_other_row)
            & (candidate != current)
        )
        proposal = positions.copy()
        proposal[chain_ids[valid], rows[valid]] = candidate[valid]
        log_hastings = np.zeros(self.n_chains, dtype=np.float64)
        log_hastings[valid] = (
            np.log(reverse_weight[valid])
            - np.log(reverse_sums[valid])
            - np.log(forward_weight[valid])
            + np.log(forward_sums[valid])
        )
        return proposal, valid, log_hastings

    def _hamiltonian_pair_proposals_cpu(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw one cheap antisymmetrized two-body event per chain."""
        if _hamiltonian_pair_proposals_numba is None:
            raise RuntimeError(
                "hamiltonian_pair sampling requires Numba."
            )
        table = self.hamiltonian_pair_cpu
        rows = self.host_rng.integers(
            0,
            self.net_cfg.k,
            size=self.n_chains,
            dtype=np.int64,
        )
        pair_slots = build_pair_slots(self.n_particles)
        incoming_slots = self.host_rng.integers(
            0,
            pair_slots.shape[0],
            size=self.n_chains,
            dtype=np.int64,
        )
        outgoing_uniform = self.host_rng.random(self.n_chains)
        return _hamiltonian_pair_proposals_numba(
            self.positions_host,
            rows,
            incoming_slots,
            outgoing_uniform,
            self.nphi,
            self.n_particles,
            self.sector_states_host,
            self.hamiltonian_pair_sorted_bitstrings,
            self.hamiltonian_pair_sorted_positions,
            pair_slots,
            table.pairs,
            table.pair_index,
            table.outgoing_indices,
            table.cumulative_weights,
            table.weight_sums,
            table.pair_weights,
        )

    def _hamiltonian_hpsi_candidates_cpu(
        self,
        positions: np.ndarray,
        rows: np.ndarray,
        n_trials: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw symmetric event candidates and their |H_xy| weights."""
        if _hamiltonian_hpsi_candidates_numba is None:
            raise RuntimeError(
                "hamiltonian_hpsi_mtm sampling requires Numba."
            )
        table = self.hamiltonian_pair_cpu
        pair_slots = build_pair_slots(self.n_particles)
        incoming_slots = self.host_rng.integers(
            0,
            pair_slots.shape[0],
            size=(self.n_chains, n_trials),
            dtype=np.int64,
        )
        outgoing_uniform = self.host_rng.random(
            (self.n_chains, n_trials)
        )
        return _hamiltonian_hpsi_candidates_numba(
            positions,
            rows,
            incoming_slots,
            outgoing_uniform,
            self.nphi,
            self.n_particles,
            self.sector_states_host,
            self.hamiltonian_pair_sorted_bitstrings,
            self.hamiltonian_pair_sorted_positions,
            pair_slots,
            table.pairs,
            table.pair_index,
            table.outgoing_indices,
            table.outgoing_counts,
            table.pair_weights,
        )

    def _hamiltonian_hpsi_candidate_logprob(
        self,
        positions: np.ndarray,
        rows: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """Evaluate all MTM candidates in one accelerator batch."""
        n_trials = candidates.shape[1]
        proposals = np.repeat(positions[:, None, :], n_trials, axis=1)
        chain_ids = np.arange(self.n_chains)[:, None]
        trial_ids = np.arange(n_trials)[None, :]
        proposals[chain_ids, trial_ids, rows[:, None]] = candidates
        packed = self.sector_states_host[proposals.reshape(-1, self.net_cfg.k)]
        logprob = _logprob_from_packed_batches(
            self.params,
            jnp.asarray(packed, dtype=jnp.uint64),
            self.net_cfg,
        )
        return np.asarray(logprob).reshape(self.n_chains, n_trials)

    def _hamiltonian_hpsi_mtm_step(self) -> float:
        """Exact multiple-try MH with candidate weights |H_xy psi(y)|."""
        if self.n_replicas != 1:
            raise ValueError(
                "hamiltonian_hpsi_mtm does not support parallel tempering."
            )
        positions = self.positions_host
        chain_ids = np.arange(self.n_chains)
        rows = self.host_rng.integers(
            0, self.net_cfg.k, size=self.n_chains, dtype=np.int64
        )
        current = positions[chain_ids, rows]
        current_logprob = np.asarray(self.logprob, dtype=np.float64)

        candidates, h_weight, valid = self._hamiltonian_hpsi_candidates_cpu(
            positions, rows, self.hpsi_candidates
        )
        candidate_logprob = self._hamiltonian_hpsi_candidate_logprob(
            positions, rows, candidates
        )
        log_weight = np.full(h_weight.shape, -np.inf, dtype=np.float64)
        positive = valid & (h_weight > 0.0) & np.isfinite(candidate_logprob)
        log_weight[positive] = (
            np.log(h_weight[positive]) + 0.5 * candidate_logprob[positive]
        )
        row_max = np.max(log_weight, axis=1)
        has_candidate = np.isfinite(row_max)
        scaled = np.zeros_like(log_weight)
        scaled[has_candidate] = np.exp(
            log_weight[has_candidate] - row_max[has_candidate, None]
        )
        weight_sum = np.sum(scaled, axis=1)
        draw = self.host_rng.random(self.n_chains) * weight_sum
        cumulative = np.cumsum(scaled, axis=1)
        selected_slot = np.sum(cumulative < draw[:, None], axis=1)
        selected_slot = np.minimum(selected_slot, self.hpsi_candidates - 1)
        selected = candidates[chain_ids, selected_slot]
        selected_h = h_weight[chain_ids, selected_slot]
        selected_logprob = candidate_logprob[chain_ids, selected_slot]

        proposed_positions = positions.copy()
        proposed_positions[chain_ids, rows] = selected
        reverse, reverse_h, reverse_valid = (
            self._hamiltonian_hpsi_candidates_cpu(
                proposed_positions, rows, self.hpsi_candidates
            )
        )
        reverse[:, -1] = current
        reverse_h[:, -1] = selected_h
        reverse_valid[:, -1] = has_candidate
        reverse_logprob = self._hamiltonian_hpsi_candidate_logprob(
            proposed_positions, rows, reverse
        )
        reverse_log_weight = np.full(
            reverse_h.shape, -np.inf, dtype=np.float64
        )
        reverse_positive = (
            reverse_valid
            & (reverse_h > 0.0)
            & np.isfinite(reverse_logprob)
        )
        reverse_log_weight[reverse_positive] = (
            np.log(reverse_h[reverse_positive])
            + 0.5 * reverse_logprob[reverse_positive]
        )

        forward_logsum = row_max + np.log(np.maximum(weight_sum, 1e-300))
        reverse_max = np.max(reverse_log_weight, axis=1)
        reverse_sum = np.zeros(self.n_chains, dtype=np.float64)
        finite_reverse = np.isfinite(reverse_max)
        reverse_sum[finite_reverse] = np.sum(
            np.exp(
                reverse_log_weight[finite_reverse]
                - reverse_max[finite_reverse, None]
            ),
            axis=1,
        )
        reverse_logsum = reverse_max + np.log(
            np.maximum(reverse_sum, 1e-300)
        )
        log_acceptance = (
            forward_logsum
            - reverse_logsum
            + 0.5 * (selected_logprob - current_logprob)
        )
        logu = np.log(self.host_rng.random(self.n_chains))
        accept = (
            has_candidate
            & np.isfinite(log_acceptance)
            & (logu < np.minimum(0.0, log_acceptance))
        )
        self.positions_host[accept] = proposed_positions[accept]
        updated_logprob = np.array(current_logprob, copy=True)
        updated_logprob[accept] = selected_logprob[accept]
        self.positions = jnp.asarray(self.positions_host, dtype=jnp.int64)
        self.logprob = jnp.asarray(updated_logprob, dtype=jnp.float64)
        rate = float(np.mean(accept[: self.n_target_chains]))
        self._local_step += 1
        self.n_accepted += int(np.sum(accept[: self.n_target_chains]))
        self.n_proposed += self.n_target_chains
        return rate

    def _momentum_scattering_proposals_numpy(
        self,
        rows: np.ndarray,
        incoming_slots: np.ndarray,
        outgoing_uniform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = self.allowed_outgoing_cpu
        positions = self.positions_host
        proposal = positions.copy()
        chain_ids = np.arange(self.n_chains)
        current_position = positions[chain_ids, rows]
        occupied = table.occupied_orbitals[current_position]
        pair_slots = table.occupied_pair_slots[incoming_slots]
        incoming_first = occupied[chain_ids, pair_slots[:, 0]]
        incoming_second = occupied[chain_ids, pair_slots[:, 1]]
        total_momentum = (
            incoming_first + incoming_second
        ) % table.nphi

        csr_rows = current_position * table.nphi + total_momentum
        starts = table.offsets[csr_rows]
        stops = table.offsets[csr_rows + 1]
        forward_count = stops - starts
        outgoing_slots = starts + np.floor(
            outgoing_uniform
            * np.maximum(forward_count, 1)
        ).astype(np.int64)
        if table.outgoing_masks.size:
            safe_slots = np.minimum(
                outgoing_slots, table.outgoing_masks.size - 1
            )
            outgoing_masks = table.outgoing_masks[safe_slots]
        else:
            outgoing_masks = np.zeros(self.n_chains, dtype=np.int64)

        incoming_masks = (
            np.left_shift(np.int64(1), incoming_first)
            | np.left_shift(np.int64(1), incoming_second)
        )
        proposed_bits = (
            table.bitstrings[current_position] & ~incoming_masks
        ) | outgoing_masks
        insertion = np.searchsorted(
            table.sorted_bitstrings, proposed_bits
        )
        safe_insertion = np.minimum(
            insertion, table.sorted_bitstrings.shape[0] - 1
        )
        candidate = table.sorted_positions[safe_insertion]
        found = (
            insertion < table.sorted_bitstrings.shape[0]
        ) & (
            table.sorted_bitstrings[safe_insertion] == proposed_bits
        )

        reverse_rows = candidate * table.nphi + total_momentum
        reverse_count = (
            table.offsets[reverse_rows + 1]
            - table.offsets[reverse_rows]
        )
        columns = np.arange(self.net_cfg.k)[None, :]
        duplicate_other_row = np.any(
            (positions == candidate[:, None])
            & (columns != rows[:, None]),
            axis=1,
        )
        valid = (
            (forward_count > 0)
            & found
            & (reverse_count > 0)
            & (~duplicate_other_row)
        )
        proposal[chain_ids[valid], rows[valid]] = candidate[valid]
        log_hastings = np.zeros(self.n_chains, dtype=np.float64)
        log_hastings[valid] = (
            np.log(forward_count[valid])
            - np.log(reverse_count[valid])
        )
        return proposal, valid, log_hastings

    def sample(
        self, n_burn: int, n_between: int, n_samples: int
    ) -> tuple[np.ndarray, float]:
        use_conserved_scan = (
            os.environ.get("NQS_CONSERVED_PAIR_SCAN") == "1"
            and self.sector_tk is not None
            and self.n_replicas == 1
            and self.net_cfg.k == 1
        )
        if use_conserved_scan:
            (
                self.key,
                self.positions,
                self.logprob,
                sampled_positions,
                accept,
                accepted_total,
            ) = _conserved_pair_sample_scan_fn(
                self.params,
                self.key,
                self.positions,
                self.logprob,
                self.chain_betas,
                self.net_cfg,
                self.n_chains,
                self.n_target_chains,
                self.nphi,
                self.n_particles,
                int(n_burn),
                int(n_between),
                int(n_samples),
            )
            sampled_positions, accept, accepted_total, final_positions = (
                jax.device_get((
                    sampled_positions, accept, accepted_total, self.positions
                ))
            )
            self.positions_host = np.asarray(
                final_positions, dtype=np.uint64
            ).reshape((-1, self.net_cfg.k))
            transition_count = n_burn + n_between * n_samples
            self._local_step += transition_count
            self.n_accepted += int(round(float(accepted_total)))
            self.n_proposed += transition_count * self.n_target_chains
            return (
                np.asarray(sampled_positions, dtype=np.uint64).reshape(
                    (-1, self.net_cfg.k)
                ),
                float(accept),
            )

        use_scan = (
            os.environ.get("NQS_SAMPLER_SCAN") == "1"
            and not self.sector_on_host
            and self.proposal_mode == "sector_uniform"
            and self.n_replicas == 1
        )
        if use_scan:
            (
                self.key,
                self.positions,
                self.logprob,
                sampled_positions,
                accept,
                accepted_total,
            ) = _sector_uniform_sample_scan_fn(
                self.params,
                self.sector_states,
                self.key,
                self.positions,
                self.logprob,
                self.chain_betas,
                self.net_cfg,
                self.n_chains,
                self.n_target_chains,
                int(n_burn),
                int(n_between),
                int(n_samples),
            )
            sampled_positions, accept, accepted_total = jax.device_get(
                (sampled_positions, accept, accepted_total)
            )
            transition_count = n_burn + n_between * n_samples
            self._local_step += transition_count
            self.n_accepted += int(round(float(accepted_total)))
            self.n_proposed += transition_count * self.n_target_chains
            return (
                np.asarray(sampled_positions, dtype=np.int64).reshape(
                    (-1, self.net_cfg.k)
                ),
                float(accept),
            )
        for _ in range(n_burn):
            self.step()
        positions = []
        accepts = []
        for _ in range(n_samples):
            for _ in range(n_between):
                accepts.append(self.step())
            positions.append(
                np.asarray(self.positions)[: self.n_target_chains]
            )
        positions_out = (
            np.concatenate(positions, axis=0)
            if positions
            else np.empty((0,), dtype=np.int64)
        )
        accept = float(np.mean(accepts)) if accepts else 0.0
        return positions_out, accept

    def acceptance_rate(self) -> float:
        return self.n_accepted / max(1, self.n_proposed)

    def swap_acceptance_rate(self) -> float:
        return self.n_swap_accepted / max(1, self.n_swap_proposed)
