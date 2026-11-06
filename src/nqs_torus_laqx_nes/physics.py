"""Torus basis and Landau-gauge Coulomb Hamiltonian construction."""

from __future__ import annotations

import itertools
import math
import operator
import os
from dataclasses import dataclass

import numpy as np

from .config import SystemConfig

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration.
    njit = None
    prange = range
    NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class BasisData:
    occupations: np.ndarray
    bitstrings: np.ndarray
    index_tk: np.ndarray
    number_tk: np.ndarray
    completion_counts: object = None

    @property
    def size(self) -> int:
        return int(self.bitstrings.shape[0])


@dataclass(frozen=True)
class SparseHamiltonianRows:
    diag: object
    cols: object
    vals: object
    mask: object

    def as_tuple(self) -> tuple[object, object, object, object]:
        return self.diag, self.cols, self.vals, self.mask

    def to_jax(self) -> "SparseHamiltonianRows":
        import jax.numpy as jnp

        return SparseHamiltonianRows(
            jnp.asarray(self.diag, dtype=jnp.complex128),
            jnp.asarray(self.cols, dtype=jnp.int64),
            jnp.asarray(self.vals, dtype=jnp.complex128),
            jnp.asarray(self.mask, dtype=jnp.float64),
        )


@dataclass(frozen=True)
class SparseHamiltonianBitstringRows:
    """Streamed Hamiltonian rows whose off-diagonal targets are bitstrings."""

    diag: object
    neighbor_bitstrings: object
    vals: object
    mask: object


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _pack_sparse_bitstring_rows_numba(
        neighbors, vals, counts, offsets, packed_neighbors, packed_vals
    ):
        for row in prange(neighbors.shape[0]):
            destination = offsets[row]
            for slot in range(counts[row]):
                packed_neighbors[destination + slot] = neighbors[row, slot]
                packed_vals[destination + slot] = vals[row, slot]


def pack_sparse_bitstring_rows(
    rows, capacity: int, value_dtype=np.complex128
):
    """Pack valid fixed-width row entries while preserving row-major order."""

    neighbors = np.asarray(rows.neighbor_bitstrings, dtype=np.uint64)
    vals = np.asarray(rows.vals, dtype=np.complex128)
    mask = np.asarray(rows.mask)
    source_width = neighbors.shape[-1]
    neighbors_2d = neighbors.reshape(-1, source_width)
    vals_2d = vals.reshape(-1, source_width)
    counts = np.count_nonzero(mask.reshape(-1, source_width), axis=1).astype(
        np.int32, copy=False
    )
    offsets = np.empty(counts.size + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    valid_count = int(offsets[-1])
    if valid_count > capacity:
        raise ValueError(
            "valid neighbor count exceeds compact capacity: "
            f"{valid_count} > {capacity}"
        )
    packed_neighbors = np.zeros(capacity, dtype=np.uint64)
    packed_vals = np.zeros(capacity, dtype=value_dtype)
    if NUMBA_AVAILABLE:
        _pack_sparse_bitstring_rows_numba(
            neighbors_2d,
            vals_2d,
            counts,
            offsets,
            packed_neighbors,
            packed_vals,
        )
    else:  # pragma: no cover - optional Numba is installed in production.
        for row, count in enumerate(counts):
            destination = offsets[row]
            packed_neighbors[destination : destination + count] = (
                neighbors_2d[row, :count]
            )
            packed_vals[destination : destination + count] = vals_2d[
                row, :count
            ]
    return packed_neighbors, packed_vals, offsets


def _validate_packed_nphi(nphi: int) -> None:
    if not 1 <= nphi <= 63:
        raise ValueError("Packed bitstrings require 1 <= Nphi <= 63.")


def bitstrings_to_occupations(bitstrings: np.ndarray, num_sites: int) -> np.ndarray:
    _validate_packed_nphi(num_sites)
    bitstrings = np.asarray(bitstrings, dtype=np.int64)
    shifts = np.arange(num_sites, dtype=np.int64)
    return ((bitstrings[:, None] >> shifts) & 1).astype(np.float64)


def count_ones_from(conf: int, start: int) -> int:
    return int(conf >> start).bit_count()


def occupation(conf: int, i: int) -> int:
    return (int(conf) >> i) & 1


def build_basis(cfg: SystemConfig) -> BasisData:
    _validate_packed_nphi(cfg.nphi)
    index = np.array(
        list(itertools.combinations(range(cfg.nphi), cfg.ne)), dtype=np.int64
    )
    base_number = index.shape[0]
    index_tk = np.zeros((cfg.nphi, base_number), dtype=np.int64)
    number_tk = np.zeros(cfg.nphi, dtype=np.int64)
    bitstrings = np.zeros(base_number, dtype=np.int64)

    for i in range(base_number):
        total_k = 0
        conf = 0
        for orb in index[i]:
            total_k += int(orb)
            conf |= 1 << int(orb)
        sector = total_k % cfg.nphi
        bitstrings[i] = conf
        index_tk[sector, number_tk[sector]] = i
        number_tk[sector] += 1
    return BasisData(index, bitstrings, index_tk, number_tk)


def _sector_completion_counts(nphi: int, ne: int) -> np.ndarray:
    """Count fixed-cardinality suffixes by momentum modulo ``nphi``."""
    counts = np.zeros((nphi + 1, ne + 1, nphi), dtype=np.int64)
    counts[nphi, 0, 0] = 1
    for start in range(nphi - 1, -1, -1):
        counts[start] = counts[start + 1]
        max_particles = min(ne, nphi - start)
        for particles in range(1, max_particles + 1):
            for momentum in range(nphi):
                counts[start, particles, momentum] += counts[
                    start + 1,
                    particles - 1,
                    (momentum - start) % nphi,
                ]
    return counts


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _unrank_one_sector_combination_numba(
        rank: int,
        nphi: int,
        ne: int,
        tk: int,
        completion_counts: np.ndarray,
        occupations: np.ndarray,
        bitstrings: np.ndarray,
    ):
        residual_rank = rank
        start = 0
        particles_left = ne
        momentum = 0
        bitstring = 0
        for position in range(ne):
            max_orbital = nphi - particles_left
            selected = start
            for orbital in range(start, max_orbital + 1):
                remaining_momentum = (tk - momentum - orbital) % nphi
                ways = completion_counts[
                    orbital + 1,
                    particles_left - 1,
                    remaining_momentum,
                ]
                if residual_rank < ways:
                    selected = orbital
                    break
                residual_rank -= ways
            occupations[rank, position] = selected
            bitstring |= 1 << selected
            momentum = (momentum + selected) % nphi
            start = selected + 1
            particles_left -= 1
        bitstrings[rank] = bitstring

    @njit(parallel=True, cache=True)
    def _unrank_sector_combinations_numba(
        nphi: int,
        ne: int,
        tk: int,
        completion_counts: np.ndarray,
        sector_size: int,
    ):
        occupations = np.empty((sector_size, ne), dtype=np.int64)
        bitstrings = np.empty(sector_size, dtype=np.int64)
        for rank in prange(sector_size):
            _unrank_one_sector_combination_numba(
                rank,
                nphi,
                ne,
                tk,
                completion_counts,
                occupations,
                bitstrings,
            )
        return occupations, bitstrings


def build_sector_basis(cfg: SystemConfig, tk: int) -> BasisData:
    """Build only one momentum sector without materializing the full basis."""
    _validate_packed_nphi(cfg.nphi)
    tk = int(tk) % cfg.nphi
    completion_counts = _sector_completion_counts(cfg.nphi, cfg.ne)
    sector_size = int(completion_counts[0, cfg.ne, tk])
    materialize = os.environ.get("NQS_MATERIALIZE_SECTOR", "1") == "1"
    if not materialize:
        occupations = None
        bitstrings = np.empty(0, dtype=np.int64)
    elif NUMBA_AVAILABLE:
        occupations, bitstrings = _unrank_sector_combinations_numba(
            cfg.nphi,
            cfg.ne,
            tk,
            completion_counts,
            sector_size,
        )
    else:  # pragma: no cover - the production environment includes Numba.
        selected = [
            combination
            for combination in itertools.combinations(range(cfg.nphi), cfg.ne)
            if sum(combination) % cfg.nphi == tk
        ]
        occupations = np.asarray(selected, dtype=np.int64)
        bitstrings = np.asarray(
            [sum(1 << int(orbital) for orbital in row) for row in occupations],
            dtype=np.int64,
        )

    number_tk = np.zeros(cfg.nphi, dtype=np.int64)
    number_tk[tk] = sector_size
    index_tk = (
        np.arange(sector_size, dtype=np.int64)[None, :]
        if materialize
        else np.zeros((1, 0), dtype=np.int64)
    )
    return BasisData(
        occupations,
        bitstrings,
        index_tk,
        number_tk,
        completion_counts,
    )


def sector_basis_bitstrings(basis: BasisData, tk: int) -> np.ndarray:
    if basis.index_tk.size == 0:
        return basis.bitstrings.astype(np.int64, copy=False)
    index_row = tk if basis.index_tk.shape[0] > 1 else 0
    basis_ids = basis.index_tk[index_row, : basis.number_tk[tk]]
    if (
        basis.index_tk.shape[0] == 1
        and basis_ids.size == basis.bitstrings.size
        and np.array_equal(basis_ids, np.arange(basis_ids.size))
    ):
        return basis.bitstrings.astype(np.int64, copy=False)
    return basis.bitstrings[basis_ids].astype(np.int64, copy=True)


def build_sector_lookup(basis: BasisData, tk: int) -> dict[int, int]:
    bitstrings = sector_basis_bitstrings(basis, tk)
    return {int(conf): int(i) for i, conf in enumerate(bitstrings)}


def generate_jlist(nphi: int) -> np.ndarray:
    out = []
    for j1 in range(nphi):
        for j2 in range(j1 + 1, nphi):
            for j3 in range(nphi):
                for j4 in range(j3 + 1, nphi):
                    if (j1 + j2 - j3 - j4) % nphi == 0:
                        out.append((j1, j2, j3, j4))
    return np.asarray(out, dtype=np.int64)


def landau_level_form_factor_polynomial(
    landau_level: int, x: float
) -> float:
    """Return the ordinary Laguerre polynomial L_n(x)."""
    if landau_level < 0:
        raise ValueError("landau_level must be non-negative.")
    if landau_level == 0:
        return 1.0
    previous = 1.0
    current = 1.0 - x
    for order in range(2, landau_level + 1):
        following = (
            (2.0 * order - 1.0 - x) * current
            - (order - 1.0) * previous
        ) / order
        previous, current = current, following
    return current


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _landau_level_form_factor_polynomial_numba(landau_level, x):
        if landau_level == 0:
            return 1.0
        previous = 1.0
        current = 1.0 - x
        for order in range(2, landau_level + 1):
            following = (
                (2.0 * order - 1.0 - x) * current
                - (order - 1.0) * previous
            ) / order
            previous = current
            current = following
        return current

    @njit(cache=True, fastmath=True, parallel=True)
    def _generate_coulomb_matrix_numba(
        nphi, ratio_rect, cutoff, landau_level
    ):
        coulomb_mat = np.zeros((nphi, nphi, nphi, nphi), dtype=np.complex128)
        pref = math.sqrt(2.0 * math.pi / (nphi * ratio_rect))
        total = nphi * nphi * nphi * nphi
        for flat in prange(total):
            j4 = flat % nphi
            tmp = flat // nphi
            j3 = tmp % nphi
            tmp = tmp // nphi
            j2 = tmp % nphi
            j1 = tmp // nphi
            if j1 == j2 or j3 == j4 or (j1 + j2 - j3 - j4) % nphi != 0:
                continue
            vq_re = 0.0
            vq_im = 0.0
            for t in range(-cutoff, cutoff):
                if (j1 - j4 - t) % nphi != 0:
                    continue
                rt = ratio_rect * t
                for s in range(-cutoff, cutoff):
                    if t == 0 and s == 0:
                        continue
                    qlb = pref * math.sqrt(s * s + rt * rt)
                    laguerre = _landau_level_form_factor_polynomial_numba(
                        landau_level, 0.5 * qlb * qlb
                    )
                    amp = (
                        0.5
                        / (qlb * nphi)
                        * laguerre
                        * laguerre
                        * math.exp(-0.5 * qlb * qlb)
                    )
                    phase = 2.0 * math.pi * s * (j1 - j3) / nphi
                    vq_re += amp * math.cos(phase)
                    vq_im -= amp * math.sin(phase)
            coulomb_mat[j1, j2, j3, j4] = vq_re + 1j * vq_im
        return coulomb_mat

    @njit(cache=True)
    def _count_ones_from_numba(conf, start):
        work = np.uint64(conf) >> np.uint64(start)
        count = 0
        while work != 0:
            count += int(work & np.uint64(1))
            work >>= np.uint64(1)
        return count

    @njit(cache=True)
    def _find_sector_position_numba(
        conf, sorted_bitstrings, sorted_positions
    ):
        position = np.searchsorted(sorted_bitstrings, conf)
        if (
            position < sorted_bitstrings.shape[0]
            and sorted_bitstrings[position] == conf
        ):
            return sorted_positions[position]
        return -1

    @njit(cache=True, parallel=True)
    def _build_sparse_rows_subset_numba(
        row_indices,
        sector_bitstrings,
        sorted_bitstrings,
        sorted_positions,
        jlist,
        coulomb_mat,
    ):
        n_rows = row_indices.shape[0]
        max_nnz_bound = jlist.shape[0]
        diag = np.zeros(n_rows, dtype=np.complex128)
        cols = np.zeros((n_rows, max_nnz_bound), dtype=np.int64)
        vals = np.zeros((n_rows, max_nnz_bound), dtype=np.complex128)
        counts = np.zeros(n_rows, dtype=np.int64)

        for row_flat in prange(n_rows):
            row_index = int(row_indices[row_flat])
            conf = int(sector_bitstrings[row_index])
            local_cols = np.full(max_nnz_bound, -1, dtype=np.int64)
            local_vals = np.zeros(max_nnz_bound, dtype=np.complex128)
            local_nnz = 0
            diag_coeff = 0.0j

            for term_idx in range(jlist.shape[0]):
                j1 = int(jlist[term_idx, 0])
                j2 = int(jlist[term_idx, 1])
                j3 = int(jlist[term_idx, 2])
                j4 = int(jlist[term_idx, 3])
                if ((conf >> j3) & 1) == 0 or ((conf >> j4) & 1) == 0:
                    continue

                sign = 1
                newconf = conf
                if _count_ones_from_numba(newconf, j4 + 1) % 2:
                    sign = -sign
                newconf &= ~(1 << j4)
                if _count_ones_from_numba(newconf, j3 + 1) % 2:
                    sign = -sign
                newconf &= ~(1 << j3)
                if ((newconf >> j1) & 1) != 0 or ((newconf >> j2) & 1) != 0:
                    continue
                if _count_ones_from_numba(newconf, j2 + 1) % 2:
                    sign = -sign
                newconf |= 1 << j2
                if _count_ones_from_numba(newconf, j1 + 1) % 2:
                    sign = -sign
                newconf |= 1 << j1

                col_idx = _find_sector_position_numba(
                    np.int64(newconf), sorted_bitstrings, sorted_positions
                )
                if col_idx < 0:
                    continue

                vq = (
                    coulomb_mat[j1, j2, j3, j4]
                    - coulomb_mat[j2, j1, j3, j4]
                    - coulomb_mat[j1, j2, j4, j3]
                    + coulomb_mat[j2, j1, j4, j3]
                )
                hij = np.conjugate(vq * sign)
                if np.abs(hij) <= 1e-14:
                    continue
                if col_idx == row_index:
                    diag_coeff += hij
                    continue

                found = -1
                for existing in range(local_nnz):
                    if local_cols[existing] == col_idx:
                        found = existing
                        break
                if found >= 0:
                    local_vals[found] += hij
                else:
                    local_cols[local_nnz] = col_idx
                    local_vals[local_nnz] = hij
                    local_nnz += 1

            write = 0
            for existing in range(local_nnz):
                value = local_vals[existing]
                if np.abs(value) > 1e-14:
                    cols[row_flat, write] = local_cols[existing]
                    vals[row_flat, write] = value
                    write += 1
            diag[row_flat] = diag_coeff
            counts[row_flat] = write
        return diag, cols, vals, counts

    @njit(cache=True, parallel=True)
    def _build_sparse_bitstring_rows_numba(
        row_bitstrings,
        jlist,
        coulomb_mat,
    ):
        n_rows = row_bitstrings.shape[0]
        max_nnz_bound = jlist.shape[0]
        diag = np.zeros(n_rows, dtype=np.complex128)
        neighbors = np.zeros((n_rows, max_nnz_bound), dtype=np.uint64)
        vals = np.zeros((n_rows, max_nnz_bound), dtype=np.complex128)
        counts = np.zeros(n_rows, dtype=np.int64)

        for row_flat in prange(n_rows):
            conf = int(row_bitstrings[row_flat])
            local_neighbors = np.zeros(max_nnz_bound, dtype=np.uint64)
            local_vals = np.zeros(max_nnz_bound, dtype=np.complex128)
            local_nnz = 0
            diag_coeff = 0.0j

            for term_idx in range(jlist.shape[0]):
                j1 = int(jlist[term_idx, 0])
                j2 = int(jlist[term_idx, 1])
                j3 = int(jlist[term_idx, 2])
                j4 = int(jlist[term_idx, 3])
                if ((conf >> j3) & 1) == 0 or ((conf >> j4) & 1) == 0:
                    continue

                sign = 1
                newconf = conf
                if _count_ones_from_numba(newconf, j4 + 1) % 2:
                    sign = -sign
                newconf &= ~(1 << j4)
                if _count_ones_from_numba(newconf, j3 + 1) % 2:
                    sign = -sign
                newconf &= ~(1 << j3)
                if ((newconf >> j1) & 1) != 0 or ((newconf >> j2) & 1) != 0:
                    continue
                if _count_ones_from_numba(newconf, j2 + 1) % 2:
                    sign = -sign
                newconf |= 1 << j2
                if _count_ones_from_numba(newconf, j1 + 1) % 2:
                    sign = -sign
                newconf |= 1 << j1

                vq = (
                    coulomb_mat[j1, j2, j3, j4]
                    - coulomb_mat[j2, j1, j3, j4]
                    - coulomb_mat[j1, j2, j4, j3]
                    + coulomb_mat[j2, j1, j4, j3]
                )
                hij = np.conjugate(vq * sign)
                if np.abs(hij) <= 1e-14:
                    continue
                if newconf == conf:
                    diag_coeff += hij
                    continue

                neighbor = np.uint64(newconf)
                found = -1
                for existing in range(local_nnz):
                    if local_neighbors[existing] == neighbor:
                        found = existing
                        break
                if found >= 0:
                    local_vals[found] += hij
                else:
                    local_neighbors[local_nnz] = neighbor
                    local_vals[local_nnz] = hij
                    local_nnz += 1

            write = 0
            for existing in range(local_nnz):
                value = local_vals[existing]
                if np.abs(value) > 1e-14:
                    neighbors[row_flat, write] = local_neighbors[existing]
                    vals[row_flat, write] = value
                    write += 1
            diag[row_flat] = diag_coeff
            counts[row_flat] = write
        return diag, neighbors, vals, counts


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _parity_numba(conf, start):
        x = np.uint64(conf) >> np.uint64(start)
        x ^= x >> np.uint64(32)
        x ^= x >> np.uint64(16)
        x ^= x >> np.uint64(8)
        x ^= x >> np.uint64(4)
        x ^= x >> np.uint64(2)
        x ^= x >> np.uint64(1)
        return int(x & np.uint64(1))

    @njit(cache=True, parallel=True)
    def _build_sparse_bitstring_rows_numba_fast(
        row_bitstrings, j1_arr, j2_arr, j3_arr, j4_arr, coulomb_vals
    ):
        n_rows = row_bitstrings.shape[0]
        n_terms = j1_arr.shape[0]
        diag = np.zeros(n_rows, dtype=np.complex128)
        neighbors = np.zeros((n_rows, n_terms), dtype=np.uint64)
        vals = np.zeros((n_rows, n_terms), dtype=np.complex128)
        counts = np.zeros(n_rows, dtype=np.int64)

        for row_flat in prange(n_rows):
            conf = int(row_bitstrings[row_flat])
            local_neighbors = np.zeros(n_terms, dtype=np.uint64)
            local_vals = np.zeros(n_terms, dtype=np.complex128)
            local_nnz = 0
            diag_coeff = 0.0j

            for term_idx in range(n_terms):
                j1 = j1_arr[term_idx]
                j2 = j2_arr[term_idx]
                j3 = j3_arr[term_idx]
                j4 = j4_arr[term_idx]
                if ((conf >> j3) & 1) == 0 or ((conf >> j4) & 1) == 0:
                    continue

                sign = 1
                newconf = conf
                if _parity_numba(newconf, j4 + 1):
                    sign = -sign
                newconf &= ~(1 << j4)
                if _parity_numba(newconf, j3 + 1):
                    sign = -sign
                newconf &= ~(1 << j3)
                if ((newconf >> j1) & 1) != 0 or ((newconf >> j2) & 1) != 0:
                    continue
                if _parity_numba(newconf, j2 + 1):
                    sign = -sign
                newconf |= 1 << j2
                if _parity_numba(newconf, j1 + 1):
                    sign = -sign
                newconf |= 1 << j1

                hij = np.conjugate(coulomb_vals[term_idx] * sign)
                if np.abs(hij) <= 1e-14:
                    continue
                if newconf == conf:
                    diag_coeff += hij
                    continue

                neighbor = np.uint64(newconf)
                found = -1
                for existing in range(local_nnz):
                    if local_neighbors[existing] == neighbor:
                        found = existing
                        break
                if found >= 0:
                    local_vals[found] += hij
                else:
                    local_neighbors[local_nnz] = neighbor
                    local_vals[local_nnz] = hij
                    local_nnz += 1

            write = 0
            for existing in range(local_nnz):
                value = local_vals[existing]
                if np.abs(value) > 1e-14:
                    neighbors[row_flat, write] = local_neighbors[existing]
                    vals[row_flat, write] = value
                    write += 1
            diag[row_flat] = diag_coeff
            counts[row_flat] = write
        return diag, neighbors, vals, counts


    @njit(cache=True, parallel=True)
    def _build_sparse_bitstring_rows_numba_fixed(
        row_bitstrings,
        j1_arr,
        j2_arr,
        j3_arr,
        j4_arr,
        coulomb_vals,
        row_width,
    ):
        """Build directly into the requested fixed width without n_terms scratch."""
        n_rows = row_bitstrings.shape[0]
        n_terms = j1_arr.shape[0]
        diag = np.zeros(n_rows, dtype=np.complex128)
        neighbors = np.zeros((n_rows, row_width), dtype=np.uint64)
        vals = np.zeros((n_rows, row_width), dtype=np.complex128)
        counts = np.zeros(n_rows, dtype=np.int64)
        overflow = np.zeros(n_rows, dtype=np.uint8)

        for row_flat in prange(n_rows):
            conf = int(row_bitstrings[row_flat])
            local_nnz = 0
            diag_coeff = 0.0j

            for term_idx in range(n_terms):
                j1 = j1_arr[term_idx]
                j2 = j2_arr[term_idx]
                j3 = j3_arr[term_idx]
                j4 = j4_arr[term_idx]
                if ((conf >> j3) & 1) == 0 or ((conf >> j4) & 1) == 0:
                    continue

                sign = 1
                newconf = conf
                if _parity_numba(newconf, j4 + 1):
                    sign = -sign
                newconf &= ~(1 << j4)
                if _parity_numba(newconf, j3 + 1):
                    sign = -sign
                newconf &= ~(1 << j3)
                if ((newconf >> j1) & 1) != 0 or ((newconf >> j2) & 1) != 0:
                    continue
                if _parity_numba(newconf, j2 + 1):
                    sign = -sign
                newconf |= 1 << j2
                if _parity_numba(newconf, j1 + 1):
                    sign = -sign
                newconf |= 1 << j1

                hij = np.conjugate(coulomb_vals[term_idx] * sign)
                if np.abs(hij) <= 1e-14:
                    continue
                if newconf == conf:
                    diag_coeff += hij
                    continue

                neighbor = np.uint64(newconf)
                found = -1
                for existing in range(local_nnz):
                    if neighbors[row_flat, existing] == neighbor:
                        found = existing
                        break
                if found >= 0:
                    vals[row_flat, found] += hij
                elif local_nnz < row_width:
                    neighbors[row_flat, local_nnz] = neighbor
                    vals[row_flat, local_nnz] = hij
                    local_nnz += 1
                else:
                    overflow[row_flat] = np.uint8(1)

            write = 0
            for existing in range(local_nnz):
                value = vals[row_flat, existing]
                if np.abs(value) > 1e-14:
                    if write != existing:
                        neighbors[row_flat, write] = neighbors[row_flat, existing]
                        vals[row_flat, write] = value
                    write += 1
            for tail in range(write, row_width):
                neighbors[row_flat, tail] = np.uint64(0)
                vals[row_flat, tail] = 0.0j
            diag[row_flat] = diag_coeff
            counts[row_flat] = write
        return diag, neighbors, vals, counts, overflow


    @njit(cache=True, parallel=True)
    def _build_sparse_bitstring_rows_numba_grouped_hash(
        row_bitstrings,
        nphi,
        pair_offsets,
        j1_grouped,
        j2_grouped,
        coulomb_grouped,
        row_width,
        hash_capacity,
    ):
        """Skip unoccupied outgoing groups and hash-deduplicate neighbors."""
        n_rows = row_bitstrings.shape[0]
        diag = np.zeros(n_rows, dtype=np.complex128)
        neighbors = np.zeros((n_rows, row_width), dtype=np.uint64)
        vals = np.zeros((n_rows, row_width), dtype=np.complex128)
        counts = np.zeros(n_rows, dtype=np.int64)
        overflow = np.zeros(n_rows, dtype=np.uint8)
        hash_slots = np.full((n_rows, hash_capacity), -1, dtype=np.int32)
        hash_mask = np.uint64(hash_capacity - 1)

        for row_flat in prange(n_rows):
            conf = int(row_bitstrings[row_flat])
            local_nnz = 0
            diag_coeff = 0.0j
            for j3 in range(nphi):
                if ((conf >> j3) & 1) == 0:
                    continue
                for j4 in range(j3 + 1, nphi):
                    if ((conf >> j4) & 1) == 0:
                        continue
                    pair = j3 * nphi + j4
                    for grouped_index in range(
                        pair_offsets[pair], pair_offsets[pair + 1]
                    ):
                        j1 = j1_grouped[grouped_index]
                        j2 = j2_grouped[grouped_index]
                        sign = 1
                        newconf = conf
                        if _parity_numba(newconf, j4 + 1):
                            sign = -sign
                        newconf &= ~(1 << j4)
                        if _parity_numba(newconf, j3 + 1):
                            sign = -sign
                        newconf &= ~(1 << j3)
                        if ((newconf >> j1) & 1) != 0 or ((newconf >> j2) & 1) != 0:
                            continue
                        if _parity_numba(newconf, j2 + 1):
                            sign = -sign
                        newconf |= 1 << j2
                        if _parity_numba(newconf, j1 + 1):
                            sign = -sign
                        newconf |= 1 << j1

                        hij = np.conjugate(coulomb_grouped[grouped_index] * sign)
                        if np.abs(hij) <= 1e-14:
                            continue
                        if newconf == conf:
                            diag_coeff += hij
                            continue

                        neighbor = np.uint64(newconf)
                        slot = np.int64(neighbor & hash_mask)
                        found = -1
                        while True:
                            existing = hash_slots[row_flat, slot]
                            if existing < 0:
                                break
                            if neighbors[row_flat, existing] == neighbor:
                                found = existing
                                break
                            slot = np.int64(
                                (slot + 1) & (hash_capacity - 1)
                            )
                        if found >= 0:
                            vals[row_flat, found] += hij
                        elif local_nnz < row_width:
                            neighbors[row_flat, local_nnz] = neighbor
                            vals[row_flat, local_nnz] = hij
                            hash_slots[row_flat, slot] = local_nnz
                            local_nnz += 1
                        else:
                            overflow[row_flat] = np.uint8(1)

            write = 0
            for existing in range(local_nnz):
                value = vals[row_flat, existing]
                if np.abs(value) > 1e-14:
                    if write != existing:
                        neighbors[row_flat, write] = neighbors[row_flat, existing]
                        vals[row_flat, write] = value
                    write += 1
            for tail in range(write, row_width):
                neighbors[row_flat, tail] = np.uint64(0)
                vals[row_flat, tail] = 0.0j
            diag[row_flat] = diag_coeff
            counts[row_flat] = write
        return diag, neighbors, vals, counts, overflow


def generate_coulomb_matrix_python(
    nphi: int,
    ratio_rect: float,
    cutoff: int,
    landau_level: int = 0,
) -> np.ndarray:
    coulomb_mat = np.zeros((nphi, nphi, nphi, nphi), dtype=np.complex128)
    pref = np.sqrt(2.0 * np.pi / (nphi * ratio_rect))
    for j1 in range(nphi):
        for j2 in range(nphi):
            if j1 == j2:
                continue
            for j3 in range(nphi):
                for j4 in range(nphi):
                    if j3 == j4 or (j1 + j2 - j3 - j4) % nphi != 0:
                        continue
                    vq = 0.0j
                    for t in range(-cutoff, cutoff):
                        if (j1 - j4 - t) % nphi != 0:
                            continue
                        for s in range(-cutoff, cutoff):
                            if t == 0 and s == 0:
                                continue
                            qlb = pref * np.sqrt(
                                s * s + (ratio_rect * t) ** 2
                            )
                            laguerre = landau_level_form_factor_polynomial(
                                landau_level, 0.5 * qlb**2
                            )
                            vq += (
                                0.5
                                / (qlb * nphi)
                                * laguerre**2
                                * np.exp(-0.5 * qlb**2)
                                * np.exp(
                                    -1j
                                    * 2.0
                                    * np.pi
                                    * s
                                    * (j1 - j3)
                                    / nphi
                                )
                            )
                    coulomb_mat[j1, j2, j3, j4] = vq
    return coulomb_mat


def antisym_coulomb(
    coulomb_mat: np.ndarray, j1: int, j2: int, j3: int, j4: int
) -> complex:
    return (
        coulomb_mat[j1, j2, j3, j4]
        - coulomb_mat[j2, j1, j3, j4]
        - coulomb_mat[j1, j2, j4, j3]
        + coulomb_mat[j2, j1, j4, j3]
    )


class TorusCoulombHamiltonian:
    """Coulomb matrix and sector-row builder with instance-local caching."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.cutoff = (
            cfg.coulomb_cut if cfg.coulomb_cut is not None else 10 * cfg.nphi
        )
        self.jlist = generate_jlist(cfg.nphi)
        self.row_cache: dict[
            tuple[int, int], tuple[complex, tuple[tuple[int, complex], ...]]
        ] = {}
        self.sector_lookup_cache: dict[int, dict[int, int]] = {}
        self.sector_bitstrings_cache: dict[int, np.ndarray] = {}
        self.sector_sorted_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.coulomb_matrix: np.ndarray | None = None
        self.precompute_method: str | None = None
        self._grouped_bitstring_terms = None

    def generate_coulomb_matrix(self) -> tuple[np.ndarray, str, int]:
        use_numba = NUMBA_AVAILABLE and not self.cfg.disable_numba_coulomb
        self.precompute_method = "numba njit parallel" if use_numba else "python"
        if use_numba:
            self.coulomb_matrix = _generate_coulomb_matrix_numba(
                self.cfg.nphi,
                self.cfg.ratio_rect,
                self.cutoff,
                self.cfg.landau_level,
            )
        else:
            self.coulomb_matrix = generate_coulomb_matrix_python(
                self.cfg.nphi,
                self.cfg.ratio_rect,
                self.cutoff,
                self.cfg.landau_level,
            )
        return self.coulomb_matrix, self.precompute_method, self.cutoff

    def _require_coulomb_matrix(self) -> np.ndarray:
        if self.coulomb_matrix is None:
            raise RuntimeError("Coulomb matrix has not been generated.")
        return self.coulomb_matrix

    def _grouped_bitstring_scattering(self):
        cached = self._grouped_bitstring_terms
        if cached is not None:
            return cached
        matrix = self._require_coulomb_matrix()
        j1, j2, j3, j4 = (
            np.ascontiguousarray(self.jlist[:, i], dtype=np.int64)
            for i in range(4)
        )
        coefficients = (
            matrix[j1, j2, j3, j4]
            - matrix[j2, j1, j3, j4]
            - matrix[j1, j2, j4, j3]
            + matrix[j2, j1, j4, j3]
        ).astype(np.complex128)
        pair_ids = j3 * self.cfg.nphi + j4
        order = np.argsort(pair_ids, kind="stable")
        grouped_ids = pair_ids[order]
        offsets = np.searchsorted(
            grouped_ids,
            np.arange(self.cfg.nphi * self.cfg.nphi + 1),
            side="left",
        ).astype(np.int64)
        cached = (
            np.ascontiguousarray(offsets),
            np.ascontiguousarray(j1[order]),
            np.ascontiguousarray(j2[order]),
            np.ascontiguousarray(coefficients[order]),
        )
        self._grouped_bitstring_terms = cached
        return cached

    def _sector_lookup(
        self, tk: int, basis: BasisData
    ) -> dict[int, int]:
        lookup = self.sector_lookup_cache.get(int(tk))
        if lookup is None:
            lookup = build_sector_lookup(basis, tk)
            self.sector_lookup_cache[int(tk)] = lookup
        return lookup

    def _sector_bitstrings(
        self, tk: int, basis: BasisData
    ) -> np.ndarray:
        bitstrings = self.sector_bitstrings_cache.get(int(tk))
        if bitstrings is None:
            bitstrings = sector_basis_bitstrings(basis, tk)
            self.sector_bitstrings_cache[int(tk)] = bitstrings
        return bitstrings

    def _sector_sorted_arrays(
        self, tk: int, basis: BasisData
    ) -> tuple[np.ndarray, np.ndarray]:
        arrays = self.sector_sorted_cache.get(int(tk))
        if arrays is None:
            sector_bitstrings = self._sector_bitstrings(tk, basis)
            order = np.argsort(sector_bitstrings)
            arrays = (
                sector_bitstrings[order].astype(np.int64, copy=False),
                order.astype(np.int64, copy=False),
            )
            self.sector_sorted_cache[int(tk)] = arrays
        return arrays

    def row_terms(
        self,
        sector_idx: int,
        tk: int,
        basis: BasisData,
        sector_lookup: dict[int, int],
    ) -> tuple[complex, tuple[tuple[int, complex], ...]]:
        cache_key = (int(tk), int(sector_idx))
        cached = self.row_cache.get(cache_key)
        if cached is not None:
            return cached

        coulomb_mat = self._require_coulomb_matrix()
        index_row = tk if basis.index_tk.shape[0] > 1 else 0
        conf = int(
            basis.bitstrings[basis.index_tk[index_row, sector_idx]]
        )
        terms: dict[int, complex] = {}
        for j1, j2, j3, j4 in self.jlist:
            if not (occupation(conf, int(j3)) and occupation(conf, int(j4))):
                continue
            sign = 1
            newconf = conf
            sign *= -1 if count_ones_from(newconf, int(j4) + 1) % 2 else 1
            newconf &= ~(1 << int(j4))
            sign *= -1 if count_ones_from(newconf, int(j3) + 1) % 2 else 1
            newconf &= ~(1 << int(j3))
            if occupation(newconf, int(j1)) or occupation(newconf, int(j2)):
                continue
            sign *= -1 if count_ones_from(newconf, int(j2) + 1) % 2 else 1
            newconf |= 1 << int(j2)
            sign *= -1 if count_ones_from(newconf, int(j1) + 1) % 2 else 1
            newconf |= 1 << int(j1)
            col_idx = sector_lookup.get(int(newconf))
            if col_idx is None:
                continue
            vq = antisym_coulomb(
                coulomb_mat, int(j1), int(j2), int(j3), int(j4)
            )
            hij = np.conjugate(vq * sign)
            if abs(hij) > 1e-14:
                terms[col_idx] = terms.get(col_idx, 0.0j) + hij

        diag_coeff = terms.pop(int(sector_idx), 0.0j)
        result = (diag_coeff, tuple((idx, val) for idx, val in terms.items()))
        self.row_cache[cache_key] = result
        return result

    def build_exact_hamiltonian(self, tk: int, basis: BasisData) -> np.ndarray:
        size = int(basis.number_tk[tk])
        sector_lookup = self._sector_lookup(tk, basis)
        h = np.zeros((size, size), dtype=np.complex128)
        for i in range(size):
            diag, offdiag = self.row_terms(i, tk, basis, sector_lookup)
            h[i, i] += diag
            for j, hij in offdiag:
                h[i, j] += hij
        return 0.5 * (h + h.conj().T)

    def build_sparse_hamiltonian_rows(
        self,
        tk: int,
        basis: BasisData,
        row_chunk_size: int = 256,
    ) -> SparseHamiltonianRows:
        size = int(basis.number_tk[tk])
        if row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive.")
        if size <= row_chunk_size:
            return self.build_sparse_hamiltonian_rows_subset(
                tk, basis, np.arange(size, dtype=np.int64)
            )

        chunks = []
        max_nnz = 1
        for start in range(0, size, row_chunk_size):
            stop = min(start + row_chunk_size, size)
            chunk = self.build_sparse_hamiltonian_rows_subset(
                tk, basis, np.arange(start, stop, dtype=np.int64)
            )
            chunks.append(chunk)
            max_nnz = max(max_nnz, int(chunk.cols.shape[-1]))

        diag = np.empty((size,), dtype=np.complex128)
        cols = np.zeros((size, max_nnz), dtype=np.int64)
        vals = np.zeros((size, max_nnz), dtype=np.complex128)
        mask = np.zeros((size, max_nnz), dtype=np.float64)
        start = 0
        for chunk in chunks:
            count = int(chunk.diag.shape[0])
            width = int(chunk.cols.shape[-1])
            stop = start + count
            diag[start:stop] = chunk.diag
            cols[start:stop, :width] = chunk.cols
            vals[start:stop, :width] = chunk.vals
            mask[start:stop, :width] = chunk.mask
            start = stop
        return SparseHamiltonianRows(diag, cols, vals, mask)

    def build_sparse_hamiltonian_csr(
        self, tk: int, basis: BasisData
    ):
        """Build the sector Hamiltonian as a SciPy CSR matrix.

        This is intentionally separate from ``build_exact_hamiltonian``:
        large sectors must never materialize an O(size**2) dense array.
        """
        try:
            from scipy.sparse import coo_matrix
        except ImportError as exc:
            raise ImportError(
                "Sparse ED requires SciPy; install scipy in the runtime environment."
            ) from exc

        rows = self.build_sparse_hamiltonian_rows(tk, basis)
        nrow, nslot = rows.cols.shape
        diagonal_ids = np.arange(nrow, dtype=np.int64)
        row_ids = np.concatenate(
            (diagonal_ids, np.repeat(diagonal_ids, nslot))
        )
        col_ids = np.concatenate((diagonal_ids, rows.cols.reshape(-1)))
        values = np.concatenate(
            (rows.diag.reshape(-1), rows.vals.reshape(-1) * rows.mask.reshape(-1))
        )
        matrix = coo_matrix(
            (values, (row_ids, col_ids)), shape=(nrow, nrow), dtype=np.complex128
        ).tocsr()
        return 0.5 * (matrix + matrix.getH())

    def sparse_exact_eigenvalues(
        self, tk: int, basis: BasisData, count: int = 8
    ) -> np.ndarray:
        """Return the lowest ``count`` ED levels without dense allocation."""
        try:
            from scipy.sparse.linalg import eigsh
        except ImportError as exc:
            raise ImportError(
                "Sparse ED requires scipy.sparse.linalg.eigsh."
            ) from exc
        matrix = self.build_sparse_hamiltonian_csr(tk, basis)
        n = matrix.shape[0]
        if n <= 2:
            return np.linalg.eigvalsh(matrix.toarray())
        k = min(max(1, int(count)), n - 1)
        values = eigsh(matrix, k=k, which="SA", return_eigenvectors=False)
        return np.sort(np.real_if_close(values).real)

    def build_sparse_hamiltonian_rows_subset(
        self,
        tk: int,
        basis: BasisData,
        row_indices: np.ndarray,
        pad_to_nnz: int | None = None,
    ) -> SparseHamiltonianRows:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        flat = row_indices.reshape(-1)
        if flat.size == 0:
            shape = row_indices.shape
            empty = np.zeros(shape + (1,), dtype=np.int64)
            return SparseHamiltonianRows(
                np.zeros(shape, dtype=np.complex128),
                empty,
                np.zeros(shape + (1,), dtype=np.complex128),
                np.zeros(shape + (1,), dtype=np.float64),
            )

        if NUMBA_AVAILABLE:
            sector_bitstrings = self._sector_bitstrings(tk, basis)
            sorted_bitstrings, sorted_positions = self._sector_sorted_arrays(
                tk, basis
            )
            diag, cols_full, vals_full, counts = (
                _build_sparse_rows_subset_numba(
                    flat,
                    sector_bitstrings,
                    sorted_bitstrings,
                    sorted_positions,
                    self.jlist,
                    self._require_coulomb_matrix(),
                )
            )
            max_nnz = max(1, int(np.max(counts)))
            if (
                pad_to_nnz is not None
                and os.environ.get("NQS_GROUPED_HASH_ROWS") == "1"
            ):
                offsets, j1_grouped, j2_grouped, coefficients_grouped = (
                    self._grouped_bitstring_scattering()
                )
                hash_capacity = 1
                while hash_capacity < 2 * pad_to_nnz:
                    hash_capacity *= 2
                diag, neighbors, vals, counts, overflow = (
                    _build_sparse_bitstring_rows_numba_grouped_hash(
                        flat,
                        self.cfg.nphi,
                        offsets,
                        j1_grouped,
                        j2_grouped,
                        coefficients_grouped,
                        pad_to_nnz,
                        hash_capacity,
                    )
                )
                if np.any(overflow):
                    raise ValueError(
                        "pad_to_nnz is smaller than a sampled Hamiltonian row."
                    )
                cols_full = neighbors
                vals_full = vals
                max_nnz = pad_to_nnz
            elif pad_to_nnz is not None:
                if max_nnz > int(pad_to_nnz):
                    raise ValueError(
                        "pad_to_nnz is smaller than a sampled Hamiltonian row."
                    )
                max_nnz = int(pad_to_nnz)
            cols = np.zeros((flat.shape[0], max_nnz), dtype=np.int64)
            vals = np.zeros((flat.shape[0], max_nnz), dtype=np.complex128)
            copied_nnz = min(max_nnz, cols_full.shape[1])
            cols[:, :copied_nnz] = cols_full[:, :copied_nnz]
            vals[:, :copied_nnz] = vals_full[:, :copied_nnz]
            mask = (
                np.arange(max_nnz, dtype=np.int64)[None, :]
                < counts[:, None]
            ).astype(np.float64, copy=False)
        else:
            sector_lookup = self._sector_lookup(tk, basis)
            diag = np.zeros(flat.shape[0], dtype=np.complex128)
            rows = []
            max_nnz = 1
            for i, row_idx in enumerate(flat):
                diag_i, offdiag = self.row_terms(
                    int(row_idx), tk, basis, sector_lookup
                )
                diag[i] = diag_i
                rows.append(offdiag)
                max_nnz = max(max_nnz, len(offdiag))
            if pad_to_nnz is not None:
                max_nnz = max(max_nnz, int(pad_to_nnz))

            cols = np.zeros((flat.shape[0], max_nnz), dtype=np.int64)
            vals = np.zeros((flat.shape[0], max_nnz), dtype=np.complex128)
            mask = np.zeros((flat.shape[0], max_nnz), dtype=np.float64)
            for i, offdiag in enumerate(rows):
                for n, (j, hij) in enumerate(offdiag):
                    cols[i, n] = int(j)
                    vals[i, n] = hij
                    mask[i, n] = 1.0

        shape = row_indices.shape
        return SparseHamiltonianRows(
            diag.reshape(shape),
            cols.reshape(shape + (max_nnz,)),
            vals.reshape(shape + (max_nnz,)),
            mask.reshape(shape + (max_nnz,)),
        )

    def build_sparse_hamiltonian_bitstring_rows(
        self,
        row_bitstrings: np.ndarray,
        pad_to_nnz: int | None = None,
    ) -> SparseHamiltonianBitstringRows:
        """Build sampled rows directly from packed states, without sector lookup."""
        _validate_packed_nphi(self.cfg.nphi)
        if pad_to_nnz is not None:
            pad_to_nnz = operator.index(pad_to_nnz)
            if pad_to_nnz <= 0:
                raise ValueError("pad_to_nnz must be positive.")
            if pad_to_nnz > self.jlist.shape[0]:
                raise ValueError(
                    "pad_to_nnz exceeds the sparse-row construction bound."
                )
        row_bitstrings = np.asarray(row_bitstrings, dtype=np.uint64)
        if np.any(row_bitstrings >> np.uint64(self.cfg.nphi)):
            raise ValueError("row_bitstrings contain orbitals outside Nphi.")
        shape = row_bitstrings.shape
        flat = row_bitstrings.reshape(-1)
        if flat.size == 0:
            empty_shape = shape + (pad_to_nnz or 1,)
            return SparseHamiltonianBitstringRows(
                np.zeros(shape, dtype=np.complex128),
                np.zeros(empty_shape, dtype=np.uint64),
                np.zeros(empty_shape, dtype=np.complex128),
                np.zeros(empty_shape, dtype=np.float64),
            )

        coulomb_mat = self._require_coulomb_matrix()
        if NUMBA_AVAILABLE:
            # Hoist antisymmetrization out of the per-row scattering loop.
            j1, j2, j3, j4 = (
                np.ascontiguousarray(self.jlist[:, i], dtype=np.int64)
                for i in range(4)
            )
            coulomb_vals = (
                coulomb_mat[j1, j2, j3, j4]
                - coulomb_mat[j2, j1, j3, j4]
                - coulomb_mat[j1, j2, j4, j3]
                + coulomb_mat[j2, j1, j4, j3]
            ).astype(np.complex128)
            if pad_to_nnz is not None:
                diag, neighbors, vals, counts, overflow = (
                    _build_sparse_bitstring_rows_numba_fixed(
                        flat,
                        j1,
                        j2,
                        j3,
                        j4,
                        coulomb_vals,
                        pad_to_nnz,
                    )
                )
                if np.any(overflow):
                    raise ValueError(
                        "pad_to_nnz is smaller than a sampled Hamiltonian row."
                    )
                max_nnz = pad_to_nnz
            else:
                diag, neighbors_full, vals_full, counts = (
                    _build_sparse_bitstring_rows_numba_fast(
                        flat, j1, j2, j3, j4, coulomb_vals
                    )
                )
                max_nnz = max(1, int(np.max(counts)))
                if pad_to_nnz is not None:
                    if max_nnz > pad_to_nnz:
                        raise ValueError(
                            "pad_to_nnz is smaller than a sampled Hamiltonian row."
                        )
                    max_nnz = pad_to_nnz
                neighbors = neighbors_full[:, :max_nnz].copy()
                vals = vals_full[:, :max_nnz].copy()
            mask = (
                np.arange(max_nnz, dtype=np.int64)[None, :]
                < counts[:, None]
            ).astype(np.float64, copy=False)
        else:  # pragma: no cover - exercised only without optional Numba.
            rows = []
            diag = np.zeros(flat.size, dtype=np.complex128)
            max_nnz = max(1, int(pad_to_nnz or 1))
            for row_i, conf_u in enumerate(flat):
                conf = int(conf_u)
                terms: dict[int, complex] = {}
                for j1, j2, j3, j4 in self.jlist:
                    j1, j2, j3, j4 = map(int, (j1, j2, j3, j4))
                    if not (occupation(conf, j3) and occupation(conf, j4)):
                        continue
                    sign = 1
                    newconf = conf
                    if count_ones_from(newconf, j4 + 1) % 2:
                        sign = -sign
                    newconf &= ~(1 << j4)
                    if count_ones_from(newconf, j3 + 1) % 2:
                        sign = -sign
                    newconf &= ~(1 << j3)
                    if occupation(newconf, j1) or occupation(newconf, j2):
                        continue
                    if count_ones_from(newconf, j2 + 1) % 2:
                        sign = -sign
                    newconf |= 1 << j2
                    if count_ones_from(newconf, j1 + 1) % 2:
                        sign = -sign
                    newconf |= 1 << j1
                    hij = np.conjugate(
                        antisym_coulomb(coulomb_mat, j1, j2, j3, j4) * sign
                    )
                    if abs(hij) <= 1e-14:
                        continue
                    if newconf == conf:
                        diag[row_i] += hij
                    else:
                        terms[newconf] = terms.get(newconf, 0.0j) + hij
                terms = {
                    state: value
                    for state, value in terms.items()
                    if abs(value) > 1e-14
                }
                rows.append(terms)
                max_nnz = max(max_nnz, len(terms))
            if pad_to_nnz is not None and max_nnz > int(pad_to_nnz):
                raise ValueError(
                    "pad_to_nnz is smaller than a sampled Hamiltonian row."
                )
            neighbors = np.zeros((flat.size, max_nnz), dtype=np.uint64)
            vals = np.zeros((flat.size, max_nnz), dtype=np.complex128)
            mask = np.zeros((flat.size, max_nnz), dtype=np.float64)
            for row_i, terms in enumerate(rows):
                for slot, (state, value) in enumerate(terms.items()):
                    neighbors[row_i, slot] = np.uint64(state)
                    vals[row_i, slot] = value
                    mask[row_i, slot] = 1.0

        return SparseHamiltonianBitstringRows(
            diag.reshape(shape),
            neighbors.reshape(shape + (max_nnz,)),
            vals.reshape(shape + (max_nnz,)),
            mask.reshape(shape + (max_nnz,)),
        )
