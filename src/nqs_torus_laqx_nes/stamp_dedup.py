"""Opt-in Numba stamp-table replacement for neighbor-state deduplication.

Retained for indexed-state profiling compatibility. The canonical baseline
uses exact row-major bitstring neighbors without deduplication and does not
install this module's optional runtime patch.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from . import optimization

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback for environments without Numba.
    njit = None


if njit is not None:

    @njit(cache=True)
    def _unique_inverse_stamp(values, stamps, slots, generation):
        unique = np.empty(values.size, dtype=np.int64)
        inverse = np.empty(values.size, dtype=np.int64)
        count = 0
        for i in range(values.size):
            value = values[i]
            if stamps[value] != generation:
                stamps[value] = generation
                slots[value] = count
                unique[count] = value
                count += 1
            inverse[i] = slots[value]
        return unique[:count], inverse

else:
    _unique_inverse_stamp = None


class _StampWorkspace:
    def __init__(self):
        self.size = 0
        self.stamps = np.empty(0, dtype=np.int32)
        self.slots = np.empty(0, dtype=np.int32)
        self.generation = 0

    def next(self, size: int):
        if size != self.size:
            self.size = size
            self.stamps = np.zeros(size, dtype=np.int32)
            self.slots = np.empty(size, dtype=np.int32)
            self.generation = 0
        if self.generation >= np.iinfo(np.int32).max:
            self.stamps.fill(0)
            self.generation = 0
        self.generation += 1
        return self.stamps, self.slots, np.int32(self.generation)


_WORKSPACE = _StampWorkspace()
_ORIGINAL = optimization._accumulate_offdiagonal


def _accumulate_offdiagonal_stamp(
    params,
    states,
    flat_neighbor_cols,
    flat_coeff,
    flat_row_ids,
    row_count: int,
    net_cfg,
    deduplicate_neighbors: bool = False,
    unique_padding_size: int | None = None,
    unique_padding_bucket_size: int | None = None,
):
    if not deduplicate_neighbors or njit is None:
        return _ORIGINAL(
            params,
            states,
            flat_neighbor_cols,
            flat_coeff,
            flat_row_ids,
            row_count,
            net_cfg,
            deduplicate_neighbors=deduplicate_neighbors,
            unique_padding_size=unique_padding_size,
            unique_padding_bucket_size=unique_padding_bucket_size,
        )

    offdiag_flat = jnp.zeros((row_count, net_cfg.k), dtype=jnp.complex128)
    cols_np = np.asarray(flat_neighbor_cols, dtype=np.int64).reshape(-1)
    if cols_np.size == 0:
        return offdiag_flat

    state_count = int(states.shape[0])
    if np.any(cols_np < 0) or np.any(cols_np >= state_count):
        raise IndexError("neighbor index is outside the staged state table")

    stamps, slots, generation = _WORKSPACE.next(state_count)
    unique_cols, inverse_np = _unique_inverse_stamp(
        cols_np, stamps, slots, generation
    )
    unique_count = int(unique_cols.size)

    padded_size = unique_count
    if unique_padding_size is not None:
        limit = int(unique_padding_size)
        if unique_count > limit:
            raise ValueError(
                "unique neighbor count exceeds fixed padding size: "
                f"{unique_count} > {limit}."
            )
        padded_size = limit
        if unique_padding_bucket_size is not None:
            bucket = int(unique_padding_bucket_size)
            padded_size = min(limit, ((unique_count + bucket - 1) // bucket) * bucket)
    if padded_size != unique_count:
        padded_cols = np.zeros(padded_size, dtype=np.int64)
        padded_cols[:unique_count] = unique_cols
        unique_cols = padded_cols

    unique_neighbor_phi = optimization.batched_wavefunction_matrix_from_positions(
        params,
        states,
        jnp.asarray(unique_cols, dtype=jnp.int64),
        net_cfg,
    ).astype(jnp.complex128)
    neighbor_phi = unique_neighbor_phi[jnp.asarray(inverse_np, dtype=jnp.int64)]
    flat_row_ids = jnp.asarray(flat_row_ids, dtype=jnp.int64)
    flat_coeff = jnp.asarray(flat_coeff)
    return offdiag_flat.at[flat_row_ids].add(flat_coeff[:, None] * neighbor_phi)


def install() -> None:
    """Install the experimental patch in the current Python process."""
    optimization._accumulate_offdiagonal = _accumulate_offdiagonal_stamp

