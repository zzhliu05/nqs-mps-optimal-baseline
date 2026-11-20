"""Compiled, per-call proposals. No wavefunction or cross-step caching."""
import numpy as np
from numba import njit, prange


def _build_impl(h, top=32, power=1.0, uniform_mix=0.1):
    rows, width = h.shape
    selected = np.full((rows, top), -1, np.int64)
    q = np.zeros((rows, width), np.float64)
    cdf = np.zeros_like(q)
    for r in prange(rows):
        magnitude = np.abs(h[r]).astype(np.float64)
        order = np.argsort(-magnitude, kind='mergesort')
        count = 0
        for k in range(min(top, width)):
            col = order[k]
            if magnitude[col] == 0:
                break
            selected[r, count] = col
            magnitude[col] = 0
            count += 1
        maximum = magnitude.max()
        if maximum == 0:
            continue
        weight_sum = 0.0
        tail_count = 0
        for j in range(width):
            if magnitude[j] > 0:
                q[r, j] = (magnitude[j] / maximum) ** power
                weight_sum += q[r, j]
                tail_count += 1
        total = 0.0
        for j in range(width):
            if magnitude[j] > 0:
                q[r, j] = (1-uniform_mix)*q[r, j]/weight_sum + uniform_mix/tail_count
                total += q[r, j]
        running = 0.0
        for j in range(width):
            q[r, j] /= total
            running += q[r, j]
            cdf[r, j] = running
        # Force the final active CDF and trailing zeros to exactly one.
        last = width - 1
        while q[r, last] == 0:
            cdf[r, last] = 1.0
            last -= 1
        cdf[r, last] = 1.0
    return selected, q, cdf


def _draw_impl(selected, q, cdf, uniforms):
    rows, samples = uniforms.shape
    top = selected.shape[1]
    columns = np.zeros((rows, top+samples), np.int64)
    weights = np.zeros((rows, top+samples), np.float64)
    for r in prange(rows):
        for k in range(top):
            if selected[r, k] >= 0:
                columns[r, k] = selected[r, k]
                weights[r, k] = 1
        if cdf[r, -1] == 0:
            continue
        for k in range(samples):
            col = np.searchsorted(cdf[r], uniforms[r, k], side='right')
            columns[r, top+k] = col
            weights[r, top+k] = 1/(samples*q[r, col])
    return columns, weights


build = njit(cache=True)(_build_impl)
build_parallel = njit(cache=False, parallel=True)(_build_impl)
draw = njit(cache=True)(_draw_impl)
draw_parallel = njit(cache=False, parallel=True)(_draw_impl)
