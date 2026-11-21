"""CPU hash row builder preserving term and first-occurrence ordering."""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def parity(x):
    x ^= x >> np.uint64(32)
    x ^= x >> np.uint64(16)
    x ^= x >> np.uint64(8)
    x ^= x >> np.uint64(4)
    return (np.uint64(0x6996) >> (x & np.uint64(15))) & np.uint64(1)


@njit(cache=True, parallel=True)
def build(states, a, b, c, d, coeff, width):
    n = len(states)
    diag = np.zeros(n, np.complex128)
    bits = np.zeros((n, width), np.uint64)
    vals = np.zeros((n, width), np.complex128)
    counts = np.zeros(n, np.int64)
    overflow = np.zeros(n, np.uint8)
    capacity = 1
    while capacity < 2 * width:
        capacity *= 2
    for r in prange(n):
        slots = np.full(capacity, -1, np.int32)
        conf = states[r]
        count = 0
        for t in range(len(a)):
            ma = np.uint64(1) << np.uint64(a[t])
            mb = np.uint64(1) << np.uint64(b[t])
            mc = np.uint64(1) << np.uint64(c[t])
            md = np.uint64(1) << np.uint64(d[t])
            if not (conf & mc) or not (conf & md):
                continue
            x = conf & ~md
            p = parity(conf >> np.uint64(d[t] + 1))
            p ^= parity(x >> np.uint64(c[t] + 1))
            x &= ~mc
            if (x & ma) or (x & mb):
                continue
            p ^= parity(x >> np.uint64(b[t] + 1))
            x |= mb
            p ^= parity(x >> np.uint64(a[t] + 1))
            x |= ma
            h = np.conjugate(coeff[t] * (-1 if p else 1))
            if abs(h) <= 1e-14:
                continue
            if x == conf:
                diag[r] += h
                continue
            # Mix high occupied orbitals into low hash bits.
            z = x ^ (x >> np.uint64(30))
            z *= np.uint64(0xbf58476d1ce4e5b9)
            z ^= z >> np.uint64(27)
            slot = np.int64(z & np.uint64(capacity - 1))
            while slots[slot] >= 0 and bits[r, slots[slot]] != x:
                slot = (slot + 1) & (capacity - 1)
            found = slots[slot]
            if found >= 0:
                vals[r, found] += h
            elif count < width:
                slots[slot] = count
                bits[r, count] = x
                vals[r, count] = h
                count += 1
            else:
                overflow[r] = 1
        write = 0
        for k in range(count):
            if abs(vals[r, k]) > 1e-14:
                bits[r, write] = bits[r, k]
                vals[r, write] = vals[r, k]
                write += 1
        for k in range(write, width):
            bits[r, k] = 0
            vals[r, k] = 0
        counts[r] = write
    return diag, bits, vals, counts, overflow
