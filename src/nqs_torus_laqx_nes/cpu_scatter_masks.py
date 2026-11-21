"""Precompute occupation and fermion-parity masks without reordering H terms."""
import numpy as np
from numba import njit, prange
from .cpu_row_hash import parity


def prepare(a, b, c, d, coeff):
    remove, create, signs, corrections = [], [], [], []
    for aa,bb,cc,dd in zip(a,b,c,d):
        aa,bb,cc,dd = map(int,(aa,bb,cc,dd))
        rm = (1<<cc) | (1<<dd)
        cr = (1<<aa) | (1<<bb)
        pm = 0
        for i in (aa,bb,cc,dd):
            pm ^= ((1<<64)-1) ^ ((1<<(i+1))-1)
        x = rm
        p = (x>>(dd+1)).bit_count() & 1
        x &= ~(1<<dd)
        p ^= (x>>(cc+1)).bit_count() & 1
        x &= ~(1<<cc)
        p ^= (x>>(bb+1)).bit_count() & 1
        x |= 1<<bb
        p ^= (x>>(aa+1)).bit_count() & 1
        remove.append(rm); create.append(cr); signs.append(pm)
        corrections.append(p ^ ((rm&pm).bit_count()&1))
    return (np.array(remove,np.uint64),np.array(create,np.uint64),
            np.array(signs,np.uint64),np.array(corrections,np.uint8),
            np.asarray(coeff,np.complex128))


@njit(cache=True)
def scatter_one(conf, remove, create, signs, corrections, coeff, width, capacity):
    bits = np.empty(width,np.uint64)
    vals = np.empty(width,np.complex128)
    slots = np.full(capacity,-1,np.int32)
    count, overflow, diag = 0, 0, 0.0j
    for t in range(len(remove)):
        if (conf & remove[t]) != remove[t]:
            continue
        x = conf & ~remove[t]
        if x & create[t]:
            continue
        p = parity(conf & signs[t]) ^ corrections[t]
        x |= create[t]
        h = np.conjugate(coeff[t] * (-1 if p else 1))
        if abs(h) <= 1e-14:
            continue
        if x == conf:
            diag += h
            continue
        z = x ^ (x >> np.uint64(30))
        z *= np.uint64(0xbf58476d1ce4e5b9)
        z ^= z >> np.uint64(27)
        slot = np.int64(z & np.uint64(capacity-1))
        while slots[slot] >= 0 and bits[slots[slot]] != x:
            slot = (slot+1) & (capacity-1)
        found = slots[slot]
        if found >= 0:
            vals[found] += h
        elif count < width:
            slots[slot] = count
            bits[count], vals[count] = x,h
            count += 1
        else:
            overflow = 1
    write = 0
    for k in range(count):
        if abs(vals[k]) > 1e-14:
            bits[write],vals[write] = bits[k],vals[k]
            write += 1
    return diag,bits[:write],vals[:write],overflow
