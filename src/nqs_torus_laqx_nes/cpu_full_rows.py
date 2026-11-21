"""Exact full-neighbor rows packed inside CPU chunks; no Top-K selection."""
from dataclasses import dataclass
import operator
import numpy as np
from numba import njit, prange
from .cpu_scatter_masks import prepare, scatter_one


@dataclass(frozen=True)
class ChunkPackedRows:
    diag: np.ndarray
    bits: np.ndarray
    vals: np.ndarray
    counts: np.ndarray
    starts: np.ndarray


@njit(parallel=True,cache=True)
def build_chunks(states,remove,create,signs,corrections,coeff,width,chunk_rows=64):
    n=len(states)
    diag=np.zeros(n,np.complex128)
    bits=np.empty(n*width,np.uint64)
    vals=np.empty(n*width,np.complex128)
    counts=np.zeros(n,np.int64)
    starts=np.zeros(n,np.int64)
    overflow=np.zeros(n,np.uint8)
    table_capacity=1
    while table_capacity<2*width:table_capacity*=2
    for chunk in prange((n+chunk_rows-1)//chunk_rows):
        first=chunk*chunk_rows
        write=first*width
        for row in range(first,min(first+chunk_rows,n)):
            diag[row],b,v,overflow[row]=scatter_one(states[row],remove,create,signs,corrections,coeff,width,table_capacity)
            starts[row]=write
            counts[row]=len(b)
            for j in range(len(b)):
                bits[write+j]=b[j]
                vals[write+j]=v[j]
            write+=len(b)
    return diag,bits,vals,counts,starts,overflow


@njit(parallel=True,cache=True)
def copy_chunks(bits,vals,counts,starts,offsets,out_bits,out_vals):
    for row in prange(len(counts)):
        for j in range(counts[row]):
            out_bits[offsets[row]+j]=bits[starts[row]+j]
            out_vals[offsets[row]+j]=vals[starts[row]+j]


def fixed_coefficients(ham):
    matrix=ham._require_coulomb_matrix()
    terms=ham.jlist
    cached=getattr(ham,'_cpu_full_mask_cache',None)
    if cached is not None and cached[0] is matrix and cached[1] is terms:
        if matrix.flags.writeable or terms.flags.writeable:
            raise ValueError('Cached Hamiltonian made writable; replace arrays to invalidate')
        return cached[2]
    a,b,c,d=[np.ascontiguousarray(terms[:,i],dtype=np.int64) for i in range(4)]
    coeff=(matrix[a,b,c,d]-matrix[b,a,c,d]-matrix[a,b,d,c]+matrix[b,a,d,c]).astype(np.complex128)
    prepared=prepare(a,b,c,d,coeff)
    matrix.flags.writeable=False
    terms.flags.writeable=False
    ham._cpu_full_mask_cache=(matrix,terms,prepared)
    return prepared


def build_rows(ham,states,width):
    diag,bits,vals,counts,starts,overflow=build_chunks(states.reshape(-1),*fixed_coefficients(ham),width)
    if np.any(overflow):
        raise ValueError('pad_to_nnz is smaller than a sampled Hamiltonian row.')
    return ChunkPackedRows(diag.reshape(states.shape),bits,vals,counts,starts)


def pack_rows(rows,capacity,value_dtype):
    capacity=operator.index(capacity)
    if not 0<=capacity<=np.iinfo(np.int32).max:
        raise ValueError('compact capacity must fit nonnegative int32')
    offsets64=np.empty(len(rows.counts)+1,np.int64)
    offsets64[0]=0
    np.cumsum(rows.counts,out=offsets64[1:])
    if offsets64[-1]>capacity:
        raise ValueError(f'valid neighbor count exceeds compact capacity: {offsets64[-1]} > {capacity}')
    offsets=offsets64.astype(np.int32)
    bits=np.zeros(capacity,np.uint64)
    vals=np.zeros(capacity,dtype=value_dtype)
    copy_chunks(rows.bits,rows.vals,rows.counts,rows.starts,offsets,bits,vals)
    return bits,vals,offsets
