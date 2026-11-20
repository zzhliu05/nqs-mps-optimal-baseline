"""Select weighted rows on CPU, before any Hamiltonian coefficients reach GPU."""
import time
import os
import numpy as np
from compiled_proposal import build, draw, build_parallel, draw_parallel


def sample_pack(rows, rng, value_dtype=np.complex128):
    start=time.perf_counter()
    bits=np.asarray(rows.neighbor_bitstrings,dtype=np.uint64)
    width=bits.shape[-1]
    bits=bits.reshape(-1,width)
    if os.environ.get('IS_FUSED_CPU_PACK')=='1':
        from fused_cpu_pack import fused_pack
        values=np.asarray(rows.vals).reshape(bits.shape)
        mask=np.asarray(rows.mask).reshape(bits.shape)
        b,v,o=fused_pack(bits,values,mask,rng.random((len(bits),32)),int(os.environ.get('IS_TOP_EXACT','32')))
        packed=(b,v.astype(value_dtype,copy=False),o)
        return packed,{'total_s':time.perf_counter()-start,'rows':len(bits),
                       'uploaded_slots':len(b),'fused':True}
    coeff=np.asarray(rows.vals).reshape(bits.shape)*np.asarray(rows.mask).reshape(bits.shape)
    prepared=time.perf_counter()
    parallel=os.environ.get('IS_PROPOSAL_PARALLEL')=='1'
    selected,q,cdf=(build_parallel if parallel else build)(coeff,32,1.0,.1)
    planned=time.perf_counter()
    columns,weights=(draw_parallel if parallel else draw)(selected,q,cdf,rng.random((len(coeff),32)))
    drawn=time.perf_counter()
    neighbors=np.take_along_axis(bits,columns,axis=1)
    values=np.take_along_axis(coeff,columns,axis=1)*weights
    neighbors=np.where(weights!=0,neighbors,0).astype(np.uint64).reshape(-1)
    values=values.astype(value_dtype).reshape(-1)
    offsets=np.arange(len(coeff)+1,dtype=np.int32)*64
    timings={'prepare_s':prepared-start,'probability_s':planned-prepared,
             'draw_s':drawn-planned,'gather_s':time.perf_counter()-drawn,
             'total_s':time.perf_counter()-start,'rows':len(coeff),
             'uploaded_slots':len(neighbors)}
    return (neighbors,values,offsets),timings
