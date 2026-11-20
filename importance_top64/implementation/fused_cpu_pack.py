"""Row-local scratch buffers, no full-batch probability or CDF tensors."""
import numpy as np
from numba import njit,prange


@njit(parallel=True,cache=True)
def fused_pack(bits,vals,mask,uniforms,top=32):
    rows,width=bits.shape
    out_bits=np.zeros((rows,top+32),np.uint64)
    out_vals=np.zeros((rows,top+32),np.complex128)
    for r in prange(rows):
        h=vals[r]*mask[r]
        magnitude=np.abs(h).astype(np.float64)
        order=np.argsort(-magnitude,kind='mergesort')
        for k in range(min(top,width)):
            col=order[k]
            if magnitude[col]==0:break
            out_bits[r,k]=bits[r,col];out_vals[r,k]=h[col]
            magnitude[col]=0
        maximum=magnitude.max()
        if maximum==0:continue
        q=np.zeros(width,np.float64);cdf=np.zeros(width,np.float64)
        weight_sum=0.;count=0
        for j in range(width):
            if magnitude[j]>0:
                q[j]=(magnitude[j]/maximum)**1.0
                weight_sum+=q[j];count+=1
        total=0.
        for j in range(width):
            if magnitude[j]>0:
                q[j]=.9*q[j]/weight_sum+.1/count
                total+=q[j]
        running=0.
        for j in range(width):
            q[j]/=total;running+=q[j];cdf[j]=running
        last=width-1
        while q[last]==0:cdf[last]=1.;last-=1
        cdf[last]=1.
        for k in range(32):
            col=np.searchsorted(cdf,uniforms[r,k],side='right')
            out_bits[r,top+k]=bits[r,col]
            out_vals[r,top+k]=h[col]*(1/(32*q[col]))
    return out_bits.reshape(-1),out_vals.reshape(-1),np.arange(rows+1,dtype=np.int32)*(top+32)
