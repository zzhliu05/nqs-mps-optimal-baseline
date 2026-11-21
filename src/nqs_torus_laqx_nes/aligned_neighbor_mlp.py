"""Physical row-stride padding only; logical LayerNorm width is unchanged."""
from functools import partial,lru_cache
from types import FunctionType
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from nqs_torus_laqx_nes import triton_kernels as tk

def _store_padded(ref, index, value, *, mask):
    rows, columns = index
    physical_mask = (rows < ref.shape[0]) & (columns < ref.shape[1])
    tk._pallas_store(ref, index, jnp.where(mask, value, 0), mask=physical_mask)

source = tk._layernorm_silu_kernel
@lru_cache(None)
def make_kernel(store_mode):
    def opaque(columns):
        return tk.plgpu.elementwise_inline_asm('mov.b32 $0, $1;',args=[columns],
            constraints='=r,r',pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct(columns.shape,columns.dtype)])[0]
    def load(ref,index,*,mask,other):
        if store_mode=='scalar_io' and len(ref.shape)==2:
            rows,cols=index
            index=(rows,opaque(cols))
        return tk._pallas_load(ref,index,mask=mask,other=other)
    def store(ref,index,value,*,mask):
        rows,columns=index
        physical=(rows<ref.shape[0])&(columns<ref.shape[1])
        if store_mode=='raw':
            tk._pallas_store(ref,index,value,mask=physical)
        elif store_mode=='split':
            tk._pallas_store(ref,index,value,mask=mask)
            tk._pallas_store(ref,index,jnp.zeros_like(value),mask=physical&~mask)
        elif store_mode=='scalar_io':
            tk._pallas_store(ref,(rows,opaque(columns)),jnp.where(mask,value,0),mask=physical)
        else:
            _store_padded(ref,index,value,mask=mask)
    return FunctionType(source.__code__,dict(source.__globals__,_pallas_store=store,_pallas_load=load),
                        source.__name__,source.__defaults__,source.__closure__)

def aligned_norm(x, scale, bias, storage,store_mode='default',block_rows=4):
    width = scale.shape[0]
    assert width <= storage <= (1 << (width-1).bit_length())
    return pl.pallas_call(partial(make_kernel(store_mode),width=width,block_rows=block_rows,
        block_width=1 << (width-1).bit_length()),
        out_shape=jax.ShapeDtypeStruct((x.shape[0],storage),jnp.bfloat16),
        grid=(pl.cdiv(x.shape[0],block_rows),),
        compiler_params=tk.plgpu.CompilerParams(num_warps=4,num_stages=1),
        name='experimental_aligned_ln')(x,scale,bias)

def generate(params,states,storage,mode='all',store_mode='default'):
    layers=params['mps']['tensor_mlp']
    assert len(layers)==4 and layers[2]['w'].shape==(397,397)
    h=tk.reuse_first_layernorm_silu(layers[0],states)
    for index,layer in enumerate(layers[1:-1],1):
        output_width=storage if mode=='all' or (index==2 and mode!='exact_norm') else layer['w'].shape[1]
        w=jnp.pad(layer['w'],((0,h.shape[-1]-layer['w'].shape[0]),
                              (0,output_width-layer['w'].shape[1])))
        b=jnp.pad(layer['b'],(0,output_width-layer['b'].shape[0]))
        h=h@w+b
        if mode=='exact_norm':
            h=tk._layernorm_silu_pallas_impl(h,layer['ln_scale'],layer['ln_bias'])
            h=jnp.pad(h,((0,0),(0,storage-h.shape[1])))
        else:
            h=aligned_norm(h,layer['ln_scale'],layer['ln_bias'],storage,store_mode)
    w=jnp.pad(layers[-1]['w'],((0,storage-layers[-1]['w'].shape[0]),(0,0)))
    h=h@w+layers[-1]['b']
    return h.reshape(states.shape[0],states.shape[1],200)
