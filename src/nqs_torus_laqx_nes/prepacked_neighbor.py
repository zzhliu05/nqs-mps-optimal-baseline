"""Pack invariant weights once per invocation, outside the counted GPU loop."""
from types import FunctionType
from types import SimpleNamespace
import jax
import jax.numpy as jnp
from nqs_torus_laqx_nes import optimization, triton_kernels as tk
from .aligned_neighbor_mlp import aligned_norm


@jax.jit
def pack(params):
    layers = params['mps']['tensor_mlp']
    assert len(layers) == 4 and layers[2]['w'].shape == (397, 397)
    return dict(original=params,
                hidden_w=jnp.pad(layers[2]['w'], ((0, 3), (0, 3))),
                hidden_b=jnp.pad(layers[2]['b'], (0, 3)),
                output_w=jnp.pad(layers[3]['w'], ((0, 3), (0, 0))))


def wavefunction(packed, states, cfg):
    if cfg.num_sites != 51 or cfg.k != 1 or cfg.mps_bond_dim != 10 or cfg.activation != 'silu':
        raise ValueError('Prepacked experiment requires N51 k1 D10 SiLU')
    layers = packed['original']['mps']['tensor_mlp']
    h = tk.reuse_first_layernorm_silu(layers[0], states)
    h = h @ layers[1]['w'] + layers[1]['b']
    h = aligned_norm(h, layers[1]['ln_scale'], layers[1]['ln_bias'], 400, 'scalar_io')
    h = h @ packed['hidden_w'] + packed['hidden_b']
    h = aligned_norm(h, layers[2]['ln_scale'], layers[2]['ln_bias'], 400, 'scalar_io')
    h = h @ packed['output_w'] + layers[3]['b']
    matrices = h.reshape(states.shape[0], states.shape[1], 200)
    return tk.persistent_realblock_triplet_tail_pallas(matrices, cfg.logphi_real_clip)


def make_candidate(tuned=False):
    source = optimization._standard_cached_neighbor_wavefunction_matrix.__wrapped__
    forward=wavefunction
    if tuned:
        def norm(x,s,b,storage,mode):
            return aligned_norm(x,s,b,storage,mode,block_rows=8)
        def final(segments,norms,*,num_warps=4):
            return tk.realblock_quad_tail_pallas(segments,norms,num_warps=2)
        original=tk.persistent_realblock_triplet_tail_pallas
        contract=FunctionType(original.__code__,dict(original.__globals__,realblock_quad_tail_pallas=final),
                              original.__name__,original.__defaults__,original.__closure__)
        contract.__kwdefaults__=original.__kwdefaults__
        proxy=SimpleNamespace(**dict(vars(tk),persistent_realblock_triplet_tail_pallas=contract))
        forward=FunctionType(wavefunction.__code__,dict(wavefunction.__globals__,aligned_norm=norm,tk=proxy))
    cloned = FunctionType(source.__code__, dict(source.__globals__, wavefunction_matrix=forward),
                          source.__name__, source.__defaults__, source.__closure__)
    cloned.__kwdefaults__ = source.__kwdefaults__
    loop = jax.jit(cloned, static_argnames=('cfg',))

    # Intentionally not jitted together: prevent re-sinking the packing into the loop.
    def checked(params, states, cfg, valid_count=None):
        if valid_count is None:
            raise ValueError('Counted neighbor execution required')
        return loop(pack(params), states, cfg, valid_count=valid_count)

    checked.packed_loop = loop
    return checked
