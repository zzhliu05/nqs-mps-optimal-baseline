"""Opt-in GPU compaction of valid proposals; proposal and RNG code unchanged."""
import ast,inspect,json,textwrap
from pathlib import Path
import numpy as np


def install(out,tile=1024):
    import jax
    import jax.numpy as jnp
    from nqs_torus_laqx_nes import sampling
    original=sampling._conserved_pair_sample_scan_fn
    logprob=sampling._logprob_from_packed_batches
    def masked(params,proposal,cfg,valid,current):
        n=proposal.shape[0]
        if n%tile:raise ValueError('sampler tile must divide chain count')
        indices=jnp.nonzero(valid,size=n,fill_value=0)[0]
        packed=proposal[indices]
        count=jnp.sum(valid.astype(jnp.int32))
        def body(i,values):
            chunk=jax.lax.dynamic_slice(packed,(i*tile,0),(tile,1))
            value=logprob(params,chunk,cfg)
            return jax.lax.dynamic_update_slice(values,value,(i*tile,))
        values=jax.lax.fori_loop(0,(count+tile-1)//tile,body,jnp.zeros_like(current))
        rank=jnp.maximum(jnp.cumsum(valid.astype(jnp.int32))-1,0)
        return jnp.where(valid,values[rank],current)

    validation={}
    def checked(params,proposal,cfg,valid,current):
        candidate=masked(params,proposal,cfg,valid,current)
        reference=logprob(params,proposal,cfg)
        def check(a,b,v):
            error=float(np.max(np.abs(a[v]-b[v]))) if np.any(v) else 0.
            if not np.isfinite(error) or error>1e-5:
                raise ValueError(f'masked sampler valid logprob error {error}')
            validation.update(valid_fraction=float(np.mean(v)),max_valid_logprob_error=error)
        jax.debug.callback(check,candidate,reference,valid,ordered=True)
        return candidate

    # Transform only the forward call. Keep original proposal, random split,
    # acceptance and scan logic verbatim, and fail if its source shape changes.
    source=textwrap.dedent(inspect.getsource(original.__wrapped__))
    class Transform(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='_logprob_from_packed_batches':
                self.count+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='_masked_forward',ctx=ast.Load()),
                    args=node.args+[ast.Name(id='valid',ctx=ast.Load()),
                                    ast.Name(id='current_logprob',ctx=ast.Load())],keywords=node.keywords),node)
            return self.generic_visit(node)
    tree=ast.parse(source);function=tree.body[0];function.decorator_list=[]
    transform=Transform();tree=ast.fix_missing_locations(transform.visit(tree))
    if transform.count!=1:raise ValueError('unexpected sampler source; refusing transform')
    static=('net_cfg','n_chains','n_target_chains','nphi','ne','n_burn','n_between','n_samples')
    def make(helper):
        namespace=dict(original.__wrapped__.__globals__,_masked_forward=helper)
        exec(compile(tree,'<isolated masked sampler>','exec'),namespace)
        return jax.jit(namespace[function.name],static_argnames=static)
    candidate=make(masked);validator=make(checked)
    signature=inspect.signature(original.__wrapped__)
    first=True
    def wrapper(*args,**kwargs):
        nonlocal first
        if first:
            bound=signature.bind(*args,**kwargs);bound.apply_defaults()
            d=dict(bound.arguments,n_burn=0,n_between=1,n_samples=1)
            reference=jax.device_get(original(**d))
            actual=jax.device_get(validator(**d))
            # Key, positions, samples and accepted counts must agree exactly.
            for i in (0,1,3,4,5):np.testing.assert_array_equal(actual[i],reference[i])
            np.testing.assert_allclose(actual[2],reference[2],rtol=0,atol=1e-5)
            validation.update(one_transition_acceptance_identical=True,tile=tile)
            (Path(out)/'masked_sampler_validation.json').write_text(json.dumps(validation,indent=2))
            first=False
        return candidate(*args,**kwargs)
    sampling._conserved_pair_sample_scan_fn=wrapper
