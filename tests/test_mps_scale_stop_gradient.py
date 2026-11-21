"""CPU regression for compensated scales, away from clipping and nodes."""
import os
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from nqs_torus_laqx_nes import network
from nqs_torus_laqx_nes import triton_kernels
from nqs_torus_laqx_nes.mps_scaling import contraction_scale


class ScaleTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get('NQS_MPS_SCALE_STOP_GRADIENT')

    def tearDown(self):
        if self.previous is None:
            os.environ.pop('NQS_MPS_SCALE_STOP_GRADIENT', None)
        else:
            os.environ['NQS_MPS_SCALE_STOP_GRADIENT'] = self.previous
        jax.clear_caches()

    def test_scale_derivative(self):
        os.environ['NQS_MPS_SCALE_STOP_GRADIENT'] = '1'
        self.assertEqual(float(jax.grad(lambda x: contraction_scale(x))(2.)), 0.)
        os.environ['NQS_MPS_SCALE_STOP_GRADIENT'] = '0'
        self.assertEqual(float(jax.grad(lambda x: contraction_scale(x))(2.)), 1.)

    def test_forward_and_gradient(self):
        rng = np.random.default_rng(51)
        for n in (5, 36, 51):
            a = jnp.asarray((np.eye(2)[None,None,None] + .03*(rng.normal(size=(2,n,2,2,2)) + 1j*rng.normal(size=(2,n,2,2,2)))).astype(np.complex64))
            for fn in (network.contract_periodic_mps, network.contract_periodic_mps_balanced, network.contract_periodic_mps_fused4, network.contract_periodic_mps_balanced_rsqrt):
                results = []
                for flag in ('0', '1'):
                    os.environ['NQS_MPS_SCALE_STOP_GRADIENT'] = flag
                    jax.clear_caches()
                    y = fn(a, 80.)
                    g = jax.grad(lambda x: jnp.sum(jnp.log(jnp.abs(fn(x,80.)))))(a)
                    results.append((np.asarray(y), np.asarray(g)))
                np.testing.assert_array_equal(results[0][0], results[1][0])
                np.testing.assert_allclose(results[0][1], results[1][1], rtol=3e-4, atol=3e-5)

    def test_fused_reference_scale_derivative(self):
        rng = np.random.default_rng(39)
        for fn, sites in ((triton_kernels._realblock_triplet_reference, 3), (triton_kernels._realblock_quad_reference, 4)):
            x = jnp.asarray(rng.normal(size=(1,sites,200)).astype(np.float32)*.1)
            os.environ['NQS_MPS_SCALE_STOP_GRADIENT'] = '1'
            grad = jax.grad(lambda a: jnp.sum(fn(a)[1]))(x)
            np.testing.assert_array_equal(np.asarray(grad), np.zeros(x.shape))


if __name__ == '__main__':
    unittest.main()
