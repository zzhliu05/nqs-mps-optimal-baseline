"""Fixed-capacity execution boundaries; a cheap deterministic wavefunction oracle."""
from dataclasses import replace
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from nqs_mps.settings import core_config, load_settings
from nqs_torus_laqx_nes import optimization as opt

ROOT = Path(os.environ.get('NQS_TEST_TEMPLATE_ROOT', Path(__file__).resolve().parents[1]))


def oracle(params, states, cfg):
    return (jnp.sum(states, axis=1, keepdims=True) + params).astype(jnp.complex64)


class CountedForwardTests(unittest.TestCase):
    def setUp(self):
        settings = load_settings(ROOT/'configs/nphi51_tk0_full_h_0p3m_counted1024_2000.json')
        self.cfg = replace(core_config(settings, ROOT/'unused-test-output').network,
                           forward_batch_size=12)
        self.states = jnp.arange(12*51, dtype=jnp.float32).reshape(12,51)
        self.env = patch.dict(os.environ, {'NQS_NEIGHBOR_FORWARD_TILE':'4'}, clear=True)
        self.env.start()
        self.forward = patch.object(opt, 'wavefunction_matrix', oracle)
        self.forward.start()
        opt._cached_neighbor_wavefunction_matrix.clear_cache()

    def tearDown(self):
        opt._cached_neighbor_wavefunction_matrix.clear_cache()
        self.forward.stop()
        self.env.stop()

    def test_counts_preserve_values_shape_and_inverse_mapping(self):
        expected = np.asarray(oracle(1.0, self.states, self.cfg))
        for count in [0,1,3,4,5,8,11,12]:
            for dtype in [jnp.int32, jnp.int64]:
                with self.subTest(count=count,dtype=str(dtype)):
                    actual = np.asarray(opt._cached_neighbor_wavefunction_matrix(
                        1.0, self.states, self.cfg, valid_count=jnp.asarray(count,dtype)))
                    self.assertEqual(actual.shape, expected.shape)
                    np.testing.assert_array_equal(actual[:count], expected[:count])
                    np.testing.assert_array_equal(actual[((count+3)//4)*4:],0)
                    if count:
                        inverse = np.arange(31)%count
                        np.testing.assert_array_equal(actual[inverse], expected[inverse])

    def test_none_uses_original_batched_path(self):
        with patch.object(opt, 'batched_wavefunction_matrix', oracle):
            actual = np.asarray(opt._cached_neighbor_wavefunction_matrix(1.0,self.states,self.cfg))
        np.testing.assert_array_equal(actual, oracle(1.0,self.states,self.cfg))

    def test_runtime_count_does_not_retrace(self):
        traces = []

        def counted_oracle(params, states, cfg):
            traces.append(states.shape)
            return oracle(params, states, cfg)

        with patch.object(opt, 'wavefunction_matrix', counted_oracle):
            opt._cached_neighbor_wavefunction_matrix(
                1.0, self.states, self.cfg, valid_count=jnp.asarray(1,jnp.int32)
            ).block_until_ready()
            initial_traces = len(traces)
            self.assertGreater(initial_traces,0)
            for count in [0,3,4,5,8,11,12]:
                opt._cached_neighbor_wavefunction_matrix(
                    1.0, self.states, self.cfg, valid_count=jnp.asarray(count,jnp.int32)
                ).block_until_ready()
                self.assertEqual(len(traces),initial_traces)

    def test_reject_invalid_count_shape_and_type(self):
        for count in [jnp.asarray([1]), jnp.asarray(1.5), jnp.asarray(True)]:
            with self.assertRaisesRegex(ValueError,'integer scalar'):
                opt._cached_neighbor_wavefunction_matrix(1.0,self.states,self.cfg,valid_count=count)

    def test_capacity_checks_and_bad_tile(self):
        for count in [-1,13]:
            with self.assertRaises(ValueError):
                opt._check_compact_dedup_capacity(count,12)
        with self.assertRaisesRegex(ValueError,'buffer capacity'):
            opt._cached_neighbor_wavefunction_matrix(1.0,self.states[:11],self.cfg,valid_count=4)


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    unittest.main()
