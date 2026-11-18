"""CPU-interpreted regressions for complete sparse Hamiltonian row sums."""
import jax
import jax.numpy as jnp
import numpy as np
import unittest

from nqs_torus_laqx_nes.triton_kernels import compact_row_reduce_pallas

jax.config.update('jax_enable_x64', True)


def check_rows_cross_tile_boundaries(width):
    lengths = np.array([0, 1, 255, 256, 257, 1023, 1024, 1025, 1400,
                        2047, 2048, 2049, 8193], dtype=np.int32)
    offsets = np.concatenate(([0], lengths.cumsum())).astype(np.int32)
    rng = np.random.default_rng(20261118)
    count = int(offsets[-1])
    coeff = (rng.normal(size=count) + 1j * rng.normal(size=count)).astype(np.complex64)
    phi = (rng.normal(size=count) + 1j * rng.normal(size=count)).astype(np.complex64)
    expected = np.array([np.sum(coeff[a:b].astype(np.complex128) * phi[a:b])
                         for a, b in zip(offsets[:-1], offsets[1:])])
    actual = np.asarray(compact_row_reduce_pallas(
        coeff, phi, offsets, block_width=width, interpret=True))
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_nonzero_contribution_after_first_tile():
    coeff = np.zeros(1400, dtype=np.complex64)
    coeff[1024:] = 1 + 2j
    actual = np.asarray(compact_row_reduce_pallas(
        coeff, np.ones_like(coeff), np.array([0, 1400], dtype=np.int32), interpret=True))
    np.testing.assert_array_equal(actual, [376 + 752j])


class CompleteRowReductionTests(unittest.TestCase):
    def test_boundaries(self):
        for width in [256, 1024, 2048]:
            with self.subTest(width=width):
                check_rows_cross_tile_boundaries(width)

    def test_tail_only(self):
        test_nonzero_contribution_after_first_tile()

    def test_reject_invalid_width(self):
        for width in [0, -1, 3]:
            with self.subTest(width=width):
                with self.assertRaisesRegex(ValueError, 'positive power of two'):
                    compact_row_reduce_pallas(jnp.ones(1), jnp.ones(1), jnp.array([0, 1]),
                                              block_width=width, interpret=True)


if __name__ == '__main__':
    unittest.main()
