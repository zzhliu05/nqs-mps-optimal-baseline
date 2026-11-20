import unittest
import numpy as np
from estimators import plan
from compiled_proposal import build, draw, build_parallel, draw_parallel


class ProposalTests(unittest.TestCase):
    def test_parallel_identical(self):
        rng=np.random.default_rng(2)
        h=rng.normal(size=(32,100))+1j*rng.normal(size=(32,100))
        h[0]=0;h[1,:]=1;h[2,4:]=0
        u=rng.random((32,32))
        for power in (0.,.5,1.,2.):
            serial=build(h,8,power)
            parallel=build_parallel(h,8,power)
            for a,b in zip(serial,parallel):np.testing.assert_array_equal(a,b)
            for a,b in zip(draw(*serial,u),draw_parallel(*parallel,u)):
                np.testing.assert_array_equal(a,b)

    def test_reference(self):
        rng = np.random.default_rng(2)
        h = rng.normal(size=(16, 100)).astype(np.float32)
        h[:, ::5] = 0
        h[0] = 0
        h[1, 4:] = 0
        for power in (0., .5, 1., 2.):
            s, q, cdf = build(h, 8, power)
            for r in range(len(h)):
                exact, tail, reference = plan(h[r], top=8, power=power)
                np.testing.assert_array_equal(s[r][s[r]>=0], exact)
                np.testing.assert_allclose(q[r], reference, atol=1e-15)
            cols, weights = draw(s, q, cdf, rng.random((len(h), 100)))
            self.assertTrue(np.isfinite(weights).all())
            self.assertTrue((weights[:2, 8:] == 0).all())

    def test_moments(self):
        h = np.array([[1., -2., 3., -.2, .8]])
        a = np.array([.2+1j, 3-.2j, -2j, 2., -.5])
        s, q, cdf = build(h, 1)
        cols, weights = draw(s, q, cdf, np.random.default_rng(7).random((1, 200000)))
        estimate = np.sum(a[cols]*weights)
        self.assertLess(abs(estimate-a.sum()), .04)


if __name__ == '__main__':
    unittest.main()
