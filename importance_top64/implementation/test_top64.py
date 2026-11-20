import unittest
import numpy as np
from fused_cpu_pack import fused_pack
from compiled_proposal import build,draw


class Top64Tests(unittest.TestCase):
    def test_reference(self):
        rng=np.random.default_rng(91)
        for width in (20,120):
            h=rng.normal(size=(8,width))+1j*rng.normal(size=(8,width))
            h[0]=0;h[1]=1;h[2,8:]=0
            bits=np.arange(1,h.size+1,dtype=np.uint64).reshape(h.shape)
            u=rng.random((len(h),32))
            s,q,cdf=build(h,64)
            col,w=draw(s,q,cdf,u)
            b,v,o=fused_pack(bits,h,np.ones(h.shape),u,64)
            np.testing.assert_array_equal(b.reshape(-1,96),np.where(w!=0,np.take_along_axis(bits,col,1),0))
            np.testing.assert_array_equal(v.reshape(-1,96),np.take_along_axis(h,col,1)*w)
            np.testing.assert_array_equal(o,np.arange(9)*96)


if __name__=='__main__':unittest.main()
