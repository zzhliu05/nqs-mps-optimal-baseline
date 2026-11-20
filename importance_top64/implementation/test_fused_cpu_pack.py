import os,unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from cpu_sample_pack import sample_pack


class FusedTests(unittest.TestCase):
    def test_reference(self):
        rng=np.random.default_rng(5)
        for width in (12,100):
            h=rng.normal(size=(8,width))+1j*rng.normal(size=(8,width))
            h[0]=0;h[1]=1;h[2,5:]=0
            mask=np.ones(h.shape);mask[:,::9]=0
            rows=SimpleNamespace(vals=h,mask=mask,
                                 neighbor_bitstrings=np.arange(1,h.size+1,dtype=np.uint64).reshape(h.shape))
            with patch.dict(os.environ,{'IS_FUSED_CPU_PACK':'0'}):
                expected,_=sample_pack(rows,np.random.default_rng(10))
            with patch.dict(os.environ,{'IS_FUSED_CPU_PACK':'1'}):
                actual,_=sample_pack(rows,np.random.default_rng(10))
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
