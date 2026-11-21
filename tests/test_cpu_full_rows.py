import os
import unittest
from unittest.mock import patch
import numpy as np
from nqs_torus_laqx_nes.cpu_scatter_masks import prepare
from nqs_torus_laqx_nes.cpu_full_rows import build_chunks,ChunkPackedRows,pack_rows
from nqs_torus_laqx_nes.physics import (
    TorusCoulombHamiltonian,SparseHamiltonianBitstringRows,pack_sparse_bitstring_rows,
    _build_sparse_bitstring_rows_numba_fixed)
from nqs_torus_laqx_nes.config import SystemConfig


class FullRowsTests(unittest.TestCase):
    def test_all_neighbors_chunk_boundaries_and_overflow(self):
        rng=np.random.default_rng(1042)
        terms=np.array([(*rng.choice(np.arange(17,51),2,replace=False),*rng.choice(17,2,replace=False)) for _ in range(1600)],np.int64)
        a,b,c,d=[np.ascontiguousarray(x) for x in terms.T]
        coeff=rng.normal(size=1600)+1j*rng.normal(size=1600)
        states=np.full(70,(1<<17)-1,np.uint64)
        prepared=prepare(a,b,c,d,coeff)
        for width in (2,1856):
            diag,bits,vals,counts,overflow=_build_sparse_bitstring_rows_numba_fixed(states,a,b,c,d,coeff,width)
            actual=build_chunks(states,*prepared,width)
            np.testing.assert_array_equal(actual[0],diag)
            np.testing.assert_array_equal(actual[3],counts)
            np.testing.assert_array_equal(actual[5],overflow)
            if width==1856:self.assertGreater(int(counts.min()),1024)
            ref=SparseHamiltonianBitstringRows(diag,bits,vals,(np.arange(width)[None,:]<counts[:,None]).astype(float))
            packed=ChunkPackedRows(*actual[:5])
            for dtype in (np.complex64,np.complex128):
                expected=pack_sparse_bitstring_rows(ref,len(states)*width,dtype)
                got=pack_sparse_bitstring_rows(packed,len(states)*width,dtype)
                for x,y in zip(got,expected):np.testing.assert_array_equal(x,y)
            with self.assertRaisesRegex(ValueError,'capacity'):pack_rows(packed,0,np.complex128)

    def test_public_cache_and_validation(self):
        ham=TorusCoulombHamiltonian(SystemConfig(nphi=9))
        rng=np.random.default_rng(414)
        ham.coulomb_matrix=rng.normal(size=(9,9,9,9))+1j*rng.normal(size=(9,9,9,9))
        states=np.array([[7],[56],[224]],np.uint64)
        with patch.dict(os.environ,{'NQS_CPU_ROW_BUILDER':'linear'}):ref=ham.build_sparse_hamiltonian_bitstring_rows(states,50)
        with patch.dict(os.environ,{'NQS_CPU_ROW_BUILDER':'masked_full'}):
            got=ham.build_sparse_hamiltonian_bitstring_rows(states,50)
            self.assertIsInstance(got,ChunkPackedRows)
            cached=ham._cpu_full_mask_cache
            ham.build_sparse_hamiltonian_bitstring_rows(states,50)
            self.assertIs(cached,ham._cpu_full_mask_cache)
            np.testing.assert_array_equal(got.diag,ref.diag)
            for x,y in zip(pack_sparse_bitstring_rows(got,150),pack_sparse_bitstring_rows(ref,150)):np.testing.assert_array_equal(x,y)
            with self.assertRaises(ValueError):ham.coulomb_matrix[0,0,0,0]=1
            ham.coulomb_matrix=ham.coulomb_matrix.copy()*2
            doubled=ham.build_sparse_hamiltonian_bitstring_rows(states,50)
            np.testing.assert_array_equal(doubled.diag,2*ref.diag)
            for x,y in zip(pack_sparse_bitstring_rows(doubled,150),pack_sparse_bitstring_rows(ref,150)):
                np.testing.assert_array_equal(x,2*y if np.iscomplexobj(y) else y)
            with self.assertRaisesRegex(ValueError,'smaller than'):ham.build_sparse_hamiltonian_bitstring_rows(states,1)
            with self.assertRaises(ValueError):ham.build_sparse_hamiltonian_bitstring_rows(np.array([512],np.uint64),50)

    def test_preset_and_guard(self):
        from pathlib import Path
        from nqs_mps.settings import load_settings,validate
        from nqs_mps.runtime import environment
        root=Path(__file__).resolve().parents[1]
        s=load_settings(root/'configs/nphi51_fullneighbor_16384_cpu.json')
        self.assertEqual(s['sampling']['chains'],16384)
        self.assertEqual(environment(s)['NQS_CPU_ROW_BUILDER'],'masked_full')
        s['execution']['neighbor_execution']='dense'
        with self.assertRaisesRegex(ValueError,'masked_full'):validate(s)

if __name__=='__main__':unittest.main()
