import unittest
import numpy as np
from nqs_torus_laqx_nes.cpu_row_hash import build


def oracle(states, terms, coeff, width):
    diag = np.zeros(len(states), complex)
    bits = np.zeros((len(states), width), np.uint64)
    vals = np.zeros((len(states), width), complex)
    counts = np.zeros(len(states), np.int64)
    overflow = np.zeros(len(states), np.uint8)
    for r, state in enumerate(states):
        state = int(state)
        values = {}
        for (a,b,c,d), h in zip(terms, coeff):
            x, sign = state, 1
            if not (x >> c & 1 and x >> d & 1):
                continue
            for site, create in ((d,False),(c,False),(b,True),(a,True)):
                if create and (x >> site & 1):
                    break
                if (x >> (site+1)).bit_count() % 2:
                    sign = -sign
                x = x | (1 << site) if create else x & ~(1 << site)
            else:
                h = np.conjugate(h * sign)
                if abs(h) <= 1e-14:
                    continue
                if x == state:
                    diag[r] += h
                elif x in values:
                    values[x] += h
                elif len(values) < width:
                    values[x] = h
                else:
                    overflow[r] = 1
        for x, h in values.items():
            if abs(h) > 1e-14:
                k = counts[r]
                bits[r,k], vals[r,k] = x,h
                counts[r] += 1
    return diag,bits,vals,counts,overflow


class HashTests(unittest.TestCase):
    def test_order_duplicates_cancellation_overflow(self):
        rng = np.random.default_rng(4)
        for orbitals in (8,51,63):
            terms = np.array([rng.choice(orbitals,4,replace=False) for _ in range(300)], np.int64)
            terms = np.concatenate((terms,terms,[[0,1,0,1]]))
            v = rng.normal(size=300) + 1j*rng.normal(size=300)
            coeff = np.concatenate((v,-v,[1+2j]))
            states = np.array([sum(1<<int(i) for i in rng.choice(orbitals,orbitals//3,replace=False)) for _ in range(20)],np.uint64)
            for width in (1,1000):
                expected = oracle(states,terms,coeff,width)
                actual = build(states,*[np.ascontiguousarray(x) for x in terms.T],coeff,width)
                for a,b in zip(actual,expected):
                    np.testing.assert_array_equal(a,b)



class IntegrationTests(unittest.TestCase):
    def test_configuration_and_rollback(self):
        from pathlib import Path
        from nqs_mps.settings import load_settings, validate
        from nqs_mps.runtime import environment
        root = Path(__file__).resolve().parents[1]
        path = root/'config.json'
        if not path.exists():
            path = root/'configs/nphi51_baseline_probe.json'
        settings = load_settings(path)
        self.assertEqual(settings['execution']['cpu_row_builder'], 'hash')
        self.assertEqual(environment(settings)['NQS_CPU_ROW_BUILDER'], 'hash')
        settings['execution']['cpu_row_builder'] = 'linear'
        validate(settings)
        self.assertEqual(environment(settings)['NQS_CPU_ROW_BUILDER'], 'linear')
        settings['execution']['cpu_row_builder'] = 'typo'
        with self.assertRaisesRegex(ValueError, 'cpu_row_builder'):
            validate(settings)

    def test_public_builder(self):
        from unittest.mock import patch
        import os
        from nqs_torus_laqx_nes.physics import TorusCoulombHamiltonian
        from nqs_torus_laqx_nes.config import SystemConfig
        ham = TorusCoulombHamiltonian(SystemConfig(nphi=9))
        rng = np.random.default_rng(42)
        ham.coulomb_matrix = rng.normal(size=(9,9,9,9)) + 1j*rng.normal(size=(9,9,9,9))
        states = np.array([[7],[56],[224]],np.uint64)
        with patch.dict(os.environ, {'NQS_CPU_ROW_BUILDER':'linear'}):
            ref = ham.build_sparse_hamiltonian_bitstring_rows(states, 50)
        with patch.dict(os.environ, {'NQS_CPU_ROW_BUILDER':'hash'}):
            with patch('nqs_torus_laqx_nes.cpu_row_hash.build', wraps=build) as spy:
                actual = ham.build_sparse_hamiltonian_bitstring_rows(states, 50)
                spy.assert_called_once()
        for key in ('diag','neighbor_bitstrings','vals','mask'):
            np.testing.assert_array_equal(getattr(actual,key),getattr(ref,key))
        for mode in ('linear','hash'):
            with patch.dict(os.environ, {'NQS_CPU_ROW_BUILDER':mode}):
                with self.assertRaisesRegex(ValueError,'smaller than'):
                    ham.build_sparse_hamiltonian_bitstring_rows(states,1)

if __name__ == '__main__':
    unittest.main()
