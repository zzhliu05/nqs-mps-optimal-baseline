import os
from pathlib import Path
import unittest
from unittest.mock import patch

from nqs_torus_laqx_nes.chain_devices import batch_starts, combine_records, evaluate_chain_batches
from nqs_mps.settings import load_settings, validate
from nqs_mps.runtime import environment


class ChainDeviceTests(unittest.TestCase):
    def test_partition(self):
        self.assertEqual(batch_starts(16384, 4096, 2), [[0, 8192], [4096, 12288]])
        for count in (0, 4096, 12288):
            with self.assertRaises(ValueError):
                batch_starts(count, 4096, 2)

    def test_order_and_statistics(self):
        records = [(i, dict(eloc_parts=[i], wf_count=4096, wf_abs_min=i,
                            wf_abs_max=i+1, wf_abs_sum=2,
                            site_bank=None, use_site_bank=False, strict_site_bank=False))
                   for i in (8192, 0, 12288, 4096)]
        result = combine_records(records)
        self.assertEqual(result['eloc_parts'], [0, 4096, 8192, 12288])
        self.assertEqual(result['wf_count'], 16384)
        self.assertEqual(result['wf_abs_sum'], 8)
        self.assertEqual(result['wf_abs_min'], 0)
        self.assertEqual(result['wf_abs_max'], 12289)

    def test_single_bypass(self):
        with patch.dict(os.environ, {'NQS_LOCAL_ENERGY_DEVICES': '1'}):
            self.assertEqual(evaluate_chain_batches(lambda **kw: kw, {'x': 7}), {'x': 7})

    def test_settings(self):
        root = Path(__file__).resolve().parents[1]
        s = load_settings(root/'configs/nphi51_fullneighbor_16384_cpu_2gpu.json')
        self.assertEqual(environment(s)['NQS_LOCAL_ENERGY_DEVICES'], '2')
        single = load_settings(root/'configs/nphi51_fullneighbor_16384_cpu.json')
        single['execution']['local_energy_devices'] = 2
        self.assertEqual(single, s)
        s['sampling']['chains'] = 12288
        with self.assertRaisesRegex(ValueError, 'device chain batches'):
            validate(s)
        s['sampling']['chains'] = 16384
        s['execution']['local_energy_devices'] = True
        with self.assertRaises(ValueError):
            validate(s)


if __name__ == '__main__':
    unittest.main()
