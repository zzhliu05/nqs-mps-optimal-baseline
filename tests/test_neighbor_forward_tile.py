"""The neighbor tile must not alter sampler batches or Hamiltonian capacity."""
import copy
from pathlib import Path
import unittest

from nqs_mps.runtime import environment
from nqs_mps.settings import core_config, load_settings, validate

ROOT = Path(__file__).resolve().parents[1]


class NeighborForwardTileTests(unittest.TestCase):
    def setUp(self):
        self.baseline = load_settings(ROOT/'configs/nphi51_tk0_full_h_0p3m_rollback_2000.json')

    def test_canonical_baseline_is_validated_fast_path(self):
        canonical = load_settings(ROOT/'configs/nphi51_tk0_full_h_0p3m_2000.json')
        counted = load_settings(ROOT/'configs/nphi51_tk0_full_h_0p3m_counted1024_2000.json')
        expected = copy.deepcopy(self.baseline)
        expected['execution'].update(neighbor_forward_tile=1024, neighbor_skip_unused=True)
        self.assertEqual(canonical, counted)
        self.assertEqual(canonical, expected)

    def test_default_preserves_environment(self):
        self.assertIsNone(self.baseline['execution']['neighbor_forward_tile'])
        self.assertNotIn('NQS_NEIGHBOR_FORWARD_TILE', environment(self.baseline))
        self.assertFalse(self.baseline['execution']['neighbor_skip_unused'])
        self.assertNotIn('NQS_NEIGHBOR_SKIP_UNUSED', environment(self.baseline))

    def test_candidate_preserves_capacity_and_other_settings(self):
        candidate = load_settings(ROOT/'configs/nphi51_tk0_full_h_0p3m_stream1024_2000.json')
        expected = copy.deepcopy(self.baseline)
        expected['execution']['neighbor_forward_tile'] = 1024
        self.assertEqual(expected, candidate)
        old, new = environment(self.baseline), environment(candidate)
        self.assertEqual(new.pop('NQS_NEIGHBOR_FORWARD_TILE'), '1024')
        self.assertEqual(old, new)
        self.assertEqual(int(new['NQS_COMPACT_DEDUP_SIZE']), 150*32768)
        config = core_config(candidate, ROOT/'test-output-unused')
        self.assertEqual(config.network.forward_batch_size, 32768)

    def test_reject_invalid_tiles(self):
        for tile in [True, 0, -1, 3, 65536, 1024.0, '1024']:
            with self.subTest(tile=tile):
                settings = copy.deepcopy(self.baseline)
                settings['execution']['neighbor_forward_tile'] = tile
                with self.assertRaises(ValueError):
                    validate(settings)

    def test_reject_bypassed_path(self):
        settings = copy.deepcopy(self.baseline)
        settings['execution']['neighbor_forward_tile'] = 1024
        settings['execution']['neighbor_execution'] = 'packed_scan'
        with self.assertRaises(ValueError):
            validate(settings)

    def test_counted_config_preserves_capacity(self):
        candidate = load_settings(ROOT/'configs/nphi51_tk0_full_h_0p3m_counted1024_2000.json')
        expected = copy.deepcopy(self.baseline)
        expected['execution'].update(neighbor_forward_tile=1024, neighbor_skip_unused=True)
        self.assertEqual(candidate, expected)
        flags = environment(candidate)
        self.assertEqual(flags['NQS_NEIGHBOR_SKIP_UNUSED'], '1')
        self.assertEqual(int(flags['NQS_COMPACT_DEDUP_SIZE']), 4915200)

    def test_counted_config_rejects_invalid_or_unsupported(self):
        for value in [None, 0, 1, 'true']:
            settings = copy.deepcopy(self.baseline)
            settings['execution']['neighbor_skip_unused'] = value
            with self.assertRaises(ValueError):
                validate(settings)
        for execution, tile in [('compact_gpu_dedup', None), ('packed_scan', 1024),
                                ('compact_dedup', 1024)]:
            settings = copy.deepcopy(self.baseline)
            settings['execution'].update(neighbor_skip_unused=True,
                                         neighbor_forward_tile=tile,
                                         neighbor_execution=execution)
            with self.assertRaises(ValueError):
                validate(settings)


if __name__ == '__main__':
    unittest.main()
