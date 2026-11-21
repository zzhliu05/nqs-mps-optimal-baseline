import copy
from pathlib import Path
import unittest
from nqs_mps.settings import load_settings, validate
from nqs_mps.runtime import environment


class PrepackedSettingsTest(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(Path(__file__).parents[1] / 'configs/nphi51_baseline_probe.json')

    def test_default_disabled(self):
        self.assertFalse(self.settings['execution']['neighbor_prepacked'])
        self.assertEqual(environment(self.settings)['NQS_NEIGHBOR_PREPACKED'], '0')

    def test_supported_and_unsupported(self):
        e = self.settings['execution']
        e.update(neighbor_prepacked=True, neighbor_skip_unused=True, neighbor_forward_tile=1024)
        validate(self.settings)
        self.assertEqual(environment(self.settings)['NQS_NEIGHBOR_PREPACKED'], '1')
        for section, key, value in [('network', 'hidden_sizes', [296,296]),
                                    ('system', 'nphi', 36),
                                    ('execution', 'neighbor_skip_unused', False),
                                    ('execution', 'neighbor_prepacked', 'true')]:
            changed = copy.deepcopy(self.settings)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                validate(changed)

    def test_tuning_requires_prepacked(self):
        self.settings['execution']['neighbor_prepacked_tuned']=True
        with self.assertRaisesRegex(ValueError,'requires neighbor_prepacked'):
            validate(self.settings)
        self.settings['execution'].update(neighbor_prepacked=True,neighbor_skip_unused=True,neighbor_forward_tile=1024)
        validate(self.settings)
        self.assertEqual(environment(self.settings)['NQS_NEIGHBOR_PREPACKED_TUNED'],'1')


if __name__ == '__main__':
    unittest.main()
