"""The adopted model is selected both by explicit and fallback configuration."""
from pathlib import Path
import tempfile
import unittest
from core import ROOT
from main import load_config


class ReleaseDefaultsTests(unittest.TestCase):
    def test_project_default_selects_weighted_model(self):
        config = load_config(ROOT/'config.toml')
        self.assertEqual(config['model'], 'roster_transformer_2026_swiss_weighted')
        self.assertEqual(config['stage'], 'finetuned')
        self.assertEqual(config['seeds'], [42, 43, 44])

    def test_missing_configuration_keeps_adopted_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory)/'absent.toml')
        self.assertEqual(config['model'], 'roster_transformer_2026_swiss_weighted')


if __name__ == '__main__':
    unittest.main()
