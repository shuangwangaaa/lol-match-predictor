import unittest
import numpy as np
import torch
from core import sigmoid
from models.deep_models import build_model
from models.tabular_models import NAMES
from pipelines.optimize import calibrate, apply_calibration
from pipelines.blend_experts import fit_slopes, make_design


class TabularModelTests(unittest.TestCase):
    def fixture(self, name):
        torch.manual_seed(17)
        model = build_model(name, {'nd': 78, 'nc': 23, 'leagues': 4, 'sequence_length': 12}).eval()
        if name == 'tabm_lite':
            torch.nn.init.normal_(model.head_weight, std=.1)
        elif name == 'ridge_series':
            torch.nn.init.normal_(model.linear.weight, std=.1)
        else:
            torch.nn.init.normal_(model.head[-1].weight, std=.1)
        values = [torch.randn(3, 78), torch.randn(3, 23), torch.randn(3, 12, 14),
                  torch.randn(3, 12, 14), torch.rand(3, 12) > .3, torch.rand(3, 12) > .3,
                  torch.tensor([0, 1, 2]), torch.randn(3)]
        values[4][0] = False
        values[5][0] = False
        return model, values

    def test_complementarity_and_padding(self):
        for name in NAMES:
            model, args = self.fixture(name)
            d, c, a, b, ma, mb, league, prior = args
            with torch.no_grad():
                original = model(*args)
                swapped = model(-d, c, b, a, mb, ma, league, -prior)
                changed = a.clone()
                changed[~ma] = 10000
                padded = model(d, c, changed, b, ma, mb, league, prior)
            self.assertTrue(torch.isfinite(original).all())
            torch.testing.assert_close(original, -swapped, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(original, padded, atol=2e-5, rtol=2e-5)

    def test_tabm_members_are_distinct_and_receive_gradients(self):
        model, args = self.fixture('tabm_lite')
        model.train()
        values = model.member_logits(*args)
        self.assertEqual(tuple(values.shape), (3, 8))
        self.assertGreater(float(values.std(1).mean().detach()), 1e-5)
        torch.nn.functional.binary_cross_entropy_with_logits(values, torch.ones_like(values)).backward()
        self.assertTrue(torch.isfinite(model.head_weight.grad).all())
        self.assertGreater(float(model.blocks[0].shared.weight.grad.abs().sum()), 0)

    def test_bounded_calibration_and_symmetry(self):
        rng = np.random.default_rng(17)
        prior = rng.normal(size=120)
        raw = prior+rng.normal(0, .3, 120)
        y = (rng.random(120) < sigmoid(prior)).astype(float)
        weights = calibrate(raw, prior, y)
        self.assertTrue(.4 <= weights[0] <= 1.6 and 0 <= weights[1] <= 1)
        np.testing.assert_allclose(apply_calibration(raw, prior, weights)+apply_calibration(-raw, -prior, weights), 1, atol=1e-12)
        x = make_design(prior, raw, prior*.9)
        blend = fit_slopes(x, y)
        self.assertLessEqual(blend[1]+blend[2], 1+1e-8)
        np.testing.assert_allclose(sigmoid(x@blend)+sigmoid(-x@blend), 1, atol=1e-12)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
