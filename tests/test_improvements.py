import unittest
import numpy as np
from pipelines.improve import Ridge, GameRidge
from core import sigmoid


class ImprovementTests(unittest.TestCase):
    def test_ridge_is_antisymmetric_after_fit(self):
        rng = np.random.default_rng(19)
        d = rng.normal(size=(120, 8))
        c = rng.normal(size=(120, 23))
        c[:, 2] = np.where(np.arange(120)%2, .6, 1)
        y = (rng.random(120) < sigmoid(d[:, 0])).astype(float)
        model = Ridge(.1).fit(d, c, d[:, 0], y, np.ones(120))
        np.testing.assert_allclose(model.logits(d, c, d[:, 0]), -model.logits(-d, c, -d[:, 0]), atol=1e-12)

    def test_scoreline_model_preserves_swap_and_series_format(self):
        rng = np.random.default_rng(20)
        d = rng.normal(0, .1, size=(120, 8))
        c = rng.normal(size=(120, 23))
        c[:, 2] = np.where(np.arange(120)%2, .6, 1)
        fractions = rng.integers(0, 4, 120)/3
        model = GameRidge(.1).fit(d, c, np.zeros(120), fractions, np.ones(120))
        p = sigmoid(model.logits(d, c, np.zeros(120)))
        swapped = sigmoid(model.logits(-d, c, np.zeros(120)))
        np.testing.assert_allclose(p+swapped, 1, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
