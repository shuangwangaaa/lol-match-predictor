"""Meaningful numerical and temporal invariants for the training pipeline."""
import unittest
from copy import deepcopy
import numpy as np
from core import Series, make_features, metrics, series_probability, sigmoid
from models.elo_mlp import ResidualMLP


def fixture(day, prefix, winner='a'):
    games = []
    for i in [1, 2]:
        games.append([{'teamid': t, 'teamname': t, 'date': f'2024-01-{day:02} 0{i}:00:00',
                       'gameid': f'{prefix}{i}', 'game': str(i), 'year': '2024', 'split': 'Spring',
                       'playoffs': '0', 'league': 'LCK', 'patch': '14.01', 'result': str(int(t == winner)),
                       'gamelength': '1800', 'golddiffat15': str(1000 if t == winner else -1000),
                       'kills': '20' if t == winner else '10'} for t in ['a', 'b']])
    return Series(games)


class ModelTests(unittest.TestCase):
    def test_series_formula(self):
        for p in [.1, .5, .7, .9]:
            self.assertAlmostEqual(series_probability(p, 3), 3*p*p-2*p**3)
            self.assertAlmostEqual(series_probability(p, 5), 10*p**3-15*p**4+6*p**5)

    def test_swap_and_no_spurious_context_advantage(self):
        rng = np.random.default_rng(7)
        model = ResidualMLP(3, 2)
        model.params['w3'][:] = rng.normal(size=(8, 1))
        d, c, o = rng.normal(size=(7, 3)), rng.normal(size=(7, 2)), rng.normal(size=7)
        np.testing.assert_allclose(sigmoid(model.logits(d, c, o))+sigmoid(model.logits(-d, c, -o)), 1, atol=1e-14)
        np.testing.assert_allclose(model.logits(d*0, c, o*0), 0, atol=1e-14)

    def test_backprop_finite_difference(self):
        rng = np.random.default_rng(5)
        model = ResidualMLP(3, 2)
        model.params['w3'][:] = rng.normal(0, .3, (8, 1))
        d, c, o, y = rng.normal(size=(6, 3)), rng.normal(size=(6, 2)), rng.normal(size=6), np.array([0, 1, 0, 1, 1, 0])
        grads = model.gradients(d, c, o, y, .01)
        def objective():
            return metrics(y, sigmoid(model.logits(d, c, o)))['log_loss']+.005*sum(np.sum(w*w) for k, w in model.params.items() if k.startswith('w'))
        for key, param in model.params.items():
            for idx in [tuple(0 for _ in param.shape), tuple(n-1 for n in param.shape)]:
                original = param[idx]
                param[idx] = original+1e-5
                plus = objective()
                param[idx] = original-1e-5
                minus = objective()
                param[idx] = original
                self.assertAlmostEqual(grads[key][idx], (plus-minus)/2e-5, places=7)

    def test_current_and_future_outcomes_do_not_change_frozen_features(self):
        series = [fixture(1, 'x'), fixture(2, 'y'), fixture(3, 'z')]
        original, _ = make_features(series)
        changed = deepcopy(series)
        # Flip every game of today's series; format stays BO3 while winner changes.
        for game in changed[1].games:
            for row in game:
                row['result'] = str(1-int(row['result']))
                row['golddiffat15'] = '99999'
        altered, _ = make_features(changed)
        for before, after in zip(original, altered):
            np.testing.assert_array_equal(before['d'], after['d'])
            np.testing.assert_array_equal(before['c'], after['c'])
            self.assertEqual(before['elo_logit'], after['elo_logit'])
        self.assertNotEqual(original[1]['y'], altered[1]['y'])

    def test_invalid_series_excluded(self):
        s = fixture(1, 'x')
        self.assertEqual(s.best_of, 3)
        s.valid_sequence = False
        self.assertIsNone(s.best_of)


if __name__ == '__main__':
    unittest.main()
