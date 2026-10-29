import json
import csv
import gzip
from pathlib import Path
import tempfile
import unittest
import numpy as np
from data_gen.regional import RegionalRatings, adjust_features, FEATURE_NAMES, region_for_league
from data_gen.features_v2 import load_players


class RegionalTests(unittest.TestCase):
    def fixture(self):
        r = RegionalRatings()
        r.assign('a', 'LCK')
        r.assign('b', 'LPL')
        r.assign('c', 'LCS')
        return r

    def test_coefficients_are_strengths_and_probability_is_complementary(self):
        r = self.fixture()
        _, _, p = r.pair('a', 'b', '2024-01-01')
        np.testing.assert_allclose(p, .4/(.4+.3))
        np.testing.assert_allclose(p+r.pair('b', 'a', '2024-01-01')[2], 1)
        np.testing.assert_allclose(r.pair('b', 'c', '2024-01-01')[2], .3/(.3+.2/3))
        weights = r.coefficients('2024-01-01')
        self.assertEqual(set(weights), {'LCK', 'LPL', 'LEC', 'LCS', 'LCP', 'CBLOL'})
        self.assertAlmostEqual(sum(v['prior'] for v in weights.values()), 1.)
        self.assertAlmostEqual(sum(v['current'] for v in weights.values()), 1.)
        self.assertAlmostEqual(weights['LEC']['prior'], .1)
        self.assertAlmostEqual(weights['LCS']['prior'], .2/3)

    def test_explicit_registry_splits_other_mass_and_rejects_invalid_prior(self):
        r = RegionalRatings({'regions': ['LCK', 'LPL', 'LEC', 'LCS', 'LCP', 'CBLOL', 'LLA']})
        weights = r.coefficients('2024-01-01')
        self.assertAlmostEqual(weights['LLA']['prior'], .05)
        self.assertAlmostEqual(sum(v['prior'] for v in weights.values()), 1.)
        for invalid in [{'others_total': .3}, {'other': .3},
                        {'regions': ['LCK', 'LPL', 'LEC']},
                        {'regions': ['LCK', 'LPL', 'LEC', 'LCS', 'LCS']}]:
            with self.assertRaises(ValueError):
                RegionalRatings(invalid)

    def test_weights_stay_positive_and_normalized_through_updates_and_decay(self):
        r = self.fixture()
        for i in range(30):
            r.update('a', 'b', '2024-01-01', i % 2, 'MSI')
            weights = r.weights('2024-01-01')
            self.assertAlmostEqual(sum(weights.values()), 1., places=14)
            self.assertTrue(all(v > 0 for v in weights.values()))
        self.assertAlmostEqual(sum(r.weights('2035-01-01').values()), 1., places=14)
        np.testing.assert_allclose(list(r.weights('2050-01-01').values()),
                                   [r.prior(name) for name in r.config['regions']], atol=1e-8)

    def test_only_known_international_games_update_region(self):
        r = self.fixture()
        r.assign('d', 'LCK')
        r.update('a', 'd', '2024-01-01', 1, 'LCK')
        self.assertEqual(r.international_updates, 0)
        self.assertAlmostEqual(r.coefficients('2024-01-01')['LCK']['current'], .4)
        r.update('a', 'b', '2024-01-02', 0, 'MSI')
        self.assertEqual(r.international_updates, 1)
        self.assertLess(r.coefficients('2024-01-02')['LCK']['current'], .4)
        self.assertGreater(r.coefficients('2024-01-02')['LPL']['current'], .3)
        r.update('a', 'unknown', '2024-01-03', 1, 'WCS')
        self.assertEqual(r.international_updates, 1)
        self.assertEqual(r.unknown_international_games, 1)

    def test_home_region_survives_cups_and_promotion_names(self):
        r = self.fixture()
        r.assign('a', 'KeSPA Cup')
        r.assign('a', 'EM')
        self.assertEqual(r.origins['a'], 'LCK')
        self.assertEqual(region_for_league('CBLOL Promotion'), 'CBLOL')
        self.assertEqual(region_for_league('LTA N Promotion'), 'LCS')
        self.assertEqual(region_for_league('LCKC'), 'LCK')
        self.assertEqual(region_for_league('LFL'), 'LEC')
        r.assign('legacy', 'LLA')
        self.assertEqual(r.origins['legacy'], 'LCS')

    def test_player_cutoff_follows_available_team_history(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'players.csv.gz'
            fields = ['date', 'game_id', 'side', 'position', 'result']
            with gzip.open(path, 'wt', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                for side in ['Blue', 'Red']:
                    for role in ['top', 'jng', 'mid', 'bot', 'sup']:
                        writer.writerow({'date': '2026-10-30', 'game_id': 'later_game', 'side': side,
                                         'position': role, 'result': str(side == 'Blue')})
            _, days, count = load_players(path, max_day='2026-10-31')
            self.assertEqual(count, 10)
            self.assertIn('2026-10-30', days)
            _, days, count = load_players(path, max_day='2026-10-26')
            self.assertEqual(count, 0)
            self.assertFalse(days)

    def test_opponent_strength_changes_form_for_same_outcome(self):
        weak, strong = self.fixture(), self.fixture()
        strong.teams['b'] = {'local': [300.]*6, 'last_day': '2024-01-01', 'history': []}
        weak.update('a', 'b', '2024-01-01', 1, 'MSI')
        strong.update('a', 'b', '2024-01-01', 1, 'MSI')
        fw, fs = weak.features('a', '2024-01-02'), strong.features('a', '2024-01-02')
        for field in ['opponent_adjusted_form_5', 'schedule_strength_5', 'schedule_adjusted_performance_5']:
            self.assertGreater(fs[FEATURE_NAMES.index(field)], fw[FEATURE_NAMES.index(field)])

    def test_read_only_forecasts_and_exact_state_restoration(self):
        r = self.fixture()
        r.update('a', 'b', '2024-01-01', 1, 'MSI')
        before = json.dumps(r.state(), sort_keys=True)
        r.features('a', '2025-01-01')
        r.coefficients('2025-01-01')
        self.assertEqual(before, json.dumps(r.state(), sort_keys=True))
        restored = RegionalRatings.from_state(json.loads(before))
        np.testing.assert_array_equal(r.features('a', '2025-01-01'), restored.features('a', '2025-01-01'))
        np.testing.assert_array_equal(r.ratings('a', '2025-01-01'), restored.ratings('a', '2025-01-01'))

    def test_region_deviations_are_bounded_and_decay_to_prior(self):
        r = RegionalRatings({'region_max_log_deviation': .01})
        r.assign('a', 'LCK')
        r.assign('b', 'LPL')
        r.update('a', 'b', '2024-01-01', 0, 'MSI')
        delta = r.regions['LCK']['delta']
        self.assertLessEqual(abs(delta), .01)
        self.assertAlmostEqual(r._region('LCK', '2024-12-31')['delta'], delta/2)

    def test_adjusted_features_swap_and_keep_region_specific_player_scale(self):
        r = self.fixture()
        d = np.arange(78)/100
        ra = dict(zip(['top', 'jng', 'mid', 'bot', 'sup'], ['a1', 'a2', 'a3', 'a4', 'a5']))
        rb = dict(zip(['top', 'jng', 'mid', 'bot', 'sup'], ['b1', 'b2', 'b3', 'b4', 'b5']))
        ratings = {n: 2000. if n.startswith('b') else 1500. for n in [*ra.values(), *rb.values()]}
        pos = adjust_features(r, 'a', 'b', '2024-01-01', d, ratings, ra, rb)
        neg = adjust_features(r, 'b', 'a', '2024-01-01', -d, ratings, rb, ra)
        np.testing.assert_allclose(pos, -neg, atol=1e-7)
        self.assertEqual(len(pos), 78+len(FEATURE_NAMES))
        self.assertGreater(pos[47], 0)  # Raw domestic player-rating inflation was removed.


if __name__ == '__main__':
    unittest.main()
