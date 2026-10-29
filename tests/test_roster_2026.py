"""Year isolation and temporal separation for the 2026-only experiment."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from data_gen.roster_data import build, ARRAY_FIELDS
from pipelines.roster_2026 import split_indices
from tests.test_model import fixture
from tests.test_roster import ROLES, game_rows


def source_fixture():
    series = [fixture(1, 'old'), fixture(1, 'jan'), fixture(2, 'next'), fixture(3, 'last')]
    for i, s in enumerate(series):
        year = '2025' if i == 0 else '2026'
        for game in s.games:
            for row in game:
                row['date'] = year+row['date'][4:]
                row['year'] = year
                row['side'] = 'Blue' if row['teamid'] == 'a' else 'Red'
    return series


def build_fixture(series, since='2026-01-01'):
    rosters = {}
    for s in series:
        for game in s.games:
            for row in game:
                players = game_rows(row['result'] == '1')
                for role in ROLES:
                    players[role]['playername'] = row['teamid']+role
                    players[role]['golddiffat15'] = row['golddiffat15']
                rosters[row['gameid'], row['side']] = players
    with patch('data_gen.roster_data.load_series', return_value=(series, {})), \
         patch('data_gen.roster_data.load_players', return_value=(rosters, {}, len(rosters)*5)):
        return build(since=since, before='2027-01-01')


class YearIsolationTests(unittest.TestCase):
    def test_past_year_outcomes_and_stats_cannot_influence_2026(self):
        source = source_fixture()
        rows, metadata = build_fixture(source)
        changed = deepcopy(source)
        for game in changed[0].games:
            for row in game:
                row['result'] = str(1-int(row['result']))
                row['golddiffat15'] = '999999'
        altered, _ = build_fixture(changed)
        self.assertEqual(metadata['updated_games'], 6)
        self.assertEqual(metadata['source_player_rows'], 60)
        self.assertTrue(all(s['date'].startswith('2026') for s in rows))
        for a, b in zip(rows, altered):
            for field in ARRAY_FIELDS:
                np.testing.assert_array_equal(a[field], b[field])
        first = rows[0]
        self.assertFalse(first['mask_a'].any())
        np.testing.assert_array_equal(first['players_a'][:, 0], 0)
        self.assertEqual(first['aux_a'][3], 0)
        unfiltered, _ = build_fixture(source, since=None)
        warmed = next(r for r in unfiltered if r['id'] == first['id'])
        self.assertTrue(warmed['mask_a'].any())

    def test_current_match_stats_do_not_enter_its_prediction(self):
        source = source_fixture()
        before, _ = build_fixture(source)
        for game in source[1].games:
            for row in game:
                row['result'] = str(1-int(row['result']))
                row['golddiffat15'] = '999999'
        after, _ = build_fixture(source)
        for field in ARRAY_FIELDS:
            np.testing.assert_array_equal(before[0][field], after[0][field])
        self.assertNotEqual(before[0]['y'], after[0]['y'])

    def test_temporal_partitions_do_not_share_series_and_reject_other_years(self):
        dates = ['2026-01-03', '2026-03-03', '2026-04-03', '2026-05-03', '2026-06-03']
        samples = [{'id': str(i), 'date': day, 'end_date': day, 'domain': 'domestic',
                    'best_of': 3, 'worlds_year': ''} for i, day in enumerate(dates)]
        ids = split_indices(samples, 'benchmark', {s['id'] for s in samples}, {'4'})
        self.assertFalse(set(ids['selection_train']) & set(ids['selection_validation']))
        for a, b in [('train', 'calibration_early'), ('train', 'calibration_late'),
                     ('train', 'evaluation'), ('calibration_early', 'calibration_late'),
                     ('calibration_late', 'evaluation')]:
            self.assertFalse(set(ids[a]) & set(ids[b]))
        samples[0]['date'] = '2025-01-03'
        with self.assertRaises(ValueError):
            split_indices(samples, 'benchmark', {s['id'] for s in samples}, {'4'})


if __name__ == '__main__':
    unittest.main()
