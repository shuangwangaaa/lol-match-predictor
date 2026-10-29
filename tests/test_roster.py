"""Behavioral checks for transferable players and bounded organization influence."""
import json
import unittest
import numpy as np
import torch
from data_gen.roster_core import RosterState, ROLES, PLAYER_NAMES, validate_roster
from data_gen.roster_data import match_input, ARRAY_FIELDS
from models.roster_models import RosterTransformer, INPUT_KEYS
from pipelines.roster_train import PlayerNormalizer


def lineup(prefix):
    return {role: prefix+role for role in ROLES}


def game_rows(won):
    return {role: {'result': str(int(won)), 'kills': '2', 'deaths': '1', 'assists': '4',
                   'goldat15': '5000', 'golddiffat15': '300', 'csdiffat15': '4',
                   'xpdiffat15': '200', 'dpm': '500', 'cspm': '7', 'wpm': '1'} for role in ROLES}


def trained_state():
    state = RosterState()
    state.regional.assign('old', 'LCK')
    state.regional.assign('opponent', 'LCK')
    state.regional.assign('renamed', 'LCK')
    state.update(lineup('a'), lineup('b'), game_rows(True), game_rows(False),
                 '2024-01-01', '14.1', 'LCK', 'old', 'opponent')
    return state


def swap_batch(batch):
    result = dict(batch)
    for name in ['players', 'history', 'mask', 'ids', 'cohesion', 'aux']:
        result[name+'_a'], result[name+'_b'] = batch[name+'_b'], batch[name+'_a']
    result['offset'] = -batch['offset']
    return result


class PlayerStateTests(unittest.TestCase):
    def test_org_change_keeps_personal_history_prior_and_chemistry(self):
        state = trained_state()
        old = match_input(state, 'old', 'opponent', lineup('a'), lineup('b'), '2024-01-02', '14.1', 3, 'LCK')
        renamed = match_input(state, 'renamed', 'opponent', lineup('a'), lineup('b'), '2024-01-02', '14.1', 3, 'LCK')
        for key in ARRAY_FIELDS:
            if not key.startswith('aux_'):
                np.testing.assert_array_equal(old[key], renamed[key])
        self.assertEqual(old['aux_a'][3], 1)
        self.assertEqual(renamed['aux_a'][3], 0)
        self.assertGreater(renamed['cohesion_a'][2], 0)

    def test_substitution_and_complete_replacement_do_not_inherit_old_skill(self):
        state = trained_state()
        old = state.arrays(lineup('a'), '2024-01-02', '14.1', 'old')
        changed = lineup('a')
        changed['mid'] = 'newmid'
        sub = state.arrays(changed, '2024-01-02', '14.1', 'old')
        mid = ROLES.index('mid')
        self.assertGreater(old['players'][mid, 0], sub['players'][mid, 0])
        self.assertEqual(sub['players'][mid, 4], 0)
        self.assertFalse(sub['history_mask'][mid].any())
        self.assertEqual(sub['cohesion'][2], 0)
        self.assertLess(sub['cohesion'][0], old['cohesion'][0])
        self.assertAlmostEqual(float(sub['auxiliary'][3]), .8)
        fresh = state.arrays(lineup('new'), '2024-01-02', '14.1', 'old')
        np.testing.assert_array_equal(fresh['players'][:, 0], 0)
        self.assertEqual(fresh['auxiliary'][3], 0)

    def test_same_win_over_stronger_opponents_has_larger_residual(self):
        weak, strong = trained_state(), trained_state()
        for name in lineup('b').values():
            strong.players[name]['ability'] = 2.
        for state in [weak, strong]:
            state.update(lineup('a'), lineup('b'), game_rows(True), game_rows(False),
                         '2024-01-02', '14.1', 'LCK', 'old', 'opponent')
        for field in ['surprise', 'opponent']:
            self.assertGreater(strong.players['atop']['history'][-1][field], weak.players['atop']['history'][-1][field])
        self.assertGreater(strong.players['atop']['ability'], weak.players['atop']['ability'])

    def test_cross_region_transfer_keeps_personal_ability_and_changes_environment(self):
        state = trained_state()
        state.regional.assign('new_lpl', 'LPL')
        a = match_input(state, 'old', 'opponent', lineup('a'), lineup('b'), '2024-01-02', '14.1', 3, 'WCS')
        b = match_input(state, 'new_lpl', 'opponent', lineup('a'), lineup('b'), '2024-01-02', '14.1', 3, 'WCS')
        np.testing.assert_array_equal(a['players_a'], b['players_a'])
        np.testing.assert_allclose(a['offset']-b['offset'], np.log(.4/.3), atol=1e-7)

    def test_read_only_and_state_round_trip(self):
        state = trained_state()
        state.regional.assign('opponent', 'LPL')
        state.update(lineup('a'), lineup('b'), game_rows(False), game_rows(True),
                     '2024-01-02', '14.1', 'MSI', 'old', 'opponent')
        before = json.dumps(state.state(), sort_keys=True)
        expected = state.arrays(lineup('a'), '2025-01-01', '15.1', 'old')
        self.assertAlmostEqual(sum(state.regional.weights('2025-01-01').values()), 1., places=14)
        self.assertEqual(before, json.dumps(state.state(), sort_keys=True))
        restored = RosterState.from_state(json.loads(before))
        for key, value in expected.items():
            np.testing.assert_array_equal(value, restored.arrays(lineup('a'), '2025-01-01', '15.1', 'old')[key])

    def test_roster_validation_rejects_missing_and_duplicate_names(self):
        for roster in [{'top': 'a'}, {r: 'same' for r in ROLES}, {**lineup('a'), 'mid': 'UNKNOWN'}]:
            with self.assertRaises(ValueError):
                validate_roster(roster)

    def test_vocabulary_and_scaler_exclude_future_players_and_values(self):
        state = trained_state()
        row = match_input(state, 'old', 'opponent', lineup('a'), lineup('b'), '2024-01-02', '14.1', 3, 'LCK')
        raw = {k: np.stack([row[k], row[k]]) for k in ARRAY_FIELDS}
        raw['players_a'][1] = 1e5
        samples = [{'league': 'LCK', 'names_a': row['names_a'], 'names_b': row['names_b']},
                   {'league': 'FutureLeague', 'names_a': list(lineup('future').values()), 'names_b': row['names_b']}]
        norm = PlayerNormalizer().fit(raw, np.array([0]), samples)
        np.testing.assert_allclose(norm.stats['players']['mean'], np.r_[row['players_a'], row['players_b']].mean(0), atol=1e-7)
        tensors = norm.transform(raw, samples, 'cpu')
        self.assertTrue(torch.all(tensors['ids_a'][1] == 0))
        self.assertEqual(tensors['league'][1], 0)
        self.assertTrue(torch.isfinite(tensors['players_a']).all())
        self.assertTrue(torch.equal(tensors['aux_a'][:, 3], torch.tensor(raw['aux_a'][:, 3])))
        fine = PlayerNormalizer().fit(raw, np.array([0]), samples, [1])
        self.assertIn('futuretop', fine.players)
        self.assertNotIn('FutureLeague', fine.leagues)


class RosterNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        torch.manual_seed(31)
        model = RosterTransformer(30, 4).eval()
        for layer in [model.head, model.matchup[-1], model.team_aux[-1]]:
            torch.nn.init.normal_(layer.weight, std=.1)
        batch = {'context': torch.randn(3, 6), 'league': torch.tensor([1, 2, 3]), 'offset': torch.randn(3)}
        for side in ['a', 'b']:
            batch.update({f'players_{side}': torch.randn(3, 5, 18), f'history_{side}': torch.randn(3, 5, 8, 14),
                          f'mask_{side}': torch.rand(3, 5, 8) > .5, f'ids_{side}': torch.randint(0, 30, (3, 5)),
                          f'cohesion_{side}': torch.randn(3, 4), f'aux_{side}': torch.randn(3, 4)})
            batch[f'mask_{side}'][0] = False
            batch[f'aux_{side}'][:, 3] = torch.rand(3)
        return model, batch

    @torch.no_grad()
    def test_swap_complementarity_with_nonzero_neural_and_auxiliary_heads(self):
        model, batch = self.fixture()
        a = model(*[batch[k] for k in INPUT_KEYS])
        swapped = swap_batch(batch)
        b = model(*[swapped[k] for k in INPUT_KEYS])
        torch.testing.assert_close(a, -b, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(a.sigmoid()+b.sigmoid(), torch.ones_like(a), atol=1e-6, rtol=0)

    @torch.no_grad()
    def test_masked_history_values_do_not_affect_prediction_even_if_all_missing(self):
        model, batch = self.fixture()
        a = model(*[batch[k] for k in INPUT_KEYS])
        for side in ['a', 'b']:
            batch[f'history_{side}'][~batch[f'mask_{side}']] = float('nan')
        b = model(*[batch[k] for k in INPUT_KEYS])
        self.assertTrue(torch.isfinite(b).all())
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    @torch.no_grad()
    def test_org_branch_is_bounded_separate_and_disabled_for_full_replacement(self):
        model, batch = self.fixture()
        before = model.decompose(*[batch[k] for k in INPUT_KEYS])
        for side in ['a', 'b']:
            batch[f'aux_{side}'][:, :3] = torch.randn(3, 3)*1000
            batch[f'aux_{side}'][:, 3] = 1
        after = model.decompose(*[batch[k] for k in INPUT_KEYS])
        torch.testing.assert_close(before['player_correction'], after['player_correction'], atol=0, rtol=0)
        self.assertTrue((after['team_auxiliary'].abs() <= .15).all())
        for side in ['a', 'b']:
            batch[f'aux_{side}'][:, 3] = 0
        empty = model.decompose(*[batch[k] for k in INPUT_KEYS])
        torch.testing.assert_close(empty['team_auxiliary'], torch.zeros(3), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
