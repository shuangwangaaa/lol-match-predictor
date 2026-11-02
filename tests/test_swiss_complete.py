from copy import deepcopy
import itertools
import json
import math
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from pipelines.swiss_complete import load_schedule, validate_schedule, supervised_loss
from pipelines.worlds_rank import BRACKET, exact_bracket


class CompleteSwissTests(unittest.TestCase):
    def test_complete_stage_accounts_for_every_match_and_team(self):
        meta, rows = load_schedule()
        records, maps = validate_schedule(rows, meta['teams'])
        self.assertEqual(len(rows), 33)
        self.assertEqual(sum(sum(r) for r in maps.values()), 102)
        self.assertEqual(sum(r[0] == 3 for r in records.values()), 8)
        self.assertEqual(records['HLE'], [2, 3])
        self.assertEqual(records['BLG'], [3, 2])

    def test_wrong_score_and_finished_team_replay_are_rejected(self):
        meta, rows = load_schedule()
        changed = deepcopy(rows)
        changed[-1]['score_b'] = 2
        with self.assertRaises(ValueError):
            validate_schedule(changed, meta['teams'])
        changed = deepcopy(rows)
        changed[-1]['a'] = 'GEN'
        with self.assertRaises(ValueError):
            validate_schedule(changed, meta['teams'])

    def test_bo3_supervision_targets_series_probability(self):
        z = torch.tensor([math.log(.7/.3)])
        loss, series, maps = supervised_loss(z, torch.ones(1), torch.tensor([3]), torch.tensor([2/3]))
        self.assertAlmostEqual(float(series), -math.log(3*.7**2-2*.7**3), places=6)
        self.assertAlmostEqual(float(loss), .75*float(series)+.25*float(maps), places=6)

    def test_sweep_provides_more_map_evidence_than_two_one_win(self):
        gradients = []
        for fraction in [2/3, 1.]:
            z = torch.tensor([0.], requires_grad=True)
            loss, _, _ = supervised_loss(z, torch.ones(1), torch.tensor([3]), torch.tensor([fraction]))
            loss.backward()
            gradients.append(float(z.grad))
        self.assertLess(gradients[1], gradients[0])


class BracketTests(unittest.TestCase):
    def schedule(self):
        return json.loads(BRACKET.read_text(encoding='utf-8'))

    def test_uniform_model_preserves_tied_ranks_and_probability_mass(self):
        result, _ = exact_bracket(self.schedule(), lambda a, b, node: .5)
        for row in result.values():
            self.assertAlmostEqual(row['champion'], 1/8)
            self.assertAlmostEqual(row['final'], 1/4)
            self.assertAlmostEqual(row['top4'], 1/2)
            self.assertAlmostEqual(row['expected_place'], 4.5)
            self.assertAlmostEqual(sum(row['placements'].values()), 1.)
        self.assertAlmostEqual(sum(r['champion'] for r in result.values()), 1.)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            exact_bracket(self.schedule(), lambda a, b, node: float('nan'))

    def test_dynamic_program_matches_all_128_joint_outcomes(self):
        schedule = self.schedule()
        nodes = [*schedule['quarterfinals'], *schedule['semifinals'], schedule['final']]
        def probability(a, b, node):
            lo, hi = sorted([a, b])
            p = .2+(sum(ord(c) for c in lo+hi+node['id']) % 61)/100
            return p if a == lo else 1-p
        result, _ = exact_bracket(schedule, probability)
        exhaustive = {team: {'champion': 0., 'final': 0., 'top4': 0.} for team in result}
        total = 0.
        for bits in itertools.product([0, 1], repeat=7):
            winners, mass = {}, 1.
            for node, bit in zip(nodes, bits):
                a = node['a'] if 'a' in node else winners[node['left']]
                b = node['b'] if 'b' in node else winners[node['right']]
                p = probability(a, b, node)
                mass *= p if bit == 0 else 1-p
                winners[node['id']] = a if bit == 0 else b
            total += mass
            for node in schedule['quarterfinals']:
                exhaustive[winners[node['id']]]['top4'] += mass
            for node in schedule['semifinals']:
                exhaustive[winners[node['id']]]['final'] += mass
            exhaustive[winners[schedule['final']['id']]]['champion'] += mass
        self.assertAlmostEqual(total, 1.)
        for team, row in exhaustive.items():
            for field, value in row.items():
                self.assertAlmostEqual(value, result[team][field], places=12)


if __name__ == '__main__':
    unittest.main()
