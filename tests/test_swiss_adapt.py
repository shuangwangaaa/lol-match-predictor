"""Tests for adaptation boundaries and non-training test labels."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from pipelines.swiss_adapt import partitions, adaptation_labels, freeze_for_adaptation, step
from tests import test_roster as roster_tests


class SwissAdaptTests(unittest.TestCase):
    def rows(self):
        return [{'round': r, 'event_date': f'2026-10-{22+r:02}', 'best_of': 1 if r <= 2 else 3,
                 'score_a': int(r % 2 == 1), 'score_b': int(r % 2 == 0)} for r in [1, 2, 3, 4]]

    def test_future_outcomes_never_become_training_labels(self):
        rows = self.rows()
        ids = partitions(rows)
        a = adaptation_labels(rows, ids['adaptation'])
        changed = deepcopy(rows)
        for i in ids['evaluation']:
            changed[i]['score_a'], changed[i]['score_b'] = changed[i]['score_b'], changed[i]['score_a']
        b = adaptation_labels(changed, partitions(changed)['adaptation'])
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isnan(a[ids['evaluation']]).all())
        self.assertFalse(set(ids['adaptation']) & set(ids['evaluation']))

    def test_training_step_rejects_test_labels_and_updates_only_unfrozen_parameters(self):
        torch.set_num_threads(2)
        model, batch = roster_tests.RosterNetworkTests().fixture()
        batch['y'] = torch.tensor([1., 0., float('nan')])
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        optimizer = torch.optim.AdamW(freeze_for_adaptation(model), lr=1e-3)
        with self.assertRaisesRegex(ValueError, 'evaluation label'):
            step(model, batch, np.array([2]), optimizer)
        step(model, batch, np.array([0, 1]), optimizer)
        changed = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                torch.testing.assert_close(before[name], p, atol=0, rtol=0)
            elif not torch.equal(before[name], p):
                changed.append(name)
        self.assertTrue(changed)

    def test_time_order_rejects_training_after_evaluation(self):
        rows = self.rows()
        rows[1]['event_date'] = '2026-10-29'
        with self.assertRaisesRegex(ValueError, 'end before evaluation'):
            partitions(rows)


if __name__ == '__main__':
    unittest.main()
