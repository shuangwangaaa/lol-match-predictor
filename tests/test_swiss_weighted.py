"""Check the relative loss weighting and adaptation boundaries."""
import unittest
import numpy as np
import torch
from pipelines import swiss_weighted as weighted
from tests import test_roster as roster_tests


class WeightedSwissTests(unittest.TestCase):
    def test_relative_weight_changes_competing_gradient_direction(self):
        gradients = []
        for weight in [.5, 10.]:
            z = torch.tensor([0.], requires_grad=True)
            loss, _, _ = weighted.objective(z, torch.ones(1), z, torch.tensor([.1]), weight, 1.)
            loss.backward()
            gradients.append(float(z.grad))
        self.assertGreater(gradients[0], 0)
        self.assertLess(gradients[1], 0)

    def test_teacher_is_detached_and_matching_distribution_has_zero_kl(self):
        teacher = torch.tensor([.2, .8], requires_grad=True)
        logits = torch.logit(teacher.detach()).requires_grad_()
        loss, _, kl = weighted.objective(torch.zeros(2), torch.ones(2), logits, teacher)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertAlmostEqual(float(kl.detach()), 0., places=6)
        torch.testing.assert_close(logits.grad, torch.zeros(2), atol=1e-7, rtol=0)

    def test_invalid_weights_and_future_labels_are_rejected(self):
        for weight in [0., -1., float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                weighted.settings({'swiss_weighted': {'swiss_weight': weight}})
        with self.assertRaisesRegex(ValueError, 'evaluation label'):
            weighted.objective(torch.zeros(1), torch.tensor([float('nan')]), torch.zeros(1), torch.ones(1)*.5)

    def test_dense_updates_while_encoders_and_auxiliary_stay_frozen(self):
        torch.set_num_threads(2)
        model, batch = roster_tests.RosterNetworkTests().fixture()
        batch['y'] = torch.tensor([1., 0., float('nan')])
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(weighted.freeze(model), lr=1e-3)
        teacher = torch.tensor([.2, .5, .8])
        cfg = weighted.settings({})
        with self.assertRaisesRegex(ValueError, 'evaluation label'):
            weighted.update(model, batch, np.array([2]), batch, teacher, np.array([0, 1]), optimizer, cfg)
        weighted.update(model, batch, np.array([0, 1]), batch, teacher, np.array([0, 1]), optimizer, cfg)
        changed = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
            elif not torch.equal(parameter, before[name]):
                changed.append(name)
        self.assertTrue(any(name.startswith('dense.') for name in changed))
        self.assertTrue(any(name.startswith('player_embedding.') for name in changed))


if __name__ == '__main__':
    unittest.main()
