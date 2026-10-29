import unittest
import torch
from models.deep_models import MatchNet, series_logits


class DeepModelTests(unittest.TestCase):
    def fixture(self, transformer):
        torch.manual_seed(10)
        model = MatchNet(leagues=4, transformer=transformer).eval()
        # Nonzero head checks the actual encoder rather than just the Elo shortcut.
        torch.nn.init.normal_(model.head[-1].weight, std=.1)
        d, c = torch.randn(3, 78), torch.randn(3, 23)
        a, b = torch.randn(3, 12, 14), torch.randn(3, 12, 14)
        ma, mb = torch.rand(3, 12) > .3, torch.rand(3, 12) > .3
        ma[0] = False
        mb[0] = False
        leagues, offset = torch.tensor([0, 1, 2]), torch.randn(3)
        return model, (d, c, a, b, ma, mb, leagues, offset)

    def test_large_parameter_counts_and_swap(self):
        for transformer in [False, True]:
            model, (d, c, a, b, ma, mb, leagues, offset) = self.fixture(transformer)
            self.assertGreater(sum(p.numel() for p in model.parameters()), 1_000_000)
            with torch.no_grad():
                pos = model(d, c, a, b, ma, mb, leagues, offset)
                neg = model(-d, c, b, a, mb, ma, leagues, -offset)
            self.assertTrue(torch.isfinite(pos).all())
            torch.testing.assert_close(pos, -neg, atol=1e-6, rtol=1e-6)

    def test_padding_has_no_information(self):
        model, args = self.fixture(True)
        d, c, a, b, ma, mb, leagues, offset = args
        changed = a.clone()
        changed[~ma] = 1000
        with torch.no_grad():
            original = model(*args)
            actual = model(d, c, changed, b, ma, mb, leagues, offset)
        torch.testing.assert_close(original, actual, atol=1e-6, rtol=1e-6)

    def test_series_link_and_gradients(self):
        values = torch.tensor([-.4, .2, .8], requires_grad=True)
        bo = torch.tensor([0, 3, 5])
        output = series_logits(values, bo)
        torch.testing.assert_close(output, -series_logits(-values, bo), atol=2e-6, rtol=1e-6)
        output.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
