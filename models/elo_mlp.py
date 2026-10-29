"""v1 baseline backbone: two-layer tanh NumPy MLP with an antisymmetric residual."""
import numpy as np
from core import sigmoid


class ResidualMLP:
    """Two tanh layers; r(d,c)=(g(d,c)-g(-d,c))/2, trained by Adam."""
    def __init__(self, nd, nc, seed=42, width=16):
        rng = np.random.default_rng(seed)
        self.nd = nd
        self.params = {'w1': rng.normal(0, .2, (nd+nc, width)), 'b1': np.zeros(width),
                       'w2': rng.normal(0, .2, (width, 8)), 'b2': np.zeros(8),
                       'w3': np.zeros((8, 1))}

    def branch(self, z):
        p = self.params
        h1 = np.tanh(z@p['w1']+p['b1'])
        h2 = np.tanh(h1@p['w2']+p['b2'])
        return (h2@p['w3']).ravel(), (z, h1, h2)

    def logits(self, d, c, offset):
        pos, _ = self.branch(np.column_stack([d, c]))
        neg, _ = self.branch(np.column_stack([-d, c]))
        return offset+(pos-neg)/2

    def gradients(self, d, c, offset, y, l2):
        pos, pc = self.branch(np.column_stack([d, c]))
        neg, nc = self.branch(np.column_stack([-d, c]))
        error = (sigmoid(offset+(pos-neg)/2)-y)/len(y)
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        for cache, sign in [(pc, .5), (nc, -.5)]:
            z, h1, h2 = cache
            dy = (sign*error)[:, None]
            grads['w3'] += h2.T@dy
            dh2 = (dy@self.params['w3'].T)*(1-h2*h2)
            grads['w2'] += h1.T@dh2
            grads['b2'] += dh2.sum(axis=0)
            dh1 = (dh2@self.params['w2'].T)*(1-h1*h1)
            grads['w1'] += z.T@dh1
            grads['b1'] += dh1.sum(axis=0)
        for key in grads:
            if key.startswith('w'):
                grads[key] += l2*self.params[key]
        return grads
