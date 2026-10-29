"""Series-probability backbones for small structured match datasets.

FT feature tokenization follows Gorishniy et al. (2021), arXiv:2106.11959.
tabm_lite is a task-specific BatchEnsemble MLP inspired by TabM
(arXiv:2410.24210), not a reproduction of the official implementation.
"""
import math
import torch
from torch import nn

NAMES = ('ridge_series', 'resnet_tabular', 'ft_transformer', 'tabm_lite')


def pooled_history(x, mask):
    weights = mask.float().unsqueeze(-1)
    n = weights.sum(1).clamp_min(1)
    mean = (x*weights).sum(1)/n
    variance = ((x-mean[:, None]).square()*weights).sum(1)/n
    return torch.cat([mean, torch.sqrt(variance+1e-6)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(width*2, width), nn.Dropout(dropout))

    def forward(self, x):
        return x+self.layers(x)


class EnsembleLinear(nn.Module):
    """Each member has diag(r_k) W diag(s_k), with shared W."""
    def __init__(self, inputs, outputs, members):
        super().__init__()
        self.shared = nn.Linear(inputs, outputs, bias=False)
        self.r = nn.Parameter(torch.empty(members, inputs).bernoulli_(.5).mul_(2).sub_(1))
        self.s = nn.Parameter(torch.empty(members, outputs).bernoulli_(.5).mul_(2).sub_(1))
        self.bias = nn.Parameter(torch.zeros(members, outputs))

    def forward(self, x):
        return self.shared(x*self.r)*self.s+self.bias


class TabularSeriesNet(nn.Module):
    output_space = 'series'

    def __init__(self, nd=78, nc=23, leagues=64, sequence_length=12, architecture='resnet_tabular',
                 width=192, dropout=.2, members=8, **unused):
        super().__init__()
        self.architecture, self.nd, self.members = architecture, nd, members
        if architecture == 'ridge_series':
            self.linear = nn.Linear(nd*2, 1, bias=False)
            nn.init.zeros_(self.linear.weight)
            return
        self.league_embedding = nn.Embedding(leagues, 8)
        nn.init.zeros_(self.league_embedding.weight[0])
        features = nd+nc+28
        if architecture == 'resnet_tabular':
            self.input = nn.Linear(features+8, width)
            self.blocks = nn.Sequential(ResidualBlock(width, dropout), ResidualBlock(width, dropout))
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        elif architecture == 'ft_transformer':
            dimension = 48
            self.token_weight = nn.Parameter(torch.randn(features, dimension)*.02)
            self.token_bias = nn.Parameter(torch.randn(features, dimension)*.02)
            self.cls = nn.Parameter(torch.zeros(1, 1, dimension))
            self.league_token = nn.Linear(8, dimension)
            blocks = []
            for _ in range(2):
                blocks.append(nn.TransformerEncoderLayer(dimension, 4, dimension*3, dropout=dropout,
                                                         activation='gelu', batch_first=True, norm_first=True))
            self.blocks = nn.ModuleList(blocks)
            self.head = nn.Sequential(nn.LayerNorm(dimension), nn.GELU(), nn.Linear(dimension, 1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        elif architecture == 'tabm_lite':
            widths = [features+8, width, width, width]
            self.blocks = nn.ModuleList([EnsembleLinear(a, b, members) for a, b in zip(widths, widths[1:])])
            self.dropout = nn.Dropout(dropout)
            self.head_weight = nn.Parameter(torch.zeros(members, width))
            self.head_bias = nn.Parameter(torch.zeros(members))
        else:
            raise ValueError(architecture)

    def branch(self, d, c, seq_a, seq_b, mask_a, mask_b, league):
        if self.architecture == 'ridge_series':
            return self.linear(torch.cat([d, d*c[:, 2:3]], dim=1)).squeeze(-1)
        sequence = pooled_history(seq_a, mask_a)-pooled_history(seq_b, mask_b)
        x = torch.cat([d, c, sequence], dim=1)
        embedding = self.league_embedding(league)
        if self.architecture == 'resnet_tabular':
            hidden = self.blocks(self.input(torch.cat([x, embedding], dim=1)))
            return self.head(hidden).squeeze(-1)
        if self.architecture == 'ft_transformer':
            tokens = x[:, :, None]*self.token_weight+self.token_bias
            cls = self.cls.expand(len(x), -1, -1)+self.league_token(embedding)[:, None]
            tokens = torch.cat([cls, tokens], dim=1)
            for block in self.blocks:
                tokens = block(tokens)
            return self.head(tokens[:, 0]).squeeze(-1)
        x = torch.cat([x, embedding], dim=1)[:, None].expand(-1, self.members, -1)
        for block in self.blocks:
            x = self.dropout(torch.nn.functional.gelu(block(x)))
        return (x*self.head_weight).sum(-1)+self.head_bias

    def member_logits(self, d, c, seq_a, seq_b, mask_a, mask_b, league, offset):
        pos = self.branch(d, c, seq_a, seq_b, mask_a, mask_b, league)
        neg = self.branch(-d, c, seq_b, seq_a, mask_b, mask_a, league)
        residual = (pos-neg)/2
        return offset[:, None]+residual if residual.ndim == 2 else offset+residual

    def forward(self, *args):
        values = self.member_logits(*args)
        if values.ndim == 1:
            return values
        return torch.logit(torch.sigmoid(values).mean(1).clamp(1e-6, 1-1e-6))


def builder(name):
    return lambda config: TabularSeriesNet(**{**config, 'architecture': name})
