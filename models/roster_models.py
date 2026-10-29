"""Shared player encoders and lineup attention, with bounded organization context."""
import torch
from torch import nn

INPUT_KEYS = ['players_a', 'players_b', 'history_a', 'history_b', 'mask_a', 'mask_b',
              'ids_a', 'ids_b', 'cohesion_a', 'cohesion_b', 'aux_a', 'aux_b', 'context', 'league', 'offset']


class RosterTransformer(nn.Module):
    def __init__(self, players, leagues, player_features=18, history_features=14, history_length=8,
                 team_aux_limit=.15):
        super().__init__()
        self.team_aux_limit = team_aux_limit
        self.player_embedding = nn.Embedding(players, 32, padding_idx=0)
        self.league_embedding = nn.Embedding(leagues, 16, padding_idx=0)
        self.history_projection = nn.Linear(history_features, 96)
        self.history_cls = nn.Parameter(torch.zeros(1, 1, 96))
        self.history_position = nn.Parameter(torch.randn(1, history_length, 96)*.02)
        self.history_blocks = nn.ModuleList([nn.TransformerEncoderLayer(96, 4, 288, dropout=.2,
                                               batch_first=True, norm_first=True, activation='gelu') for _ in range(2)])
        self.player_projection = nn.Sequential(nn.Linear(player_features+96+32, 128), nn.LayerNorm(128), nn.GELU())
        self.role_embedding = nn.Parameter(torch.randn(1, 5, 128)*.02)
        self.roster_cls = nn.Parameter(torch.zeros(1, 1, 128))
        self.cohesion_projection = nn.Linear(4, 128)
        self.roster_blocks = nn.ModuleList([nn.TransformerEncoderLayer(128, 4, 384, dropout=.2,
                                              batch_first=True, norm_first=True, activation='gelu') for _ in range(2)])
        layers, width = [], 128+5*player_features+4+6+16
        for hidden in [512, 256, 128]:
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(.25)]
            width = hidden
        self.dense = nn.Sequential(*layers)
        self.head = nn.Linear(128, 1)
        self.matchup = nn.Sequential(nn.Linear(128*2+6+16, 256), nn.GELU(), nn.Dropout(.2), nn.Linear(256, 1))
        self.team_aux = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))
        for module in [self.head, self.matchup[-1], self.team_aux[-1]]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, players, history, mask, ids, cohesion, context, league_embedding):
        batch, roles, length, _ = history.shape
        history = history.masked_fill(~mask[..., None], 0)
        tokens = self.history_projection(history.reshape(batch*roles, length, -1))+self.history_position
        tokens = torch.cat([self.history_cls.expand(batch*roles, -1, -1), tokens], dim=1)
        valid = torch.cat([torch.ones((batch*roles, 1), device=mask.device, dtype=torch.bool), mask.reshape(batch*roles, length)], dim=1)
        for block in self.history_blocks:
            tokens = block(tokens, src_key_padding_mask=~valid)
        temporal = tokens[:, 0].reshape(batch, roles, -1)
        players_encoded = self.player_projection(torch.cat([players, temporal, self.player_embedding(ids)], dim=-1))
        cls = self.roster_cls.expand(batch, -1, -1)+self.cohesion_projection(cohesion)[:, None]
        tokens = torch.cat([cls, players_encoded+self.role_embedding], dim=1)
        for block in self.roster_blocks:
            tokens = block(tokens)
        return self.dense(torch.cat([tokens[:, 0], players.flatten(1), cohesion, context, league_embedding], dim=1))

    def decompose(self, players_a, players_b, history_a, history_b, mask_a, mask_b,
                  ids_a, ids_b, cohesion_a, cohesion_b, aux_a, aux_b, context, league, offset):
        embedding = self.league_embedding(league)
        batch = len(players_a)
        packed = self.encode(torch.cat([players_a, players_b]), torch.cat([history_a, history_b]),
                             torch.cat([mask_a, mask_b]), torch.cat([ids_a, ids_b]),
                             torch.cat([cohesion_a, cohesion_b]), context.repeat(2, 1), embedding.repeat(2, 1))
        a, b = packed[:batch], packed[batch:]
        d = a-b
        shared = torch.cat([d.abs(), context, embedding], dim=1)
        matchup = (self.matchup(torch.cat([d, shared], dim=1))
                   -self.matchup(torch.cat([-d, shared], dim=1))).squeeze(-1)/2
        player_correction = self.head(a).squeeze(-1)-self.head(b).squeeze(-1)+matchup
        gate_a, gate_b = aux_a[:, 3].clamp(0, 1), aux_b[:, 3].clamp(0, 1)
        auxiliary = self.team_aux_limit*(gate_a*torch.tanh(self.team_aux(aux_a).squeeze(-1))
                                        -gate_b*torch.tanh(self.team_aux(aux_b).squeeze(-1)))/2
        return {'player_prior': offset, 'player_correction': player_correction,
                'team_auxiliary': auxiliary, 'logit': offset+player_correction+auxiliary}

    def forward(self, *args):
        return self.decompose(*args)['logit']
