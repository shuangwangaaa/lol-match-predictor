"""Large dense and dense+temporal Transformer probability models."""
import torch
from torch import nn


class MatchNet(nn.Module):
    def __init__(self, nd=78, nc=23, leagues=64, sequence_length=12, transformer=False):
        super().__init__()
        self.transformer_enabled = transformer
        self.league_embedding = nn.Embedding(leagues, 16)
        width = nd+nc+sequence_length*14+sequence_length+16
        layers = []
        for hidden in [1024, 768, 512, 256]:
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(.3)]
            width = hidden
        self.dense = nn.Sequential(*layers)
        if transformer:
            self.summary_projection = nn.Linear(nd+nc+16, 128)
            self.history_projection = nn.Linear(14, 128)
            self.role_projection = nn.Linear(5, 128)
            self.position = nn.Parameter(torch.randn(1, sequence_length, 128)*.02)
            self.side_embedding = nn.Parameter(torch.randn(2, 1, 128)*.02)
            self.role_embedding = nn.Parameter(torch.randn(1, 5, 128)*.02)
            layer = nn.TransformerEncoderLayer(128, 8, 512, dropout=.2, activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 4, norm=nn.LayerNorm(128), enable_nested_tensor=False)
            # Encoder clones otherwise begin with identical weights.
            for block in self.encoder.layers:
                for module in block.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
            width += 128
        self.head = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(.2), nn.Linear(128, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def branch(self, d, c, seq_a, seq_b, mask_a, mask_b, league):
        emb = self.league_embedding(league)
        x = torch.cat([d, c, (seq_a-seq_b).flatten(1), mask_a.float()-mask_b.float(), emb], dim=1)
        dense = self.dense(x)
        if self.transformer_enabled:
            summary = self.summary_projection(torch.cat([d, c, emb], dim=1))[:, None, :]
            a = self.history_projection(seq_a)+self.position+self.side_embedding[0]
            b = self.history_projection(seq_b)+self.position+self.side_embedding[1]
            roles = self.role_projection(d[:, 47:72].reshape(-1, 5, 5))+self.role_embedding
            tokens = torch.cat([summary, a, b, roles], dim=1)
            valid = torch.cat([torch.ones((len(d), 1), dtype=torch.bool, device=d.device), mask_a, mask_b,
                               torch.ones((len(d), 5), dtype=torch.bool, device=d.device)], dim=1)
            encoded = self.encoder(tokens, src_key_padding_mask=~valid)[:, 0]
            dense = torch.cat([dense, encoded], dim=1)
        return self.head(dense).squeeze(-1)

    def forward(self, d, c, seq_a, seq_b, mask_a, mask_b, league, offset):
        # Antisymmetrize the complete network, including the temporal encoder.
        seq_a = seq_a.masked_fill(~mask_a[:, :, None], 0)
        seq_b = seq_b.masked_fill(~mask_b[:, :, None], 0)
        positive = self.branch(d, c, seq_a, seq_b, mask_a, mask_b, league)
        negative = self.branch(-d, c, seq_b, seq_a, mask_b, mask_a, league)
        return offset+(positive-negative)/2


def series_logits(game_logits, best_of):
    p = torch.sigmoid(game_logits.float())
    p3 = p*p*(3-2*p)
    p5 = p*p*p*(10-15*p+6*p*p)
    value = torch.where(best_of == 3, p3, torch.where(best_of == 5, p5, p)).clamp(1e-6, 1-1e-6)
    return torch.logit(value)


# Every backbone is registered under a name and built from the same shared feature
# config, so train and test swap structures by name alone.
def _dense(config):
    return MatchNet(nd=config['nd'], nc=config['nc'], leagues=config['leagues'],
                    sequence_length=config['sequence_length'], transformer=False)


def _dense_transformer(config):
    return MatchNet(nd=config['nd'], nc=config['nc'], leagues=config['leagues'],
                    sequence_length=config['sequence_length'], transformer=True)


from models.tabular_models import NAMES as TABULAR_MODELS, builder as tabular_builder

MODEL_BUILDERS = {'dense': _dense, 'dense_transformer': _dense_transformer,
                  **{name: tabular_builder(name) for name in TABULAR_MODELS}}
MODEL_NAMES = tuple(MODEL_BUILDERS)


def build_model(name, config):
    """Instantiate the registered backbone whose name is given."""
    if name not in MODEL_BUILDERS:
        raise ValueError(f'unknown model {name!r}; registered: {list(MODEL_NAMES)}')
    return MODEL_BUILDERS[name](config)
