"""Shared core: data loading, causal features, metrics and calibration."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from collections import Counter, defaultdict, deque
import csv
from dataclasses import dataclass, field
from datetime import datetime
import math
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DIFF_NAMES = ['elo', 'recent_win_rate', 'opponent_adjusted_form', 'gold_diff_15',
              'gold_15_availability', 'kills_per_minute', 'dragon_share', 'tower_share',
              'duration', 'experience', 'rest_days', 'patch_experience']
LEAGUES = ['LCK', 'LPL', 'LEC', 'LCS', 'LTA N', 'LTA S', 'LTA', 'LCP', 'PCS', 'VCS', 'MSI', 'WCS', 'Worlds', 'FST', 'EWC']
CONTEXT_NAMES = ['patch_major', 'patch_minor', 'best_of', 'mean_experience'] + LEAGUES


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def logit(p):
    p = np.clip(p, 1e-8, 1-1e-8)
    return np.log(p / (1-p))


def series_probability(p, best_of):
    """IID constant-game-p baseline; no claim to model draft or side changes."""
    target = (best_of + 1) // 2
    return sum(math.comb(best_of, k) * p**k * (1-p)**(best_of-k)
               for k in range(target, best_of+1))


def metrics(y, p):
    p = np.clip(np.asarray(p), 1e-8, 1-1e-8)
    y = np.asarray(y)
    bins, ece = [], 0.0
    for i in range(10):
        keep = (p >= i/10) & (p < (i+1)/10 if i < 9 else p <= 1)
        if keep.any():
            n = int(keep.sum())
            confidence, frequency = float(p[keep].mean()), float(y[keep].mean())
            ece += n / len(y) * abs(confidence-frequency)
            bins.append({'lower': i/10, 'n': n, 'prediction': confidence, 'observed': frequency})
    return {'n': len(y), 'log_loss': float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            'brier': float(np.mean((p-y)**2)), 'accuracy': float(np.mean((p >= .5) == y)),
            'ece_10_bins': ece, 'calibration_bins': bins}


def number(row, key, default=0.0):
    try:
        v = float(row.get(key, ''))
        return v if math.isfinite(v) else default
    except (ValueError, TypeError):
        return default


def team_key(row):
    return row.get('teamid') or row['teamname']


@dataclass
class Series:
    games: list
    valid_sequence: bool = True

    @property
    def first(self):
        return self.games[0]

    @property
    def start(self):
        return self.first[0]['date'][:10]

    @property
    def end(self):
        return self.games[-1][0]['date'][:10]

    @property
    def teams(self):
        return sorted([team_key(r) for r in self.first])

    @property
    def wins(self):
        return Counter(team_key(r) for game in self.games for r in game if number(r, 'result') == 1)

    @property
    def best_of(self):
        # Retrospective format annotation, not a measured feature. BO1 is excluded
        # because one observed game cannot distinguish BO1 from an incomplete BOx.
        wins = sorted([self.wins[t] for t in self.teams])
        target = wins[-1]
        if not self.valid_sequence or target not in (2, 3) or wins[0] >= target:
            return None
        declared = {re.sub(r'[^0-9]', '', r.get('bo_format', '')) for game in self.games for r in game}
        declared.discard('')
        if declared and declared != {str(2*target-1)}:
            return None
        running = Counter()
        for game in self.games[:-1]:
            running.update(team_key(r) for r in game if number(r, 'result') == 1)
            if max(running.values()) >= target:
                return None
        return 2*target-1


def load_series(path):
    games, audit = defaultdict(list), Counter()
    with Path(path).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['date'][:10] > '2026-10-27':
                audit['future_rows_removed'] += 1
                continue
            games[row['gameid']].append(row)
    grouped = defaultdict(list)
    for game_id, rows in games.items():
        if len(rows) != 2 or len({team_key(r) for r in rows}) != 2 or {number(r, 'result', -1) for r in rows} != {0, 1}:
            audit['invalid_games'] += 1
            continue
        if len({r['date'] for r in rows}) != 1:
            audit['date_mismatch_games'] += 1
            continue
        row = rows[0]
        key = (row['year'], row['league'], row['split'], row['playoffs'], tuple(sorted(team_key(r) for r in rows)))
        grouped[key].append(rows)
    series = []
    for games_in_group in grouped.values():
        current = None
        for game in sorted(games_in_group, key=lambda x: (x[0]['date'], x[0]['gameid'])):
            game_num = int(number(game[0], 'game'))
            previous = current.games[-1][0] if current else None
            gap = (datetime.fromisoformat(game[0]['date']) - datetime.fromisoformat(previous['date'])).total_seconds()/3600 if previous else 1e9
            if current is None or game_num == 1 or gap > 18:
                current = Series([game], game_num == 1)
                series.append(current)
            else:
                current.valid_sequence &= game_num == int(number(previous, 'game')) + 1
                current.games.append(game)
    series.sort(key=lambda s: (s.start, s.first[0]['date'], s.first[0]['gameid']))
    # Drop the snapshot's final day from targets, since series may still be incomplete.
    last_day = max(s.end for s in series)
    audit['series_total'] = len(series)
    audit['games_total'] = sum(len(s.games) for s in series)
    audit['bo3_targets'] = sum(s.best_of == 3 and s.end < last_day for s in series)
    audit['bo5_targets'] = sum(s.best_of == 5 and s.end < last_day for s in series)
    audit['excluded_single_game_or_ambiguous_series'] = sum(s.best_of is None for s in series)
    audit['excluded_last_day_targets'] = sum(s.best_of is not None and s.end == last_day for s in series)
    audit['targets_with_provider_format'] = sum(s.best_of is not None and s.end < last_day and bool(s.first[0].get('bo_format')) for s in series)
    audit['target_last_day_exclusive'] = last_day
    return series, dict(audit)


@dataclass
class TeamState:
    rating: float = 1500.0
    history: deque = field(default_factory=lambda: deque(maxlen=15))
    last_day: str = ''
    roster: set = field(default_factory=set)
    last_year: int = 2024

    def elo(self, day):
        # Annual regression is deterministic and uses no future outcomes.
        years = max(0, int(day[:4])-self.last_year)
        return 1500+(self.rating-1500)*(.75**years)

    def features(self, day, roster, patch):
        h = list(self.history)
        mean = lambda key, default: float(np.mean([x[key] for x in h])) if h else default
        rest = min(60, (datetime.fromisoformat(day)-datetime.fromisoformat(self.last_day)).days) if self.last_day else 60
        return np.array([self.elo(day)/400, mean('win', .5), mean('residual', 0),
                         mean('gd15', 0)/3000, mean('gd_available', 0), mean('kpm', .5),
                         mean('dragons', .5), mean('towers', .5), mean('duration', 1800)/1800,
                         math.log1p(len(h))/3, rest/30,
                         sum(x['patch'] == patch for x in h)/15], dtype=float)


def make_features(series, k_factor=24):
    states = defaultdict(TeamState)
    starts, ends = defaultdict(list), defaultdict(list)
    for s in series:
        starts[s.start].append(s)
        ends[s.end].append(s)
    samples = []
    cutoff = max(ends)
    for day in sorted(set(starts) | set(ends)):
        # All series beginning today use yesterday's completed series only.
        for s in starts[day]:
            if s.best_of is None or s.end >= cutoff:
                continue
            a, b = s.teams
            rows = {team_key(r): r for r in s.first}
            patch = rows[a].get('patch', '')
            league = rows[a]['league']
            ra, rb = set(rows[a].get('roster', '').split('|'))-{''}, set(rows[b].get('roster', '').split('|'))-{''}
            xa, xb = states[a].features(day, ra, patch), states[b].features(day, rb, patch)
            d = xa-xb
            minor = float(patch.split('.')[1]) if '.' in patch else 0
            major = float(patch.split('.')[0]) if '.' in patch else 0
            c = np.array([major/20, minor/30, s.best_of/5, (xa[9]+xb[9])/2]+[float(league == l) for l in LEAGUES])
            p_game = float(sigmoid(math.log(10)*d[0]))
            p_series = series_probability(p_game, s.best_of)
            samples.append({'id': s.first[0]['gameid'], 'date': day, 'end_date': s.end,
                            'league': league, 'team_a': rows[a]['teamname'], 'team_b': rows[b]['teamname'],
                            'team_a_id': a, 'team_b_id': b, 'best_of': s.best_of,
                            'patch': patch, 'y': int(s.wins[a] > s.wins[b]),
                            'format_source': 'provider' if rows[a].get('bo_format') else 'inferred_from_terminal_score',
                            'd': d, 'c': c, 'elo_logit': float(logit(p_series)),
                            'a_history': len(states[a].history), 'b_history': len(states[b].history)})
        for s in sorted(ends[day], key=lambda s: s.games[-1][0]['date']):
            for game in s.games:
                row_a, row_b = sorted(game, key=team_key)
                a, b = team_key(row_a), team_key(row_b)
                sa, sb = states[a], states[b]
                ea, eb = sa.elo(day), sb.elo(day)
                p = float(sigmoid(math.log(10)/400*(ea-eb)))
                y = number(row_a, 'result')
                sa.rating, sb.rating = ea+k_factor*(y-p), eb-k_factor*(y-p)
                for row, opponent, state, expected in [(row_a, row_b, sa, p), (row_b, row_a, sb, 1-p)]:
                    dr, odr = number(row, 'dragons'), number(opponent, 'dragons')
                    tw, otw = number(row, 'towers'), number(opponent, 'towers')
                    duration = max(1, number(row, 'gamelength', 1800))
                    gd_available = bool(row.get('golddiffat15')) and number(row, 'goldat15') > 0
                    state.history.append({'win': number(row, 'result'), 'residual': number(row, 'result')-expected,
                                          'gd15': number(row, 'golddiffat15') if gd_available else 0, 'gd_available': float(gd_available),
                                          'kpm': number(row, 'kills')/duration*60,
                                          'dragons': dr/(dr+odr) if dr+odr else .5,
                                          'towers': tw/(tw+otw) if tw+otw else .5,
                                          'duration': duration, 'patch': row.get('patch', '')})
                    state.last_day, state.last_year = day, int(day[:4])
                    state.roster = set(row.get('roster', '').split('|'))-{''}
    return samples, states


def fit_temperature(logits, y):
    candidates = np.exp(np.linspace(math.log(.4), math.log(3), 181))
    losses = [metrics(y, sigmoid(logits/t))['log_loss'] for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def bootstrap(y, p, baseline, seed=2026):
    rng = np.random.default_rng(seed)
    loss = lambda q: -(y*np.log(np.clip(q, 1e-8, 1))+(1-y)*np.log(np.clip(1-q, 1e-8, 1)))
    delta = loss(p)-loss(baseline)
    draws = np.array([delta[rng.integers(0, len(y), len(y))].mean() for _ in range(2000)])
    return {'delta_log_loss_mlp_minus_elo': float(delta.mean()),
            'paired_series_bootstrap_95_ci': np.quantile(draws, [.025, .975]).tolist(),
            'note': 'Descriptive paired series bootstrap; team/time correlations are not modelled.'}
