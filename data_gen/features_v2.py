"""Causal team/player state features, frozen before the first game of a series."""
import csv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import numpy as np
from core import ROOT, LEAGUES, load_series, make_features, number, team_key, sigmoid, logit, series_probability

KS = [8, 16, 24, 40, 64, 96]
ROLES = ['top', 'jng', 'mid', 'bot', 'sup']


def player_key(row):
    name = row.get('playername', '').strip().casefold()
    return name if name not in {'', 'missing', 'unknown', 'nan'} else None


@dataclass
class History:
    ratings: np.ndarray = field(default_factory=lambda: np.full(len(KS), 1500.0))
    series_rating: float = 1500.0
    games: deque = field(default_factory=lambda: deque(maxlen=50))
    series: deque = field(default_factory=lambda: deque(maxlen=10))
    last_day: str = ''
    year: int = 2024
    roster: set = field(default_factory=set)
    roster_by_role: dict = field(default_factory=dict)

    def elo(self, day):
        return 1500+(self.ratings-1500)*(.75**max(0, int(day[:4])-self.year))

    def features(self, day, patch):
        values, names = [], []
        for k, value in zip(KS, self.elo(day)):
            names.append(f'elo_k{k}')
            values.append(value/400)
        values.append((self.series_rating-1500)/400)
        names.append('series_elo')
        for window in [5, 15, 40]:
            h = list(self.games)[-window:]
            for stat, default in [('win', .5), ('residual', 0), ('gd15', 0), ('gd_known', 0),
                                  ('kill_margin', 0), ('dragons', .5), ('towers', .5), ('duration', 1),
                                  ('opp_elo', 0), ('win_speed', 0), ('loss_speed', 0)]:
                values.append(float(np.mean([x[stat] for x in h])) if h else default)
                names.append(f'{stat}_last{window}')
        for window in [3, 8]:
            recent = list(self.series)[-window:]
            values.append(float(np.mean(recent)) if recent else .5)
            names.append(f'series_wr_last{window}')
        rest = min(90, (datetime.fromisoformat(day)-datetime.fromisoformat(self.last_day)).days) if self.last_day else 90
        values.extend([math.log1p(len(self.games)), rest/30, sum(x['patch'] == patch for x in self.games)/20,
                       sum(x['day'][:4] == day[:4] for x in self.games)/30])
        names.extend(['experience', 'rest', 'patch_experience', 'season_experience'])
        return np.array(values), names


def load_players(path, max_day=None):
    if not Path(path).exists():
        return {}, {}, 0
    groups = defaultdict(list)
    rows_count = 0
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if max_day is None or row['date'][:10] <= max_day:
                groups[row['game_id']].append(row)
                rows_count += 1
    by_day = defaultdict(list)
    roster = {}
    for game_id, rows in groups.items():
        sides = {side: [r for r in rows if r['side'] == side] for side in ['Blue', 'Red']}
        if any(len(s) != 5 or {r['position'] for r in s} != set(ROLES) for s in sides.values()):
            continue
        if {r['result'].lower() for r in rows} != {'true', 'false'}:
            continue
        if len({r['date'] for r in rows}) != 1 or any(len({r['result'].lower() for r in side}) != 1 for side in sides.values()):
            continue
        by_day[rows[0]['date'][:10]].append(rows)
        for side, players in sides.items():
            roster[game_id, side] = {r['position']: r for r in players}
    return roster, by_day, rows_count


def player_vector(roster, last, player_elo, player_history):
    names, values, identities = [], [], set()
    for role in ROLES:
        key = player_key(roster.get(role, {}))
        if key:
            identities.add(key)
        h = list(player_history.get(key, [])) if key else []
        mean = lambda f, default: float(np.mean([r[f] for r in h])) if h else default
        values.extend([(player_elo.get(key, 1500)-1500)/400 if key else 0, mean('win', .5),
                       mean('gd15', 0), mean('cs15', 0), math.log1p(len(h))])
        names.extend([f'{role}_player_elo', f'{role}_player_wr', f'{role}_player_gd15',
                      f'{role}_player_cs15', f'{role}_player_history'])
    values.extend([len(identities)/5, len(identities & last)/5])
    names.extend(['roster_known', 'roster_retained'])
    strengths = [(player_elo.get(p, 1500)-1500)/400 for p in sorted(identities)]
    current_strength = float(np.mean(strengths)) if strengths else 0
    previous_strength = float(np.mean([(player_elo.get(p, 1500)-1500)/400 for p in sorted(last)])) if last else current_strength
    values.extend([current_strength, min(strengths) if strengths else 0,
                   current_strength-previous_strength,
                   sum(len(player_history.get(p, [])) < 5 for p in identities)/5])
    names.extend(['player_mean_strength', 'player_floor_strength', 'roster_strength_change', 'rookie_fraction'])
    return np.array(values), names, identities


def history_sequence(state, day, patch, length=12):
    history = list(state.games)[-length:]
    values = np.zeros((length, 14), dtype=np.float32)
    valid = np.zeros(length, dtype=bool)
    keys = ['win', 'residual', 'gd15', 'gd_known', 'kill_margin', 'dragons', 'towers', 'duration', 'opp_elo', 'win_speed', 'loss_speed']
    for i, game in enumerate(history, length-len(history)):
        age = min(180, (datetime.fromisoformat(day)-datetime.fromisoformat(game['day'])).days)/90
        values[i] = [game[key] for key in keys]+[age, float(game['patch'] == patch), float(game['day'][:4] == day[:4])]
        valid[i] = True
    return values, valid


def build(include_players=True, before=None, save_snapshot=False, team_path=None, include_first_game=False, sequence_length=0,
          regional_config=None):
    from data_gen.regional import RegionalRatings, adjust_features, FEATURE_NAMES
    regional = RegionalRatings(regional_config) if regional_config is not None else None
    series, audit = load_series(team_path or ROOT/'data'/'teams.csv')
    if before:
        series = [s for s in series if s.end < before]
    old, _ = make_features(series)
    old_by_id = {s['id']: s for s in old}
    if include_first_game:
        last_day = max(s.end for s in series)
        for s in series:
            sid = s.first[0]['gameid']
            if sid in old_by_id or not s.valid_sequence or s.end >= last_day:
                continue
            a, b = s.teams
            rows = {team_key(r): r for r in s.first}
            # Single observed game is labelled ONLY as a first-map target; no BO1 claim.
            old_by_id[sid] = {'id': sid, 'date': s.start, 'end_date': s.end, 'league': rows[a]['league'],
                             'team_a': rows[a]['teamname'], 'team_b': rows[b]['teamname'],
                             'team_a_id': a, 'team_b_id': b, 'best_of': 0, 'patch': rows[a].get('patch', ''),
                             'y': int(number(rows[a], 'result')), 'format_source': 'first_map_only'}
    starts, ends = defaultdict(list), defaultdict(list)
    for s in series:
        starts[s.start].append(s)
        ends[s.end].append(s)
    rosters, player_days, player_rows = load_players(ROOT/'data'/'players.csv.gz', max_day=max(s.end for s in series)) if include_players else ({}, {}, 0)
    if before:
        player_days = {day: rows for day, rows in player_days.items() if day < before}
    teams, player_elo, player_history = defaultdict(History), defaultdict(lambda: 1500.0), defaultdict(lambda: deque(maxlen=20))
    head_to_head = defaultdict(lambda: deque(maxlen=8))
    output, diff_names, common_names = [], None, None

    def player_features(row, day):
        roster = rosters.get((row['gameid'], row['side']), {})
        return player_vector(roster, teams[team_key(row)].roster, player_elo, player_history)

    days = sorted(set(starts) | set(ends) | set(player_days))
    for day in days:
        if regional is not None:
            # Today's scheduled domestic participation is known before outcomes.
            for s in starts[day]:
                for row in s.first:
                    regional.assign(team_key(row), row['league'])
        for s in starts[day]:
            if s.first[0]['gameid'] not in old_by_id:
                continue
            a, b = s.teams
            rows = {team_key(r): r for r in s.first}
            patch = rows[a].get('patch', '')
            xa, names = teams[a].features(day, patch)
            xb, _ = teams[b].features(day, patch)
            d = xa-xb
            pair = head_to_head[a, b]
            d = np.r_[d, np.mean(pair) if pair else 0]
            names = names+['head_to_head_margin']
            if include_players:
                pa, pnames, _ = player_features(rows[a], day)
                pb, _, _ = player_features(rows[b], day)
                d = np.r_[d, pa-pb]
                names += pnames
            if regional is not None:
                ra = {r: player_key(p) for r, p in rosters.get((rows[a]['gameid'], rows[a]['side']), {}).items()}
                rb = {r: player_key(p) for r, p in rosters.get((rows[b]['gameid'], rows[b]['side']), {}).items()}
                d = adjust_features(regional, a, b, day, d, player_elo if include_players else None, ra, rb)
                names = names+FEATURE_NAMES
            sample = dict(old_by_id[s.first[0]['gameid']])
            trusted_series = s.best_of is not None
            sample['map_fraction'] = s.wins[a]/len(s.games) if trusted_series else number(rows[a], 'result')
            sample['game_count'] = len(s.games) if trusted_series else 1
            sample['first_map_y'] = int(number(rows[a], 'result'))
            if 'c' not in sample:
                parts = patch.split('.')
                base = [float(parts[0])/20 if len(parts) == 2 else 0, float(parts[1])/30 if len(parts) == 2 else 0, 0,
                        (math.log1p(min(15, len(teams[a].games)))+math.log1p(min(15, len(teams[b].games))))/6]
                sample['c'] = np.array(base+[float(rows[a]['league'] == league) for league in LEAGUES])
                sample['elo_logit'] = math.log(10)*(xa[2]-xb[2])
            common = np.r_[sample['c'], number(rows[a], 'playoffs'),
                           (xa[-4]+xb[-4])/2, abs(d[2]) if regional is not None else abs(xa[2]-xb[2]),
                           (xa[-2]+xb[-2])/2]
            if regional is not None:
                sample['elo_logit'] = math.log(10)*float(d[2])
            cnames = ['patch_major', 'patch_minor', 'best_of', 'mean_experience']+LEAGUES+['playoffs', 'mean_history', 'absolute_elo_gap', 'mean_patch_experience']
            sample.update({'d': d, 'c': common, 'k_logits': np.array([logit(series_probability(float(sigmoid(math.log(10)*x)), s.best_of or 1)) for x in d[:len(KS)]])})
            if sequence_length:
                sample['seq_a'], sample['mask_a'] = history_sequence(teams[a], day, patch, sequence_length)
                sample['seq_b'], sample['mask_b'] = history_sequence(teams[b], day, patch, sequence_length)
            output.append(sample)
            diff_names, common_names = names, cnames
        # Team state changes only after completed series; safe for target snapshots.
        for s in sorted(ends[day], key=lambda s: s.games[-1][0]['date']):
            for game in s.games:
                ra, rb = sorted(game, key=team_key)
                a, b = team_key(ra), team_key(rb)
                sa, sb = teams[a], teams[b]
                ea, eb = sa.elo(day), sb.elo(day)
                p = sigmoid(math.log(10)/400*(ea-eb))
                y = number(ra, 'result')
                sa.ratings, sb.ratings = ea+np.array(KS)*(y-p), eb-np.array(KS)*(y-p)
                global_a, global_b, global_expected = regional.pair(a, b, day) if regional is not None else (None, None, None)
                head_to_head[a, b].append(2*y-1)
                head_to_head[b, a].append(1-2*y)
                for row, opponent, state, opprating, expected in [(ra, rb, sa, eb[2], p[2]), (rb, ra, sb, ea[2], 1-p[2])]:
                    duration = max(1, number(row, 'gamelength', 1800))
                    win = number(row, 'result')
                    known = bool(row.get('golddiffat15')) and number(row, 'goldat15') > 0
                    dr, od = number(row, 'dragons'), number(opponent, 'dragons')
                    tw, ot = number(row, 'towers'), number(opponent, 'towers')
                    state.games.append({'day': day, 'patch': row.get('patch', ''), 'win': win, 'residual': win-expected,
                                        'gd15': number(row, 'golddiffat15')/3000 if known else 0, 'gd_known': float(known),
                                        'kill_margin': (number(row, 'kills')-number(opponent, 'kills'))/20,
                                        'dragons': dr/(dr+od) if dr+od else .5, 'towers': tw/(tw+ot) if tw+ot else .5,
                                        'duration': duration/1800, 'opp_elo': (opprating-1500)/400,
                                        'win_speed': win*1800/duration, 'loss_speed': (1-win)*1800/duration})
                    if regional is not None:
                        state.games[-1]['residual'] = win-float(global_expected[2] if state is sa else 1-global_expected[2])
                        state.games[-1]['opp_elo'] = float(((global_b if state is sa else global_a)[2]-1500)/400)
                    state.last_day, state.year = day, int(day[:4])
                    roster = rosters.get((row['gameid'], row['side']), {})
                    state.roster = {player_key(r) for r in roster.values()}-{None}
                    state.roster_by_role = {role: player_key(r) for role, r in roster.items()}
                if regional is not None:
                    regional.update(a, b, day, y, ra['league'])
            if s.best_of:
                a, b = s.teams
                sa, sb = teams[a], teams[b]
                y = float(s.wins[a] > s.wins[b])
                p = float(sigmoid(math.log(10)/400*(sa.series_rating-sb.series_rating)))
                sa.series_rating += 32*(y-p)
                sb.series_rating -= 32*(y-p)
                sa.series.append(y)
                sb.series.append(1-y)
        # For player state, only completed previous-day games enter later snapshots.
        for rows in sorted(player_days.get(day, []), key=lambda rows: (rows[0]['date'], rows[0]['game_id'])):
            left = [r for r in rows if r['side'] == 'Blue']
            right = [r for r in rows if r['side'] == 'Red']
            ea = np.mean([player_elo[player_key(r)] if player_key(r) else 1500 for r in left])
            eb = np.mean([player_elo[player_key(r)] if player_key(r) else 1500 for r in right])
            expected = float(sigmoid(math.log(10)/400*(ea-eb)))
            for group, p in [(left, expected), (right, 1-expected)]:
                win = float(group[0]['result'].lower() == 'true')
                for r in group:
                    key = player_key(r)
                    if key is None:
                        continue
                    player_elo[key] += 24*(win-p)
                    known = number(r, 'goldat15') > 0
                    player_history[key].append({'win': win, 'gd15': number(r, 'golddiffat15')/1000 if known else 0,
                                                'cs15': number(r, 'csdiffat15')/20 if known else 0})
    if save_snapshot:
        snapshot = {
            'snapshot_last_day': max(s.end for s in series),
            'aliases': {r['teamname']: team_key(r) for s in series for g in s.games for r in g},
            'teams': {key: {'ratings': state.ratings.tolist(), 'series_rating': state.series_rating,
                            'games': list(state.games), 'series': list(state.series), 'last_day': state.last_day,
                            'year': state.year, 'roster': sorted(state.roster), 'roster_by_role': state.roster_by_role}
                      for key, state in teams.items()},
            'player_elo': dict(player_elo), 'player_history': {k: list(v) for k, v in player_history.items()},
            'head_to_head': [{'a': a, 'b': b, 'outcomes': list(v)} for (a, b), v in head_to_head.items()],
        }
        if regional is not None:
            snapshot['regional'] = regional.state()
        target = Path(save_snapshot) if isinstance(save_snapshot, (str, Path)) else ROOT/'artifacts_v2'/'snapshot.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(snapshot, ensure_ascii=False), encoding='utf-8')
    metadata = {'difference': diff_names, 'context': common_names, 'player_rows': player_rows,
                'players': len(player_history), 'audit': audit}
    if regional is not None:
        metadata['regional'] = {'config': regional.config, 'coefficients': regional.coefficients(days[-1]),
                                'international_updates': regional.international_updates,
                                'unknown_international_games': regional.unknown_international_games}
    return output, metadata


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--without-players', action='store_true')
    args = parser.parse_args()
    samples, metadata = build(not args.without_players, save_snapshot=True)
    out = ROOT/'artifacts_v2'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'features.npz', d=np.stack([s['d'] for s in samples]), c=np.stack([s['c'] for s in samples]),
                        elo=np.array([s['elo_logit'] for s in samples]), k_logits=np.stack([s['k_logits'] for s in samples]),
                        y=np.array([s['y'] for s in samples]), map_fraction=np.array([s['map_fraction'] for s in samples]),
                        game_count=np.array([s['game_count'] for s in samples]))
    rows = [{k: v for k, v in s.items() if k not in {'d', 'c', 'k_logits'}} for s in samples]
    (out/'samples.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'feature_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'samples': len(samples), 'diff_features': len(metadata['difference']),
                      'context_features': len(metadata['context']), 'player_rows': metadata['player_rows']}), flush=True)
