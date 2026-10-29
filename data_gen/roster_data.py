"""Freeze ten player histories before each series, then update completed games."""
from collections import defaultdict
import json
import math
import numpy as np
from core import ROOT, LEAGUES, load_series, make_features, number, team_key
from data_gen.features_v2 import load_players, player_key
from data_gen.deep_data import WORLD_WINDOWS, DOMESTIC
from data_gen.roster_core import RosterState, ROLES, PLAYER_NAMES, HISTORY_NAMES, COHESION_NAMES, AUX_NAMES

ARRAY_FIELDS = ['players_a', 'players_b', 'history_a', 'history_b', 'mask_a', 'mask_b',
                'cohesion_a', 'cohesion_b', 'aux_a', 'aux_b', 'context', 'offset']


def match_input(state, a, b, roster_a, roster_b, day, patch, best_of, league, playoffs=0,
                region_a=None, region_b=None):
    region_a = state.regional.origins.get(a) if region_a is None else region_a
    region_b = state.regional.origins.get(b) if region_b is None else region_b
    aa = state.arrays(roster_a, day, patch, a)
    bb = state.arrays(roster_b, day, patch, b)
    major, minor = (map(float, patch.split('.')) if '.' in patch else (0., 0.))
    la, lb = state.regional.log_strength(region_a, day), state.regional.log_strength(region_b, day)
    context = np.array([major/30, minor/30, best_of/5, playoffs,
                        (aa['players'][:, 4].mean()+bb['players'][:, 4].mean())/2, abs(la-lb)], dtype=np.float32)
    output = {'context': context, 'offset': np.float32(state.roster_strength(roster_a, day, region_a)
                                                      -state.roster_strength(roster_b, day, region_b))}
    for side, value in [('a', aa), ('b', bb)]:
        output['players_'+side] = value['players']
        output['history_'+side] = value['history']
        output['mask_'+side] = value['history_mask']
        output['cohesion_'+side] = value['cohesion']
        output['aux_'+side] = value['auxiliary']
        output['names_'+side] = value['player_names']
    return output


def build(regional_config=None, player_config=None, before=None, save_snapshot=None, since=None):
    series, audit = load_series(ROOT/'data'/'teams_all.csv')
    if since:
        series = [s for s in series if s.start >= since]
    if before:
        series = [s for s in series if s.end < before]
    if not series:
        raise ValueError('No completed series in the requested history window.')
    last_day = max(s.end for s in series)
    labels, _ = make_features(series)
    labels = {s['id']: s for s in labels}
    for s in series:
        sid = s.first[0]['gameid']
        if sid in labels or not s.valid_sequence or s.end >= last_day:
            continue
        a, b = s.teams
        rows = {team_key(row): row for row in s.first}
        labels[sid] = {'id': sid, 'date': s.start, 'end_date': s.end, 'league': rows[a]['league'],
                       'team_a': rows[a]['teamname'], 'team_b': rows[b]['teamname'],
                       'team_a_id': a, 'team_b_id': b, 'best_of': 0, 'patch': rows[a].get('patch', ''),
                       'y': int(number(rows[a], 'result')), 'format_source': 'first_map_only'}
    rosters, _, player_rows = load_players(ROOT/'data'/'players.csv.gz', max_day=last_day)
    if since:
        game_ids = {game[0]['gameid'] for s in series for game in s.games}
        rosters = {key: value for key, value in rosters.items() if key[0] in game_ids}
        player_rows = sum(len(value) for value in rosters.values())
    starts, ends = defaultdict(list), defaultdict(list)
    for s in series:
        starts[s.start].append(s)
        ends[s.end].append(s)
    state = RosterState(regional_config, player_config)
    samples, aliases = [], {}
    missing_target_slots = 0
    skipped_updates = 0
    for day in sorted(set(starts) | set(ends)):
        for s in starts[day]:
            for row in s.first:
                state.regional.assign(team_key(row), row['league'])
                aliases[row['teamname']] = team_key(row)
        for s in starts[day]:
            sid = s.first[0]['gameid']
            if sid not in labels:
                continue
            a, b = s.teams
            rows = {team_key(row): row for row in s.first}
            target_rosters = []
            for team in [a, b]:
                present = rosters.get((sid, rows[team]['side']), {})
                current = {role: player_key(row) for role, row in present.items()}
                if not current:
                    current = state.team_lineups.get(team, {})
                target_rosters.append(current)
                missing_target_slots += sum(not current.get(role) for role in ROLES)
            sample = {k: v for k, v in labels[sid].items() if k not in {'d', 'c', 'elo_logit', 'a_history', 'b_history'}}
            sample.update(match_input(state, a, b, *target_rosters, day, rows[a].get('patch', ''),
                                       sample['best_of'], rows[a]['league'], number(rows[a], 'playoffs')))
            sample['map_fraction'] = s.wins[a]/len(s.games) if s.best_of is not None else number(rows[a], 'result')
            sample['game_count'] = len(s.games) if s.best_of is not None else 1
            sample['first_map_y'] = int(number(rows[a], 'result'))
            year = sample['date'][:4]
            first, last, _ = WORLD_WINDOWS[year]
            worlds = sample['league'] == 'WCS' and first <= sample['date'] <= last
            sample['domain'] = 'worlds' if worlds else 'domestic' if sample['league'] in DOMESTIC else 'other'
            sample['worlds_year'] = year if worlds else ''
            samples.append(sample)
        # No target on this day can see this block's outcomes or individual stats.
        for s in sorted(ends[day], key=lambda s: s.games[-1][0]['date']):
            for game in s.games:
                ra, rb = sorted(game, key=team_key)
                a, b = team_key(ra), team_key(rb)
                pa = rosters.get((ra['gameid'], ra['side']), {})
                pb = rosters.get((rb['gameid'], rb['side']), {})
                if set(pa) != set(ROLES) or set(pb) != set(ROLES):
                    skipped_updates += 1
                    continue
                roster_a = {role: player_key(row) for role, row in pa.items()}
                roster_b = {role: player_key(row) for role, row in pb.items()}
                state.update(roster_a, roster_b, pa, pb, day, ra.get('patch', ''), ra['league'], a, b)
    if save_snapshot:
        snapshot = {'snapshot_last_day': last_day, 'history_since': since, 'aliases': aliases, 'state': state.state()}
        save_snapshot.write_text(json.dumps(snapshot, ensure_ascii=False), encoding='utf-8')
    return samples, {'player_features': PLAYER_NAMES, 'history_features': HISTORY_NAMES,
                      'cohesion_features': COHESION_NAMES, 'team_auxiliary_features': AUX_NAMES,
                      'players': len(state.players), 'updated_games': state.updated_games,
                      'source_player_rows': player_rows, 'missing_target_player_slots': missing_target_slots,
                      'skipped_updates_missing_rosters': skipped_updates,
                      'history_window': {'since_inclusive': since, 'before_exclusive': before,
                                         'first_day': min(s.start for s in series), 'last_day': last_day,
                                         'series': len(series), 'games': sum(len(s.games) for s in series)},
                      'region_weights': state.regional.coefficients(last_day), 'audit': audit,
                      'identity_source': 'case-folded player handle, no stable player ID supplied by source',
                      'lineup_assumption': 'historical first-game starters assumed available pre-BP; live inference defaults to last-known unless supplied'}
