"""Forecast with V2; lineups may be supplied, otherwise last-known lineups are explicit."""
import argparse
from collections import deque
from datetime import datetime
import json
import math
import pickle
from pipelines.improve import ROOT, OUT, sigmoid, logit, series_probability
import numpy as np
from data_gen.features_v2 import History, LEAGUES, ROLES, player_vector


def forecast_vectors(snapshot, a, b, day, patch, best_of, league, playoffs=0, roster_a=None, roster_b=None):
    def restore(key):
        raw = snapshot['teams'][key]
        return History(ratings=np.array(raw['ratings']), series_rating=raw['series_rating'],
                       games=deque(raw['games'], maxlen=50), series=deque(raw['series'], maxlen=10),
                       last_day=raw['last_day'], year=raw['year'], roster=set(raw['roster']),
                       roster_by_role=raw['roster_by_role'])
    sa, sb = restore(a), restore(b)
    xa, _ = sa.features(day, patch)
    xb, _ = sb.features(day, patch)
    h2h = next((r['outcomes'] for r in snapshot['head_to_head'] if r['a'] == a and r['b'] == b), [])
    def lineup(state, provided):
        names = state.roster_by_role if provided is None else provided
        return {role: {'playername': names.get(role) or ''} for role in ROLES}, names
    ra, names_a = lineup(sa, roster_a)
    rb, names_b = lineup(sb, roster_b)
    pa, _, _ = player_vector(ra, sa.roster, snapshot['player_elo'], snapshot['player_history'])
    pb, _, _ = player_vector(rb, sb.roster, snapshot['player_elo'], snapshot['player_history'])
    d = np.r_[xa-xb, np.mean(h2h) if h2h else 0, pa-pb]
    major, minor = map(float, patch.split('.'))
    mean_old_experience = (math.log1p(min(15, len(sa.games)))+math.log1p(min(15, len(sb.games))))/6
    c = np.array([major/20, minor/30, best_of/5, mean_old_experience]+[float(league == l) for l in LEAGUES]+
                 [playoffs, (xa[-4]+xb[-4])/2, abs(xa[2]-xb[2]), (xa[-2]+xb[-2])/2])
    offsets = np.array([logit(series_probability(float(sigmoid(math.log(10)*delta)), best_of)) for delta in d[:6]])
    return d, c, offsets, names_a, names_b


def score(archive, d, c, offsets, objective='winner_model'):
    selected = archive['selection'][objective]
    specs = {s['name']: s for s in archive['specs']}
    def raw(name):
        spec = specs[name]
        if spec['kind'] == 'ensemble':
            return float(np.mean([raw(component) for component in spec['components']]))
        if spec['kind'] == 'elo':
            return float(offsets[spec['k_index']])
        model = archive['models'][name]
        return float(model.logits(d[None, :spec['dim']], c[None, :], np.array([offsets[spec['k_index']]]))[0])
    p = float(sigmoid(raw(selected)/archive['temperatures'][selected]))
    return p, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('team_a')
    parser.add_argument('team_b')
    parser.add_argument('--date', required=True)
    parser.add_argument('--patch', required=True)
    parser.add_argument('--best-of', type=int, choices=[3, 5], required=True)
    parser.add_argument('--league', choices=LEAGUES, required=True)
    parser.add_argument('--playoffs', type=int, choices=[0, 1], default=0)
    parser.add_argument('--roster-a', help='JSON mapping top,jng,mid,bot,sup to announced player names')
    parser.add_argument('--roster-b', help='JSON mapping top,jng,mid,bot,sup to announced player names')
    args = parser.parse_args()
    snapshot = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    if args.date <= snapshot['snapshot_last_day']:
        parser.error('Saved state contains this day or later; historical results are in benchmark_predictions.csv.')
    aliases = {name.casefold(): key for name, key in snapshot['aliases'].items()}
    a = args.team_a if args.team_a in snapshot['teams'] else aliases.get(args.team_a.casefold())
    b = args.team_b if args.team_b in snapshot['teams'] else aliases.get(args.team_b.casefold())
    if a not in snapshot['teams'] or b not in snapshot['teams'] or a == b:
        parser.error('Choose two distinct known teams; see snapshot.json aliases.')
    def parse_lineup(text):
        if text is None:
            return None
        value = json.loads(text)
        if set(value) != set(ROLES) or not all(isinstance(v, str) for v in value.values()):
            parser.error('A supplied lineup must contain all five roles and player names.')
        return value
    d, c, offsets, roster_a, roster_b = forecast_vectors(snapshot, a, b, args.date, args.patch,
                                                        args.best_of, args.league, args.playoffs,
                                                        parse_lineup(args.roster_a), parse_lineup(args.roster_b))
    with (OUT/'models.pkl').open('rb') as stream:
        archive = pickle.load(stream)
    winner_p, winner_name = score(archive, d, c, offsets)
    probability_p, probability_name = score(archive, d, c, offsets, 'probability_model')
    print(json.dumps({'team_a': args.team_a, 'team_b': args.team_b,
                      'p_a_win': winner_p, 'p_b_win': 1-winner_p,
                      'winner_model': winner_name, 'probability_model': probability_name,
                      'probability_model_p_a': probability_p,
                      'snapshot_last_day': snapshot['snapshot_last_day'],
                      'snapshot_age_days': (datetime.fromisoformat(args.date)-datetime.fromisoformat(snapshot['snapshot_last_day'])).days,
                      'lineup_a': roster_a, 'lineup_b': roster_b,
                      'lineup_status': 'Supplied lineups are assumed announced; omitted lineups use last-known historical players.'},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
