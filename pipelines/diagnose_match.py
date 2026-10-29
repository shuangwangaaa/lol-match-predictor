"""Inspect a frozen forecast without fitting weights or changing model selection."""
import csv
import json
from collections import Counter
import numpy as np
import torch
from core import ROOT, sigmoid, logit
from main import Normalizer, load_model, make_input, predict, raw_series


def main():
    torch.set_num_threads(4)
    day, patch, bo = '2026-10-28', '26.20', 3
    reports, inputs = {}, {}
    for label, folder in [('major', 'artifacts_optimized'), ('deep', 'artifacts_deep')]:
        s = json.loads((ROOT/folder/'snapshot.json').read_text(encoding='utf-8'))
        a, b = [s['aliases'][n] for n in ['Invictus Gaming', 'LYON']]
        raw, _, _ = make_input(s, a, b, day, patch, bo, 'WCS')
        inputs[label] = raw
        teams = {}
        for name, key in [('Invictus Gaming', a), ('LYON', b)]:
            t = s['teams'][key]
            teams[name] = {
                'team_id': key, 'ratings_k8_16_24_40_64_96': t['ratings'],
                'series_rating': t['series_rating'], 'last_day': t['last_day'],
                'last5': {k: float(np.mean([g[k] for g in t['games'][-5:]]))
                          for k in ['win', 'gd15', 'opp_elo']},
                'player_elo': {role: {'name': player, 'elo': s['player_elo'].get(player, 1500)}
                               for role, player in t['roster_by_role'].items()},
            }
        reports[label] = {'snapshot_last_day': s['snapshot_last_day'], 'teams': teams,
                          'elo_game_logit': float(raw['offset'][0]),
                          'elo_game_probability_ig': float(sigmoid(raw['offset'])[0]),
                          'elo_series_probability_ig': float(sigmoid(raw_series(raw['offset'], np.array([bo])))[0])}
    members = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in [42, 43, 44]:
        checkpoint = torch.load(ROOT/'artifacts_deep'/f'dense_transformer_seed{seed}'/'finetuned.pt',
                                map_location='cpu', weights_only=True)
        tensors = Normalizer.from_state(checkpoint['normalizer']).transform(inputs['deep'], [{'league': 'WCS'}], device)
        model = load_model(checkpoint, 'dense_transformer', device)
        z = float(predict(model, tensors, np.array([0]))[0])
        p = float(sigmoid(raw_series(np.array([z], dtype=np.float32), np.array([bo])))[0])
        members.append({'seed': seed, 'elo_game_logit': float(inputs['deep']['offset'][0]),
                        'network_game_logit_correction': z-float(inputs['deep']['offset'][0]),
                        'game_probability_ig': float(sigmoid(z)), 'series_probability_ig': p})
    report = json.loads((ROOT/'artifacts_optimized'/'blend_metrics.json').read_text(encoding='utf-8'))
    weights = report['weights']['recalibrated_transformer']
    prior = float(raw_series(inputs['major']['offset'], np.array([bo]))[0])
    deep = float(logit(np.mean([m['series_probability_ig'] for m in members])))
    final_z = weights[0]*prior+weights[2]*(deep-prior)
    rows = list(csv.DictReader((ROOT/'data'/'teams.csv').open(encoding='utf-8-sig')))
    summaries = {}
    for name in ['Invictus Gaming', 'LYON']:
        selected = [r for r in rows if r['teamname'] == name and r['date'][:4] == '2026'
                    and r['date'][:10] <= '2026-10-26']
        by_league = {}
        for league in sorted({r['league'] for r in selected}):
            league_rows = [r for r in selected if r['league'] == league]
            by_league[league] = {'games': len(league_rows), 'wins': sum(float(r['result']) for r in league_rows)}
        summaries[name] = by_league
    result = {'date': day, 'patch': patch, 'snapshots': reports, 'members': members,
              'deep_ensemble_probability_ig': float(sigmoid(deep)), 'calibration_weights': weights,
              'prior_logit_contribution': weights[0]*prior,
              'neural_residual_logit_contribution': weights[2]*(deep-prior),
              'final_probability_ig': float(sigmoid(final_z)), '2026_source_league_counts': summaries,
              'note': 'League WCS counts may include regional qualifiers; this is an audit, not a retraining run.'}
    target = ROOT/'artifacts_optimized'/'diagnosis_ig_lyon_2026-10-28.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
