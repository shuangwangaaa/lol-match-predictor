"""Calibrate and blend the historical best tree and fixed Transformer expert.

Expert weights are fitted on April/May data only. A forward split within that
period decides whether the blend is recommended, before benchmark calculation.
"""
import csv
import json
import pickle
import numpy as np
from scipy.optimize import minimize
import torch
from core import ROOT, metrics, sigmoid, logit
from main import Normalizer, load_model, predict, raw_series, make_input
from pipelines.optimize import OUT, table_arrays

NAMES = ('expert_blend', 'calibrated_transformer')


def fit_slopes(x, y, target=None):
    target = np.array(target if target is not None else [1., .35, .35])
    def objective(w):
        z = x@w
        return (np.mean(np.logaddexp(0, z)-y*z)+.01*np.sum((w-target)**2),
                x.T@(sigmoid(z)-y)/len(y)+.02*(w-target))
    fitted = minimize(objective, target, jac=True, bounds=[(.4, 1.6), (0., 1.), (0., 1.)],
                       constraints=[{'type': 'ineq', 'fun': lambda w: 1-w[1]-w[2],
                                     'jac': lambda w: np.array([0., -1., -1.])}], method='SLSQP',
                       options={'maxiter': 200, 'ftol': 1e-10})
    if not fitted.success:
        raise RuntimeError(f'Expert calibration failed: {fitted.message}')
    return fitted.x


def make_design(prior, tree, deep):
    return np.c_[prior, tree-prior, deep-prior]


def run():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with np.load(OUT/'data.npz', allow_pickle=False) as stream:
        major = {k: stream[k] for k in stream.files}
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    dates = np.array([s['date'] for s in samples])
    cal = np.flatnonzero((dates >= '2026-04-01') & (dates < '2026-05-24'))
    early = np.flatnonzero((dates >= '2026-04-01') & (dates < '2026-05-01'))
    late = np.flatnonzero((dates >= '2026-05-01') & (dates < '2026-05-24'))
    tree_folder = OUT/'hist_gbm_seed42'
    tree_metadata = json.loads((tree_folder/'metadata.json').read_text(encoding='utf-8'))
    tree_norm = Normalizer.from_state(tree_metadata['normalizer'])
    tt = tree_norm.transform(major, samples, device)
    features = table_arrays(tt)
    swapped = features.copy()
    swapped[:, :78] *= -1
    swapped[:, 101:] *= -1
    with (tree_folder/'model.pkl').open('rb') as stream:
        tree_model = pickle.load(stream)
    tree = (tree_model.decision_function(features)-tree_model.decision_function(swapped))/2
    del tt
    deep_samples = json.loads((ROOT/'artifacts_deep'/'samples.json').read_text(encoding='utf-8'))
    index = {s['id']: i for i, s in enumerate(deep_samples)}
    deep_ids = np.array([index[s['id']] for s in samples])
    with np.load(ROOT/'artifacts_deep'/'data.npz', allow_pickle=False) as stream:
        deep_data = {k: stream[k] for k in stream.files}
    probabilities = []
    for seed in [42, 43, 44]:
        checkpoint = torch.load(ROOT/'artifacts_deep'/f'dense_transformer_seed{seed}'/'finetuned.pt',
                                map_location='cpu', weights_only=True)
        normalizer = Normalizer.from_state(checkpoint['normalizer'])
        tensors = normalizer.transform(deep_data, deep_samples, device)
        model = load_model(checkpoint, 'dense_transformer', device)
        game = predict(model, tensors, deep_ids)
        probabilities.append(sigmoid(raw_series(game, major['best_of'])))
        del model, tensors
    deep = logit(np.mean(probabilities, axis=0))
    prior = major['prior'].astype(float)
    design = make_design(prior, tree, deep)
    # Fit/select calibrator type on an earlier/later forward split, never on 412.
    blend_early = fit_slopes(design[early], major['y'][early])
    deep_design = make_design(prior, prior, deep)
    tree_design = make_design(prior, tree, prior)
    deep_early = fit_slopes(deep_design[early], major['y'][early], [1., 0., .5])
    tree_early = fit_slopes(tree_design[early], major['y'][early], [1., .5, 0.])
    choices = {'expert_blend': (design, blend_early), 'recalibrated_transformer': (deep_design, deep_early),
               'recalibrated_tree': (tree_design, tree_early)}
    validation = {name: {k: v for k, v in metrics(major['y'][late], sigmoid(x[late]@w)).items() if k != 'calibration_bins'}
                  for name, (x, w) in choices.items()}
    selected = min(validation, key=lambda name: validation[name]['log_loss'])
    protocol = {'experts': ['past-only selected monotonic HistGradientBoosting', 'fixed prior dense_transformer finetuned seeds42/43/44'],
                'calibration_train': {'first': '2026-04-01', 'last_exclusive': '2026-05-01', 'n': len(early)},
                'calibration_forward_validation': {'first': '2026-05-01', 'last_exclusive': '2026-05-24', 'n': len(late)},
                'selection': selected, 'validation': validation,
                'benchmark_status': 'previously inspected retrospective benchmark; not new independent evidence'}
    (OUT/'blend_selection.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    weights = {'expert_blend': fit_slopes(design[cal], major['y'][cal]),
               'recalibrated_transformer': fit_slopes(deep_design[cal], major['y'][cal], [1., 0., .5]),
               'recalibrated_tree': fit_slopes(tree_design[cal], major['y'][cal], [1., .5, 0.])}
    old = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    ids = np.array([i for i, sample in enumerate(samples) if sample['id'] in old])
    predictions = {'expert_blend': sigmoid(design@weights['expert_blend']),
                   'recalibrated_transformer': sigmoid(deep_design@weights['recalibrated_transformer']),
                   'recalibrated_tree': sigmoid(tree_design@weights['recalibrated_tree'])}
    results = {name: metrics(major['y'][ids], p[ids]) for name, p in predictions.items()}
    output = {'protocol': protocol, 'weights': {key: v.tolist() for key, v in weights.items()},
              'results': results, 'recommended': selected}
    (OUT/'blend_metrics.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    np.savez_compressed(OUT/'expert_logits.npz', prior=prior, tree=tree, deep=deep, dates=dates, y=major['y'])
    rows = []
    for i in ids:
        row = {k: samples[i][k] for k in ['id', 'date', 'league', 'team_a', 'team_b', 'best_of', 'y']}
        row.update({name: float(p[i]) for name, p in predictions.items()})
        rows.append(row)
    with (OUT/'blend_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('FORWARD CALIBRATION', json.dumps(protocol), flush=True)
    print('BLEND RESULTS', json.dumps({name: {key: value for key, value in score.items() if key != 'calibration_bins'} for name, score in results.items()}), flush=True)
    return output


def test(args, config, parser):
    if config['stage'] != 'finetuned':
        parser.error('These calibration weights belong to the finetuned Transformer; choose --stage finetuned.')
    if sorted(config['seeds']) != [42, 43, 44]:
        parser.error('These calibration weights require the fitted seed ensemble [42, 43, 44].')
    major_snapshot = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    deep_snapshot = json.loads((ROOT/'artifacts_deep'/'snapshot.json').read_text(encoding='utf-8'))
    if args.date <= max(major_snapshot['snapshot_last_day'], deep_snapshot['snapshot_last_day']):
        parser.error('Saved snapshots contain this date or later. Use stored predictions for historical dates.')
    aliases = {name.casefold(): key for name, key in major_snapshot['aliases'].items()}
    a, b = aliases.get(args.team_a.casefold(), args.team_a), aliases.get(args.team_b.casefold(), args.team_b)
    if a not in major_snapshot['teams'] or b not in major_snapshot['teams'] or a == b:
        parser.error('Choose two distinct known teams.')
    def roster(value):
        if value is None:
            return None
        result = json.loads(value)
        if set(result) != {'top', 'jng', 'mid', 'bot', 'sup'} or not all(isinstance(v, str) for v in result.values()):
            parser.error('A lineup must contain names for top,jng,mid,bot,sup.')
        return result
    provided_a, provided_b = roster(args.roster_a), roster(args.roster_b)
    inputs, ra, rb = make_input(major_snapshot, a, b, args.date, args.patch, args.best_of, args.league,
                                args.playoffs, provided_a, provided_b)
    prior = raw_series(inputs['offset'], np.array([args.best_of]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metadata = json.loads((OUT/'hist_gbm_seed42'/'metadata.json').read_text(encoding='utf-8'))
    normalized = Normalizer.from_state(metadata['normalizer']).transform(inputs, [{'league': args.league}], device)
    x = table_arrays(normalized)
    swap = x.copy()
    swap[:, :78] *= -1
    swap[:, 101:] *= -1
    with (OUT/'hist_gbm_seed42'/'model.pkl').open('rb') as stream:
        tree_model = pickle.load(stream)
    tree = (tree_model.decision_function(x)-tree_model.decision_function(swap))/2
    # Deep expert uses its own all-league historical state, matching its training.
    deep_inputs, _, _ = make_input(deep_snapshot, a, b, args.date, args.patch, args.best_of, args.league,
                                    args.playoffs, provided_a, provided_b)
    members = []
    for seed in [42, 43, 44]:
        checkpoint = torch.load(ROOT/'artifacts_deep'/f'dense_transformer_seed{seed}'/'finetuned.pt',
                                map_location='cpu', weights_only=True)
        tensors = Normalizer.from_state(checkpoint['normalizer']).transform(deep_inputs, [{'league': args.league}], device)
        model = load_model(checkpoint, 'dense_transformer', device)
        game = predict(model, tensors, np.array([0]))
        members.append(sigmoid(raw_series(game, np.array([args.best_of]))))
    deep = logit(np.mean(members, axis=0))
    report = json.loads((OUT/'blend_metrics.json').read_text(encoding='utf-8'))
    variant = 'expert_blend' if config['model'] == 'expert_blend' else 'recalibrated_transformer'
    design = make_design(prior, tree if variant == 'expert_blend' else prior, deep)
    value = float(sigmoid(design@np.array(report['weights'][variant]))[0])
    print(json.dumps({'team_a': args.team_a, 'team_b': args.team_b, 'model': config['model'],
                      'p_a_win': value, 'p_b_win': 1-value,
                      'snapshot_last_day': major_snapshot['snapshot_last_day'], 'lineup_a': ra, 'lineup_b': rb,
                      'lineup_status': 'Supplied or last-known historical lineup; current announcements are not fetched.',
                      'selection_basis': 'forward calibration validation before the retrospective benchmark'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    run()
