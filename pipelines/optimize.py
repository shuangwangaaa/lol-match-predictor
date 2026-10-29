"""Past-only rolling model selection and calibrated direct-series prediction."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timedelta
import hashlib
import json
import os
import inspect
from pathlib import Path
import random
import time
import numpy as np
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
import torch
from torch.nn import functional as F
from core import ROOT, metrics, sigmoid, logit
from data_gen.deep_data import DOMESTIC, WORLD_WINDOWS
from data_gen.features_v2 import build
from models.deep_models import build_model, series_logits
from models.tabular_models import NAMES, pooled_history
from main import Normalizer, seed_everything, make_input

OUT = ROOT/'artifacts_optimized'
FOLDS = [('2025-01-01', '2025-05-01'), ('2025-05-01', '2025-09-01'),
         ('2025-09-01', '2026-01-01'), ('2026-01-01', '2026-04-01')]


def prepare():
    OUT.mkdir(exist_ok=True)
    samples, metadata = build(sequence_length=12, save_snapshot=OUT/'snapshot.json')
    fields = ['d', 'c', 'seq_a', 'seq_b', 'mask_a', 'mask_b']
    raw = {key: np.stack([s[key] for s in samples]) for key in fields}
    raw.update(y=np.array([s['y'] for s in samples], dtype='float32'),
               best_of=np.array([s['best_of'] for s in samples], dtype='int64'),
               offset=np.array([np.log(10)*s['d'][2] for s in samples], dtype='float32'),
               prior=np.array([s['k_logits'][2] for s in samples], dtype='float32'))
    for key in ['d', 'c', 'seq_a', 'seq_b']:
        raw[key] = raw[key].astype('float32')
    np.savez_compressed(OUT/'data.npz', **raw)
    rows = []
    for sample in samples:
        row = {k: v for k, v in sample.items() if k not in set(fields+['k_logits'])}
        window = WORLD_WINDOWS[row['date'][:4]]
        row['domain'] = 'worlds' if row['league'] == 'WCS' and window[0] <= row['date'] <= window[1] else 'domestic' if row['league'] in DOMESTIC else 'other'
        rows.append(row)
    (OUT/'samples.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT/'feature_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT/'preparation.json').write_text(json.dumps(source_fingerprint(), indent=2), encoding='utf-8')
    return raw, rows


def source_fingerprint():
    return {name: hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest() for name in ['teams.csv', 'players.csv.gz']}


def calibrate(raw_logits, prior, y):
    """Two bounded, regularized slopes; zero intercept preserves complementarity."""
    if len(y) == 0:
        raise ValueError('Calibration needs past completed series.')
    x = np.c_[prior, raw_logits-prior].astype(float)
    target = np.array([1., .5])
    def objective(w):
        z = x@w
        loss = np.mean(np.logaddexp(0, z)-y*z)+.01*np.sum((w-target)**2)
        gradient = x.T@(sigmoid(z)-y)/len(y)+.02*(w-target)
        return loss, gradient
    fit = minimize(objective, target, jac=True, bounds=[(.4, 1.6), (0., 1.)], method='L-BFGS-B')
    if not fit.success and not np.isfinite(fit.fun):
        raise RuntimeError(f'Calibration failed: {fit.message}')
    return fit.x


def apply_calibration(raw_logits, prior, weights):
    return sigmoid(weights[0]*prior+weights[1]*(raw_logits-prior))


def model_inputs(tensors, idx):
    return [tensors[k][idx] for k in ['d', 'c', 'seq_a', 'seq_b', 'mask_a', 'mask_b', 'league', 'prior']]


@torch.no_grad()
def infer(model, tensors, ids):
    model.eval()
    values = []
    device = tensors['d'].device
    for start in range(0, len(ids), 256):
        index = torch.as_tensor(ids[start:start+256], device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            values.append(model(*model_inputs(tensors, index)).float().cpu().numpy())
    return np.concatenate(values)


def table_arrays(tensors):
    d, c = tensors['d'], tensors['c']
    h = pooled_history(tensors['seq_a'], tensors['mask_a'])-pooled_history(tensors['seq_b'], tensors['mask_b'])
    return torch.cat([d, c, h], 1).cpu().numpy()


def sample_weights(samples, ids, half_life=365):
    dates = np.array([samples[i]['date'] for i in ids], dtype='datetime64[D]')
    ages = (dates.max()-dates).astype('timedelta64[D]').astype(float)
    return np.exp2(-ages/half_life)


def optimize_neural(model, tensors, train_ids, val_ids, samples, spec, max_epochs=45, fixed=None, fine=False):
    device = tensors['d'].device
    ids = torch.as_tensor(train_ids, device=device)
    weights = torch.as_tensor(sample_weights(samples, train_ids), device=device, dtype=torch.float32)
    if fine:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith('head') or name.startswith('linear'))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4 if fine else spec.get('lr', 3e-4), weight_decay=spec.get('decay', .05))
    best, best_epoch, best_state, history = float('inf'), 0, None, []
    targets = tensors['y']
    epochs = fixed or max_epochs
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(ids), device=device)
        losses = []
        for begin in range(0, len(ids), 128):
            take = order[begin:begin+128]
            batch = ids[take]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                values = model.member_logits(*model_inputs(tensors, batch))
            if values.ndim == 1:
                values = values[:, None]
            labels = targets[batch, None].expand_as(values)
            loss_rows = F.binary_cross_entropy_with_logits(values.float(), labels.float(), reduction='none').mean(1)
            loss = (loss_rows*weights[take]).sum()/weights[take].sum()
            # Explicitly shrink unsupported corrections to the dynamic prior.
            loss += .003*(values.float()-tensors['prior'][batch, None]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite neural loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = metrics(targets[val_ids].cpu().numpy(), sigmoid(infer(model, tensors, val_ids)))['log_loss'] if val_ids is not None else None
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation_log_loss': validation})
        if fixed is None and validation < best-1e-5:
            best, best_epoch = validation, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if fixed is None and epoch-best_epoch >= 6:
            break
    if fixed is None:
        model.load_state_dict(best_state)
    return (fixed or best_epoch), history


def train_one(spec, raw, samples, train_ids, fine_ids, seed, device):
    seed_everything(seed)
    normalizer = Normalizer().fit(raw, train_ids, samples)
    tensors = normalizer.transform(raw, samples, device)
    config = {'nd': raw['d'].shape[1], 'nc': raw['c'].shape[1], 'leagues': len(normalizer.vocabulary)+1,
              'sequence_length': 12, 'dropout': spec.get('dropout', .2), 'members': 8, 'width': 192}
    trace = {}
    if spec['model'] == 'hist_gbm':
        x = table_arrays(tensors)
        mirrored = x.copy()
        mirrored[:, :78] *= -1
        mirrored[:, 101:] *= -1
        constraints = [1 if i < 7 and spec['monotonic'] else 0 for i in range(x.shape[1])]
        model = HistGradientBoostingClassifier(max_iter=180, learning_rate=.04, max_leaf_nodes=spec['leaves'],
                                               l2_regularization=20, min_samples_leaf=30, early_stopping=False,
                                               monotonic_cst=constraints, random_state=seed)
        y = raw['y'][train_ids]
        w = sample_weights(samples, train_ids)
        model.fit(np.r_[x[train_ids], mirrored[train_ids]], np.r_[y, 1-y], sample_weight=np.r_[w, w])
        logits = (model.decision_function(x)-model.decision_function(mirrored))/2
        return {'kind': 'hist_gbm', 'model': model, 'normalizer': normalizer, 'config': config,
                'pretrained': logits, 'finetuned': logits, 'trace': trace}
    model = build_model(spec['model'], config).to(device)
    if spec['model'] == 'ridge_series':
        d, c = tensors['d'][train_ids].cpu().numpy(), tensors['c'][train_ids].cpu().numpy()
        design = np.c_[d, d*c[:, 2:3]].astype(float)
        w = sample_weights(samples, train_ids)
        w /= w.sum()
        coefficients = np.zeros(design.shape[1])
        regularization = spec['regularization']
        for _ in range(30):
            p = sigmoid(raw['prior'][train_ids]+design@coefficients)
            grad = design.T@(w*(p-raw['y'][train_ids]))+regularization*coefficients
            hessian = design.T@((w*p*(1-p))[:, None]*design)+regularization*np.eye(len(coefficients))
            step = np.linalg.solve(hessian, grad)
            coefficients -= step
            if np.max(np.abs(step)) < 1e-6:
                break
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor(coefficients[None], dtype=torch.float32, device=device))
    else:
        dates = sorted({samples[i]['date'] for i in train_ids})
        boundary = dates[int(.8*len(dates))]
        inner_train = np.array([i for i in train_ids if samples[i]['end_date'] < boundary])
        inner_val = np.array([i for i in train_ids if samples[i]['date'] >= boundary])
        epoch, trace['selection'] = optimize_neural(model, tensors, inner_train, inner_val, samples, spec)
        seed_everything(seed)
        model = build_model(spec['model'], config).to(device)
        _, trace['refit'] = optimize_neural(model, tensors, train_ids, None, samples, spec, fixed=epoch)
        trace['pretrain_epochs'] = epoch
    before_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
    before = infer(model, tensors, np.arange(len(samples)))
    if len(fine_ids):
        _, trace['fine'] = optimize_neural(model, tensors, fine_ids, None, samples, spec, fixed=5, fine=True)
    after = infer(model, tensors, np.arange(len(samples)))
    result = {'kind': 'torch', 'model': model, 'normalizer': normalizer, 'config': config,
              'before_state': before_state, 'pretrained': before, 'finetuned': after, 'trace': trace}
    del tensors
    return result


def specifications(models=None):
    values = [{'name': f'ridge_r{r}', 'model': 'ridge_series', 'regularization': r} for r in [.1, 1., 3.]]
    for family in ['resnet_tabular', 'ft_transformer', 'tabm_lite']:
        for dropout in [.1, .3]:
            values.append({'name': f'{family}_drop{dropout}', 'model': family, 'dropout': dropout,
                           'decay': .05, 'lr': .0001 if family == 'ft_transformer' else .0003})
    values += [{'name': f'hist_gbm_l{leaves}_mono{monotonic}', 'model': 'hist_gbm', 'leaves': leaves,
                'monotonic': monotonic} for leaves, monotonic in [(7, True), (15, True), (7, False)]]
    return [v for v in values if models is None or v['model'] in models]


def run(models=None, seeds=(42, 43, 44), refresh=False):
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prepared = OUT/'preparation.json'
    stale = not prepared.exists() or json.loads(prepared.read_text(encoding='utf-8')) != source_fingerprint()
    if refresh or stale or not (OUT/'data.npz').exists():
        prepare()
    with np.load(OUT/'data.npz', allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    domains = np.array([s['domain'] for s in samples])
    dates, ends = np.array([s['date'] for s in samples]), np.array([s['end_date'] for s in samples])
    specs = specifications(models)
    digest = hashlib.sha256()
    digest.update((OUT/'data.npz').read_bytes())
    digest.update((ROOT/'models'/'tabular_models.py').read_bytes())
    for function in [calibrate, optimize_neural, train_one, specifications, Normalizer]:
        digest.update(inspect.getsource(function).encode())
    digest.update(json.dumps(FOLDS).encode())
    fingerprint = digest.hexdigest()
    manifest = {'models': specs, 'folds': FOLDS, 'seed_search': 42, 'final_seeds': list(seeds),
                'objective': 'minimum pooled past-only calibrated log loss; accuracy is secondary',
                'pretrain_domain': 'major domestic leagues', 'fine_domain': 'past Worlds event windows only',
                'fine_tune': '5 epochs, head only; frozen representation',
                'fold_calibration': 'last 45 days, extended to at least 80 past series when offseason data are sparse; grouped by date',
                'final_train_end_exclusive': '2026-04-01', 'final_calibration_end_exclusive': '2026-05-24',
                'benchmark_status': '412 previously inspected series; retrospective only',
                'loss': 'direct series BCE; independent-game conversion only supplies a prior',
                'calibration': 'bounded two-slope prior/residual shrinkage, zero intercept'}
    manifest['data_and_training_fingerprint'] = fingerprint
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    development = []
    start = time.monotonic()
    for spec in specs:
        cache = OUT/'development'/f'{spec["name"]}_{fingerprint[:12]}.json'
        if cache.exists():
            development.extend(json.loads(cache.read_text(encoding='utf-8')))
            continue
        records = {stage: {'y': [], 'p': [], 'folds': []} for stage in ['pretrained', 'finetuned']}
        for first, last in FOLDS:
            cut = (datetime.fromisoformat(first)-timedelta(days=45)).date().isoformat()
            if np.sum((dates >= cut) & (ends < first)) < 60:
                past = np.flatnonzero(ends < first)
                cut = min(cut, samples[past[-80]]['date'])
            train_ids = np.flatnonzero((domains == 'domestic') & (ends < cut))
            fine_ids = np.flatnonzero((domains == 'worlds') & (ends < cut))
            cal_ids = np.flatnonzero((dates >= cut) & (ends < first))
            val_ids = np.flatnonzero((dates >= first) & (ends < last))
            fitted = train_one(spec, raw, samples, train_ids, fine_ids, 42, device)
            for stage in records:
                weights = calibrate(fitted[stage][cal_ids], raw['prior'][cal_ids], raw['y'][cal_ids])
                probability = apply_calibration(fitted[stage][val_ids], raw['prior'][val_ids], weights)
                record = records[stage]
                record['y'].extend(raw['y'][val_ids].tolist())
                record['p'].extend(probability.tolist())
                record['folds'].append({'start': first, 'end': last, 'train': len(train_ids), 'fine': len(fine_ids),
                                        'calibration': len(cal_ids), 'calibrator': weights.tolist(),
                                        **{k: v for k, v in metrics(raw['y'][val_ids], probability).items() if k != 'calibration_bins'}})
            del fitted
        rows = [{'spec': spec, 'stage': stage, 'folds': values['folds'],
                 **{k: v for k, v in metrics(values['y'], values['p']).items() if k != 'calibration_bins'}}
                for stage, values in records.items() if stage == 'pretrained' or spec['model'] != 'hist_gbm']
        development.extend(rows)
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print('DEVELOPMENT', json.dumps([{k: v for k, v in row.items() if k != 'folds'} for row in rows]),
              'elapsed_s', round(time.monotonic()-start, 1), flush=True)
    ranked = sorted(development, key=lambda x: x['log_loss'])
    family_best = {family: min([r for r in ranked if r['spec']['model'] == family], key=lambda r: r['log_loss'])
                   for family in sorted({r['spec']['model'] for r in ranked})}
    selection = {'best': ranked[0], 'family_best': family_best, 'development': ranked,
                 'benchmark_not_used_for_selection': True, 'benchmark_is_previously_inspected': True}
    # The complete selection is written before calculating any benchmark metric.
    (OUT/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print('SELECTED', ranked[0]['spec']['name'], ranked[0]['stage'], flush=True)
    train_ids = np.flatnonzero((domains == 'domestic') & (ends < '2026-04-01'))
    fine_ids = np.flatnonzero((domains == 'worlds') & (ends < '2026-04-01'))
    cal_ids = np.flatnonzero((dates >= '2026-04-01') & (ends < '2026-05-24'))
    old = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    test_ids = np.array([i for i, s in enumerate(samples) if s['id'] in old])
    assert len(test_ids) == 412 and all(int(old[samples[i]['id']]['y']) == raw['y'][i] for i in test_ids)
    rows, results = [], {}
    import pickle
    for family, choice in family_best.items():
        spec, stage = choice['spec'], choice['stage']
        probs = []
        for seed in ([42] if family == 'hist_gbm' else seeds):
            fitted = train_one(spec, raw, samples, train_ids, fine_ids, seed, device)
            folder = OUT/f'{family}_seed{seed}'
            folder.mkdir(exist_ok=True)
            calibrators = {name: calibrate(fitted[name][cal_ids], raw['prior'][cal_ids], raw['y'][cal_ids]) for name in ['pretrained', 'finetuned']}
            probability = apply_calibration(fitted[stage][test_ids], raw['prior'][test_ids], calibrators[stage])
            probs.append(probability)
            normalizer = fitted['normalizer'].state()
            if family == 'hist_gbm':
                with (folder/'model.pkl').open('wb') as stream:
                    pickle.dump(fitted['model'], stream)
            else:
                for saved_stage in ['pretrained', 'finetuned']:
                    state = fitted['before_state'] if saved_stage == 'pretrained' else fitted['model'].state_dict()
                    torch.save({'state_dict': state, 'model': family, 'config': fitted['config'], 'normalizer': normalizer,
                                'calibrator': calibrators[saved_stage].tolist(), 'stage': saved_stage,
                                'manifest': manifest}, folder/f'{saved_stage}.pt')
            metadata = {'model': family, 'spec': spec, 'selected_stage': stage, 'seed': seed, 'normalizer': normalizer,
                        'calibrators': {key: v.tolist() for key, v in calibrators.items()}, 'training': fitted['trace'],
                        'data_and_training_fingerprint': fingerprint}
            (folder/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            del fitted
        mean = np.mean(probs, axis=0)
        results[family] = {'stage': stage, 'development_log_loss': choice['log_loss'], 'seed_policy': 'equal weights',
                           **metrics(raw['y'][test_ids], mean)}
        for i, probability in zip(test_ids, mean):
            rows.append({'model': family, **{k: samples[i][k] for k in ['id', 'date', 'league', 'team_a', 'team_b', 'best_of', 'y']}, 'p_a': float(probability)})
    baseline = np.array([float(old[samples[i]['id']]['mlp_calibrated']) for i in test_ids])
    results['v1_mlp'] = metrics(raw['y'][test_ids], baseline)
    selected_family = ranked[0]['spec']['model']
    report = {'selection': selected_family, 'selection_stage': ranked[0]['stage'], 'manifest': manifest,
              'train_count': len(train_ids), 'fine_count': len(fine_ids), 'calibration_count': len(cal_ids),
              'results': results, 'dataset_samples': len(samples)}
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (OUT/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('RESULTS', json.dumps({key: {k: v for k, v in value.items() if k != 'calibration_bins'} for key, value in results.items()}), flush=True)
    return report


def test(args, config, parser):
    import pickle
    snapshot = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    if args.date <= snapshot['snapshot_last_day']:
        parser.error('Saved state already contains this day; use predictions.csv for historical evaluation.')
    aliases = {name.casefold(): key for name, key in snapshot['aliases'].items()}
    a, b = aliases.get(args.team_a.casefold(), args.team_a), aliases.get(args.team_b.casefold(), args.team_b)
    if a not in snapshot['teams'] or b not in snapshot['teams'] or a == b:
        parser.error('Choose two distinct known teams.')
    def roster(value):
        if value is None:
            return None
        result = json.loads(value)
        if set(result) != {'top', 'jng', 'mid', 'bot', 'sup'} or not all(isinstance(v, str) for v in result.values()):
            parser.error('A supplied lineup needs all five role names.')
        return result
    inputs, ra, rb = make_input(snapshot, a, b, args.date, args.patch, args.best_of, args.league, args.playoffs,
                                 roster(args.roster_a), roster(args.roster_b))
    inputs['prior'] = series_logits(torch.tensor(inputs['offset']), torch.tensor([args.best_of])).numpy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probabilities = []
    for seed in ([42] if config['model'] == 'hist_gbm' else config['seeds']):
        folder = OUT/f'{config["model"]}_seed{seed}'
        metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        normalizer = Normalizer.from_state(metadata['normalizer'])
        tensors = normalizer.transform(inputs, [{'league': args.league}], device)
        stage = 'pretrained' if config['model'] == 'hist_gbm' else config['stage']
        if config['model'] == 'hist_gbm':
            with (folder/'model.pkl').open('rb') as stream:
                model = pickle.load(stream)
            x = table_arrays(tensors)
            mirror = x.copy()
            mirror[:, :78] *= -1
            mirror[:, 101:] *= -1
            logits = (model.decision_function(x)-model.decision_function(mirror))/2
        else:
            checkpoint = torch.load(folder/f'{stage}.pt', map_location='cpu', weights_only=True)
            model = build_model(config['model'], checkpoint['config']).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            logits = infer(model, tensors, [0])
        probabilities.append(float(apply_calibration(logits, inputs['prior'], metadata['calibrators'][stage])[0]))
    value = float(np.mean(probabilities))
    print(json.dumps({'team_a': args.team_a, 'team_b': args.team_b, 'model': config['model'], 'stage': stage,
                      'p_a_win': value, 'p_b_win': 1-value, 'seed_probabilities': probabilities,
                      'snapshot_last_day': snapshot['snapshot_last_day'], 'lineup_a': ra, 'lineup_b': rb,
                      'lineup_status': 'Supplied or last-known historical lineup; not verified against current announcements.'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=[*NAMES, 'hist_gbm'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    run(args.models, args.seeds, args.refresh)
