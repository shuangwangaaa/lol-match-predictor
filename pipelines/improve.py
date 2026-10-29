"""Choose improvements on past-only rolling folds; report the reused benchmark honestly."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'.deps'))
import csv
import json
import math
import pickle
import numpy as np
from core import ROOT, metrics, sigmoid, logit, fit_temperature, bootstrap, series_probability

OUT = ROOT/'artifacts_v2'
FOLDS = [('2025-01-01', '2025-05-01'), ('2025-05-01', '2025-09-01'),
         ('2025-09-01', '2026-01-01'), ('2026-01-01', '2026-04-01')]
CAL_START, TEST_START = '2026-04-01', '2026-05-24'


class Ridge:
    def __init__(self, regularization, half_life=365):
        self.regularization, self.half_life = regularization, half_life

    def transform(self, d, c):
        d = np.clip(d/self.scale, -7, 7)
        return np.column_stack([d]+[d*((c[:, i]-self.center[j])/self.cscale[j])[:, None] for j, i in enumerate([2, 19])])

    def fit(self, d, c, offset, y, weights):
        self.scale = np.maximum(np.sqrt(np.mean(d*d, axis=0)), .05)
        self.center = c[:, [2, 19]].mean(axis=0)
        self.cscale = np.maximum(c[:, [2, 19]].std(axis=0), .1)
        x = self.transform(d, c)
        self.coef = np.zeros(x.shape[1])
        weights = weights/weights.sum()
        for step in range(30):
            p = sigmoid(offset+x@self.coef)
            gradient = x.T@(weights*(p-y))+self.regularization*self.coef
            hessian = x.T@((weights*p*(1-p))[:, None]*x)+np.eye(x.shape[1])*self.regularization
            delta = np.linalg.solve(hessian, gradient)
            self.coef -= delta
            if np.max(np.abs(delta)) < 1e-6:
                break
        return self

    def logits(self, d, c, offset):
        return offset+self.transform(d, c)@self.coef


class Trees:
    def __init__(self, depth, iterations=400, residual=False):
        from catboost import CatBoostClassifier
        self.residual = residual
        self.model = CatBoostClassifier(depth=depth, iterations=iterations, learning_rate=.035,
                                        l2_leaf_reg=15, loss_function='Logloss', random_seed=42,
                                        thread_count=4, allow_writing_files=False, verbose=False)

    def fit(self, d, c, offset, y, weights):
        from catboost import Pool
        pos, neg = np.c_[d, c, np.abs(d)], np.c_[-d, c, np.abs(d)]
        pool = Pool(np.r_[pos, neg], np.r_[y, 1-y], weight=np.r_[weights, weights],
                    baseline=np.r_[offset, -offset] if self.residual else None)
        self.model.fit(pool)
        return self

    def logits(self, d, c, offset):
        pos = self.model.predict(np.c_[d, c, np.abs(d)], prediction_type='RawFormulaVal')
        neg = self.model.predict(np.c_[-d, c, np.abs(d)], prediction_type='RawFormulaVal')
        return (offset if self.residual else 0)+(pos-neg)/2


class GameRidge(Ridge):
    """Use the complete scoreline likelihood, keeping pre-series features frozen."""
    def fit(self, d, c, offset, y, weights):
        return super().fit(d, c, math.log(10)*d[:, 2], y, weights)

    def logits(self, d, c, offset):
        map_p = sigmoid(super().logits(d, c, math.log(10)*d[:, 2]))
        output = np.empty(len(d))
        for bo in [3, 5]:
            keep = np.rint(c[:, 2]*5).astype(int) == bo
            output[keep] = logit(series_probability(map_p[keep], bo))
        return output


def candidates(nd):
    specs = []
    for j, k in enumerate([8, 16, 24, 40, 64, 96]):
        specs.append({'name': f'elo_k{k}', 'kind': 'elo', 'k_index': j, 'dim': nd, 'half_life': 365})
    dimensions = [('team', 47)] + ([('players', nd)] if nd > 47 else [])
    for label, width in dimensions:
        for reg in [.01, .1, 1.0]:
            for half_life in [180, 730]:
                specs.append({'name': f'ridge_{label}_r{reg}_h{half_life}', 'kind': 'ridge',
                              'regularization': reg, 'half_life': half_life, 'dim': width, 'k_index': 2})
        for reg in [.03, .1, 1.0]:
            specs.append({'name': f'map_ridge_{label}_r{reg}', 'kind': 'map_ridge',
                          'regularization': reg, 'half_life': 365, 'dim': width, 'k_index': 2})
        for depth in [3, 4, 5]:
            specs.append({'name': f'catboost_{label}_d{depth}', 'kind': 'catboost',
                          'depth': depth, 'half_life': 365, 'dim': width, 'k_index': 2})
        for depth in [3, 4]:
            specs.append({'name': f'catboost_residual_{label}_d{depth}', 'kind': 'catboost',
                          'residual': True, 'depth': depth, 'half_life': 365, 'dim': width, 'k_index': 2})
    return specs


def train_candidate(spec, train_idx, d, c, offsets, y, dates, game_y, game_n):
    if spec['kind'] == 'elo':
        return None
    if spec['kind'] == 'ridge':
        model = Ridge(spec['regularization'])
    elif spec['kind'] == 'map_ridge':
        model = GameRidge(spec['regularization'])
    else:
        model = Trees(spec['depth'], residual=spec.get('residual', False))
    train_dates = dates[train_idx].astype('datetime64[D]')
    age = (train_dates.max()-train_dates).astype('timedelta64[D]').astype(float)
    weights = np.exp2(-age/spec['half_life'])
    targets = y[train_idx]
    if spec['kind'] == 'map_ridge':
        targets = game_y[train_idx]
        weights = weights*game_n[train_idx]
    return model.fit(d[train_idx, :spec['dim']], c[train_idx], offsets[train_idx, spec['k_index']], targets, weights)


def predict(spec, model, idx, d, c, offsets):
    if spec['kind'] == 'elo':
        return offsets[idx, spec['k_index']]
    return model.logits(d[idx, :spec['dim']], c[idx], offsets[idx, spec['k_index']])


def compact(y, p):
    return {k: v for k, v in metrics(y, p).items() if k != 'calibration_bins'}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--development-only', action='store_true')
    parser.add_argument('--skip-trees', action='store_true')
    args = parser.parse_args()
    data = np.load(OUT/'features.npz', allow_pickle=False)
    d, c, y, offsets = data['d'], data['c'], data['y'], data['k_logits']
    game_y, game_n = data['map_fraction'], data['game_count']
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    dates = np.array([s['date'] for s in samples])
    end_dates = np.array([s['end_date'] for s in samples])
    specs = candidates(d.shape[1])
    if args.skip_trees:
        specs = [s for s in specs if s['kind'] != 'catboost']
    manifest = {'selection_folds': FOLDS, 'train_end_exclusive': CAL_START,
                'calibration_start': CAL_START, 'benchmark_start': TEST_START, 'candidates': specs,
                'selection_objectives': ['minimum rolling log loss', 'maximum rolling accuracy; log loss breaks ties'],
                'benchmark_status': 'Previously inspected V1 test set; V2 comparison is retrospective, not a new untouched test.'}
    (OUT/'search_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('SEARCH', len(specs), 'models x', len(FOLDS), 'past-only folds', flush=True)
    oof_idx = np.concatenate([np.flatnonzero((dates >= start) & (dates < end) & (end_dates < end)) for start, end in FOLDS])
    oof = {}
    scores = []
    for spec in specs:
        predictions, folds = [], []
        for start, end in FOLDS:
            train_idx = np.flatnonzero(end_dates < start)
            val_idx = np.flatnonzero((dates >= start) & (dates < end) & (end_dates < end))
            model = train_candidate(spec, train_idx, d, c, offsets, y, dates, game_y, game_n)
            raw = predict(spec, model, val_idx, d, c, offsets)
            predictions.extend(raw)
            folds.append({'start': start, 'end': end, 'n_train': len(train_idx), **compact(y[val_idx], sigmoid(raw))})
        raw = np.array(predictions)
        oof[spec['name']] = raw
        score = {'name': spec['name'], **compact(y[oof_idx], sigmoid(raw)), 'folds': folds}
        scores.append(score)
        print('DEV', json.dumps({k: v for k, v in score.items() if k != 'folds'}), flush=True)
    if args.development_only:
        (OUT/'team_development.json').write_text(json.dumps(sorted(scores, key=lambda s: s['log_loss']), indent=2), encoding='utf-8')
        print('DEVELOPMENT ONLY: benchmark not evaluated', flush=True)
        return
    # Fixed family ensembles, all evaluated on the same past-only OOF rows.
    families = {}
    for kind in ['elo', 'ridge', 'map_ridge', 'catboost']:
        eligible = [s for s in scores if next(p for p in specs if p['name'] == s['name'])['kind'] == kind]
        families[kind] = min(eligible, key=lambda s: s['log_loss'])['name']
    ensembles = []
    for left, right in [('elo', 'ridge'), ('elo', 'catboost'), ('ridge', 'catboost'), ('elo', 'map_ridge'), ('map_ridge', 'catboost')]:
        name = f'ensemble_{left}_{right}'
        components = [families[left], families[right]]
        raw = np.mean([oof[n] for n in components], axis=0)
        oof[name] = raw
        spec = {'name': name, 'kind': 'ensemble', 'components': components}
        ensembles.append(spec)
        scores.append({'name': name, **compact(y[oof_idx], sigmoid(raw))})
    specs += ensembles
    best_loss = min(scores, key=lambda s: s['log_loss'])['name']
    best_accuracy = min(scores, key=lambda s: (-s['accuracy'], s['log_loss']))['name']
    selection = {'probability_model': best_loss, 'winner_model': best_accuracy, 'family_best': families,
                 'development_scores': sorted(scores, key=lambda s: s['log_loss']),
                 'benchmark_was_not_used_in_selection': True,
                 'benchmark_is_reused_from_v1': True}
    # Persist the selected configuration BEFORE calculating any benchmark score.
    (OUT/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print('SELECTED', best_loss, best_accuracy, flush=True)
    train_idx = np.flatnonzero(end_dates < CAL_START)
    cal_idx = np.flatnonzero((dates >= CAL_START) & (dates < TEST_START) & (end_dates < TEST_START))
    test_idx = np.flatnonzero(dates >= TEST_START)
    models, temperatures, raw_cache = {}, {}, {}
    spec_by_name = {s['name']: s for s in specs}
    def final(name):
        if name in raw_cache:
            return raw_cache[name]
        spec = spec_by_name[name]
        if spec['kind'] == 'ensemble':
            values = [final(n) for n in spec['components']]
            pair = tuple(np.mean([v[i] for v in values], axis=0) for i in [0, 1])
        else:
            model = train_candidate(spec, train_idx, d, c, offsets, y, dates, game_y, game_n)
            models[name] = model
            pair = (predict(spec, model, cal_idx, d, c, offsets), predict(spec, model, test_idx, d, c, offsets))
        raw_cache[name] = pair
        return pair
    report_models, probabilities = {}, {}
    report_names = list(dict.fromkeys(['elo_k24', families['elo'], families['ridge'], families['map_ridge'], families['catboost'], best_loss, best_accuracy]))
    for name in report_names:
        cal_logits, test_logits = final(name)
        temp = fit_temperature(cal_logits, y[cal_idx])
        temperatures[name] = temp
        prob = sigmoid(test_logits/temp)
        probabilities[name] = prob
        report_models[name] = {'temperature': temp, **metrics(y[test_idx], prob)}
    old_rows = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    assert {samples[i]['id'] for i in test_idx} == set(old_rows), 'Benchmark changed; do not compare unlike populations.'
    old_p = np.array([float(old_rows[samples[i]['id']]['mlp_calibrated']) for i in test_idx])
    probabilities['v1_mlp'] = old_p
    report_models['v1_mlp'] = metrics(y[test_idx], old_p)
    comparison = bootstrap(y[test_idx], probabilities[best_accuracy], old_p)
    # Accuracy uncertainty is paired by series, descriptive only.
    rng = np.random.default_rng(2026)
    delta = ((probabilities[best_accuracy] >= .5) == y[test_idx]).astype(float)-((old_p >= .5) == y[test_idx]).astype(float)
    accuracy_ci = np.quantile([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(2000)], [.025, .975]).tolist()
    rows = []
    for j, i in enumerate(test_idx):
        row = {key: samples[i][key] for key in ['id', 'date', 'league', 'team_a', 'team_b', 'best_of', 'y']}
        row.update({name: float(p[j]) for name, p in probabilities.items()})
        rows.append(row)
    with (OUT/'benchmark_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'selection': selection, 'manifest': manifest,
              'final_train': {'n': len(train_idx), 'last_date': str(max(dates[train_idx]))},
              'calibration': {'n': len(cal_idx), 'first_date': str(min(dates[cal_idx])), 'last_date': str(max(dates[cal_idx]))},
              'benchmark': {'n': len(test_idx), 'first_date': str(min(dates[test_idx])), 'last_date': str(max(dates[test_idx]))},
              'results': report_models, 'comparison_to_v1': comparison,
              'accuracy_delta': float(delta.mean()), 'accuracy_delta_paired_95_ci': accuracy_ci,
              'by_league': {league: {'n': sum(samples[i]['league'] == league for i in test_idx),
                                       **compact(y[test_idx][np.array([samples[i]['league'] == league for i in test_idx])],
                                                 probabilities[best_accuracy][np.array([samples[i]['league'] == league for i in test_idx])])}
                            for league in sorted({samples[i]['league'] for i in test_idx})}}
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (OUT/'models.pkl').open('wb') as stream:
        pickle.dump({'models': models, 'temperatures': temperatures, 'specs': specs, 'selection': selection}, stream)
    print('BENCHMARK', json.dumps({k: {m: v for m, v in value.items() if m != 'calibration_bins'} for k, value in report_models.items()}), flush=True)
    print('IMPROVEMENT', json.dumps({'accuracy_delta': report['accuracy_delta'], 'ci': accuracy_ci, **comparison}), flush=True)


if __name__ == '__main__':
    # Persist importable class names, so saved models load in another process.
    sys.modules['improve'] = sys.modules[__name__]
    for cls in [Ridge, GameRidge, Trees]:
        cls.__module__ = 'improve'
    main()
