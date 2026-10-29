"""Rebuild real chronological prefixes and restore new region-aware checkpoints."""
import csv
import json
import numpy as np
import torch
from core import ROOT, sigmoid
from data_gen.features_v2 import build
from data_gen.regional import RegionalRatings, DEFAULTS
from main import Normalizer, load_model, predict, raw_series
from pipelines.regional_train import OUT, forecast


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    config = report['protocol']['feature_signature']['config']
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    index = {s['id']: i for i, s in enumerate(samples)}
    with np.load(OUT/'data.npz', allow_pickle=False) as stream:
        raw = {k: stream[k] for k in stream.files}
    prefix, _ = build(team_path=ROOT/'data'/'teams_all.csv', include_first_game=True,
                       sequence_length=12, before='2026-01-01', regional_config=config)
    for sample in prefix:
        i = index[sample['id']]
        for key in ['d', 'c', 'seq_a', 'seq_b', 'mask_a', 'mask_b']:
            np.testing.assert_array_equal(np.asarray(sample[key], dtype=raw[key].dtype), raw[key][i])
    snapshot = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    r = RegionalRatings.from_state(snapshot['regional'])
    assert r.unknown_international_games == 0
    for day in [snapshot['snapshot_last_day'], '2026-10-28', '2027-10-28']:
        weights = r.weights(day)
        np.testing.assert_allclose(sum(weights.values()), 1., atol=1e-14)
        assert all(value > 0 for value in weights.values())
    np.testing.assert_allclose(sum(r.prior(name) for name in r.config['regions']), 1., atol=1e-14)
    ids = np.array([i for i, s in enumerate(samples) if s['date'] >= '2026-05-24' and s['best_of'] >= 3])[:64]
    restored = 0
    for seed in report['protocol']['seeds']:
        c = torch.load(OUT/f'seed{seed}'/'finetuned.pt', map_location='cpu', weights_only=True)
        assert c['feature_signature'] == report['protocol']['feature_signature']
        assert {**DEFAULTS, **c['regional_config']} == config
        assert c['normalizer']['vocabulary']['WCS'] > 0
        norm = Normalizer.from_state(c['normalizer'])
        tensors = norm.transform(raw, samples, device)
        model = load_model(c, 'dense_transformer', device)
        game = predict(model, tensors, ids)
        with np.load(OUT/f'seed{seed}'/'predictions.npz', allow_pickle=False) as stream:
            np.testing.assert_allclose(sigmoid(raw_series(game, raw['best_of'][ids])), stream['series_probability'][ids], atol=.002)
        batch = [tensors[k][ids] for k in ['d', 'c', 'seq_a', 'seq_b', 'mask_a', 'mask_b', 'league', 'offset']]
        with torch.no_grad():
            a = model(*batch)
            d, context, sa, sb, ma, mb, league, offset = batch
            b = model(-d, context, sb, sa, mb, ma, league, -offset)
        torch.testing.assert_close(a, -b, atol=2e-5, rtol=2e-5)
        restored += 1
    before = json.dumps(r.state(), sort_keys=True)
    r.coefficients('2026-10-28')
    assert json.dumps(r.state(), sort_keys=True) == before
    original = {x['id']: x for x in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    new = list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf-8')))
    assert len(new) == 412 and {x['id'] for x in new} == set(original)
    assert all(int(x['y']) == int(original[x['id']]['y']) for x in new)
    assert report['selection']['validation_end_exclusive'] <= min(s['date'] for s in new)
    forecasts = [forecast(a, b, '2026-10-28', '26.20', 3, 'WCS')
                 for a, b in [('Invictus Gaming', 'LYON'), ('LYON', 'Invictus Gaming'), ('Movistar KOI', 'T1')]]
    assert all(p['p_a_win'] == p['neural_p_a'] for p in forecasts)
    assert all(abs(p['region_weight_sum']-1) < 1e-14 for p in forecasts)
    np.testing.assert_allclose(forecasts[0]['p_a_win']+forecasts[1]['p_a_win'], 1, atol=1e-6)
    try:
        forecast('Invictus Gaming', 'LYON', '2026-10-25', '26.20', 3, 'WCS')
    except ValueError:
        pass
    else:
        raise AssertionError('Historical snapshot guard did not reject date.')
    (OUT/'predictions_worlds_2026-10-28.json').write_text(json.dumps(forecasts, indent=2, ensure_ascii=False), encoding='utf-8')
    result = {'prefix_samples': len(prefix), 'restored_models': restored, 'benchmark_membership': 412,
              'prefix_features_exact': True, 'model_swap_complementarity': True,
              'unknown_international_games': 0, 'historical_guard': True,
              'calibration_selection_before_benchmark': True, 'forecast_read_only': True,
              'region_weights_positive_and_sum_to_one': True}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
