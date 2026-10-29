"""Real chronological prefix, restored model and transfer checks for player core."""
import csv
import json
import numpy as np
import torch
from core import ROOT, sigmoid
from main import load_config, raw_series
from data_gen.roster_core import RosterState
from data_gen.roster_data import build, ARRAY_FIELDS
from models.roster_models import RosterTransformer, INPUT_KEYS
from pipelines.roster_train import OUT, PlayerNormalizer, infer, inputs, forecast, test as cli_test, signature
from tests.test_roster import swap_batch


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    protocol = report['protocol']
    # Source fingerprints record training provenance; compatibility of later,
    # backward-compatible builders is verified by actual prefix reconstruction.
    current = signature(load_config(ROOT/'config.toml'))
    for key in ['regional', 'players']:
        assert current[key] == protocol['feature_signature'][key]
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    index = {s['id']: i for i, s in enumerate(samples)}
    with np.load(OUT/'data.npz', allow_pickle=False) as stream:
        raw = {k: stream[k] for k in stream.files}
    prefix, _ = build(protocol['feature_signature']['regional'], protocol['feature_signature']['players'], before='2026-01-01')
    for sample in prefix:
        i = index[sample['id']]
        for key in ARRAY_FIELDS:
            np.testing.assert_array_equal(np.asarray(sample[key], dtype=raw[key].dtype), raw[key][i])
    print(json.dumps({'prefix_features_exact': len(prefix)}), flush=True)
    snapshot = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    state = RosterState.from_state(snapshot['state'])
    before = json.dumps(state.state(), sort_keys=True)
    for day in [snapshot['snapshot_last_day'], '2026-10-28', '2027-10-28']:
        assert abs(sum(state.regional.weights(day).values())-1) < 1e-14
    assert json.dumps(state.state(), sort_keys=True) == before
    ids = np.array([i for i, s in enumerate(samples) if s['date'] >= '2026-05-24' and s['best_of'] >= 3])[:64]
    restored, max_error, parameters = 0, 0., None
    for seed in protocol['seeds']:
        checkpoint = torch.load(OUT/f'seed{seed}'/'finetuned.pt', map_location='cpu', weights_only=True)
        assert checkpoint['feature_signature'] == protocol['feature_signature']
        assert checkpoint['stage'] == 'finetuned'
        norm = PlayerNormalizer.from_state(checkpoint['normalizer'])
        tensors = norm.transform(raw, samples, device)
        model = RosterTransformer(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        total, core, aux = infer(model, tensors, ids, return_components=True)
        assert max(abs(aux)) <= state.config['team_aux_limit']+1e-6
        with np.load(OUT/f'seed{seed}'/'predictions.npz', allow_pickle=False) as stream:
            p = sigmoid(raw_series(total, raw['best_of'][ids]))
            np.testing.assert_allclose(p, stream['series_probability'][ids], atol=.002, rtol=0)
            max_error = max(max_error, float(max(abs(p-stream['series_probability'][ids]))))
        # Verify no post-training future identities entered the learned dictionary.
        training = [s for s in samples if (s['domain'] == 'domestic' and s['end_date'] < '2026-04-01')
                    or s['worlds_year'] in {'2024', '2025'}]
        allowed = {n for s in training for side in ['a', 'b'] for n in s['names_'+side] if n}
        assert set(norm.players) == allowed
        batch = dict(zip(INPUT_KEYS, inputs(tensors, ids[:8])))
        swapped = swap_batch(batch)
        with torch.no_grad():
            pos = model(*[batch[k] for k in INPUT_KEYS])
            neg = model(*[swapped[k] for k in INPUT_KEYS])
            torch.testing.assert_close(pos, -neg, atol=2e-5, rtol=2e-5)
            a = model.decompose(*[batch[k] for k in INPUT_KEYS])
            for side in ['a', 'b']:
                batch['aux_'+side] = torch.zeros_like(batch['aux_'+side])
            b = model.decompose(*[batch[k] for k in INPUT_KEYS])
            torch.testing.assert_close(a['player_correction'], b['player_correction'], atol=0, rtol=0)
        parameters = sum(p.numel() for p in model.parameters())
        restored += 1
        del model, tensors
    original = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    new = list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf-8')))
    assert len(new) == 412 and {r['id'] for r in new} == set(original)
    assert all(int(r['y']) == int(original[r['id']]['y']) for r in new)
    assert report['selection']['validation_end_exclusive'] <= min(s['date'] for s in new)
    forecasts = [forecast(a, b, '2026-10-28', '26.20', 3) for a, b in
                 [('Invictus Gaming', 'LYON'), ('LYON', 'Invictus Gaming')]]
    np.testing.assert_allclose(forecasts[0]['p_a_win']+forecasts[1]['p_a_win'], 1., atol=1e-6)
    renamed = forecast('NewOrganization', 'LYON', '2026-10-28', '26.20', 3,
                       roster_a=forecasts[0]['lineup_a'], region_a=forecasts[0]['regions'][0])
    np.testing.assert_allclose(renamed['player_only_p_a_win'], forecasts[0]['player_only_p_a_win'], atol=1e-7)
    assert renamed['seed_decompositions'][0]['team_auxiliary_logit'] != forecasts[0]['seed_decompositions'][0]['team_auxiliary_logit']
    both_new = forecast('NewA', 'NewB', '2026-10-28', '26.20', 3,
                        roster_a=forecasts[0]['lineup_a'], roster_b=forecasts[0]['lineup_b'],
                        region_a=forecasts[0]['regions'][0], region_b=forecasts[0]['regions'][1])
    assert all(s['team_auxiliary_logit'] == 0 for s in both_new['seed_decompositions'])
    assert both_new['p_a_win'] == both_new['player_only_p_a_win']
    np.testing.assert_allclose(both_new['player_only_p_a_win'], forecasts[0]['player_only_p_a_win'], atol=1e-7)
    try:
        forecast('Invictus Gaming', 'LYON', '2026-10-25', '26.20', 3)
    except ValueError:
        pass
    else:
        raise AssertionError('Historical snapshot guard failed')
    class Reject:
        def error(self, message):
            raise ValueError(message)
    config = load_config(ROOT/'config.toml')
    config['players']['team_aux_limit'] = .1
    try:
        cli_test(None, config, Reject())
    except ValueError as exc:
        assert 'configuration changed' in str(exc)
    else:
        raise AssertionError('Configuration guard failed')
    result = {'prefix_samples': len(prefix), 'all_player_features_exact': True, 'restored_models': restored,
              'parameters_per_model': parameters, 'max_restored_series_probability_error': max_error,
              'benchmark_membership_and_labels': 412, 'model_swap_complementarity': True,
              'team_rename_keeps_player_core': True, 'new_organizations_supported': True,
              'new_organization_has_zero_own_auxiliary': True, 'auxiliary_logit_bounded': True,
              'player_vocabulary_training_only': True, 'region_weights_sum_to_one': True,
              'calibration_selection_before_benchmark': True, 'historical_guard': True,
              'configuration_guard': True, 'state_queries_read_only': True}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT/'transfer_example.json').write_text(json.dumps({'original': forecasts[0], 'renamed': renamed,
                                                       'both_new_organizations': both_new}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
