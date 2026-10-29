"""Verify year isolation, chronological prefixes and all six saved checkpoints."""
import json
import numpy as np
import torch
from core import ROOT, metrics, sigmoid
from main import load_config, raw_series
from data_gen.roster_core import RosterState
from data_gen.roster_data import ARRAY_FIELDS, build
from models.roster_models import RosterTransformer, INPUT_KEYS
from pipelines import roster_train as shared
from pipelines.roster_2026 import OUT, MODEL, PROFILES, signature, split_indices, forecast
from tests.test_roster import swap_batch


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    with np.load(OUT/'data.npz', allow_pickle=False) as stream:
        raw = {k: stream[k] for k in stream.files}
    config = load_config(ROOT/'config.toml')
    expected_signature = signature(config)
    snapshot = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    state = RosterState.from_state(snapshot['state'])
    assert snapshot['history_since'] == '2026-01-01'
    assert all(s['date'][:4] == s['end_date'][:4] == '2026' for s in samples)
    assert all('2026-01-01' <= row['day'] < '2027-01-01' for p in state.players.values() for row in p['history'])
    assert all('2026-01-01' <= p['last_day'] < '2027-01-01' for p in state.players.values())
    first = min(s['date'] for s in samples)
    for i, sample in enumerate(samples):
        if sample['date'] == first:
            assert not raw['mask_a'][i].any() and not raw['mask_b'][i].any()
            np.testing.assert_array_equal(raw['players_a'][i, :, 0], 0)
    prefix, _ = build(config.get('regional'), config.get('players'), since='2026-01-01', before='2026-07-01')
    index = {s['id']: i for i, s in enumerate(samples)}
    for s in prefix:
        i = index[s['id']]
        for key in ARRAY_FIELDS:
            np.testing.assert_array_equal(np.asarray(s[key], dtype=raw[key].dtype), raw[key][i])
    print(json.dumps({'2026_prefix_exact': len(prefix)}), flush=True)
    major = {s['id'] for s in json.loads((ROOT/'artifacts_optimized'/'samples.json').read_text(encoding='utf-8'))}
    import csv
    original = {r['id'] for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    restored, restoration_error, profile_counts = 0, 0., {}
    for profile in PROFILES:
        folder = OUT/'benchmark' if profile == 'benchmark' else OUT
        report = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        protocol = report['protocol']
        provenance = protocol['feature_signature']
        for key in ['regional', 'players', 'history_scope']:
            assert provenance[key] == expected_signature[key]
        # Report/example corrections can change pipeline source hashes without
        # changing trained tensors. Verify feature sources and actual restoration;
        # retain checkpoint provenance rather than rewriting training hashes.
        for path, digest in provenance['sources'].items():
            if not path.startswith('pipelines/'):
                assert expected_signature['sources'][path] == digest
        assert not protocol['older_checkpoint_used'] and not protocol['older_match_history_used']
        assert protocol['worlds_finetuning_n'] == protocol['worlds_2026_optimizer_labels_used'] == 0
        ids = split_indices(samples, profile, major, original)
        training = ids['train']
        allowed = {n for i in training for side in ['a', 'b'] for n in samples[i]['names_'+side] if n}
        score_ids = ids['evaluation']
        if profile == 'benchmark':
            assert {samples[i]['id'] for i in score_ids} == original and len(score_ids) == 412
        profile_counts[profile] = {k: len(v) for k, v in ids.items()}
        check_ids = np.r_[score_ids[:32], training[:8]]
        for seed in protocol['seeds']:
            checkpoint = torch.load(folder/f'seed{seed}'/'pretrained.pt', map_location='cpu', weights_only=True)
            assert checkpoint['feature_signature'] == provenance
            assert checkpoint['stage'] == 'pretrained' and checkpoint['worlds_finetuning_n'] == 0
            norm = shared.PlayerNormalizer.from_state(checkpoint['normalizer'])
            assert set(norm.players) == allowed
            tensors = norm.transform(raw, samples, device)
            model = RosterTransformer(**checkpoint['config']).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            total, core, aux = shared.infer(model, tensors, check_ids, return_components=True)
            assert max(abs(aux)) <= .150001
            with np.load(folder/f'seed{seed}'/'predictions.npz', allow_pickle=False) as stream:
                p = sigmoid(raw_series(total, raw['best_of'][check_ids]))
                delta = float(max(abs(p-stream['series_probability'][check_ids])))
                restoration_error = max(restoration_error, delta)
                np.testing.assert_allclose(p, stream['series_probability'][check_ids], atol=.002, rtol=0)
            batch = dict(zip(INPUT_KEYS, shared.inputs(tensors, check_ids[:8])))
            swapped = swap_batch(batch)
            with torch.no_grad():
                a = model(*[batch[k] for k in INPUT_KEYS])
                b = model(*[swapped[k] for k in INPUT_KEYS])
                torch.testing.assert_close(a, -b, atol=2e-5, rtol=2e-5)
            restored += 1
            del model, tensors
        with np.load(folder/'ensemble_predictions.npz', allow_pickle=False) as stream:
            actual = metrics(raw['y'][score_ids], stream['probability'][score_ids])
        assert actual == report['results'][MODEL]
    a = forecast('Invictus Gaming', 'LYON', '2026-10-28', '26.20', 3)
    b = forecast('LYON', 'Invictus Gaming', '2026-10-28', '26.20', 3)
    assert a['model'] == MODEL and a['history_since'] == '2026-01-01'
    np.testing.assert_allclose(a['p_a_win']+b['p_a_win'], 1., atol=1e-6)
    try:
        forecast('Invictus Gaming', 'LYON', '2026-10-25', '26.20', 3)
    except ValueError:
        pass
    else:
        raise AssertionError('Historical snapshot guard failed')
    for day in ['2026-10-26', '2027-01-01']:
        assert abs(sum(state.regional.weights(day).values())-1) < 1e-14
    result = {'history_2026_only': True, 'player_vocabulary_training_only': True,
              'no_old_checkpoint_or_worlds_finetune': True, 'feature_prefix_exact': len(prefix),
              'restored_models': restored, 'max_restored_probability_error': restoration_error,
              'temporal_partitions': profile_counts, 'benchmark_membership': 412,
              'model_swap_complementarity': True, 'auxiliary_bounded': True,
              'region_weights_sum_to_one': True, 'historical_guard': True}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
