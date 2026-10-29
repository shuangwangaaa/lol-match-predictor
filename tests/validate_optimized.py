"""Check causal features, artifact restoration, and exact benchmark membership."""
import csv
import json
import pickle
import numpy as np
import torch
from core import ROOT
from data_gen.features_v2 import build
from models.deep_models import build_model
from main import Normalizer
from pipelines.optimize import OUT, infer, table_arrays, apply_calibration


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with np.load(OUT/'data.npz', allow_pickle=False) as stream:
        raw = {key: stream[key] for key in stream.files}
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    lookup = {s['id']: i for i, s in enumerate(samples)}
    assert len(lookup) == len(samples)
    prefix, _ = build(sequence_length=12, before='2026-01-01')
    for sample in prefix:
        i = lookup[sample['id']]
        for key in ['d', 'c', 'seq_a', 'seq_b', 'mask_a', 'mask_b']:
            np.testing.assert_array_equal(np.asarray(sample[key], dtype=raw[key].dtype), raw[key][i])
    report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    predictions = list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf-8')))
    originals = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    restored = 0
    for family, result in report['results'].items():
        if family == 'v1_mlp':
            continue
        rows = [r for r in predictions if r['model'] == family]
        assert len(rows) == 412 and {r['id'] for r in rows} == set(originals)
        assert all(r['y'] == originals[r['id']]['y'] for r in rows)
        selected = rows[:32]
        ids = np.array([lookup[r['id']] for r in selected])
        probabilities = []
        for folder in sorted(OUT.glob(f'{family}_seed*')):
            metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            normalizer = Normalizer.from_state(metadata['normalizer'])
            tensors = normalizer.transform(raw, samples, device)
            stage = metadata['selected_stage']
            if family == 'hist_gbm':
                with (folder/'model.pkl').open('rb') as stream:
                    model = pickle.load(stream)
                x = table_arrays(tensors)[ids]
                mirror = x.copy()
                mirror[:, :78] *= -1
                mirror[:, 101:] *= -1
                values = (model.decision_function(x)-model.decision_function(mirror))/2
            else:
                checkpoint = torch.load(folder/f'{stage}.pt', weights_only=True, map_location='cpu')
                model = build_model(family, checkpoint['config']).to(device)
                model.load_state_dict(checkpoint['state_dict'])
                values = infer(model, tensors, ids)
            probabilities.append(apply_calibration(values, raw['prior'][ids], metadata['calibrators'][stage]))
            restored += 1
            del model, tensors
        np.testing.assert_allclose(np.mean(probabilities, axis=0), [float(r['p_a']) for r in selected], atol=2e-3, rtol=2e-3)
    blended = list(csv.DictReader((OUT/'blend_predictions.csv').open(encoding='utf-8')))
    assert len(blended) == 412 and {r['id'] for r in blended} == set(originals)
    blend_selection = json.loads((OUT/'blend_selection.json').read_text(encoding='utf-8'))
    assert blend_selection['calibration_forward_validation']['last_exclusive'] <= min(r['date'] for r in blended)
    for r in blended:
        for name in ['expert_blend', 'recalibrated_transformer', 'recalibrated_tree']:
            assert 0 < float(r[name]) < 1
    output = {'real_past_prefix_samples_checked': len(prefix), 'restored_models': restored,
              'benchmark_ids_and_labels_unchanged': 412,
              'restored_predictions_match_saved_probabilities': True,
              'fusion_selection_precedes_benchmark': True,
              'recommendation_from_forward_validation': blend_selection['selection']}
    (OUT/'verification.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
