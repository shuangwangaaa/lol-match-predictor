"""Audit the dense/Transformer experiment and restore the saved checkpoints."""
import csv
import gzip
import hashlib
import json
import unittest
import numpy as np
import torch
from data_gen.deep_data import OUT, ROOT
from data_gen.features_v2 import build
from models.deep_models import series_logits
from main import Normalizer, load_model, predict


def main():
    suite = unittest.defaultTestLoader.loadTestsFromNames(['tests.test_deep', 'tests.test_model'])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    raw = np.load(OUT/'data.npz', allow_pickle=False)
    feature_arrays = {key: raw[key] for key in ['d', 'c', 'seq_a', 'seq_b', 'mask_a', 'mask_b']}
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    index = {s['id']: i for i, s in enumerate(samples)}
    assert len(index) == len(samples)
    prefix, _ = build(team_path=ROOT/'data'/'teams_all.csv', include_first_game=True, sequence_length=12, before='2026-01-01')
    for sample in prefix:
        i = index[sample['id']]
        for key in ['d', 'c', 'seq_a', 'seq_b', 'mask_a', 'mask_b']:
            expected = feature_arrays[key][i]
            np.testing.assert_array_equal(np.asarray(sample[key], dtype=expected.dtype), expected)
    files = 0
    for metadata_file in ['sources.json', 'player_sources.json']:
        for source in json.loads((ROOT/'data'/metadata_file).read_text(encoding='utf-8')):
            path = ROOT/'data'/'raw'/source['url'].split('/')[-1]
            payload = path.read_bytes()
            assert hashlib.sha256(payload).hexdigest() == source['sha256_gzip']
            assert hashlib.sha256(gzip.decompress(payload)).hexdigest() == source['sha256_uncompressed']
            files += 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    restored, changed = 0, []
    for folder in sorted(OUT.glob('*_seed*')):
        checkpoint = torch.load(folder/'finetuned.pt', map_location='cpu', weights_only=True)
        model = load_model(checkpoint, folder.name.split('_seed')[0], device)
        tensor = Normalizer.from_state(checkpoint['normalizer']).transform(raw, samples, device)
        rows = list(csv.DictReader((folder/'predictions.csv').open(encoding='utf-8')))
        ids = np.array([index[r['id']] for r in rows[:8]])
        values = predict(model, tensor, ids)
        assert np.isfinite(values).all()
        logits = series_logits(torch.tensor(values), torch.tensor(raw['best_of'][ids]))
        actual = torch.sigmoid(logits/checkpoint['temperatures']['series']).numpy()
        expected = np.array([float(r['worlds_finetuned_series']) for r in rows[:8]])
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)
        # Distinguish actual fine-tuning from copying the pretraining checkpoint.
        pre = torch.load(folder/'pretrained.pt', map_location='cpu', weights_only=True)
        difference = sum(float(torch.sum((checkpoint['state_dict'][k]-pre['state_dict'][k])**2)) for k in checkpoint['state_dict'])
        assert difference > 0
        assert checkpoint['protocol']['worlds_2026_optimizer_labels_used'] == 0
        restored += 1
        changed.append({'run': folder.name, 'finetune_squared_weight_change': difference})
        del model, tensor
    output = {'unit_tests_passed': result.testsRun, 'raw_sources_sha256_verified': files,
              'real_grouped_prefix_invariance_passed': len(prefix), 'unique_grouped_samples': len(samples),
              'trained_and_finetuned_checkpoints_restored': restored,
              'finetune_is_actual_weight_update': changed,
              'worlds_2026_optimizer_labels_used': 0}
    (OUT/'verification.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2), flush=True)


if __name__ == '__main__':
    main()
