"""Audit saved real-data artifacts without retraining or tuning the model."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from core import ROOT, load_series, make_features


def main():
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_model')
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    sources = json.loads((ROOT/'data'/'sources.json').read_text(encoding='utf-8'))
    for source in sources:
        raw = (ROOT/'data'/'raw'/Path(source['url']).name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['sha256_gzip']
        assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == source['sha256_uncompressed']
    series, _ = load_series(ROOT/'data'/'teams.csv')
    full, _ = make_features(series)
    prefix, _ = make_features([s for s in series if s.end < '2026-01-01'])
    lookup = {s['id']: s for s in full}
    for sample in prefix:
        actual = lookup[sample['id']]
        np.testing.assert_array_equal(sample['d'], actual['d'])
        np.testing.assert_array_equal(sample['c'], actual['c'])
        assert sample['elo_logit'] == actual['elo_logit']
    with (ROOT/'artifacts'/'series_features.csv').open(encoding='utf-8', newline='') as stream:
        features = list(csv.DictReader(stream))
    assert len({s['id'] for s in features}) == len(features)
    metrics = json.loads((ROOT/'artifacts'/'metrics.json').read_text(encoding='utf-8'))
    assert metrics['swap_max_error'] < 1e-12
    with (ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == metrics['splits']['test']['n']
    for row in predictions:
        for name in metrics['test']:
            assert 0 < float(row[name]) < 1
    output = {'unit_tests_passed': result.testsRun, 'source_files_sha256_verified': len(sources),
              'real_series_prefix_invariance_passed': len(prefix),
              'unique_series_features': len(features), 'test_predictions_verified': len(predictions),
              'swap_max_error': metrics['swap_max_error']}
    (ROOT/'artifacts'/'verification.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
