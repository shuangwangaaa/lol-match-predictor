"""Verify the final release using only final weights, snapshot and configuration."""
import hashlib
import json
from unittest.mock import patch
import numpy as np
import torch
from core import ROOT
from main import load_config
from pipelines import roster_train as shared
from pipelines import swiss_weighted as weighted
from pipelines import swiss_complete as full


def main():
    config = load_config(ROOT/'config.toml')
    assert config['model'] == full.MODEL
    manifest = json.loads((ROOT/'docs/release_manifest.json').read_text(encoding='utf-8-sig'))
    for name, expected in manifest['files'].items():
        path = ROOT/name
        assert path.stat().st_size == expected['bytes'], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name
    report = json.loads((full.OUT/'metrics.json').read_text(encoding='utf-8'))
    expected_signature = report['protocol']['feature_signature']
    count, hashes = [], {}
    for seed in [42, 43, 44]:
        path = full.OUT/f'seed{seed}'/'finetuned.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['stage'] == 'finetuned'
        assert checkpoint['feature_signature'] == expected_signature
        count.append(sum(p.numel() for p in checkpoint['state_dict'].values()))
        hashes[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Inference remains usable if the original training assets are unavailable.
    absent = ROOT/'__not_distributed_training_assets__'
    with patch.object(weighted, 'BASE', absent), patch.object(weighted, 'REPLAY', absent), patch.object(weighted, 'WEAK', absent):
        a = shared.forecast('Top Esports', 'Invictus Gaming', '2026-11-03', '16.20', 5,
                            playoffs=1, artifact_dir=full.OUT, model_name=full.MODEL)
        b = shared.forecast('Invictus Gaming', 'Top Esports', '2026-11-03', '16.20', 5,
                            playoffs=1, artifact_dir=full.OUT, model_name=full.MODEL)
    np.testing.assert_allclose(a['p_a_win']+b['p_a_win'], 1., atol=2e-5)
    rows = json.loads((full.OUT/'rankings.json').read_text(encoding='utf-8'))['quarterfinals']
    saved = next(r['p_a'] for r in rows if r['a'] == 'TES' and r['b'] == 'IG')
    np.testing.assert_allclose(a['p_a_win'], saved, atol=2e-3 if not torch.cuda.is_available() else 2e-7, rtol=0)
    try:
        shared.forecast('Top Esports', 'Invictus Gaming', '2026-10-31', '16.20', 5,
                        artifact_dir=full.OUT, model_name=full.MODEL)
    except ValueError as exc:
        assert 'Snapshot includes' in str(exc)
    else:
        raise AssertionError('Historical date guard did not reject a leaking snapshot')
    result = {'date': '2026-11-02', 'model': full.MODEL, 'verified_release_files': len(manifest['files']),
              'final_checkpoint_sha256': hashes, 'parameters_per_seed': count,
              'inference_does_not_load_parent_or_replay_assets': True,
              'snapshot_last_day': a['snapshot_last_day'], 'historical_date_guard': True,
              'swap_probability_sum': a['p_a_win']+b['p_a_win'],
              'tes_vs_ig_p_tes': a['p_a_win'],
              'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    (ROOT/'docs/release_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
