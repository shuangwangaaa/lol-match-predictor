"""Validate full-stage inputs, immutable parents, three fits and scheduled forecasts."""
import json
import numpy as np
import torch
from core import ROOT
from main import seed_everything
from models.roster_models import RosterTransformer
from pipelines import roster_train as shared
from pipelines import swiss_weighted as weighted
from pipelines import swiss_complete as full
from pipelines.worlds_rank import exact_bracket


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, rows, raw, samples, preparation = full.prepare()
    report = json.loads((full.OUT/'metrics.json').read_text(encoding='utf-8'))
    protocol = report['protocol']
    assert len(rows) == 33 and len(raw['y']) == 33
    assert sum(r['score_a']+r['score_b'] for r in rows) == 51
    np.testing.assert_array_equal(raw['y'], [r['score_a'] > r['score_b'] for r in rows])
    np.testing.assert_allclose(raw['map_fraction'], [r['score_a']/(r['score_a']+r['score_b']) for r in rows])
    with np.load(weighted.REPLAY/'inputs.npz', allow_pickle=False) as stream:
        for key in shared.ARRAY_FIELDS:
            np.testing.assert_array_equal(raw[key][:24], stream[key][:24])
    original_snapshot = json.loads((weighted.BASE/'snapshot.json').read_text(encoding='utf-8'))
    snapshot = json.loads((full.OUT/'snapshot.json').read_text(encoding='utf-8'))
    assert snapshot['snapshot_last_day'] == '2026-10-31'
    assert snapshot['state']['updated_games']-original_snapshot['state']['updated_games'] == 22
    assert len(set(snapshot['post_swiss_update']['new_games'])) == 22
    assert max(row['input_date'] for row in samples) <= '2026-10-31'
    assert report['independent_test_n'] == 0 and protocol['Swiss_stage_is_training_data']
    changes = {}
    for seed in protocol['seeds']:
        path = weighted.BASE/f'seed{seed}/pretrained.pt'
        assert full.sha(path) == protocol['parent_sha256'][str(seed)]
        assert full.sha(weighted.OUT/f'seed{seed}/finetuned.pt') == protocol['previous_weighted_sha256'][str(seed)]
        parent = torch.load(path, map_location='cpu', weights_only=True)
        checkpoint = torch.load(full.OUT/f'seed{seed}/finetuned.pt', map_location='cpu', weights_only=True)
        assert checkpoint['normalizer'] == parent['normalizer']
        model = RosterTransformer(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        weighted.freeze(model)
        changed = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                torch.testing.assert_close(p, parent['state_dict'][name].to(device), atol=0, rtol=0)
            elif not torch.equal(p.detach().cpu(), parent['state_dict'][name].cpu()):
                changed.append(name)
        assert all(any(n.startswith(prefix) for n in changed) for prefix in ['dense.', 'player_embedding.', 'head.'])
        changes[str(seed)] = len(changed)
        tensors = shared.PlayerNormalizer.from_state(checkpoint['normalizer']).transform(raw, samples, device)
        with np.load(full.OUT/f'seed{seed}/predictions.npz', allow_pickle=False) as stream:
            np.testing.assert_allclose(shared.infer(model, tensors, np.arange(33)), stream['training_game_logits'], atol=1e-7, rtol=0)
        del model, tensors
    # Reproduce seed 42 with the exact recorded setup, including teacher replay.
    cfg = protocol['settings']
    replay_raw, replay_samples, _ = weighted.rehearsal(cfg)
    seed_everything(42)
    _, model, tensors, replay, teacher, optimizer = weighted.setup(42, raw, samples, replay_raw, replay_samples, device, cfg)
    bo = torch.as_tensor(raw['best_of'], device=device)
    map_fraction = torch.as_tensor(raw['map_fraction'], device=device)
    for _ in range(protocol['epochs']):
        full.fit_step(model, tensors, replay, teacher, optimizer, cfg, bo, map_fraction)
    checkpoint = torch.load(full.OUT/'seed42/finetuned.pt', map_location='cpu', weights_only=True)
    max_diff = 0.
    for name, p in model.state_dict().items():
        expected = checkpoint['state_dict'][name].cpu()
        torch.testing.assert_close(p.detach().cpu(), expected, atol=1e-7, rtol=1e-6)
        max_diff = max(max_diff, float((p.detach().cpu()-expected).abs().max()))
    rankings = json.loads((full.OUT/'rankings.json').read_text(encoding='utf-8'))
    assert np.isclose(rankings['mass_checks']['champions'], 1.)
    assert np.isclose(rankings['mass_checks']['finalists'], 2.)
    assert np.isclose(rankings['mass_checks']['semifinalists'], 4.)
    assert all(row['champion_probability'] == 0 for row in rankings['all_16_strength_and_standings'] if not row['qualified'])
    # Confirm the published QF probabilities match ordinary CLI inference.
    meta, _ = full.load_schedule()
    for row in rankings['quarterfinals']:
        actual = shared.forecast(meta['teams'][row['a']], meta['teams'][row['b']], row['start_utc'][:10], '16.20', 5,
                                 playoffs=1, artifact_dir=full.OUT, model_name=full.MODEL)
        np.testing.assert_allclose(actual['p_a_win'], row['p_a'], atol=2e-7, rtol=0)
    assert report['calibration_weights'] == json.loads((weighted.BASE/'metrics.json').read_text(encoding='utf-8'))['calibration_weights']
    result = {'model': full.MODEL, 'training_series': 33, 'training_maps': 51, 'all_33_labels_present': True,
              'first_24_inputs_unchanged': True, 'new_history_maps_updated_once': 22, 'snapshot_last_day': '2026-10-31',
              'all_3_saved_predictions_restored': True, 'changed_tensors': changes, 'frozen_parameters_exact': True,
              'original_parent_and_R1_R2_weighted_checkpoints_unchanged': True, 'calibrator_and_normalizer_unchanged': True,
              'seed42_reproduction_max_parameter_difference': max_diff, 'knockout_probabilities_mass_conserved': True,
              'all_4_QFs_match_normal_inference': True, 'eliminated_teams_have_zero_title_probability': True,
              'independent_test_matches': 0}
    (full.OUT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
