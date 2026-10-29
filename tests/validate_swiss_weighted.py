"""Restore weighted artifacts and reproduce training with reversed future outcomes."""
from copy import deepcopy
import hashlib
import json
import numpy as np
import torch
from core import metrics
from main import seed_everything
from models.roster_models import RosterTransformer, INPUT_KEYS
from pipelines import roster_train as shared
from pipelines import swiss_weighted as weighted
from tests.test_roster import swap_batch


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = json.loads((weighted.OUT/'metrics.json').read_text(encoding='utf-8'))
    protocol, cfg = report['protocol'], report['protocol']['settings']
    base_report = json.loads((weighted.BASE/'metrics.json').read_text(encoding='utf-8'))
    assert report['calibration_weights'] == base_report['calibration_weights']
    rows = json.loads((weighted.REPLAY/'predictions.json').read_text(encoding='utf-8'))
    samples = json.loads((weighted.REPLAY/'samples.json').read_text(encoding='utf-8'))
    with np.load(weighted.REPLAY/'inputs.npz', allow_pickle=False) as stream:
        raw = {key: stream[key] for key in shared.ARRAY_FIELDS}
    split = weighted.partitions(rows)
    raw['y'] = weighted.adaptation_labels(rows, split['adaptation'])
    assert len(split['adaptation']) == 16 and len(split['evaluation']) == 10
    assert np.isnan(raw['y'][split['evaluation']]).all()
    replay_raw, replay_samples, replay_ids = weighted.rehearsal(cfg)
    assert replay_ids.tolist() == protocol['replay_indices']
    assert all(r['domain'] == 'domestic' and '2026-01-01' <= r['date'] <= r['end_date'] < '2026-09-01'
               for r in replay_samples)
    assert hashlib.sha256((weighted.REPLAY/'inputs.npz').read_bytes()).hexdigest() == protocol['input_sha256']
    assert hashlib.sha256((weighted.BASE/'data.npz').read_bytes()).hexdigest() == protocol['replay_input_sha256']
    changed_counts = {}
    for seed in protocol['seeds']:
        path = weighted.BASE/f'seed{seed}'/'pretrained.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol['parent_sha256'][str(seed)]
        parent = torch.load(path, map_location='cpu', weights_only=True)
        checkpoint = torch.load(weighted.OUT/f'seed{seed}'/'finetuned.pt', map_location='cpu', weights_only=True)
        assert checkpoint['normalizer'] == parent['normalizer']
        assert checkpoint['stage'] == 'finetuned'
        model = RosterTransformer(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        weighted.freeze(model)
        changed = []
        for name, p in model.named_parameters():
            original = parent['state_dict'][name].to(device)
            if not p.requires_grad:
                torch.testing.assert_close(p, original, atol=0, rtol=0)
            elif not torch.equal(p, original):
                changed.append(name)
        assert all(any(name.startswith(prefix) for name in changed)
                   for prefix in ['dense.', 'player_embedding.', 'head.'])
        changed_counts[str(seed)] = len(changed)
        norm = shared.PlayerNormalizer.from_state(checkpoint['normalizer'])
        tensors = norm.transform(raw, samples, device)
        actual = shared.infer(model, tensors, np.arange(len(rows)))
        with np.load(weighted.OUT/f'seed{seed}'/'predictions.npz', allow_pickle=False) as stream:
            np.testing.assert_allclose(actual, stream['after_game_logits'], atol=1e-7, rtol=0)
        batch = dict(zip(INPUT_KEYS, shared.inputs(tensors, split['evaluation'])))
        reverse = swap_batch(batch)
        with torch.no_grad():
            a = model(*[batch[key] for key in INPUT_KEYS])
            b = model(*[reverse[key] for key in INPUT_KEYS])
        torch.testing.assert_close(a, -b, atol=2e-5, rtol=2e-5)
        del model, tensors
    selection = json.loads((weighted.OUT/'selection.json').read_text(encoding='utf-8'))
    best = min(selection['trace'], key=lambda row: row['R2_validation_log_loss'])
    assert best['epoch'] == selection['selected_positive_epochs']
    assert not selection['R3_R4_used_for_selection']
    poisoned = deepcopy(rows)
    for i in split['evaluation']:
        poisoned[i]['score_a'], poisoned[i]['score_b'] = poisoned[i]['score_b'], poisoned[i]['score_a']
    poisoned_raw = {**raw, 'y': weighted.adaptation_labels(poisoned, split['adaptation'])}
    np.testing.assert_array_equal(raw['y'], poisoned_raw['y'])
    seed = protocol['selection_seed']
    seed_everything(seed)
    _, model, tensors, replay, teacher, optimizer = weighted.setup(
        seed, poisoned_raw, samples, replay_raw, replay_samples, device, cfg)
    assert not teacher.requires_grad
    fixed_teacher = teacher.clone()
    shared.infer(model, tensors, np.arange(len(rows)))
    for _ in range(selection['selected_positive_epochs']):
        weighted.update(model, tensors, split['adaptation'], replay, teacher,
                        weighted.draw_replay(teacher, cfg), optimizer, cfg)
    torch.testing.assert_close(teacher, fixed_teacher, atol=0, rtol=0)
    checkpoint = torch.load(weighted.OUT/f'seed{seed}'/'finetuned.pt', map_location='cpu', weights_only=True)
    max_diff = 0.
    for name, value in model.state_dict().items():
        actual, expected = value.detach().cpu(), checkpoint['state_dict'][name].cpu()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        max_diff = max(max_diff, float((actual-expected).abs().max()))
    output = json.loads((weighted.OUT/'predictions.json').read_text(encoding='utf-8'))
    assert len(output) == 10 and all(row['round'] in (3, 4) for row in output)
    y = np.array([row['score_a'] > row['score_b'] for row in output])
    for name, field in [('base', 'base_p_a'), ('weak', 'weak_p_a'), ('weighted', 'weighted_p_a')]:
        score = metrics(y, [row[field] for row in output])
        for key in ['n', 'accuracy', 'log_loss', 'brier']:
            np.testing.assert_allclose(score[key], report['results']['R3_R4'][name][key], atol=1e-10)
    result = {'restored_models': 3, 'adaptation_matches': 16, 'evaluation_matches': 10,
              'replay_only_original_2026_training_matches': True, 'teacher_unchanged': True,
              'parent_checkpoint_hashes_unchanged': True, 'normalizer_and_calibrator_unchanged': True,
              'frozen_parameters_exact': True, 'changed_tensor_counts': changed_counts,
              'saved_predictions_reproduced': True, 'probability_complementarity': True,
              'selected_epochs_from_R2_only': selection['selected_positive_epochs'],
              'reversed_future_labels_max_parameter_difference': max_diff,
              'evaluation_excludes_R1_R2': True}
    (weighted.OUT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
