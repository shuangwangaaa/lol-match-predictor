"""Restore adapted models and reproduce a fit with poisoned future labels."""
from copy import deepcopy
import json
import hashlib
import numpy as np
import torch
from core import metrics
from main import seed_everything
from models.roster_models import INPUT_KEYS, RosterTransformer
from pipelines import roster_train as shared
from pipelines.swiss_adapt import OUT, BASE, REPLAY, partitions, adaptation_labels, freeze_for_adaptation, step, load_parent
from tests.test_roster import swap_batch


def main():
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    parent_report = json.loads((BASE/'metrics.json').read_text(encoding='utf-8'))
    assert report['calibration_weights'] == parent_report['calibration_weights']
    rows = json.loads((REPLAY/'predictions.json').read_text(encoding='utf-8'))
    samples = json.loads((REPLAY/'samples.json').read_text(encoding='utf-8'))
    with np.load(REPLAY/'inputs.npz', allow_pickle=False) as stream:
        raw = {k: stream[k] for k in shared.ARRAY_FIELDS}
    splits = partitions(rows)
    raw['y'] = adaptation_labels(rows, splits['adaptation'])
    assert len(splits['adaptation']) == 16 and len(splits['evaluation']) == 10
    assert np.isnan(raw['y'][splits['evaluation']]).all()
    changed_tensors = {}
    for seed in report['protocol']['seeds']:
        base_path = BASE/f'seed{seed}'/'pretrained.pt'
        assert hashlib.sha256(base_path.read_bytes()).hexdigest() == report['protocol']['parent_sha256'][str(seed)]
        parent, _, _ = load_parent(seed, raw, samples, device)
        checkpoint = torch.load(OUT/f'seed{seed}'/'finetuned.pt', map_location='cpu', weights_only=True)
        assert checkpoint['stage'] == 'finetuned'
        assert checkpoint['normalizer'] == parent['normalizer']
        model = RosterTransformer(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        freeze_for_adaptation(model)
        changed = []
        for name, p in model.named_parameters():
            original = parent['state_dict'][name].to(device)
            if not p.requires_grad:
                torch.testing.assert_close(p, original, atol=0, rtol=0)
            elif not torch.equal(p, original):
                changed.append(name)
        assert changed and any(name.startswith('player_embedding.') for name in changed)
        assert any(name.startswith('head.') for name in changed)
        changed_tensors[str(seed)] = changed
        norm = shared.PlayerNormalizer.from_state(checkpoint['normalizer'])
        tensors = norm.transform(raw, samples, device)
        actual = shared.infer(model, tensors, np.arange(len(rows)))
        with np.load(OUT/f'seed{seed}'/'predictions.npz', allow_pickle=False) as stream:
            np.testing.assert_allclose(actual, stream['after_game_logits'], atol=1e-7, rtol=0)
        batch = dict(zip(INPUT_KEYS, shared.inputs(tensors, splits['evaluation'])))
        swapped = swap_batch(batch)
        with torch.no_grad():
            a = model(*[batch[k] for k in INPUT_KEYS])
            b = model(*[swapped[k] for k in INPUT_KEYS])
            torch.testing.assert_close(a, -b, atol=2e-5, rtol=2e-5)
        del model, tensors
    selection = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    best = min(selection['trace'], key=lambda x: x['R2_validation_log_loss'])
    assert best['epoch'] == selection['selected_positive_epochs']
    # Reproduce the actual first-seed fit after reversing R3/R4 outcomes.
    poisoned = deepcopy(rows)
    for i in splits['evaluation']:
        poisoned[i]['score_a'], poisoned[i]['score_b'] = poisoned[i]['score_b'], poisoned[i]['score_a']
    poisoned_raw = {**raw, 'y': adaptation_labels(poisoned, splits['adaptation'])}
    np.testing.assert_array_equal(poisoned_raw['y'], raw['y'])
    seed = report['protocol']['selection_seed']
    seed_everything(seed)
    _, model, tensors = load_parent(seed, poisoned_raw, samples, device)
    shared.infer(model, tensors, np.arange(len(rows)))
    optimizer = torch.optim.AdamW(freeze_for_adaptation(model),
                                  lr=report['protocol']['learning_rate'], weight_decay=report['protocol']['weight_decay'])
    for _ in range(selection['selected_positive_epochs']):
        step(model, tensors, splits['adaptation'], optimizer)
    checkpoint = torch.load(OUT/f'seed{seed}'/'finetuned.pt', map_location='cpu', weights_only=True)
    max_diff = 0.
    for name, value in model.state_dict().items():
        actual = value.detach().cpu()
        torch.testing.assert_close(actual, checkpoint['state_dict'][name].cpu(), atol=1e-7, rtol=1e-6)
        max_diff = max(max_diff, float((actual-checkpoint['state_dict'][name].cpu()).abs().max()))
    output = json.loads((OUT/'predictions.json').read_text(encoding='utf-8'))
    assert len(output) == 10 and all(r['round'] in (3, 4) for r in output)
    y = np.array([r['score_a'] > r['score_b'] for r in output])
    for key, field in [('before', 'before_p_a'), ('after', 'after_p_a')]:
        score = metrics(y, [r[field] for r in output])
        for name in ['n', 'accuracy', 'log_loss', 'brier']:
            np.testing.assert_allclose(score[name], report['results']['R3_R4'][key][name], atol=1e-10)
    result = {'adaptation_matches': 16, 'evaluation_matches': 10, 'restored_models': 3,
              'selected_epochs_from_R2_only': selection['selected_positive_epochs'],
              'calibrator_and_normalizer_unchanged': True, 'frozen_parameters_exact': True,
              'actual_changed_tensors': changed_tensors,
              'reversed_R3_R4_labels_do_not_change_training': True,
              'reproduced_training_max_parameter_difference': max_diff,
              'parent_checkpoint_hashes_unchanged': True, 'probability_complementarity': True,
              'saved_predictions_reproduced': True, 'evaluation_excludes_R1_R2': True}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
