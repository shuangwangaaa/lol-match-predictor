"""Check real pre-day inputs, immutable weights, predictions and score separation."""
from copy import deepcopy
import hashlib
import json
import numpy as np
import torch
from core import ROOT, metrics
from data_gen.roster_data import ARRAY_FIELDS
from pipelines.swiss_replay import REPORT, OUT, load_schedule, replay_inputs, predict


def main():
    torch.set_num_threads(4)
    schedule, matches = load_schedule()
    assert len(matches) == 26
    assert [sum(row['round'] == r for row in matches) for r in [1, 2, 3, 4]] == [8, 8, 8, 2]
    assert sum(row['best_of'] == 1 for row in matches) == 20
    results = json.loads((REPORT/'predictions.json').read_text(encoding='utf-8'))
    samples = json.loads((REPORT/'samples.json').read_text(encoding='utf-8'))
    summary = json.loads((REPORT/'summary.json').read_text(encoding='utf-8'))
    with np.load(REPORT/'inputs.npz', allow_pickle=False) as stream:
        raw = {k: stream[k] for k in ARRAY_FIELDS}
    original = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    index = {s['id']: i for i, s in enumerate(original)}
    checked = 0
    with np.load(OUT/'data.npz', allow_pickle=False) as stream:
        for i, r in enumerate(results):
            j = index.get(r['first_game_id'])
            if j is None:
                continue
            reverse = r['team_a'] != original[j]['team_a']
            for field in ARRAY_FIELDS:
                source = field
                if reverse and field.endswith(('_a', '_b')):
                    source = field[:-1]+('b' if field.endswith('_a') else 'a')
                value = np.array(stream[source][j])
                if reverse and field == 'offset':
                    value = -value
                if field == 'context':
                    # Old singleton targets had unknown series format 0. Official
                    # Swiss BO1 metadata supplies the pre-match known format 1.
                    value[2] = r['best_of']/5
                np.testing.assert_array_equal(raw[field][i], value)
            checked += 1
    assert checked == 20
    # Four last-CSV-day matches are now eligible because official results prove
    # their complete format/score. A full replay rebuilds their pre-day state.
    regenerated, regen_samples, _ = replay_inputs(schedule, matches)
    for key in ARRAY_FIELDS:
        np.testing.assert_array_equal(raw[key], regenerated[key])
    assert samples == regen_samples
    # The final two outcomes exist only in the external results manifest.
    # Reversing those labels must not change any frozen input or player identity.
    flipped = deepcopy(matches)
    for match in flipped[-2:]:
        match['score_a'], match['score_b'] = match['score_b'], match['score_a']
    independent, changed_samples, _ = replay_inputs(schedule, flipped)
    for key in ARRAY_FIELDS:
        np.testing.assert_array_equal(raw[key], independent[key])
    assert samples == changed_samples
    bo = np.array([r['best_of'] for r in matches])
    p, _ = predict(raw, samples, bo)
    np.testing.assert_allclose(p, [r['p_a_win'] for r in results], atol=1e-7, rtol=0)
    swapped = dict(raw)
    for field in ['players', 'history', 'mask', 'cohesion', 'aux']:
        swapped[field+'_a'], swapped[field+'_b'] = raw[field+'_b'], raw[field+'_a']
    swapped['offset'] = -raw['offset']
    swapped_samples = [{**s, 'names_a': s['names_b'], 'names_b': s['names_a']} for s in samples]
    neg, _ = predict(swapped, swapped_samples, bo)
    np.testing.assert_allclose(p+neg, 1, atol=1e-6, rtol=0)
    y = np.array([r['score_a'] > r['score_b'] for r in matches])
    for label, keep in [('all', np.ones(26, dtype=bool)), ('BO1', bo == 1), ('BO3', bo == 3)]:
        score = metrics(y[keep], p[keep])
        for key in ['n', 'accuracy', 'log_loss', 'brier']:
            np.testing.assert_allclose(score[key], summary['metrics'][label][key], atol=1e-10)
    for seed, expected in summary['checkpoint_sha256'].items():
        actual = hashlib.sha256((OUT/f'seed{seed}'/'pretrained.pt').read_bytes()).hexdigest()
        assert actual == expected
    verification = {'completed_swiss_matches': 26, 'bo1': 20, 'bo3': 6,
                    'existing_pre_day_feature_matches': checked,
                    'reconstructed_pre_day_inputs_exact': True, 'last_day_complete_matches_restored': 4,
                    'final_two_external_labels_do_not_change_features': True,
                    'swapped_probabilities_complementary': True, 'saved_predictions_reproduced': True,
                    'checkpoint_hashes_unchanged': True, 'metrics_recomputed': True}
    (REPORT/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
