"""Use all completed Swiss series and map results, then forecast the fixed bracket."""
from collections import Counter, defaultdict
from datetime import date
import csv
import gzip
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F
from core import ROOT, team_key, metrics, sigmoid, logit
from data_gen.features_v2 import load_players, player_key
from data_gen.roster_core import RosterState, ROLES
from data_gen.roster_data import match_input
from models.deep_models import series_logits
from pipelines import roster_train as shared
from pipelines import swiss_weighted as weighted

OUT = ROOT/'artifacts_roster_2026_swiss_all'
MODEL = 'roster_transformer_2026_swiss_all'
COMPLETE = ROOT/'data/swiss_2026_complete.json'
DOWNLOADS = ROOT/'data/raw/post_swiss_2026_11_02'
EPOCHS = 24
SERIES_WEIGHT = .75


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_schedule():
    meta = json.loads(COMPLETE.read_text(encoding='utf-8'))
    initial = json.loads((ROOT/meta['initial_manifest']).read_text(encoding='utf-8'))
    rows = [dict(zip(initial['fields'], row)) for row in initial['matches']]
    rows += [dict(zip(meta['fields'], row)) for row in meta['additional_matches']]
    validate_schedule(rows, initial['teams'])
    return {**meta, 'teams': initial['teams']}, rows


def validate_schedule(rows, teams):
    records = {name: [0, 0] for name in teams}
    maps = {name: [0, 0] for name in teams}
    seen = set()
    for round_id in range(1, 6):
        playing = set()
        for row in [r for r in rows if r['round'] == round_id]:
            a, b = row['a'], row['b']
            if a == b or a in playing or b in playing or a not in records or b not in records:
                raise ValueError('Duplicate or unknown team in Swiss round')
            if records[a] != records[b] or max(records[a]) >= 3:
                raise ValueError('Illegal Swiss pool or a finished team plays again')
            expected_bo = 3 if 2 in records[a] else 1
            target = row['best_of']//2+1
            if row['best_of'] != expected_bo or max(row['score_a'], row['score_b']) != target or min(row['score_a'], row['score_b']) >= target:
                raise ValueError('Incomplete or wrong-format Swiss series')
            pair = frozenset([a, b])
            if pair in seen:
                raise ValueError('Unexpected repeated Swiss pairing')
            seen.add(pair)
            playing.update([a, b])
            win = row['score_a'] > row['score_b']
            records[a][0 if win else 1] += 1
            records[b][1 if win else 0] += 1
            maps[a][0] += row['score_a']; maps[a][1] += row['score_b']
            maps[b][0] += row['score_b']; maps[b][1] += row['score_a']
    if len(rows) != 33 or sum(r['score_a']+r['score_b'] for r in rows) != 51:
        raise ValueError('Complete Swiss stage must contain 33 series and 51 maps')
    if sum(r[0] == 3 for r in records.values()) != 8 or any(max(r) != 3 for r in records.values()):
        raise ValueError('Incomplete final Swiss standings')
    return records, maps


def provider_matches(meta, rows):
    with gzip.open(DOWNLOADS/'teams.csv.gz', 'rt', encoding='utf-8-sig', newline='') as stream:
        source = [r for r in csv.DictReader(stream) if r['league'] in ('WLDs', 'WCS')
                  and '2026-10-23' <= r['date'][:10] <= '2026-11-01']
    games = defaultdict(list)
    for item in source:
        row = { {'game_id': 'gameid', 'game_number': 'game', 'team_name': 'teamname', 'team_id': 'teamid'}.get(k, k): v
                for k, v in item.items() }
        row['league'] = 'WCS'
        row['result'] = str(int(row['result'].lower() == 'true'))
        games[row['gameid']].append(row)
    if len(games) != 51 or any(len(g) != 2 for g in games.values()):
        raise ValueError('Provider Swiss games are incomplete or duplicated')
    assigned, output = set(), []
    for match in rows:
        pair = {meta['teams'][match[k]] for k in ['a', 'b']}
        candidate = [g for g in games.values() if {r['teamname'] for r in g} == pair
                     and abs((date.fromisoformat(g[0]['date'][:10])-date.fromisoformat(match['event_date'])).days) <= 1]
        candidate.sort(key=lambda g: (g[0]['date'], int(g[0]['game'])))
        if len(candidate) != match['score_a']+match['score_b']:
            raise ValueError(f'Official score and provider game count differ: {match}')
        score = Counter()
        for n, game in enumerate(candidate, 1):
            if game[0]['gameid'] in assigned or any(int(r['game']) != n for r in game):
                raise ValueError('Duplicate map or noncontiguous map order')
            if {r['result'] for r in game} != {'0', '1'} or len({r['date'] for r in game}) != 1:
                raise ValueError('Inconsistent provider map result or timestamp')
            assigned.add(game[0]['gameid'])
            for r in game:
                score[r['teamname']] += int(r['result'])
        if (score[meta['teams'][match['a']]], score[meta['teams'][match['b']]]) != (match['score_a'], match['score_b']):
            raise ValueError('Official and provider winners differ')
        output.append(candidate)
    if assigned != set(games):
        raise ValueError('Unassigned Swiss games remain')
    return output


def prepare():
    OUT.mkdir(exist_ok=True)
    meta, rows = load_schedule()
    signature = {'complete_manifest': sha(COMPLETE), 'initial_manifest': sha(ROOT/meta['initial_manifest']),
                 'old_inputs': sha(weighted.REPLAY/'inputs.npz'), 'old_snapshot': sha(weighted.BASE/'snapshot.json')}
    if (OUT/'preparation.json').exists():
        cache = json.loads((OUT/'preparation.json').read_text(encoding='utf-8'))
        if cache['signature'] != signature or sha(OUT/'training_inputs.npz') != cache['inputs_sha256'] or sha(OUT/'snapshot.json') != cache['snapshot_sha256']:
            raise ValueError('Cached full-Swiss data changed; rebuild in a new output directory')
        with np.load(OUT/'training_inputs.npz', allow_pickle=False) as stream:
            raw = {k: stream[k] for k in stream.files}
        samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
        return meta, rows, raw, samples, cache
    grouped = provider_matches(meta, rows)
    rosters, _, _ = load_players(DOWNLOADS/'players.csv.gz', max_day='2026-11-01')
    for games in grouped:
        for game in games:
            for r in game:
                players = rosters.get((r['gameid'], r['side']), {})
                if set(players) != set(ROLES) or any(player_key(p) is None for p in players.values()):
                    raise ValueError('Incomplete player statistics for a Swiss map')
                if any(int(str(p['result']).lower() == 'true') != int(r['result']) for p in players.values()):
                    raise ValueError('Team/player map results disagree')
    old_snapshot = json.loads((weighted.BASE/'snapshot.json').read_text(encoding='utf-8'))
    state = RosterState.from_state(old_snapshot['state'])
    aliases = dict(old_snapshot['aliases'])
    old_cutoff = old_snapshot['snapshot_last_day']
    with np.load(weighted.REPLAY/'inputs.npz', allow_pickle=False) as stream:
        old = {key: stream[key] for key in shared.ARRAY_FIELDS}
    old_samples = json.loads((weighted.REPLAY/'samples.json').read_text(encoding='utf-8'))
    old_predictions = json.loads((weighted.REPLAY/'predictions.json').read_text(encoding='utf-8'))
    frozen, samples = {}, {}
    starts, ends = defaultdict(list), defaultdict(list)
    for i, games in enumerate(grouped):
        day, end = games[0][0]['date'][:10], games[-1][0]['date'][:10]
        if end <= old_cutoff:
            if i >= 24 or [rows[i][k] for k in ['a', 'b', 'score_a', 'score_b']] != [old_predictions[i][k] for k in ['a', 'b', 'score_a', 'score_b']]:
                raise ValueError('Immutable first 24 Swiss inputs do not align')
            frozen[i] = {key: old[key][i] for key in shared.ARRAY_FIELDS}
            samples[i] = {**old_samples[i], **rows[i], 'input_date': day,
                          'feature_origin': 'immutable original pre-day input', 'game_ids': [g[0]['gameid'] for g in games]}
        else:
            if day <= old_cutoff:
                raise ValueError('Series crosses original snapshot cutoff')
            starts[day].append(i); ends[end].append(i)
    updated, first_new_count = [], state.updated_games
    for day in sorted(set(starts) | set(ends)):
        for i in starts[day]:
            for r in grouped[i][0]:
                name, key = r['teamname'], team_key(r)
                if name in aliases and aliases[name] != key:
                    raise ValueError('Provider changed a team identity')
                aliases[name] = key
                state.regional.assign(key, 'WCS')
            by_name = {r['teamname']: r for r in grouped[i][0]}
            aa, bb = [by_name[meta['teams'][rows[i][k]]] for k in ['a', 'b']]
            lineup_a, lineup_b = [{role: player_key(p) for role, p in rosters[r['gameid'], r['side']].items()} for r in [aa, bb]]
            feature = match_input(state, team_key(aa), team_key(bb), lineup_a, lineup_b, day, aa['patch'], rows[i]['best_of'], 'WCS', 0)
            frozen[i] = feature
            samples[i] = {**rows[i], 'league': 'WCS', 'names_a': feature['names_a'], 'names_b': feature['names_b'],
                          'input_date': day, 'patch': aa['patch'], 'lineup_a': lineup_a, 'lineup_b': lineup_b,
                          'history_cutoff_exclusive': day, 'feature_origin': 'actual first-game lineup, pre-day history',
                          'game_ids': [g[0]['gameid'] for g in grouped[i]]}
        # Every starting series on this day is frozen before any same-day update.
        for i in sorted(ends[day], key=lambda j: grouped[j][-1][0]['date']):
            for game in grouped[i]:
                aa, bb = sorted(game, key=team_key)
                pa, pb = [rosters[r['gameid'], r['side']] for r in [aa, bb]]
                la, lb = [{role: player_key(p) for role, p in players.items()} for players in [pa, pb]]
                state.update(la, lb, pa, pb, day, aa['patch'], 'WCS', team_key(aa), team_key(bb))
                updated.append(aa['gameid'])
    if len(frozen) != 33 or len(updated) != 22 or len(set(updated)) != 22 or state.updated_games-first_new_count != 22:
        raise ValueError('Full-stage features or once-only new history updates are incomplete')
    raw = {key: np.stack([frozen[i][key] for i in range(33)]) for key in shared.ARRAY_FIELDS}
    raw.update(y=np.array([r['score_a'] > r['score_b'] for r in rows], dtype='float32'),
               map_fraction=np.array([r['score_a']/(r['score_a']+r['score_b']) for r in rows], dtype='float32'),
               best_of=np.array([r['best_of'] for r in rows], dtype='int64'))
    snapshot = {**old_snapshot, 'snapshot_last_day': max(g[-1][0]['date'][:10] for g in grouped),
                'aliases': aliases, 'state': state.state(), 'post_swiss_update': {'old_cutoff': old_cutoff, 'new_games': updated}}
    (OUT/'snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(OUT/'training_inputs.npz', **raw)
    sample_list = [samples[i] for i in range(33)]
    (OUT/'samples.json').write_text(json.dumps(sample_list, ensure_ascii=False, indent=2), encoding='utf-8')
    source = json.loads((DOWNLOADS/'sources.json').read_text(encoding='utf-8'))
    cache = {'signature': signature, 'inputs_sha256': sha(OUT/'training_inputs.npz'), 'snapshot_sha256': sha(OUT/'snapshot.json'),
             'sources': source, 'series': 33, 'maps': 51, 'old_swiss_maps': 29, 'new_history_maps': 22,
             'snapshot_last_day': snapshot['snapshot_last_day'], 'supervised_features': 'all pre-series/pre-day; first 24 preserved exactly',
             'bracket_or_knockout_outcomes_used': False}
    (OUT/'preparation.json').write_text(json.dumps(cache, indent=2), encoding='utf-8')
    return meta, rows, raw, sample_list, cache


def supervised_loss(z, labels, best_of, map_fraction):
    series_bce = F.binary_cross_entropy_with_logits(series_logits(z, best_of), labels)
    map_bce = F.binary_cross_entropy_with_logits(z, map_fraction)
    return SERIES_WEIGHT*series_bce+(1-SERIES_WEIGHT)*map_bce, series_bce, map_bce


def fit_step(model, tensors, replay, teacher, optimizer, cfg, best_of, map_fraction):
    model.eval(); model.dense.train(); model.matchup.train()
    optimizer.zero_grad(set_to_none=True)
    ids = np.arange(len(best_of))
    z = model(*shared.inputs(tensors, ids))
    supervision, series, maps = supervised_loss(z, tensors['y'], best_of, map_fraction)
    replay_ids = weighted.draw_replay(teacher, cfg)
    replay_z = model(*shared.inputs(replay, replay_ids))
    _, _, kl = weighted.objective(z, tensors['y'], replay_z, teacher[replay_ids])
    loss = (cfg['swiss_weight']*supervision+cfg['replay_weight']*kl)/(cfg['swiss_weight']+cfg['replay_weight'])
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite full-Swiss objective')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    optimizer.step()
    return {'objective': float(loss.detach()), 'series_bce': float(series.detach()),
            'map_bce': float(maps.detach()), 'replay_kl': float(kl.detach())}


def run(config):
    from main import seed_everything, raw_series
    from pipelines.optimize import apply_calibration
    from pipelines.worlds_rank import rank
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta, rows, raw, samples, cache = prepare()
    cfg = weighted.settings(config)
    original = json.loads((weighted.BASE/'metrics.json').read_text(encoding='utf-8'))
    if sorted(config['seeds']) != sorted(original['protocol']['seeds']):
        raise ValueError('Use the saved three parent seeds')
    replay_raw, replay_samples, replay_ids = weighted.rehearsal(cfg)
    protocol = {'model': MODEL, 'checkpoint_stage': 'finetuned', 'seeds': config['seeds'],
                'feature_signature': original['protocol']['feature_signature'], 'settings': cfg,
                'training_series': 33, 'training_maps': 51, 'training_rounds': [1, 2, 3, 4, 5],
                'epochs': EPOCHS, 'epochs_source': 'fixed 24 from previously selected R1/R2 protocol; no new result-based tuning',
                'starting_point': 'original 2026 domestic parent per seed; no cumulative R1/R2 double training',
                'supervised_loss': '.75 * series BCE after IID BO transform + .25 * map-fraction BCE; equal series means',
                'series_weight': SERIES_WEIGHT, 'swiss_to_teacher_weight': [cfg['swiss_weight'], cfg['replay_weight']],
                'replay_indices': replay_ids.tolist(), 'trainable_prefixes': list(weighted.PREFIXES),
                'calibration_refitted': False, 'evaluation_n': 0, 'Swiss_stage_is_training_data': True,
                'parent_sha256': {str(s): sha(weighted.BASE/f'seed{s}/pretrained.pt') for s in config['seeds']},
                'previous_weighted_sha256': {str(s): sha(weighted.OUT/f'seed{s}/finetuned.pt') for s in config['seeds']},
                'data_preparation': cache, 'source_sha256': sha(ROOT/'pipelines/swiss_complete.py'),
                'status': 'all Swiss outcomes used for training; in-sample fits are not predictive validation'}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    bo = torch.as_tensor(raw['best_of'], device=device)
    map_fraction = torch.as_tensor(raw['map_fraction'], device=device)
    history, probabilities = {}, []
    for seed in config['seeds']:
        seed_everything(seed)
        parent, model, tensors, replay, teacher, optimizer = weighted.setup(seed, raw, samples, replay_raw, replay_samples, device, cfg)
        history[str(seed)] = []
        for epoch in range(1, EPOCHS+1):
            history[str(seed)].append(fit_step(model, tensors, replay, teacher, optimizer, cfg, bo, map_fraction))
        folder = OUT/f'seed{seed}'; folder.mkdir(exist_ok=True)
        torch.save({**parent, 'state_dict': model.state_dict(), 'stage': 'finetuned', 'adaptation_protocol': protocol}, folder/'finetuned.pt')
        z = shared.infer(model, tensors, np.arange(len(rows)))
        np.savez_compressed(folder/'predictions.npz', training_game_logits=z)
        probabilities.append(sigmoid(raw_series(z, raw['best_of'])))
        print(json.dumps({'seed': seed, 'epochs': EPOCHS, 'series': 33, 'maps': 51,
                          'first': history[str(seed)][0], 'last': history[str(seed)][-1]}), flush=True)
        del model, tensors, replay, teacher, optimizer
    p = apply_calibration(logit(np.mean(probabilities, axis=0)), raw_series(raw['offset'], raw['best_of']), original['calibration_weights'])
    fit_metrics = metrics(raw['y'], p)
    report = {'protocol': protocol, 'calibration_weights': original['calibration_weights'],
              'training_fit_only': fit_metrics, 'results': {}, 'independent_test_n': 0}
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT/'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    fitted = [{**r, 'p_a_win_in_sample': float(p[i]), 'training_label': True} for i, r in enumerate(rows)]
    (OUT/'training_fit.json').write_text(json.dumps(fitted, indent=2), encoding='utf-8')
    rank()


def test(args, config, parser):
    return shared.test(args, config, parser, artifact_dir=OUT, model_name=MODEL)


if __name__ == '__main__':
    import sys
    from main import load_config
    if sys.argv[1:] == ['prepare']:
        *_, cache = prepare()
        print(json.dumps(cache, ensure_ascii=False), flush=True)
    else:
        run(load_config(ROOT/'config.toml'))
