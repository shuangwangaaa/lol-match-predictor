"""Replay completed Swiss series using locked 2026-only weights and pre-day state."""
from collections import defaultdict
from datetime import date
import csv
import hashlib
import json
import numpy as np
import torch
from core import ROOT, load_series, team_key, number, sigmoid, logit, metrics
from data_gen.features_v2 import load_players, player_key
from data_gen.roster_core import RosterState, ROLES
from data_gen.roster_data import ARRAY_FIELDS, match_input
from models.roster_models import RosterTransformer
from pipelines import roster_train as shared
from pipelines.roster_2026 import OUT, MODEL

REPORT = OUT/'swiss_replay'
SCHEDULE = ROOT/'data'/'swiss_2026_results.json'


def load_schedule():
    value = json.loads(SCHEDULE.read_text(encoding='utf-8'))
    rows = [dict(zip(value['fields'], row)) for row in value['matches']]
    seen = set()
    for row in rows:
        key = (row['round'], frozenset([row['a'], row['b']]))
        if key in seen or row['a'] == row['b'] or row['best_of'] not in (1, 3):
            raise ValueError('Invalid or duplicate Swiss schedule entry')
        seen.add(key)
        target = row['best_of']//2+1
        if max(row['score_a'], row['score_b']) != target or min(row['score_a'], row['score_b']) >= target:
            raise ValueError('Schedule contains an unfinished match')
    return value, rows


def replay_inputs(schedule, matches):
    """Outcomes are used only after all targets starting on a day are frozen."""
    report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    expected = report['protocol']['feature_signature']
    for path in ['data/teams_all.csv', 'data/players.csv.gz', 'data_gen/roster_core.py', 'models/roster_models.py']:
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected['sources'][path]:
            raise ValueError(f'Training source changed: {path}. Replay needs the matching historical sources.')
    series, _ = load_series(ROOT/'data'/'teams_all.csv')
    series = [s for s in series if s.start >= '2026-01-01' and s.end < '2027-01-01']
    last_day = max(s.end for s in series)
    rosters, _, _ = load_players(ROOT/'data'/'players.csv.gz', max_day=last_day)
    swiss = [s for s in series if s.first[0]['league'] == 'WCS' and s.start >= '2026-10-23']
    targets = {}
    for i, match in enumerate(matches):
        pair = {schedule['teams'][match['a']], schedule['teams'][match['b']]}
        candidates = [s for s in swiss if {r['teamname'] for r in s.first} == pair
                      and abs((date.fromisoformat(s.start)-date.fromisoformat(match['event_date'])).days) <= 1]
        if len(candidates) > 1:
            raise ValueError('Ambiguous source match')
        if candidates:
            source = candidates[0]
            by_name = {r['teamname']: team_key(r) for r in source.first}
            actual = (source.wins[by_name[schedule['teams'][match['a']]]],
                      source.wins[by_name[schedule['teams'][match['b']]]])
            if actual != (match['score_a'], match['score_b']) or len(source.games) != sum(actual):
                raise ValueError('Official score and complete local game records disagree')
            targets[source.first[0]['gameid']] = i
        elif match['event_date'] <= last_day:
            raise ValueError('Missing historical game before source cutoff')
    if len(targets) != len(swiss):
        raise ValueError('Some local Swiss series are absent from the official completed-match list')
    starts, ends = defaultdict(list), defaultdict(list)
    for s in series:
        starts[s.start].append(s)
        ends[s.end].append(s)
    state = RosterState(expected['regional'], expected['players'])
    frozen, metadata = {}, {}
    for day in sorted(set(starts) | set(ends)):
        for s in starts[day]:
            for row in s.first:
                state.regional.assign(team_key(row), row['league'])
        for s in starts[day]:
            i = targets.get(s.first[0]['gameid'])
            if i is None:
                continue
            match = matches[i]
            by_name = {r['teamname']: r for r in s.first}
            a, b = (by_name[schedule['teams'][match[k]]] for k in ['a', 'b'])
            ra = {role: player_key(row) for role, row in rosters.get((a['gameid'], a['side']), {}).items()}
            rb = {role: player_key(row) for role, row in rosters.get((b['gameid'], b['side']), {}).items()}
            if set(ra) != set(ROLES) or set(rb) != set(ROLES):
                raise ValueError('Incomplete starting roster for a completed Swiss match')
            patch = a['patch']
            frozen[i] = match_input(state, team_key(a), team_key(b), ra, rb, day, patch, match['best_of'], 'WCS', 0)
            metadata[i] = {'input_date': day, 'patch': patch, 'first_game_id': a['gameid'],
                           'lineup_a': ra, 'lineup_b': rb, 'lineup_source': 'actual first-game starters, assumed announced pre-BP',
                           'outcome_source': 'official bracket cross-checked against local complete games'}
        for s in sorted(ends[day], key=lambda s: s.games[-1][0]['date']):
            for game in s.games:
                a, b = sorted(game, key=team_key)
                pa = rosters.get((a['gameid'], a['side']), {})
                pb = rosters.get((b['gameid'], b['side']), {})
                if set(pa) != set(ROLES) or set(pb) != set(ROLES):
                    continue
                ra, rb = ({r: player_key(row) for r, row in players.items()} for players in [pa, pb])
                state.update(ra, rb, pa, pb, day, a.get('patch', ''), a['league'], team_key(a), team_key(b))
    saved = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    if state.state() != saved['state']:
        raise ValueError('Reconstructed historical state differs from the trained data snapshot')
    aliases = {name.casefold(): key for name, key in saved['aliases'].items()}
    # CSV stops on Oct 26. There are no completed Oct 27 Swiss matches; for the
    # two Oct 28 matches use the same immutable Oct 26 state, never their scores.
    # Their patch follows the observed tournament patch, not calendar year 26.
    patches = {row['patch'] for row in metadata.values()}
    if len(patches) != 1:
        raise ValueError('Explicit patch confirmation needed for matches beyond CSV cutoff')
    patch = next(iter(patches))
    for i, match in enumerate(matches):
        if i in frozen:
            continue
        a, b = [aliases[schedule['teams'][match[k]].casefold()] for k in ['a', 'b']]
        ra, rb = state.team_lineups[a], state.team_lineups[b]
        frozen[i] = match_input(state, a, b, ra, rb, match['event_date'], patch, match['best_of'], 'WCS', 0)
        metadata[i] = {'input_date': match['event_date'], 'patch': patch, 'first_game_id': None,
                       'lineup_a': ra, 'lineup_b': rb, 'lineup_source': 'last-known starters as of '+last_day,
                       'outcome_source': schedule['source'], 'history_cutoff': last_day,
                       'patch_source': 'last observed patch in same tournament'}
    rows = [frozen[i] for i in range(len(matches))]
    raw = {k: np.stack([row[k] for row in rows]) for k in ARRAY_FIELDS}
    samples = [{'league': 'WCS', 'names_a': row['names_a'], 'names_b': row['names_b']} for row in rows]
    return raw, samples, [metadata[i] for i in range(len(matches))]


def predict(raw, samples, best_of):
    from main import raw_series
    from pipelines.optimize import apply_calibration
    report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    probabilities, seed_logits = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in report['protocol']['seeds']:
        checkpoint = torch.load(OUT/f'seed{seed}'/'pretrained.pt', map_location='cpu', weights_only=True)
        if checkpoint['feature_signature'] != report['protocol']['feature_signature'] or checkpoint['stage'] != 'pretrained':
            raise ValueError('Checkpoint/calibrator provenance mismatch')
        model = RosterTransformer(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        norm = shared.PlayerNormalizer.from_state(checkpoint['normalizer'])
        tensors = norm.transform(raw, samples, device)
        values = shared.infer(model, tensors, np.arange(len(samples)))
        probabilities.append(sigmoid(raw_series(values, best_of)))
        seed_logits.append(values)
    prior = raw_series(raw['offset'], best_of)
    p = apply_calibration(logit(np.mean(probabilities, axis=0)), prior, report['calibration_weights'])
    return p, np.stack(seed_logits)


def main():
    torch.set_num_threads(4)
    REPORT.mkdir(exist_ok=True)
    schedule, matches = load_schedule()
    raw, samples, provenance = replay_inputs(schedule, matches)
    bo = np.array([row['best_of'] for row in matches])
    probability, seed_logits = predict(raw, samples, bo)
    results = []
    for i, match in enumerate(matches):
        p = float(probability[i])
        winner = match['a'] if match['score_a'] > match['score_b'] else match['b']
        pick = match['a'] if p >= .5 else match['b']
        results.append({**match, 'team_a': schedule['teams'][match['a']], 'team_b': schedule['teams'][match['b']],
                        'p_a_win': p, 'p_b_win': 1-p, 'predicted_winner': pick, 'actual_winner': winner,
                        'correct': pick == winner, **provenance[i]})
    scores = {}
    for name, keep in [('all', np.ones(len(results), dtype=bool)), ('BO1', bo == 1), ('BO3', bo == 3)]:
        y = np.array([r['score_a'] > r['score_b'] for r in results])[keep]
        scores[name] = {**metrics(y, probability[keep]), 'correct': sum(r['correct'] for i, r in enumerate(results) if keep[i])}
    for round_no in sorted({r['round'] for r in results}):
        subset = [r for r in results if r['round'] == round_no]
        scores[f'round_{round_no}'] = {'n': len(subset), 'correct': sum(r['correct'] for r in subset),
                                      'accuracy': np.mean([r['correct'] for r in subset]).item()}
    model_report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    summary = {'model': MODEL, 'seeds': model_report['protocol']['seeds'],
               'scope': 'Swiss only, through event date 2026-10-28; play-ins excluded',
               'metrics': scores, 'official_results_source': schedule['source'],
               'source_csv_last_day': '2026-10-26', 'full_local_matches': 24, 'official_score_only_matches': 2,
               'training_protocol': model_report['protocol']['time_boundaries'],
               'calibration': 'Existing BO3/BO5 series-logit slopes also applied to BO1; no Swiss-label refit and no dedicated BO1 calibration validation',
               'temporal_protocol': 'All targets on a CSV day are frozen before that day completed games update state; no same-series outcomes or postgame stats enter its input',
               'retrospective': True, 'no_retraining_or_calibrator_refitting': True,
               'corrections': ['Previous 8 Worlds-series metric included 6 play-in BO5, not 8 Swiss matches',
                               'Previous Oct 28 forecasts supplied patch 26.20; corrected to observed tournament patch 16.20',
                               'Last CSV day four confirmed complete Swiss matches restored using official format and score'],
               'checkpoint_sha256': {str(seed): hashlib.sha256((OUT/f'seed{seed}'/'pretrained.pt').read_bytes()).hexdigest() for seed in model_report['protocol']['seeds']}}
    (REPORT/'predictions.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    (REPORT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(REPORT/'inputs.npz', **raw, seed_game_logits=seed_logits)
    (REPORT/'samples.json').write_text(json.dumps(samples), encoding='utf-8')
    fields = ['round', 'event_date', 'a', 'b', 'best_of', 'p_a_win', 'p_b_win', 'predicted_winner',
              'score_a', 'score_b', 'actual_winner', 'correct', 'input_date', 'patch', 'lineup_source', 'outcome_source']
    with (REPORT/'predictions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    lines = ['# 瑞士轮逐场回放：2026 年限定选手模型', '',
             f'共 {scores["all"]["n"]} 场，命中 {scores["all"]["correct"]} 场（{scores["all"]["accuracy"]:.2%}）；BO1 {scores["BO1"]["correct"]}/{scores["BO1"]["n"]}，BO3 {scores["BO3"]["correct"]}/{scores["BO3"]["n"]}。', '',
             '模型权重与校准器均未重新拟合。每场按实际 BO 赛制和开赛前历史构造输入；BO3 胜率为整场胜率。BO1 沿用现有系列赛校准斜率，尚无单独 BO1 校准验证。', '',
             '此前 8 场世界赛统计含 6 场入围赛，不能视为瑞士轮成绩；本次仅统计官方瑞士轮。此前两场预测的版本参数写成 26.20，原始瑞士轮数据为 16.20，本次按 16.20 重算。', '',
             '本地 CSV 截至 10 月 26 日，覆盖前 24 场。10 月 28 日两场使用截至 26 日的同一只读状态与最后已知阵容，实际比分由官方赛程补充；没有将其比分输入特征。赛事日期与 CSV 日期因时区可能相差一天。', '',
             f'赛程与比分：[LoL Esports 官方瑞士轮]({schedule["source"]})。历史统计：[ChainCC Open Dataset](https://chaincc.lol/free/data)，CC BY 4.0。', '',
             '| 轮次 | 日期 | 对阵 | 前队胜率 | 后队胜率 | 预测赢家 | 实际比分 | 命中 |',
             '|---|---|---|---:|---:|---|---|---|']
    for r in results:
        lines.append(f'| R{r["round"]} | {r["event_date"][5:]} | {r["a"]}–{r["b"]} | {r["p_a_win"]:.2%} | {r["p_b_win"]:.2%} | {r["predicted_winner"]} | {r["score_a"]}:{r["score_b"]} | {"✓" if r["correct"] else "✗"} |')
    lines += ['', '已查看赛果后的回顾性分析，不属于事前盲测。样本量有限，不能用此次准确率证明泛化收益。完整阵容、来源与概率见 predictions.json；冻结输入及权重哈希一并保存。']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k in ['model', 'metrics', 'full_local_matches', 'official_score_only_matches']}, ensure_ascii=False), flush=True)
    print('\n'.join(lines[12:]), flush=True)


if __name__ == '__main__':
    main()
