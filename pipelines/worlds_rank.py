"""Exact placement probabilities for the verified Worlds knockout bracket."""
from datetime import datetime, timedelta
import itertools
import json
import math
import numpy as np
import torch
from core import ROOT, sigmoid, logit
from data_gen.roster_core import RosterState
from data_gen.roster_data import match_input
from models.roster_models import RosterTransformer
from pipelines import roster_train as shared

BRACKET = ROOT/'data/worlds_2026_knockouts.json'


def exact_bracket(schedule, probability):
    distributions, top4 = {}, {}
    teams = [t for node in schedule['quarterfinals'] for t in [node['a'], node['b']]]
    if len(teams) != 8 or len(set(teams)) != 8:
        raise ValueError('The knockout bracket requires eight distinct teams')
    for node in schedule['quarterfinals']:
        a, b = node['a'], node['b']
        p = float(probability(a, b, node))
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid match probability')
        distributions[node['id']] = {a: p, b: 1-p}
        top4.update(distributions[node['id']])
    def combine(node):
        left, right = distributions[node['left']], distributions[node['right']]
        if set(left) & set(right):
            raise ValueError('A team appears on both sides of a bracket node')
        result = {t: 0. for t in [*left, *right]}
        for a, pa in left.items():
            for b, pb in right.items():
                p = float(probability(a, b, node))
                if not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError('Invalid match probability')
                result[a] += pa*pb*p
                result[b] += pa*pb*(1-p)
        if not np.isclose(sum(result.values()), 1.):
            raise ValueError('Bracket probability mass is not conserved')
        distributions[node['id']] = result
        return result
    final_probability = {}
    for node in schedule['semifinals']:
        final_probability.update(combine(node))
    champion = combine(schedule['final'])
    result = {}
    for team in teams:
        first, finalists, semifinalists = champion[team], final_probability[team], top4[team]
        placements = {'1': first, '2': finalists-first, '3-4': semifinalists-finalists, '5-8': 1-semifinalists}
        result[team] = {'champion': first, 'final': finalists, 'top4': semifinalists, 'placements': placements,
                        'expected_place': first+2*placements['2']+3.5*placements['3-4']+6.5*placements['5-8']}
    return result, distributions


def standings(rows, teams):
    from pipelines.swiss_complete import validate_schedule
    records, maps = validate_schedule(rows, teams)
    result = []
    for team in teams:
        opponents = [r['b'] if r['a'] == team else r['a'] for r in rows if team in (r['a'], r['b'])]
        wins, losses = records[team]
        placement = '1-8 (pending)' if wins == 3 else {2: '9-11', 1: '12-14', 0: '15-16'}[wins]
        result.append({'team': team, 'series_wins': wins, 'series_losses': losses,
                       'map_wins': maps[team][0], 'map_losses': maps[team][1],
                       'opponents': opponents, 'buchholz_opponent_final_wins': sum(records[t][0] for t in opponents),
                       'qualified': wins == 3, 'placement': placement})
    return result


def forecast_requests(folder, requests, meta):
    """Load each seed once; simulate no future player stats or fabricated wins."""
    from main import raw_series
    from pipelines.optimize import apply_calibration
    report = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    snapshot = json.loads((folder/'snapshot.json').read_text(encoding='utf-8'))
    state = RosterState.from_state(snapshot['state'])
    aliases = {name.casefold(): value for name, value in snapshot['aliases'].items()}
    features, samples, best_of = [], [], []
    for request in requests:
        if request['day'] <= snapshot['snapshot_last_day']:
            raise ValueError('Forecast date must be later than the full-stage snapshot')
        a, b = [aliases[meta['teams'][request[k]].casefold()] for k in ['a', 'b']]
        ra, rb = state.team_lineups[a], state.team_lineups[b]
        feature = match_input(state, a, b, ra, rb, request['day'], '16.20', 5, 'WCS', 1)
        features.append(feature)
        samples.append({'league': 'WCS', 'names_a': feature['names_a'], 'names_b': feature['names_b']})
        best_of.append(5)
    raw = {k: np.stack([r[k] for r in features]) for k in shared.ARRAY_FIELDS}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_probability = []
    for seed in report['protocol']['seeds']:
        checkpoint = torch.load(folder/f'seed{seed}/finetuned.pt', map_location='cpu', weights_only=True)
        if checkpoint['feature_signature'] != report['protocol']['feature_signature']:
            raise ValueError('Forecast checkpoint provenance differs')
        norm = shared.PlayerNormalizer.from_state(checkpoint['normalizer'])
        tensors = norm.transform(raw, samples, device)
        model = RosterTransformer(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        z = shared.infer(model, tensors, np.arange(len(requests)))
        seed_probability.append(sigmoid(raw_series(z, np.array(best_of))))
        del model, tensors
    prior = raw_series(raw['offset'], np.array(best_of))
    seed_probability = np.array(seed_probability)
    calibrated = apply_calibration(logit(seed_probability.mean(0)), prior, report['calibration_weights'])
    per_seed = [apply_calibration(logit(p), prior, report['calibration_weights']) for p in seed_probability]
    return calibrated, per_seed


def rank():
    from pipelines.swiss_complete import OUT, load_schedule, sha
    torch.set_num_threads(4)
    meta, rows = load_schedule()
    schedule = json.loads(BRACKET.read_text(encoding='utf-8'))
    snapshot = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    known = standings(rows, meta['teams'])
    teams = sorted(meta['teams'])
    requests, lookup = [], {}
    def register(a, b, day):
        a, b = sorted([a, b])
        key = (a, b, day)
        if key not in lookup:
            lookup[key] = len(requests)
            requests.append({'a': a, 'b': b, 'day': day})
    # Date-matched conditional probabilities for every possible future pairing.
    descendants = {}
    for node in schedule['quarterfinals']:
        descendants[node['id']] = [node['a'], node['b']]
        register(node['a'], node['b'], node['start_utc'][:10])
    for node in [*schedule['semifinals'], schedule['final']]:
        left, right = descendants[node['left']], descendants[node['right']]
        for a, b in itertools.product(left, right):
            register(a, b, node['start_utc'][:10])
        descendants[node['id']] = left+right
    # A separate same-date round-robin comparison estimates strength, not placement.
    for a, b in itertools.combinations(teams, 2):
        register(a, b, '2026-11-02')
    p, per_seed = forecast_requests(OUT, requests, meta)
    def probability(a, b, node, values=p):
        lo, hi = sorted([a, b])
        day = node.get('day', node.get('start_utc', '')[:10])
        value = float(values[lookup[lo, hi, day]])
        return value if a == lo else 1-value
    result, distributions = exact_bracket(schedule, probability)
    seed_results = [exact_bracket(schedule, lambda a, b, node, v=values: probability(a, b, node, v))[0]
                    for values in per_seed]
    ranking = []
    for team, value in sorted(result.items(), key=lambda item: -item[1]['champion']):
        ranking.append({'team': team, **value,
                        'seed_champion_min': min(r[team]['champion'] for r in seed_results),
                        'seed_champion_max': max(r[team]['champion'] for r in seed_results),
                        'ranking_basis': 'descending championship probability under fixed schedule'})
    qf = []
    for node in schedule['quarterfinals']:
        value = probability(node['a'], node['b'], node)
        beijing = datetime.fromisoformat(node['start_utc'].replace('Z', '+00:00'))+timedelta(hours=8)
        qf.append({**node, 'start_beijing': beijing.strftime('%Y-%m-%d %H:%M'), 'p_a': value,
                   'p_b': 1-value, 'pick': node['a'] if value >= .5 else node['b']})
    power = []
    for team in teams:
        average = float(np.mean([probability(team, other, {'day': '2026-11-02'}) for other in teams if other != team]))
        record = next(r for r in known if r['team'] == team)
        power.append({**record, 'mean_bo5_win_vs_other_15': average,
                      'champion_probability': result[team]['champion'] if team in result else 0.})
    power.sort(key=lambda row: -row['mean_bo5_win_vs_other_15'])
    for i, row in enumerate(power, 1):
        row['model_strength_rank'] = i
    output = {'as_of': '2026-11-02', 'model': 'roster_transformer_2026_swiss_all',
              'snapshot_last_day': snapshot['snapshot_last_day'], 'training_series': 33, 'training_maps': 51,
              'method': 'exact probability propagation over seven BO5s, equivalent to enumerating 128 bracket outcomes',
              'fixed_inputs': 'Latest actual lineups; no simulated future player statistics; UTC date-dependent decay only',
              'calibrator': 'unchanged parent calibrator; no fresh out-of-sample calibration test',
              'ranking_definition': 'title-probability ordering for qualified teams; actual tied placement tiers for eliminated teams',
              'bracket_source': schedule, 'bracket_sha256': sha(BRACKET), 'quarterfinals': qf,
              'title_ranking': ranking, 'all_16_strength_and_standings': power,
              'node_distributions': distributions,
              'mass_checks': {'champions': sum(r['champion'] for r in ranking),
                              'finalists': sum(r['final'] for r in ranking), 'semifinalists': sum(r['top4'] for r in ranking)},
              'seed_range_note': 'Variation among three fitted seeds only; not a confidence interval',
              'independent_test_matches': 0}
    (OUT/'rankings.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'forecast_requests.json').write_text(json.dumps([{**r, 'p_a': float(p[i]), 'per_seed': [float(v[i]) for v in per_seed]}
                                                         for i, r in enumerate(requests)], indent=2), encoding='utf-8')
    write_report(output, OUT)
    print('RANKINGS '+json.dumps({'quarterfinals': qf, 'title_ranking': ranking}, ensure_ascii=False), flush=True)
    return output


def write_report(output, folder):
    lines = ['# 全瑞士轮训练后的世界赛排名预测', '',
             '截至 2026-11-02，模型已将瑞士轮全部 33 场系列赛、51 个小局的结果加入训练。选手与对手实力状态更新至 10 月 31 日。以下是基于已公布淘汰赛对阵的前瞻概率；瑞士轮已属于训练集，不能再用作独立准确率。', '',
             '## 八强对阵', '', '| 北京时间 | BO5 对阵 | 前队胜率 | 后队胜率 | 预测优势方 |', '|---|---|---:|---:|---|']
    for r in output['quarterfinals']:
        lines.append(f'| {r["start_beijing"]} | {r["a"]} vs {r["b"]} | {r["p_a"]:.2%} | {r["p_b"]:.2%} | {r["pick"]} |')
    lines += ['', '赛程路径：TES/IG 的胜者与 GEN/LYON 的胜者争夺一个决赛名额，DK/BLG 的胜者与 T1/KC 的胜者争夺另一个。决赛为北京时间 11 月 15 日 03:00。赛程与赛制核对自 [LoL Esports](https://lolesports.com/en-US/leagues/worlds)，上下半区核对自 [Riot 对阵图的转载](https://gamedaily.com/games/lol-worlds-2026-knockouts-bracket)。', '',
              '## 按实际路径计算的排名', '',
              '下表按夺冠概率排序，用于比较当前赛程下的夺冠机会。模型实力排名另列于后文；对阵难度会使两种排序不同。期望名次对并列 3–4 和 5–8 名分别采用 3.5 和 6.5，不意味着赛事会产生唯一第 3 或第 5 名。', '',
              '| 预测顺序 | 队伍 | 冠军 | 亚军 | 3–4 名 | 5–8 名 | 进四强 | 进决赛 | 期望名次 |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for i, r in enumerate(output['title_ranking'], 1):
        x = r['placements']
        lines.append(f'| {i} | {r["team"]} | {x["1"]:.2%} | {x["2"]:.2%} | {x["3-4"]:.2%} | {x["5-8"]:.2%} | {r["top4"]:.2%} | {r["final"]:.2%} | {r["expected_place"]:.2f} |')
    lines += ['', '## 十六队当前实力与已知名次', '',
              '实力排序使用同一天、同一 BO5 环境下对另外 15 队的平均模型胜率。它是模型比较分数，不是赛事官方排名。已淘汰队伍的夺冠概率固定为 0，不能因模型实力较高重新进入淘汰赛。', '',
              '| 实力顺序 | 队伍 | 对其他队平均 BO5 胜率 | 瑞士轮战绩 | 小局战绩 | 已知赛事名次 |',
              '|---:|---|---:|---|---|---|']
    for r in output['all_16_strength_and_standings']:
        lines.append(f'| {r["model_strength_rank"]} | {r["team"]} | {r["mean_bo5_win_vs_other_15"]:.2%} | {r["series_wins"]}–{r["series_losses"]} | {r["map_wins"]}–{r["map_losses"]} | {r["placement"]} |')
    lines += ['', '## 训练与解释边界', '',
              '从同一 2026 联赛底座重新开始，避免在 R1/R2 微调权重上再次累计相同标签。沿用已选定 24 轮、10:1 世界赛/联赛教师约束、3e-4 学习率及原可训练层。监督损失为 75% 系列赛胜负 BCE 与 25% 单局胜率对实际小局胜率的 BCE，按系列赛等权；所以 2:0 和 2:1 提供不同信息，较长系列赛不会仅因局数更多而压过 BO1。', '',
              '前 24 场原始赛前特征逐值保留；后 9 场从 10 月 26 日快照继续构建，全部目标先于同日结果更新。22 个新增小局只更新一次。全部 51 个小局的队伍比分与选手结果交叉检查；没有伪造缺失 KDA 或经济统计。', '',
              '用精确动态规划传播七场 BO5 的全部可能结果，不会把每场优势方强行固定成胜者。所有未来对阵采用瑞士轮结束时最后已知阵容，日期只影响既有状态的时间衰减；不虚构未来对线统计。三个种子的概率先集成后沿用原校准器。', '',
              '当前没有新的独立测试集，也没有重新验证校准效果；夺冠概率取决于固定单局胜率的 BO5 假设，未覆盖 BP、选边、Fearless Draft 和真实阵容变化。种子间范围只是三个拟合结果的差异，不是置信区间。', '',
              '来源：[官方瑞士轮比分](https://lolesports.com/en-GB/tournament/115660540725177488/stage/115660540725570706)、[ChainCC 数据](https://chaincc.lol/free/data)（2026-11-02 获取，CC BY 4.0）。完整来源哈希见 preparation.json，训练设置见 protocol.json。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    rank()
