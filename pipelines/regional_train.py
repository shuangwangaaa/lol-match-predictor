"""Retrain Dense/Transformer on causal region-anchored, opponent-adjusted features."""
import csv
import hashlib
import json
import math
import numpy as np
import torch
from core import ROOT, metrics, sigmoid, logit, fit_temperature
from data_gen.features_v2 import build
from data_gen.deep_data import DOMESTIC, WORLD_WINDOWS
from data_gen.regional import DEFAULTS, RegionalRatings

OUT = ROOT/'artifacts_regional_simplex'


def fingerprint(config):
    paths = ['data/teams_all.csv', 'data/players.csv.gz', 'data_gen/features_v2.py',
             'data_gen/regional.py', 'models/deep_models.py', 'main.py']
    return {'config': {**DEFAULTS, **config},
            'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}


def prepare(config):
    OUT.mkdir(exist_ok=True)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT/'data'/'raw').glob('chaincc-teams-*.csv.gz'))}
    manifest = OUT/'team_sources.json'
    if not (ROOT/'data'/'teams_all.csv').exists() or not manifest.exists() or json.loads(manifest.read_text(encoding='utf-8')) != source_hashes:
        from data_gen.deep_data import write_all_teams
        write_all_teams()
        manifest.write_text(json.dumps(source_hashes, indent=2), encoding='utf-8')
    signature = fingerprint(config)
    if (OUT/'preparation.json').exists() and json.loads((OUT/'preparation.json').read_text(encoding='utf-8')) == signature:
        with np.load(OUT/'data.npz', allow_pickle=False) as stream:
            raw = {key: stream[key] for key in stream.files}
        return raw, json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    samples, metadata = build(team_path=ROOT/'data'/'teams_all.csv', include_first_game=True,
                              sequence_length=12, save_snapshot=OUT/'snapshot.json', regional_config=config)
    arrays = ['d', 'c', 'seq_a', 'seq_b', 'mask_a', 'mask_b']
    raw = {k: np.stack([s[k] for s in samples]) for k in arrays}
    for k in ['d', 'c', 'seq_a', 'seq_b']:
        raw[k] = raw[k].astype('float32')
    raw.update(y=np.array([s['y'] for s in samples], dtype='float32'),
               first_map_y=np.array([s['first_map_y'] for s in samples], dtype='float32'),
               map_target=np.array([s['map_fraction'] for s in samples], dtype='float32'),
               game_count=np.array([s['game_count'] for s in samples], dtype='float32'),
               best_of=np.array([s['best_of'] for s in samples], dtype='int64'),
               offset=np.array([math.log(10)*s['d'][2] for s in samples], dtype='float32'))
    rows = []
    for sample in samples:
        row = {k: v for k, v in sample.items() if k not in set(arrays+['k_logits'])}
        first, last, _ = WORLD_WINDOWS[row['date'][:4]]
        worlds = row['league'] == 'WCS' and first <= row['date'] <= last
        row['domain'] = 'worlds' if worlds else 'domestic' if row['league'] in DOMESTIC else 'other'
        row['worlds_year'] = row['date'][:4] if worlds else ''
        rows.append(row)
    np.savez_compressed(OUT/'data.npz', **raw)
    (OUT/'samples.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'feature_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'preparation.json').write_text(json.dumps(signature, indent=2), encoding='utf-8')
    print(json.dumps({'prepared': len(rows), 'difference_features': raw['d'].shape[1],
                      'international_updates': metadata['regional']['international_updates'],
                      'unknown_international_games': metadata['regional']['unknown_international_games']}), flush=True)
    return raw, rows


def normalizer(raw, ids, samples):
    from main import Normalizer
    n = Normalizer().fit(raw, ids, samples)
    # Reserve a real Worlds embedding using the known competition schema.
    # The pretraining data are domestic; this row first learns in Worlds fine-tuning.
    if 'WCS' not in n.vocabulary:
        n.vocabulary['WCS'] = len(n.vocabulary)+1
    return n


def run(config):
    from main import seed_everything, fit_stage, predict, raw_series
    from models.deep_models import build_model
    from pipelines.optimize import calibrate, apply_calibration
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw, samples = prepare(config.get('regional', {}))
    dates = np.array([s['date'] for s in samples])
    ends = np.array([s['end_date'] for s in samples])
    domestic = np.array([s['domain'] == 'domestic' for s in samples])
    years = np.array([s['worlds_year'] for s in samples])
    choose_train = np.flatnonzero(domestic & (ends < '2025-07-01'))
    choose_val = np.flatnonzero(domestic & (dates >= '2025-07-01') & (ends < '2025-10-14'))
    choose_fine = np.flatnonzero(years == '2024')
    choose_fine_val = np.flatnonzero(years == '2025')
    final_train = np.flatnonzero(domestic & (ends < '2026-04-01'))
    final_fine = np.flatnonzero((years == '2024') | (years == '2025'))
    major_ids = {s['id'] for s in json.loads((ROOT/'artifacts_optimized'/'samples.json').read_text(encoding='utf-8'))}
    major = np.array([s['id'] in major_ids for s in samples])
    trusted = raw['best_of'] >= 3
    early = np.flatnonzero(major & trusted & (dates >= '2026-04-01') & (ends < '2026-05-01'))
    late = np.flatnonzero(major & trusted & (dates >= '2026-05-01') & (ends < '2026-05-24'))
    cal = np.r_[early, late]
    # Epochs, coefficients and calibrator selection are locked before reporting benchmark metrics.
    protocol = {'prior_interpretation': 'LCK=.4,LPL=.3,LEC=.1; other registered regions equally share .2. All region weights sum to one at all dates.',
                'region_registry': {**DEFAULTS, **config.get('regional', {})}['regions'],
                'region_updates': 'Known cross-region matches only; bounded, mean-reverting logit updates, followed by softmax across a fixed registry; weights always sum to one',
                'domestic_train_end_exclusive': '2026-04-01', 'domestic_train_n': len(final_train),
                'worlds_finetune_years': [2024, 2025], 'worlds_finetune_n': len(final_fine),
                'worlds_2026_optimizer_labels_used': 0,
                'selection': '2025 domestic/Worlds selects training epochs; April->May 2026 selects calibration family',
                'calibration_early_n': len(early), 'calibration_late_n': len(late),
                'seeds': config['seeds'], 'max_pretrain_epochs': config['max_pretrain_epochs'],
                'max_finetune_epochs': config['max_finetune_epochs'],
                'feature_signature': fingerprint(config.get('regional', {})),
                'status': 'Region prior design requested after inspecting IG-LYON and previous benchmarks; all comparisons retrospective.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    probabilities, traces, choices = [], {}, None
    for seed in config['seeds']:
        seed_everything(seed)
        folder = OUT/f'seed{seed}'
        folder.mkdir(exist_ok=True)
        if choices is None:
            norm = normalizer(raw, choose_train, samples)
            tensors = norm.transform(raw, samples, device)
            dimensions = {'nd': raw['d'].shape[1], 'nc': raw['c'].shape[1],
                          'leagues': len(norm.vocabulary)+1, 'sequence_length': 12}
            model = build_model('dense_transformer', dimensions).to(device)
            pre = fit_stage(model, tensors, choose_train, choose_val, 'regional_selection_pretrain',
                            config['max_pretrain_epochs'], 4)
            fine = fit_stage(model, tensors, choose_fine, choose_fine_val, 'worlds_finetune',
                             config['max_finetune_epochs'], 4)
            choices = {'pretrain_epochs': pre['best_epoch'], 'finetune_epochs': fine['best_epoch'],
                       'selection_seed': seed, 'remaining_seeds': 'Fixed epochs chosen by first seed; no test selection.'}
            traces['selection'] = [pre, fine]
            (OUT/'choices.json').write_text(json.dumps(choices, indent=2), encoding='utf-8')
            del model, tensors
        seed_everything(seed)
        norm = normalizer(raw, final_train, samples)
        tensors = norm.transform(raw, samples, device)
        dimensions = {'nd': raw['d'].shape[1], 'nc': raw['c'].shape[1],
                      'leagues': len(norm.vocabulary)+1, 'sequence_length': 12}
        model = build_model('dense_transformer', dimensions).to(device)
        pre = fit_stage(model, tensors, final_train, None, 'regional_final_pretrain',
                        choices['pretrain_epochs'], 0, fixed_epochs=choices['pretrain_epochs'])
        def save(stage):
            torch.save({'state_dict': model.state_dict(), 'config': dimensions, 'normalizer': norm.state(),
                        'stage': stage, 'regional_config': config.get('regional', {}),
                        'feature_signature': protocol['feature_signature']}, folder/f'{stage}.pt')
        save('pretrained')
        fine = fit_stage(model, tensors, final_fine, None, 'worlds_finetune',
                         choices['finetune_epochs'], 0, fixed_epochs=choices['finetune_epochs'])
        save('finetuned')
        logits = predict(model, tensors, np.arange(len(samples)))
        p = sigmoid(raw_series(logits, raw['best_of']))
        probabilities.append(p)
        np.savez_compressed(folder/'predictions.npz', game_logits=logits, series_probability=p)
        traces[str(seed)] = [pre, fine]
        del model, tensors
        print(json.dumps({'seed_complete': seed}), flush=True)
    deep = logit(np.mean(probabilities, axis=0))
    prior = raw_series(raw['offset'], raw['best_of'])
    early_weights = calibrate(deep[early], prior[early], raw['y'][early])
    early_temperature = fit_temperature(prior[early], raw['y'][early])
    validation = {'regional_transformer': metrics(raw['y'][late], apply_calibration(deep[late], prior[late], early_weights)),
                  'regional_elo': metrics(raw['y'][late], sigmoid(prior[late]/early_temperature))}
    selected = min(validation, key=lambda k: validation[k]['log_loss'])
    selection = {'selected': selected, 'validation': validation,
                 'validation_end_exclusive': '2026-05-24', 'seeds': config['seeds'],
                 'choices': choices, 'selection_has_seen_benchmark': False}
    (OUT/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    weights = calibrate(deep[cal], prior[cal], raw['y'][cal])
    temperature = fit_temperature(prior[cal], raw['y'][cal])
    neural_p = apply_calibration(deep, prior, weights)
    elo_p = sigmoid(prior/temperature)
    final_p = neural_p if selected == 'regional_transformer' else elo_p
    # Benchmark is only accessed for reporting after all choices above are persisted.
    old = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    benchmark = np.array([i for i, s in enumerate(samples) if s['id'] in old])
    assert len(benchmark) == 412
    assert all(int(old[samples[i]['id']]['y']) == int(raw['y'][i]) for i in benchmark)
    worlds = np.flatnonzero((years == '2026') & trusted)
    inter = np.flatnonzero(major & trusted & ~domestic & (dates >= '2026-05-24'))
    results = {}
    for name, p in [('regional_transformer', neural_p), ('regional_elo', elo_p), ('selected', final_p)]:
        results[name] = {'reused_412': metrics(raw['y'][benchmark], p[benchmark]),
                         'worlds_2026_series': metrics(raw['y'][worlds], p[worlds]),
                         'international_since_2026_05_24': metrics(raw['y'][inter], p[inter])}
    report = {'protocol': protocol, 'selection': selection, 'calibration_weights': weights.tolist(),
              'elo_temperature': float(temperature), 'results': results,
              'parameters_per_network': sum(v.numel() for v in torch.load(OUT/f'seed{config["seeds"][0]}'/'finetuned.pt', map_location='cpu', weights_only=True)['state_dict'].values())}
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT/'training_history.json').write_text(json.dumps(traces, indent=2), encoding='utf-8')
    np.savez_compressed(OUT/'ensemble_predictions.npz', deep=deep, prior=prior, probability=final_p)
    rows = [{**{k: samples[i][k] for k in ['id', 'date', 'league', 'team_a', 'team_b', 'best_of', 'y']},
             'regional_elo': float(elo_p[i]), 'regional_transformer': float(neural_p[i]),
             'selected_probability': float(final_p[i])} for i in benchmark]
    with (OUT/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('REGIONAL_RESULTS', json.dumps({'selected': selected, 'results': {k: {n: {m: v for m, v in score.items() if m != 'calibration_bins'} for n, score in value.items()} for k, value in results.items()}}), flush=True)
    write_report(report)
    return report


def forecast(a_name, b_name, day, patch, best_of, league, playoffs=0, roster_a=None, roster_b=None):
    from main import Normalizer, make_input, load_model, predict, raw_series
    torch.set_num_threads(4)
    report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    snapshot = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    if day <= snapshot['snapshot_last_day']:
        raise ValueError('Saved snapshot includes this date or later; use historical stored predictions.')
    aliases = {k.casefold(): v for k, v in snapshot['aliases'].items()}
    a, b = aliases.get(a_name.casefold(), a_name), aliases.get(b_name.casefold(), b_name)
    if a not in snapshot['teams'] or b not in snapshot['teams'] or a == b:
        raise ValueError('Choose two distinct known teams.')
    raw, ra, rb = make_input(snapshot, a, b, day, patch, best_of, league, playoffs, roster_a, roster_b)
    prior = raw_series(raw['offset'], np.array([best_of]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    members = []
    for seed in report['protocol']['seeds']:
        checkpoint = torch.load(OUT/f'seed{seed}'/'finetuned.pt', map_location='cpu', weights_only=True)
        tensors = Normalizer.from_state(checkpoint['normalizer']).transform(raw, [{'league': league}], device)
        model = load_model(checkpoint, 'dense_transformer', device)
        logits = predict(model, tensors, np.array([0]))
        members.append(float(sigmoid(raw_series(logits, np.array([best_of])))[0]))
    from pipelines.optimize import apply_calibration
    deep = np.array([float(logit(np.mean(members)))])
    neural = float(apply_calibration(deep, prior, report['calibration_weights'])[0])
    elo = float(sigmoid(prior/report['elo_temperature'])[0])
    selected = report['selection']['selected']
    # The explicitly named neural model always returns its neural prediction.
    # The validation-preferred Elo baseline remains visible rather than silently
    # replacing the requested Dense/Transformer architecture.
    p = neural
    regional = RegionalRatings.from_state(snapshot['regional'])
    regions = [regional.origins.get(team) for team in [a, b]]
    coefficients = regional.coefficients(day)
    return {'model': 'regional_transformer', 'selected_component': 'regional_transformer',
            'validation_preferred_new_candidate': selected, 'date': day, 'patch': patch,
            'league': league, 'best_of': best_of, 'team_a': a_name, 'team_b': b_name,
            'p_a_win': p, 'p_b_win': 1-p, 'regional_elo_p_a': elo, 'neural_p_a': neural,
            'seed_probabilities_a': members, 'regions': regions,
            'region_coefficients': {r: coefficients.get(r) for r in regions},
            'all_region_weights': {r: value['current'] for r, value in coefficients.items()},
            'region_weight_sum': sum(value['current'] for value in coefficients.values()),
            'snapshot_last_day': snapshot['snapshot_last_day'], 'lineup_a': ra, 'lineup_b': rb,
            'lineup_status': 'Supplied or last-known; current official lineup not fetched.'}


def test(args, config, parser):
    if config['stage'] != 'finetuned':
        parser.error('Regional calibration requires --stage finetuned.')
    def roster(value):
        if value is None:
            return None
        data = json.loads(value)
        if set(data) != {'top', 'jng', 'mid', 'bot', 'sup'} or not all(isinstance(n, str) for n in data.values()):
            raise ValueError('Roster needs names for top,jng,mid,bot,sup.')
        return {k: v.casefold() for k, v in data.items()}
    try:
        report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
        if sorted(config['seeds']) != sorted(report['protocol']['seeds']):
            parser.error('Use the saved ensemble seeds; changing them requires refitting calibration.')
        if {**DEFAULTS, **config.get('regional', {})} != report['protocol']['feature_signature']['config']:
            parser.error('Regional coefficients/settings changed. Retrain before using the saved model.')
        output = forecast(args.team_a, args.team_b, args.date, args.patch, args.best_of, args.league,
                           args.playoffs, roster(args.roster_a), roster(args.roster_b))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, ensure_ascii=False, indent=2))


def write_report(report):
    result = report['results']['regional_transformer']['reused_412']
    baseline = report['results']['regional_elo']['reused_412']
    metadata = json.loads((OUT/'feature_metadata.json').read_text(encoding='utf-8'))
    lines = ['# 赛区分层评分与对手强度修正', '',
             '归一化先验：LCK 0.4、LPL 0.3、LEC 0.1，其余已登记赛区平分 0.2。全部初始权重之和为 1。', '',
             '默认固定名单为 LCK、LPL、LEC、LCS、LCP、CBLOL，其余三个各为 0.2/3。为统一回放历史，LTA N、历史 LLA 归入 LCS；PCS/VCS/LJL/LCO 归入 LCP，LTA S 归入 CBLOL，欧洲地区联赛归入 LEC。可在 [regional].regions 中单列 LLA，此时四个其余赛区各为 0.05。未知体系不新增权重项，也不更新赛区层。', '',
             '每个时点的权重 w = softmax(log(先验权重) + 历史更新偏移)。赛区偏移受界限约束，随时间回归先验；初始化、赛后更新与衰减后，所有权重均严格为正且总和为 1。', '',
             '全局队伍实力 = 赛区 log 权重（相对其余赛区几何均值）× 400/ln(10) + 队伍内部 Elo。联赛结果更新队伍；已知不同赛区的国际赛更新赛区权重。权重不等于该赛区队伍对任意对手的胜率。', '',
             '近期表现使用赛前预期残差、赛程对手强度、对手强度加胜负得分和预期胜率，分别计算近 5/15/40 局的时间衰减均值。对手评分在当时冻结，不使用后来实力。', '',
             '旧选手 Elo 的绝对均值改为本队全局实力锚点，只保留 25% 的队内相对差异，避免将同队胜场重复解释为独立个人实力。', '',
             'Dense/Transformer 从头重新训练，保持联赛预训练和 2024/2025 世界赛微调；2026 世界赛标签未参与梯度优化。WCS 拥有独立的赛事 embedding。', '',
             f'新候选的内部比较：2026 年 4 月拟合校准、5 月验证，更偏好 **{report["selection"]["selected"]}**。CLI 明确指定 regional_transformer 时始终返回神经网络结果，并同时显示 Elo 对照及验证偏好。最终校准用两个月数据拟合。', '',
             f'新神经网络同一 412 场回顾性评估：Accuracy **{result["accuracy"]:.2%}**，Log Loss **{result["log_loss"]:.6f}**，Brier **{result["brier"]:.6f}**。', '',
             f'新赛区 Elo 对照：Accuracy **{baseline["accuracy"]:.2%}**，Log Loss **{baseline["log_loss"]:.6f}**。', '',
             '旧默认 calibrated_transformer 为 69.17%、0.593892；旧默认保留，本次修正后的 regional_transformer 可显式运行。先验定义已根据用户澄清更改，不能复用旧的未归一化模型权重。', '',
             '**这些比赛以及 IG–LYON 已在设计前查看过，结果只能作为回顾性检查，不能证明未见比赛上的提升。**', '',
             '## 当前赛区权重', '', '| 赛区 | 先验 | 当前 | 国际单局更新数 |', '|---|---:|---:|---:|']
    for region, value in metadata['regional']['coefficients'].items():
        if value['international_games'] or region in {'LCK', 'LPL', 'LEC', 'LCS', 'LCP', 'CBLOL'}:
            lines.append(f'| {region} | {value["prior"]:.6f} | {value["current"]:.6f} | {value["international_games"]} |')
    coefficients = metadata['regional']['coefficients']
    lines += [f'| 合计 | {sum(v["prior"] for v in coefficients.values()):.6f} | {sum(v["current"] for v in coefficients.values()):.6f} | — |']
    lines += ['', '## 使用', '', '命令行加 --model regional_transformer，或在 config.toml 中设置同名模型，显式运行新神经网络。修改 [regional] 后运行：', '',
              '```powershell', "& 'C:\\Users\\ASUS\\.conda\\envs\\cxz\\python.exe' main.py train --model regional_transformer "
              f'--max-pretrain-epochs {report["protocol"]["max_pretrain_epochs"]} --max-finetune-epochs {report["protocol"]["max_finetune_epochs"]}',
              "& 'C:\\Users\\ASUS\\.conda\\envs\\cxz\\python.exe' main.py test 'Invictus Gaming' LYON --model regional_transformer --date 2026-10-28 --patch 26.20 --best-of 3", '```', '',
              '比赛历史仍只截至 2026-10-26；未知赛区保持中性基线并跳过赛区级更新。全局 BP、逐局选边和抽签尚未建模。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
