"""Strict 2026-only player model: matched historical benchmark and Worlds forecast."""
import csv
import hashlib
import json
import shutil
import numpy as np
import torch
from core import ROOT, sigmoid, logit, metrics
from data_gen.roster_data import ARRAY_FIELDS, build
from models.roster_models import RosterTransformer
from pipelines import roster_train as shared

OUT = ROOT/'artifacts_roster_2026'
MODEL = 'roster_transformer_2026'
PROFILES = {
    'benchmark': {'selection_train_end': '2026-03-01', 'train_end': '2026-04-01',
                  'calibration_early_end': '2026-05-01', 'test_start': '2026-05-24'},
    'worlds': {'selection_train_end': '2026-07-01', 'train_end': '2026-09-01',
               'calibration_early_end': '2026-10-01', 'test_start': '2026-10-15'},
}


def split_indices(samples, profile, major, benchmark_ids):
    p = PROFILES[profile]
    dates = np.array([s['date'] for s in samples])
    ends = np.array([s['end_date'] for s in samples])
    if not all('2026-01-01' <= s['date'] <= s['end_date'] < '2027-01-01' for s in samples):
        raise ValueError('Every source series must lie wholly within 2026.')
    domestic = np.array([s['domain'] == 'domestic' for s in samples])
    trusted = np.array([s['best_of'] >= 3 for s in samples])
    main = np.array([s['id'] in major for s in samples])
    masks = {
        'selection_train': domestic & (ends < p['selection_train_end']),
        'selection_validation': domestic & (dates >= p['selection_train_end']) & (ends < p['train_end']),
        'train': domestic & (ends < p['train_end']),
        'calibration_early': main & trusted & (dates >= p['train_end']) & (ends < p['calibration_early_end']),
        'calibration_late': main & trusted & (dates >= p['calibration_early_end']) & (ends < p['test_start']),
        'evaluation': np.array([s['id'] in benchmark_ids for s in samples]) if profile == 'benchmark'
                      else np.array([s['worlds_year'] == '2026' for s in samples]) & trusted,
    }
    result = {k: np.flatnonzero(v) for k, v in masks.items()}
    for name, ids in result.items():
        if not len(ids):
            raise ValueError(f'Empty {profile} partition: {name}')
    if any(dates[i] < p['test_start'] for i in result['evaluation']):
        raise ValueError('Evaluation series precedes the frozen model cutoff.')
    return result


def signature(config):
    value = shared.signature(config)
    value['history_scope'] = {'since_inclusive': '2026-01-01', 'before_exclusive': '2027-01-01',
                              'cold_start': True, 'older_checkpoint_used': False}
    for path in ['pipelines/roster_2026.py', 'pipelines/roster_train.py']:
        value['sources'][path] = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    return value


def prepare(config):
    from data_gen.deep_data import write_all_teams
    OUT.mkdir(exist_ok=True)
    write_all_teams()
    fingerprint = signature(config)
    if (OUT/'preparation.json').exists() and json.loads((OUT/'preparation.json').read_text(encoding='utf-8')) == fingerprint:
        with np.load(OUT/'data.npz', allow_pickle=False) as stream:
            raw = {k: stream[k] for k in stream.files}
        return raw, json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    samples, metadata = build(config.get('regional'), config.get('players'), since='2026-01-01',
                               before='2027-01-01', save_snapshot=OUT/'snapshot.json')
    raw = {k: np.stack([s[k] for s in samples]) for k in ARRAY_FIELDS}
    for field, source, dtype in [('y', 'y', 'float32'), ('map_target', 'map_fraction', 'float32'),
                                 ('game_count', 'game_count', 'float32'), ('best_of', 'best_of', 'int64')]:
        raw[field] = np.array([s[source] for s in samples], dtype=dtype)
    rows = [{k: v for k, v in s.items() if k not in ARRAY_FIELDS} for s in samples]
    np.savez_compressed(OUT/'data.npz', **raw)
    for name, value in [('samples', rows), ('feature_metadata', metadata), ('preparation', fingerprint)]:
        (OUT/f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'prepared_2026': len(rows), 'history_window': metadata['history_window'],
                      'players': metadata['players'], 'updated_games': metadata['updated_games']}), flush=True)
    return raw, rows


def train_profile(config, raw, samples, profile):
    from main import seed_everything, raw_series
    from pipelines.optimize import calibrate, apply_calibration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = OUT/'benchmark' if profile == 'benchmark' else OUT
    folder.mkdir(exist_ok=True)
    if profile == 'benchmark':
        shutil.copyfile(OUT/'snapshot.json', folder/'snapshot.json')
    old_predictions = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    major = {s['id'] for s in json.loads((ROOT/'artifacts_optimized'/'samples.json').read_text(encoding='utf-8'))}
    partitions = split_indices(samples, profile, major, set(old_predictions))
    protocol = {'model': MODEL, 'profile': profile, 'feature_signature': signature(config),
                'seeds': config['seeds'], 'checkpoint_stage': 'pretrained',
                'training_scope': '2026 only; random initialization and history cold-start on January 1',
                'domestic_pretraining_n': len(partitions['train']), 'worlds_finetuning_n': 0,
                'worlds_finetune_years': [], 'worlds_2026_optimizer_labels_used': 0,
                'older_checkpoint_used': False, 'older_match_history_used': False,
                'time_boundaries': PROFILES[profile], 'partition_sizes': {k: len(v) for k, v in partitions.items()},
                'selection_max_epochs': config['max_pretrain_epochs'], 'parameters_architecture': 'Unchanged shared player/lineup Transformer and Dense; player vocabulary fitted within this training window',
                'device': str(device), 'torch': str(torch.__version__),
                'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                'status': 'Previously inspected retrospective evaluation, not independent validation'}
    (folder/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print('PROTOCOL', json.dumps(protocol), flush=True)
    traces, choices, all_p, core_p = {}, None, [], []
    for seed in config['seeds']:
        seed_everything(seed)
        if choices is None:
            norm = shared.PlayerNormalizer().fit(raw, partitions['selection_train'], samples)
            tensors = norm.transform(raw, samples, device)
            dims = {'players': len(norm.players)+1, 'leagues': len(norm.leagues)+1,
                    'history_length': raw['history_a'].shape[2],
                    'team_aux_limit': {**shared.DEFAULTS, **config.get('players', {})}['team_aux_limit']}
            model = RosterTransformer(**dims).to(device)
            selected = shared.fit(model, tensors, partitions['selection_train'], partitions['selection_validation'],
                                  profile+'_2026_selection', config['max_pretrain_epochs'])
            choices = {'pretrain_epochs': selected['epochs'], 'finetune_epochs': 0, 'selection_seed': seed}
            traces['selection'] = selected
            (folder/'choices.json').write_text(json.dumps(choices, indent=2), encoding='utf-8')
            del model, tensors
        seed_everything(seed)
        norm = shared.PlayerNormalizer().fit(raw, partitions['train'], samples)
        tensors = norm.transform(raw, samples, device)
        dims = {'players': len(norm.players)+1, 'leagues': len(norm.leagues)+1,
                'history_length': raw['history_a'].shape[2],
                'team_aux_limit': {**shared.DEFAULTS, **config.get('players', {})}['team_aux_limit']}
        model = RosterTransformer(**dims).to(device)
        traces[str(seed)] = shared.fit(model, tensors, partitions['train'], None, profile+'_2026_final',
                                       choices['pretrain_epochs'], fixed=True)
        seed_dir = folder/f'seed{seed}'
        seed_dir.mkdir(exist_ok=True)
        torch.save({'config': dims, 'state_dict': model.state_dict(), 'normalizer': norm.state(),
                    'feature_signature': protocol['feature_signature'], 'stage': 'pretrained',
                    'training_scope': protocol['training_scope'], 'worlds_finetuning_n': 0}, seed_dir/'pretrained.pt')
        total, core, aux = shared.infer(model, tensors, np.arange(len(samples)), return_components=True)
        p, cp = sigmoid(raw_series(total, raw['best_of'])), sigmoid(raw_series(core, raw['best_of']))
        all_p.append(p)
        core_p.append(cp)
        np.savez_compressed(seed_dir/'predictions.npz', game_logits=total, core_game_logits=core,
                            auxiliary_game_logits=aux, series_probability=p)
        print(json.dumps({'profile': profile, 'seed_complete': seed, 'parameters': sum(p.numel() for p in model.parameters())}), flush=True)
        del model, tensors
    deep, core = logit(np.mean(all_p, axis=0)), logit(np.mean(core_p, axis=0))
    prior = raw_series(raw['offset'], raw['best_of'])
    early, late = partitions['calibration_early'], partitions['calibration_late']
    early_weights = calibrate(deep[early], prior[early], raw['y'][early])
    selection = {'choices': choices, 'validation_end_exclusive': PROFILES[profile]['test_start'],
                 'forward_validation': metrics(raw['y'][late], apply_calibration(deep[late], prior[late], early_weights)),
                 'evaluation_used_for_epoch_or_calibrator_selection': False}
    # Calibration choices are fixed before the evaluation labels are accessed.
    (folder/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    cal = np.r_[early, late]
    weights = calibrate(deep[cal], prior[cal], raw['y'][cal])
    probability = apply_calibration(deep, prior, weights)
    no_team = apply_calibration(core, prior, weights)
    evaluate = partitions['evaluation']
    if profile == 'benchmark':
        assert len(evaluate) == 412 and {samples[i]['id'] for i in evaluate} == set(old_predictions)
        assert all(raw['y'][i] == int(old_predictions[samples[i]['id']]['y']) for i in evaluate)
    old = {r['id']: r for r in csv.DictReader((ROOT/'artifacts_roster'/'predictions.csv').open(encoding='utf-8'))}
    comparison_ids = [i for i in evaluate if samples[i]['id'] in old]
    if len(comparison_ids) != len(evaluate):
        raise ValueError('Missing matched multi-year comparison probabilities')
    original = np.array([float(old[samples[i]['id']]['probability']) for i in evaluate])
    assert all(raw['y'][i] == int(old[samples[i]['id']]['y']) for i in evaluate)
    results = {MODEL: metrics(raw['y'][evaluate], probability[evaluate]),
               'player_only_ablation': metrics(raw['y'][evaluate], no_team[evaluate]),
               'multiyear_roster_same_matches': metrics(raw['y'][evaluate], original)}
    report = {'protocol': protocol, 'selection': selection, 'calibration_weights': weights.tolist(), 'results': results,
              'ablation': 'Identical trained weights/calibrator, organization auxiliary output set to zero'}
    for name, value in [('metrics', report), ('training_history', traces)]:
        (folder/f'{name}.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    np.savez_compressed(folder/'ensemble_predictions.npz', deep=deep, core=core, prior=prior,
                        probability=probability, evaluation_ids=evaluate)
    rows = [{**{k: samples[i][k] for k in ['id', 'date', 'league', 'team_a', 'team_b', 'best_of', 'y']},
             'probability': float(probability[i]), 'player_only': float(no_team[i]),
             'multiyear_probability': float(original[j])} for j, i in enumerate(evaluate)]
    with (folder/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('RESULT', json.dumps({'profile': profile, 'results': {k: {n: v for n, v in score.items() if n != 'calibration_bins'} for k, score in results.items()}}), flush=True)
    return report


def forecast(*args, profile='worlds', **kwargs):
    folder = OUT/'benchmark' if profile == 'benchmark' else OUT
    return shared.forecast(*args, **kwargs, artifact_dir=folder, model_name=MODEL)


def test(args, config, parser):
    actual = dict(config)
    if args.stage is None:
        actual['stage'] = 'pretrained'
    return shared.test(args, actual, parser, artifact_dir=OUT, model_name=MODEL)


def write_report():
    reports = {p: json.loads(((OUT/'benchmark' if p == 'benchmark' else OUT)/'metrics.json').read_text(encoding='utf-8')) for p in PROFILES}
    metadata = json.loads((OUT/'feature_metadata.json').read_text(encoding='utf-8'))
    lines = ['# 2026 年数据限定实验', '',
             '沿用个人历史 Transformer、五位置阵容 Transformer 与 Dense 架构。网络从随机权重开始，个人能力、阵容磨合、组织历史与赛区动态偏移从 2026-01-01 冷启动；2024/2025 比赛与旧 checkpoint 均未使用。人工指定六赛区先验保留。', '',
             '没有可用的往年世界赛微调样本，本实验关闭世界赛微调；2026 世界赛标签不参与梯度训练或校准。个人近期状态仍按已完成比赛滚动更新。worlds 评估范围包含入围赛与瑞士轮；仅瑞士轮结果请见 swiss_replay/report.md。', '',
             f'有效历史单局 {metadata["updated_games"]:,} 场，选手 handle {metadata["players"]:,} 个。源文件保留多年份，但在构建目标与任何历史状态之前筛选 2026 年系列赛；选手行只保留这些单局。', '',
             '| 实验 | 联赛训练时间 | 样本数 | 校准时间 | 评估 |', '|---|---|---:|---|---|']
    for profile, r in reports.items():
        p = r['protocol']
        boundary = p['time_boundaries']
        lines.append(f'| {profile} | 2026-01-01 至 {boundary["train_end"]}（不含） | {p["domestic_pretraining_n"]} | {boundary["train_end"]} 至 {boundary["test_start"]}（不含） | {r["results"][MODEL]["n"]} 场 |')
    lines += ['', '同一 412 场采用相同训练/校准截止日，衡量移除旧年份及历史世界赛微调后的整体变化；不是单独识别“年份”的因果实验。worlds 版本允许更多 2026 年训练样本，仅在训练与校准之后的世界赛上比较，不把早期已用于训练的比赛当作测试。', '',
              '| 评估 | 模型 | Accuracy | Log Loss ↓ | Brier ↓ |', '|---|---|---:|---:|---:|']
    for profile, report in reports.items():
        for name in [MODEL, 'multiyear_roster_same_matches']:
            score = report['results'][name]
            lines.append(f'| {profile} | {name} | {score["accuracy"]:.2%} | {score["log_loss"]:.6f} | {score["brier"]:.6f} |')
    predictions_path = OUT/'predictions_2026-10-28.json'
    if predictions_path.exists():
        forecasts = json.loads(predictions_path.read_text(encoding='utf-8'))
        lines += ['', '## 10 月 28 日两场 BO3 回放', '', '| 对阵 | 前者胜率 | 后者胜率 |', '|---|---:|---:|']
        for row in forecasts:
            lines.append(f'| {row["team_a"]} vs {row["team_b"]} | {row["p_a_win"]:.2%} | {row["p_b_win"]:.2%} |')
    lines += ['', '这些赛果与原评估集合已查看，全部为回顾性实验。世界赛 BO3/BO5 样本很少，不能据此确定泛化收益。预测使用截至 10 月 26 日的历史和最后已知阵容，没有核对当天官方首发。', '',
              '## 复现', '', '```powershell',
              "& 'C:\\Users\\ASUS\\.conda\\envs\\cxz\\python.exe' main.py train --model roster_transformer_2026 --max-pretrain-epochs 24",
              "& 'C:\\Users\\ASUS\\.conda\\envs\\cxz\\python.exe' main.py test 'Invictus Gaming' LYON --model roster_transformer_2026 --date 2026-10-28 --patch 16.20 --best-of 3",
              "& 'C:\\Users\\ASUS\\.conda\\envs\\cxz\\python.exe' -m tests.validate_roster_2026", '```', '',
              '默认配置与多年份权重保持原样；显式模型名 roster_transformer_2026 使用 worlds 版本。世界赛微调关闭，保存阶段为 pretrained；无需传 --stage，若指定则使用 --stage pretrained。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run(config):
    torch.set_num_threads(4)
    raw, samples = prepare(config)
    for profile in PROFILES:
        train_profile(config, raw, samples, profile)
    tournament_rows = [s for s in samples if s['worlds_year'] == '2026']
    observed_patch = max(tournament_rows, key=lambda s: s['date'])['patch']
    forecasts = [forecast(a, b, '2026-10-28', observed_patch, 3) for a, b in
                 [('Movistar KOI', 'T1'), ('Invictus Gaming', 'LYON')]]
    (OUT/'predictions_2026-10-28.json').write_text(json.dumps(forecasts, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report()
    print('FORECASTS', json.dumps([{k: row[k] for k in ['team_a', 'team_b', 'p_a_win', 'p_b_win']} for row in forecasts]), flush=True)
