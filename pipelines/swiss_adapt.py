"""Adapt the 2026-only model on Swiss R1/R2; evaluate only R3/R4."""
import hashlib
import json
import shutil
import csv
import numpy as np
import torch
from torch.nn import functional as F
from core import ROOT, metrics, sigmoid, logit
from models.roster_models import RosterTransformer
from pipelines import roster_train as shared
from pipelines.swiss_replay import REPORT as REPLAY
from pipelines.roster_2026 import OUT as BASE

OUT = ROOT/'artifacts_roster_2026_swiss'
MODEL = 'roster_transformer_2026_swiss'
MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = .03
TRAINABLE_PREFIXES = ('player_embedding.', 'head.', 'matchup.')


def partitions(rows):
    result = {key: np.array([i for i, r in enumerate(rows) if r['round'] in rounds], dtype=int)
              for key, rounds in [('selection_train', {1}), ('selection_validation', {2}),
                                  ('adaptation', {1, 2}), ('evaluation', {3, 4})]}
    if any(not len(value) for value in result.values()):
        raise ValueError('All four Swiss round partitions are required')
    if any(rows[i]['best_of'] != 1 for i in result['adaptation']):
        raise ValueError('This experiment expects R1/R2 BO1 supervision')
    if set(result['adaptation']) & set(result['evaluation']):
        raise ValueError('Training/evaluation overlap')
    if max(rows[i]['event_date'] for i in result['adaptation']) >= min(rows[i]['event_date'] for i in result['evaluation']):
        raise ValueError('Adaptation must end before evaluation begins')
    return result


def adaptation_labels(rows, ids):
    # Test labels remain NaN inside training tensors, making an accidental test
    # batch fail rather than silently use R3/R4 results.
    labels = np.full(len(rows), np.nan, dtype='float32')
    for i in ids:
        labels[i] = float(rows[i]['score_a'] > rows[i]['score_b'])
    return labels


def freeze_for_adaptation(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAINABLE_PREFIXES))
    return [p for p in model.parameters() if p.requires_grad]


def step(model, tensors, ids, optimizer):
    # Frozen encoders and Dense retain eval-mode dropout behavior. The matchup
    # prediction head alone uses its existing dropout regularization.
    model.eval()
    model.matchup.train()
    target = tensors['y'][ids]
    if not torch.isfinite(target).all():
        raise ValueError('An evaluation label entered adaptation')
    optimizer.zero_grad(set_to_none=True)
    value = F.binary_cross_entropy_with_logits(model(*shared.inputs(tensors, ids)), target)
    if not torch.isfinite(value):
        raise RuntimeError('Non-finite Swiss adaptation loss')
    value.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    optimizer.step()
    return float(value.detach())


def series_probability(game_logits, prior, bo, weights):
    from main import raw_series
    from pipelines.optimize import apply_calibration
    return apply_calibration(raw_series(game_logits, bo), raw_series(prior, bo), weights)


def load_parent(seed, raw, samples, device):
    checkpoint = torch.load(BASE/f'seed{seed}'/'pretrained.pt', map_location='cpu', weights_only=True)
    model = RosterTransformer(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    norm = shared.PlayerNormalizer.from_state(checkpoint['normalizer'])
    return checkpoint, model, norm.transform(raw, samples, device)


def run(config):
    from main import seed_everything, raw_series
    from pipelines.optimize import apply_calibration
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUT.mkdir(exist_ok=True)
    parent = json.loads((BASE/'metrics.json').read_text(encoding='utf-8'))
    if sorted(config['seeds']) != sorted(parent['protocol']['seeds']):
        raise ValueError('Use the three saved parent-model seeds')
    rows = json.loads((REPLAY/'predictions.json').read_text(encoding='utf-8'))
    samples = json.loads((REPLAY/'samples.json').read_text(encoding='utf-8'))
    with np.load(REPLAY/'inputs.npz', allow_pickle=False) as stream:
        raw = {k: stream[k] for k in shared.ARRAY_FIELDS}
    splits = partitions(rows)
    raw['y'] = adaptation_labels(rows, splits['adaptation'])
    protocol = {'model': MODEL, 'feature_signature': parent['protocol']['feature_signature'],
                'checkpoint_stage': 'finetuned', 'seeds': config['seeds'],
                'training_scope': '2026 domestic parent plus Swiss R1/R2 only',
                'parent_model': 'roster_transformer_2026',
                'parent_sha256': {str(seed): hashlib.sha256((BASE/f'seed{seed}'/'pretrained.pt').read_bytes()).hexdigest() for seed in config['seeds']},
                'adaptation_n': len(splits['adaptation']), 'adaptation_rounds': [1, 2],
                'selection_train_round': 1, 'selection_validation_round': 2,
                'selection_seed': config['seeds'][0], 'selection_max_epochs': MAX_EPOCHS,
                'selection_rule': 'Minimum R2 calibrated log loss among positive epochs 1..20, first seed only; epoch 0 retained as control',
                'evaluation_n': len(splits['evaluation']), 'evaluation_rounds': [3, 4],
                'test_labels_available_to_training': False, 'test_labels_in_training_tensors': 'NaN',
                'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
                'trainable_parameter_prefixes': list(TRAINABLE_PREFIXES),
                'frozen': 'Temporal/lineup Transformers, Dense, league/role embeddings, organization auxiliary, normalizer and identity vocabulary',
                'calibration_refitted': False, 'calibration_weights': parent['calibration_weights'],
                'dynamic_history': 'Identical frozen replay inputs as baseline, no extra replay of wins into ratings',
                'adaptation_indices': splits['adaptation'].tolist(), 'evaluation_indices': splits['evaluation'].tolist(),
                'input_sha256': hashlib.sha256((REPLAY/'inputs.npz').read_bytes()).hexdigest(),
                'training_source_sha256': hashlib.sha256((ROOT/'pipelines'/'swiss_adapt.py').read_bytes()).hexdigest(),
                'status': 'R3/R4 results previously inspected; retrospective experiment, not independent validation'}
    # Save the protocol before model selection or R3/R4 inference.
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    weights = parent['calibration_weights']
    first_seed = config['seeds'][0]
    seed_everything(first_seed)
    checkpoint, model, tensors = load_parent(first_seed, raw, samples, device)
    optimizer = torch.optim.AdamW(freeze_for_adaptation(model), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_ids, val_ids = splits['selection_train'], splits['selection_validation']
    def validation():
        z = shared.infer(model, tensors, val_ids)
        p = series_probability(z, raw['offset'][val_ids], np.ones(len(val_ids), dtype=int), weights)
        return metrics(raw['y'][val_ids], p)['log_loss']
    control, trace, best_loss, epochs = validation(), [], float('inf'), None
    for epoch in range(1, MAX_EPOCHS+1):
        loss = step(model, tensors, train_ids, optimizer)
        score = validation()
        trace.append({'epoch': epoch, 'R1_training_bce': loss, 'R2_validation_log_loss': score})
        if score < best_loss:
            best_loss, epochs = score, epoch
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({'selection_epoch': epoch, 'R1_training_bce': loss, 'R2_validation_log_loss': score}), flush=True)
    selection = {'selected_positive_epochs': epochs, 'R2_zero_update_control_log_loss': control,
                 'R2_selected_log_loss': best_loss, 'selection_seed': first_seed,
                 'positive_update_required_for_experiment': True,
                 'R3_R4_used_for_selection': False, 'trace': trace}
    (OUT/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    del model, tensors
    before, after, traces = [], [], {}
    bo = np.array([r['best_of'] for r in rows])
    for seed in config['seeds']:
        seed_everything(seed)
        checkpoint, model, tensors = load_parent(seed, raw, samples, device)
        if checkpoint['feature_signature'] != protocol['feature_signature']:
            raise ValueError('Parent checkpoint/calibrator provenance mismatch')
        z_before = shared.infer(model, tensors, np.arange(len(rows)))
        optimizer = torch.optim.AdamW(freeze_for_adaptation(model), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        traces[str(seed)] = [step(model, tensors, splits['adaptation'], optimizer) for _ in range(epochs)]
        folder = OUT/f'seed{seed}'
        folder.mkdir(exist_ok=True)
        torch.save({**checkpoint, 'state_dict': model.state_dict(), 'stage': 'finetuned',
                    'adaptation_protocol': protocol, 'parent_sha256': protocol['parent_sha256'][str(seed)]}, folder/'finetuned.pt')
        z_after = shared.infer(model, tensors, np.arange(len(rows)))
        before.append(sigmoid(raw_series(z_before, bo)))
        after.append(sigmoid(raw_series(z_after, bo)))
        np.savez_compressed(folder/'predictions.npz', before_game_logits=z_before, after_game_logits=z_after)
        print(json.dumps({'seed': seed, 'epochs': epochs,
                          'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                          'first_training_bce': traces[str(seed)][0], 'last_training_bce': traces[str(seed)][-1]}), flush=True)
        del model, tensors
    prior = raw_series(raw['offset'], bo)
    p_before = apply_calibration(logit(np.mean(before, axis=0)), prior, weights)
    p_after = apply_calibration(logit(np.mean(after, axis=0)), prior, weights)
    np.testing.assert_allclose(p_before, [r['p_a_win'] for r in rows], atol=1e-7, rtol=0)
    # R3/R4 outcomes are first used here, after choices and final weights exist.
    ids = splits['evaluation']
    y = np.array([r['score_a'] > r['score_b'] for r in rows])
    results = {}
    for group, keep in [('R3_R4', ids), ('R3', np.array([i for i in ids if rows[i]['round'] == 3])),
                         ('R4', np.array([i for i in ids if rows[i]['round'] == 4]))]:
        results[group] = {name: {**metrics(y[keep], p[keep]), 'correct': int(((p[keep] >= .5) == y[keep]).sum())}
                          for name, p in [('before', p_before), ('after', p_after)]}
    report = {'protocol': protocol, 'selection': {k: v for k, v in selection.items() if k != 'trace'},
              'calibration_weights': weights, 'results': results}
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT/'training_history.json').write_text(json.dumps(traces, indent=2), encoding='utf-8')
    shutil.copyfile(BASE/'snapshot.json', OUT/'snapshot.json')
    output = []
    for i in ids:
        r = rows[i]
        winner = r['actual_winner']
        pa, pb = float(p_before[i]), float(p_after[i])
        a = r['a'] if pa >= .5 else r['b']
        b = r['a'] if pb >= .5 else r['b']
        output.append({**r, 'before_p_a': pa, 'after_p_a': pb, 'after_p_b': 1-pb,
                       'before_pick': a, 'after_pick': b, 'before_correct': a == winner, 'after_correct': b == winner})
    (OUT/'predictions.json').write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    fields = ['round', 'event_date', 'a', 'b', 'best_of', 'before_p_a', 'after_p_a', 'after_p_b',
              'before_pick', 'after_pick', 'score_a', 'score_b', 'before_correct', 'after_correct']
    with (OUT/'predictions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(output)
    write_report(report, output)
    print('ADAPT_RESULTS', json.dumps({group: {name: {k: v for k, v in score.items() if k != 'calibration_bins'} for name, score in value.items()} for group, value in results.items()}), flush=True)


def write_report(report, rows):
    choice = report['selection']
    lines = ['# R1/R2 成绩加入神经网络微调', '',
             '从已保存的 2026 年限定三种子模型开始，使用瑞士轮 R1/R2 的 16 场 BO1 更新选手 embedding 与预测头。冻结时间/阵容 Transformer、Dense、组织辅助、标准化、选手词典与原校准器。动态评分输入沿用原回放，未将同一赛果重复更新进评分。', '',
             f'先用 R1 的 8 场训练、R2 的 8 场验证；第一种子在 1—20 个正轮次中选出 {choice["selected_positive_epochs"]} 轮，再各自从原 checkpoint 重启，用 R1+R2 的 16 场微调三个种子。学习率 1e-4，AdamW weight_decay=.03；没有使用 R3/R4 选择轮次或校准。', '',
             f'R2 选轮次时，不更新的对照 Log Loss 为 {choice["R2_zero_update_control_log_loss"]:.6f}，所选正轮次为 {choice["R2_selected_log_loss"]:.6f}。不更新对照仅用于说明；本实验按请求执行正轮次微调。', '',
             'R1/R2 已参与训练，不能再作为测试。下表仅统计 R3/R4 的 10 场，日期、阵容、版本和历史输入与原回放相同。', '',
             '| 评估 | 模型 | 命中 | Accuracy | Log Loss ↓ | Brier ↓ |', '|---|---|---:|---:|---:|---:|']
    for group, value in report['results'].items():
        for name, score in value.items():
            lines.append(f'| {group} | {name} | {score["correct"]}/{score["n"]} | {score["accuracy"]:.2%} | {score["log_loss"]:.6f} | {score["brier"]:.6f} |')
    lines += ['', '| 轮次 | 对阵 | 原前队胜率 | 微调后前队胜率 | 原预测 | 新预测 | 实际比分 | 原命中 | 新命中 |',
              '|---|---|---:|---:|---|---|---|---|---|']
    for r in rows:
        lines.append(f'| R{r["round"]} | {r["a"]}–{r["b"]} | {r["before_p_a"]:.2%} | {r["after_p_a"]:.2%} | {r["before_pick"]} | {r["after_pick"]} | {r["score_a"]}:{r["score_b"]} | {"✓" if r["before_correct"] else "✗"} | {"✓" if r["after_correct"] else "✗"} |')
    lines += ['', 'R3/R4 赛果在前轮分析中已经查看，本次属于用户指定的回顾性适应实验。只有 16 场适应数据、8 场轮次选择验证及 10 场评估，不能确认稳定泛化收益。BO1 沿用原系列赛校准斜率；原模型和默认配置保留。', '',
              '```powershell',
              "& 'C:\\Users\\ASUS\\.conda\\envs\\cxz\\python.exe' main.py train --model roster_transformer_2026_swiss",
              "& 'C:\\Users\\ASUS\\.conda\\envs\\cxz\\python.exe' main.py test 'Invictus Gaming' LYON --model roster_transformer_2026_swiss --date 2026-10-28 --patch 16.20 --best-of 3", '```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def test(args, config, parser):
    return shared.test(args, config, parser, artifact_dir=OUT, model_name=MODEL)


if __name__ == '__main__':
    from main import load_config
    run(load_config(ROOT/'config.toml'))
