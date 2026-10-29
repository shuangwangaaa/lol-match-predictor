"""Stronger Swiss adaptation, weighted against a domestic distillation constraint."""
import csv
import hashlib
import json
import math
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from core import ROOT, metrics, logit, sigmoid
from pipelines import roster_train as shared
from pipelines.swiss_adapt import BASE, REPLAY, partitions, adaptation_labels, load_parent, series_probability

OUT = ROOT/'artifacts_roster_2026_swiss_weighted'
WEAK = ROOT/'artifacts_roster_2026_swiss'
MODEL = 'roster_transformer_2026_swiss_weighted'
DEFAULTS = {'swiss_weight': 10., 'replay_weight': 1., 'learning_rate': 3e-4,
            'weight_decay': .03, 'max_epochs': 60, 'replay_pool': 256, 'replay_batch': 32}
PREFIXES = ('player_embedding.', 'dense.', 'head.', 'matchup.')


def settings(config):
    value = {**DEFAULTS, **config.get('swiss_weighted', {})}
    if set(value) != set(DEFAULTS):
        raise ValueError('Unknown swiss_weighted setting')
    for name in ['swiss_weight', 'replay_weight', 'learning_rate']:
        if not math.isfinite(value[name]) or value[name] <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not math.isfinite(value['weight_decay']) or value['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and non-negative')
    for name in ['max_epochs', 'replay_pool', 'replay_batch']:
        if not isinstance(value[name], int) or value[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if value['replay_batch'] > value['replay_pool']:
        raise ValueError('replay_batch cannot exceed replay_pool')
    return value


def freeze(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIXES))
    return [p for p in model.parameters() if p.requires_grad]


def objective(swiss_logits, y, replay_logits, teacher_probability, swiss_weight=10., replay_weight=1.):
    if swiss_weight <= 0 or replay_weight <= 0 or not math.isfinite(swiss_weight+replay_weight):
        raise ValueError('Loss weights must be finite and positive')
    if not torch.isfinite(y).all():
        raise ValueError('An evaluation label entered adaptation')
    swiss_bce = F.binary_cross_entropy_with_logits(swiss_logits, y)
    # Bernoulli KL(teacher || student); entropy is constant w.r.t. student.
    teacher = teacher_probability.detach().clamp(1e-6, 1-1e-6)
    entropy = -(teacher*teacher.log()+(1-teacher)*torch.log1p(-teacher)).mean()
    replay_kl = F.binary_cross_entropy_with_logits(replay_logits, teacher)-entropy
    loss = (swiss_weight*swiss_bce+replay_weight*replay_kl)/(swiss_weight+replay_weight)
    return loss, swiss_bce, replay_kl


def update(model, tensors, ids, replay_tensors, teacher, replay_ids, optimizer, cfg):
    model.eval()
    model.dense.train()
    model.matchup.train()
    target = tensors['y'][ids]
    if not torch.isfinite(target).all():
        raise ValueError('An evaluation label entered adaptation')
    optimizer.zero_grad(set_to_none=True)
    a = model(*shared.inputs(tensors, ids))
    b = model(*shared.inputs(replay_tensors, replay_ids))
    value, swiss, replay = objective(a, target, b, teacher[replay_ids], cfg['swiss_weight'], cfg['replay_weight'])
    if not torch.isfinite(value):
        raise RuntimeError('Non-finite weighted adaptation loss')
    value.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    optimizer.step()
    return {'objective': float(value.detach()), 'swiss_bce': float(swiss.detach()), 'replay_kl': float(replay.detach())}


def rehearsal(cfg):
    samples = json.loads((BASE/'samples.json').read_text(encoding='utf-8'))
    eligible = np.array([i for i, r in enumerate(samples) if r['domain'] == 'domestic'
                         and '2026-01-01' <= r['date'] <= r['end_date'] < '2026-09-01'])
    if len(eligible) < cfg['replay_pool']:
        raise ValueError('Not enough original domestic training samples for replay')
    ids = np.sort(np.random.default_rng(20261029).choice(eligible, cfg['replay_pool'], replace=False))
    with np.load(BASE/'data.npz', allow_pickle=False) as stream:
        raw = {k: stream[k][ids] for k in shared.ARRAY_FIELDS}
    return raw, [samples[i] for i in ids], ids


def setup(seed, raw, samples, replay_raw, replay_samples, device, cfg):
    parent, model, tensors = load_parent(seed, raw, samples, device)
    norm = shared.PlayerNormalizer.from_state(parent['normalizer'])
    replay = norm.transform(replay_raw, replay_samples, device)
    model.eval()
    with torch.no_grad():
        pool = np.arange(len(replay_samples))
        teacher = model(*shared.inputs(replay, pool)).sigmoid().detach()
    optimizer = torch.optim.AdamW(freeze(model), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    return parent, model, tensors, replay, teacher, optimizer


def draw_replay(teacher, cfg):
    return torch.randperm(len(teacher), device=teacher.device)[:cfg['replay_batch']]


def run(config):
    from main import seed_everything, raw_series
    from pipelines.optimize import apply_calibration
    cfg = settings(config)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parent_report = json.loads((BASE/'metrics.json').read_text(encoding='utf-8'))
    if sorted(config['seeds']) != sorted(parent_report['protocol']['seeds']):
        raise ValueError('Use the three saved parent seeds')
    rows = json.loads((REPLAY/'predictions.json').read_text(encoding='utf-8'))
    samples = json.loads((REPLAY/'samples.json').read_text(encoding='utf-8'))
    with np.load(REPLAY/'inputs.npz', allow_pickle=False) as stream:
        raw = {k: stream[k] for k in shared.ARRAY_FIELDS}
    splits = partitions(rows)
    raw['y'] = adaptation_labels(rows, splits['adaptation'])
    replay_raw, replay_samples, replay_ids = rehearsal(cfg)
    OUT.mkdir(exist_ok=True)
    weights = parent_report['calibration_weights']
    protocol = {'model': MODEL, 'feature_signature': parent_report['protocol']['feature_signature'],
                'checkpoint_stage': 'finetuned', 'seeds': config['seeds'], 'settings': cfg,
                'parent_sha256': {str(s): hashlib.sha256((BASE/f'seed{s}'/'pretrained.pt').read_bytes()).hexdigest() for s in config['seeds']},
                'training_scope': '2026 domestic parent; R1/R2 supervised adaptation; original Jan-Aug domestic teacher distillation only',
                'objective': '(swiss_weight * mean Swiss BCE + replay_weight * mean domestic teacher KL) / sum weights',
                'weight_unit': 'Weights between group-mean losses, not per-row sampling weights or multiplying the whole loss',
                'teacher': 'Immutable original parent, per-game probability; no new domestic labels used',
                'trainable_prefixes': list(PREFIXES), 'frozen': 'Transformers, role/league encodings, organization auxiliary, normalizer and player vocabulary',
                'selection_seed': config['seeds'][0], 'selection_train_round': 1, 'selection_validation_round': 2,
                'selection_rule': 'Minimum R2 calibrated log loss over positive epochs; weight 10, LR and budget fixed before R3/R4 inference',
                'adaptation_indices': splits['adaptation'].tolist(), 'adaptation_n': 16,
                'evaluation_indices': splits['evaluation'].tolist(), 'evaluation_n': 10,
                'replay_indices': replay_ids.tolist(), 'replay_sampling_seed': 20261029,
                'calibration_refitted': False, 'calibration_weights': weights,
                'evaluation_labels_in_training_tensors': 'NaN', 'R3_R4_used_for_selection': False,
                'input_sha256': hashlib.sha256((REPLAY/'inputs.npz').read_bytes()).hexdigest(),
                'replay_input_sha256': hashlib.sha256((BASE/'data.npz').read_bytes()).hexdigest(),
                'source_sha256': hashlib.sha256((ROOT/'pipelines/swiss_weighted.py').read_bytes()).hexdigest(),
                'status': 'Previously inspected R3/R4 results; retrospective user-requested experiment; multiple settings changed from weak adaptation'}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    seed = config['seeds'][0]
    seed_everything(seed)
    _, model, tensors, replay, teacher, optimizer = setup(seed, raw, samples, replay_raw, replay_samples, device, cfg)
    val = splits['selection_validation']
    def validation():
        z = shared.infer(model, tensors, val)
        p = series_probability(z, raw['offset'][val], np.ones(len(val), dtype=int), weights)
        return metrics(raw['y'][val], p)['log_loss']
    control = validation()
    trace, best_loss, best_epoch = [], float('inf'), None
    for epoch in range(1, cfg['max_epochs']+1):
        losses = update(model, tensors, splits['selection_train'], replay, teacher,
                        draw_replay(teacher, cfg), optimizer, cfg)
        score = validation()
        trace.append({'epoch': epoch, 'R2_validation_log_loss': score, **losses})
        if score < best_loss:
            best_loss, best_epoch = score, epoch
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({'epoch': epoch, 'R2_validation_log_loss': score, **losses}), flush=True)
    choice = {'selected_positive_epochs': best_epoch, 'R2_zero_update_control_log_loss': control,
              'R2_selected_log_loss': best_loss, 'selection_seed': seed,
              'R3_R4_used_for_selection': False, 'trace': trace}
    (OUT/'selection.json').write_text(json.dumps(choice, indent=2), encoding='utf-8')
    del model, tensors, replay, teacher, optimizer
    before, after, traces = [], [], {}
    bo = np.array([r['best_of'] for r in rows])
    for seed in config['seeds']:
        seed_everything(seed)
        parent, model, tensors, replay, teacher, optimizer = setup(seed, raw, samples, replay_raw, replay_samples, device, cfg)
        if parent['feature_signature'] != protocol['feature_signature']:
            raise ValueError('Parent provenance mismatch')
        base_logits = shared.infer(model, tensors, np.arange(len(rows)))
        values = []
        for epoch in range(1, best_epoch+1):
            values.append(update(model, tensors, splits['adaptation'], replay, teacher,
                                 draw_replay(teacher, cfg), optimizer, cfg))
        traces[str(seed)] = values
        folder = OUT/f'seed{seed}'
        folder.mkdir(exist_ok=True)
        torch.save({**parent, 'state_dict': model.state_dict(), 'stage': 'finetuned',
                    'adaptation_protocol': protocol, 'parent_sha256': protocol['parent_sha256'][str(seed)]}, folder/'finetuned.pt')
        adapted_logits = shared.infer(model, tensors, np.arange(len(rows)))
        before.append(sigmoid(raw_series(base_logits, bo)))
        after.append(sigmoid(raw_series(adapted_logits, bo)))
        np.savez_compressed(folder/'predictions.npz', before_game_logits=base_logits, after_game_logits=adapted_logits)
        print(json.dumps({'seed': seed, 'epochs': best_epoch,
                          'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                          'first': values[0], 'last': values[-1]}), flush=True)
        del model, tensors, replay, teacher, optimizer
    prior = raw_series(raw['offset'], bo)
    p_before = apply_calibration(logit(np.mean(before, axis=0)), prior, weights)
    p_after = apply_calibration(logit(np.mean(after, axis=0)), prior, weights)
    np.testing.assert_allclose(p_before, [r['p_a_win'] for r in rows], atol=1e-7, rtol=0)
    # Evaluation labels and old weak-version scores are accessed only after
    # weighted selection, final fitting and saved predictions are complete.
    ids = splits['evaluation']
    y = np.array([r['score_a'] > r['score_b'] for r in rows])
    old_rows = json.loads((WEAK/'predictions.json').read_text(encoding='utf-8'))
    old_map = {(r['round'], r['a'], r['b']): r for r in old_rows}
    weak = np.full(len(rows), np.nan)
    for i in ids:
        weak[i] = old_map[(rows[i]['round'], rows[i]['a'], rows[i]['b'])]['after_p_a']
    results = {}
    for group, take in [('R3_R4', ids), ('R3', np.array([i for i in ids if rows[i]['round'] == 3])),
                        ('R4', np.array([i for i in ids if rows[i]['round'] == 4]))]:
        results[group] = {name: {**metrics(y[take], p[take]), 'correct': int(((p[take] >= .5) == y[take]).sum())}
                          for name, p in [('base', p_before), ('weak', weak), ('weighted', p_after)]}
    report = {'protocol': protocol, 'selection': {k: v for k, v in choice.items() if k != 'trace'},
              'calibration_weights': weights, 'results': results}
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT/'training_history.json').write_text(json.dumps(traces, indent=2), encoding='utf-8')
    shutil.copyfile(BASE/'snapshot.json', OUT/'snapshot.json')
    output = []
    for i in ids:
        r = rows[i]
        before_pick = r['a'] if p_before[i] >= .5 else r['b']
        after_pick = r['a'] if p_after[i] >= .5 else r['b']
        output.append({**r, 'base_p_a': float(p_before[i]), 'weak_p_a': float(weak[i]),
                       'weighted_p_a': float(p_after[i]), 'weighted_p_b': float(1-p_after[i]),
                       'base_pick': before_pick, 'weighted_pick': after_pick,
                       'base_correct': before_pick == r['actual_winner'], 'weighted_correct': after_pick == r['actual_winner']})
    (OUT/'predictions.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['round', 'a', 'b', 'best_of', 'base_p_a', 'weak_p_a', 'weighted_p_a', 'weighted_p_b',
              'base_pick', 'weighted_pick', 'score_a', 'score_b', 'base_correct', 'weighted_correct']
    with (OUT/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(output)
    write_report(report, output)
    print('WEIGHTED_RESULTS', json.dumps({group: {name: {k: v for k, v in score.items() if k != 'calibration_bins'} for name, score in value.items()} for group, value in results.items()}), flush=True)


def write_report(report, rows):
    cfg = report['protocol']['settings']
    selected = report['selection']
    lines = ['# 提高瑞士轮微调数据影响', '',
             f'使用固定权重：世界赛监督 BCE : 2026 年联赛教师 KL 约束 = {cfg["swiss_weight"]:g}:{cfg["replay_weight"]:g}。两项各自取组内均值，再加权；该比值不是每场采样权重，也不是把整个损失乘常数。', '',
             '此前微调批次已经 100% 来自 R1/R2，单纯放大整体损失不能可靠增强 AdamW 更新。因此本实验同时放开 Dense，保留选手 embedding 与预测头更新，Transformer 和组织辅助仍冻结。原 2026 年模型作为固定教师，联赛回放只约束遗忘，不再读取联赛真值标签。', '',
             f'学习率从 1e-4 提至 {cfg["learning_rate"]:g}，每步回放 {cfg["replay_batch"]} 场，固定池 {cfg["replay_pool"]} 场，均来自 2026 年 9 月 1 日前的原联赛训练集。先用 R1 训练、R2 验证，在最多 {cfg["max_epochs"]} 轮中选出 {selected["selected_positive_epochs"]} 轮，再从同一原模型重启，以 R1+R2 共 16 场训练三个种子。', '',
             f'R2 选择验证：原始 Log Loss {selected["R2_zero_update_control_log_loss"]:.6f}，选中轮次 {selected["R2_selected_log_loss"]:.6f}。权重、学习率及最大轮数在读取本实验 R3/R4 结果前已固定。', '',
             '标准化、选手词典、历史特征与校准器沿用原版。没有重复更新动态评分，没有用 R3/R4 梯度训练或挑参数。本实验同时改变权重、学习率、可训练层和轮数预算，不能将变化单独归因于损失权重。', '',
             '| 评估 | 版本 | 命中 | Accuracy | Log Loss ↓ | Brier ↓ |', '|---|---|---:|---:|---:|---:|']
    for group, value in report['results'].items():
        for name, score in value.items():
            lines.append(f'| {group} | {name} | {score["correct"]}/{score["n"]} | {score["accuracy"]:.2%} | {score["log_loss"]:.6f} | {score["brier"]:.6f} |')
    lines += ['', '| 轮次 | 对阵 | 原版前队胜率 | 轻微调前队胜率 | 加权版前队胜率 | 加权版预测 | 实际比分 | 命中 |',
              '|---|---|---:|---:|---:|---|---|---|']
    for r in rows:
        lines.append(f'| R{r["round"]} | {r["a"]}–{r["b"]} | {r["base_p_a"]:.2%} | {r["weak_p_a"]:.2%} | {r["weighted_p_a"]:.2%} | {r["weighted_pick"]} | {r["score_a"]}:{r["score_b"]} | {"✓" if r["weighted_correct"] else "✗"} |')
    lines += ['', 'R1/R2 的 16 场已参与适应，不计入测试。R3/R4 共 10 场，而且赛果已在先前回合查看，这是回顾性试验，不能证明稳定提升。旧权重与默认模型保留。', '',
              '```powershell', 'python main.py train --model roster_transformer_2026_swiss_weighted',
              "python main.py test 'Invictus Gaming' LYON --model roster_transformer_2026_swiss_weighted --date 2026-10-28 --patch 16.20 --best-of 3", '```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def test(args, config, parser):
    return shared.test(args, config, parser, artifact_dir=OUT, model_name=MODEL)
