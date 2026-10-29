"""Write the complete comparison, selection rationale, and uncertainty report."""
import csv
from datetime import datetime
import json
import numpy as np
from core import ROOT, metrics
from pipelines.optimize import OUT


def week_bootstrap(rows, new, old):
    y = np.array([int(r['y']) for r in rows])
    loss = lambda p: -(y*np.log(np.clip(p, 1e-8, 1))+(1-y)*np.log(np.clip(1-p, 1e-8, 1)))
    delta = loss(new)-loss(old)
    accuracy = ((new >= .5) == y).astype(float)-((old >= .5) == y).astype(float)
    weeks = np.array([datetime.fromisoformat(r['date']).strftime('%G-%V') for r in rows])
    blocks = [np.flatnonzero(weeks == week) for week in sorted(set(weeks))]
    rng = np.random.default_rng(20261028)
    draws = []
    for _ in range(2000):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        draws.append([delta[idx].mean(), accuracy[idx].mean()])
    draws = np.array(draws)
    return {'log_loss_delta': float(delta.mean()), 'log_loss_delta_95_ci': np.quantile(draws[:, 0], [.025, .975]).tolist(),
            'accuracy_delta': float(accuracy.mean()), 'accuracy_delta_95_ci': np.quantile(draws[:, 1], [.025, .975]).tolist(),
            'week_blocks': len(blocks), 'status': 'descriptive retrospective week-block bootstrap; not new independent evidence'}


def main():
    report = json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    blend = json.loads((OUT/'blend_metrics.json').read_text(encoding='utf-8'))
    selection = json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    deep = json.loads((ROOT/'artifacts_deep'/'summary.json').read_text(encoding='utf-8'))
    rows = list(csv.DictReader((OUT/'blend_predictions.csv').open(encoding='utf-8')))
    baseline = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    original = np.array([float(baseline[r['id']]['mlp_calibrated']) for r in rows])
    comparisons = {name: week_bootstrap(rows, np.array([float(r[name]) for r in rows]), original)
                   for name in ['expert_blend', 'recalibrated_transformer']}
    (OUT/'uncertainty.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
    lines = ['# 结构化数据模型优化报告', '',
             '本轮沿用整理后的 main.py / config.toml / models / pipelines / tests 结构，使用现有 cxz 环境，无需安装新依赖。', '',
             '## 主要结果', '',
             '默认模型为 `calibrated_transformer`，由测试期之前的前向校准验证选择。`expert_blend` 在回顾性集合上表现更好，保留为研究候选；没有据此覆盖默认模型。', '',
             '| 模型 | 准确率 ↑ | Log Loss ↓ | Brier ↓ | ECE ↓ |', '| --- | ---: | ---: | ---: | ---: |']
    comparisons_table = [
        ('原始 Elo + 小 MLP', report['results']['v1_mlp']),
        ('上一轮 Dense + Transformer（三种子）', deep['results']['dense_transformer_worlds_finetuned_ensemble']['reused_412_series']),
        ('本轮默认：重新校准 Transformer', blend['results']['recalibrated_transformer']),
        ('单调约束梯度提升树', report['results']['hist_gbm']),
        ('候选：树 + Transformer 专家融合', blend['results']['expert_blend']),
    ]
    for label, score in comparisons_table:
        lines.append(f"| {label} | {score['accuracy']:.2%} | {score['log_loss']:.4f} | {score['brier']:.4f} | {score['ece_10_bins']:.4f} |")
    lines += ['', '**比较集合始终是相同的 412 场、相同标签；该集合此前已多次查看，因此不是新的独立测试。** 这次候选融合提高了命中率并降低 Log Loss，但 ECE 没有同步改善，不能称为所有指标均胜出。', '',
              '## 实际训练和比较的模型', '',
              '| 家族 | 历史滚动回测 Log Loss | 回顾性 412 场准确率 | 回顾性 Log Loss |', '| --- | ---: | ---: | ---: |']
    for name in ['ridge_series', 'resnet_tabular', 'ft_transformer', 'tabm_lite', 'hist_gbm']:
        value = report['results'][name]
        lines.append(f"| {name} | {value['development_log_loss']:.4f} | {value['accuracy']:.2%} | {value['log_loss']:.4f} |")
    lines += ['', '- `ridge_series`：带 Elo 先验的正则化逻辑回归，用作强小样本基线。',
              '- `resnet_tabular`：较紧凑的表格残差网络，减少大 Dense 的自由度。',
              '- `ft_transformer`：把数值特征逐列编码为 token，学习表格特征之间的关系。',
              '- `tabm_lite`：自实现的 BatchEnsemble MLP，共享主权重、成员保留缩放和独立输出，训练时分别监督 8 个成员；它是参考 TabM 思路的任务变体，不是官方实现复现。',
              '- `hist_gbm`：直方图梯度提升分类器；比较不同叶数与单调约束，Elo 差异列约束为优势增加不降低胜率。', '',
              '四折回测覆盖 1,250 场 2025 年至 2026 年 3 月的比赛，比较 12 种设置及微调前后状态。树模型是四折概率质量赢家，新的神经网络没有稳定超过它。', '',
              '## 训练和防泄漏安排', '',
              '1. 使用原有主要联赛的 2,851 场完整系列赛特征，并重新生成双方最近 12 局历史。没有通过剔除难预测比赛提高准确率。',
              '2. 每折仅用外层验证之前的联赛样本训练，内部按时间留出选择训练轮数；重新拟合后才预测外层区间。',
              '3. 独立校准段为验证之前 45 天；休赛期不足 60 场时向前扩展到至少 80 场，并按日期整体划分。校准段不进入训练或世界赛微调。',
              '4. 世界赛微调只使用当时已结束的世界赛，冻结主干，仅更新预测头五轮；未使用 2026 世界赛标签训练。',
              f"5. 最终联赛训练 {report['train_count']} 场；历届世界赛微调 {report['fine_count']} 场；2026-04-01 至 2026-05-23 的 {report['calibration_count']} 场仅用于校准。", '',
              '新模型直接用系列赛标签做二元交叉熵，不再强制把学习输出按独立小局假设换算为系列赛胜率；Elo 的独立模型只作为先验。与上一轮大模型对比时，还同时切换为主要联赛子集、改变损失与校准方式，因此这不是只替换网络结构的严格消融。', '',
              '校准层对“动态先验 + 模型残差”使用有边界且正则化的斜率，不加入截距，因此保持交换双方概率互补。专家融合对树模型与既有三种子 Transformer 的输出做同样约束；两个专家分别使用与其训练一致的历史快照。', '',
              '## 为什么默认不是回顾性最高分候选', '',
              '融合与校准层先在 2026-04-01 至 2026-04-30 的 179 场拟合，再在 2026-05-01 至 2026-05-23 的 150 场前向验证。选择在计算 412 场结果之前写入 blend_selection.json。', '',
              '| 校准候选 | 前向验证 Log Loss ↓ | 准确率 ↑ |', '| --- | ---: | ---: |']
    for name, value in blend['protocol']['validation'].items():
        lines.append(f"| {name} | {value['log_loss']:.4f} | {value['accuracy']:.2%} |")
    lines += ['', '前向验证选中重新校准 Transformer，所以 config.toml 默认值设为 calibrated_transformer。融合候选在 412 场更好属于后续观察，仍可显式指定 expert_blend 做前瞻验证。', '',
              '## 不确定性', '']
    for name, value in comparisons.items():
        lo, hi = value['log_loss_delta_95_ci']
        lines.append(f"- {name} 相对原始小 MLP 的 Log Loss 差值为 {value['log_loss_delta']:.4f}，按比赛周分块 bootstrap 的描述性 95% 区间为 [{lo:.4f}, {hi:.4f}]。")
    lines += ['', '区间包含零时，不能声称具有稳定优势。即使区间不含零，多次查看同一集合也限制推断；还需要未来比赛验证。', '',
              '## 调用', '', '```powershell', r"$cxzPython = 'C:\Users\ASUS\.conda\envs\cxz\python.exe'",
              '& $cxzPython main.py optimize', '& $cxzPython main.py test T1 Gen.G --best-of 5 --patch 16.20 --date 2026-10-28',
              '& $cxzPython main.py test T1 Gen.G --model expert_blend --best-of 5 --patch 16.20 --date 2026-10-28',
              '& $cxzPython main.py test T1 Gen.G --model hist_gbm --stage pretrained --best-of 5 --patch 16.20 --date 2026-10-28',
              '& $cxzPython -m unittest discover -s tests -v', '& $cxzPython -m tests.validate_optimized',
              '& $cxzPython -m pipelines.report_optimized', '```', '',
              '示例不表示当天有这场比赛。实际日期、赛制与版本须对应赛程；没有输入首发时使用最后已知阵容。历史日期必须读取已冻结预测，不能用最新状态回填。', '',
              '`main.py train --model calibrated_transformer` 或 `expert_blend` 只重新拟合校准/融合层，复用既有骨干；其三种子和 finetuned 阶段必须与所拟合校准一致。新表格神经网络通过模型注册表和优化流水线接入，旧模型及产物均保留。', '',
              '## 产物', '', '- `selection.json`：四折搜索排名与每折分数。',
              '- `blend_selection.json`：前向校准选择与日期。', '- `metrics.json` / `blend_metrics.json`：全部分数和融合权重。',
              '- `predictions.csv` / `blend_predictions.csv`：逐场预测。', '- `verification.json`：因果特征、重载和集合一致性检查。',
              '- 各模型目录：预训练与微调权重、归一化参数和训练曲线。', '',
              '缓存包含数据和训练实现指纹；原始输入变化或模型训练实现变化后不会静默复用旧搜索结果。', '',
              '## 来源和边界', '',
              '- [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)：表格残差网络与 FT-Transformer 思路。',
              '- [TabM](https://arxiv.org/abs/2410.24210)：参数共享集成思路，本项目只实现任务变体。',
              '- [HistGradientBoostingClassifier 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)。',
              '- 数据：ChainCC. League of Legends esports match dataset. [开放数据主页](https://chaincc.lol/free/data)，CC BY 4.0。实际最新记录为 2026-10-26。', '',
              '仍继承系列赛赛制回溯重建、选手名字标识可能重名/更名、首发赛前已知假设等限制。未建模 BP、Fearless Draft 的逐局英雄占用、未来选边；未接入官方阵容公告。', '']
    (OUT/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(OUT/'report.md')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
