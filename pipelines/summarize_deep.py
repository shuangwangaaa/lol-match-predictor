"""Report every run and equal-weight seed ensembles; never select the best test seed."""
import csv
import json
import numpy as np
from core import ROOT, metrics, bootstrap

OUT = ROOT/'artifacts_deep'


def main():
    old = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    world = {s['id']: s for s in samples if s['domain'] == 'worlds' and s['worlds_year'] == '2026'}
    scopes = {'reused_412_series': list(old), 'worlds_2026_first_map': list(world),
              'worlds_2026_series': [sid for sid, s in world.items() if s['best_of'] >= 3]}
    results, run_metrics, prediction_rows = {}, {}, []
    for name in ['dense', 'dense_transformer']:
        runs = []
        for seed in [42, 43, 44]:
            folder = OUT/f'{name}_seed{seed}'
            runs.append({r['id']: r for r in csv.DictReader((folder/'predictions.csv').open(encoding='utf-8'))})
            run_metrics[f'{name}_seed{seed}'] = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        for stage in ['pretrained', 'worlds_finetuned']:
            key = f'{name}_{stage}_ensemble'
            results[key] = {}
            for scope, ids in scopes.items():
                first_map = scope.endswith('first_map')
                column = stage+('_first_map' if first_map else '_series')
                y = np.array([int(runs[0][sid]['first_map_y' if first_map else 'y']) for sid in ids])
                p = np.mean([[float(run[sid][column]) for sid in ids] for run in runs], axis=0)
                results[key][scope] = metrics(y, p)
                if scope == 'reused_412_series':
                    baseline = np.array([float(old[sid]['mlp_calibrated']) for sid in ids])
                    comparison = bootstrap(y, p, baseline)
                    comparison['delta_log_loss_new_minus_v1'] = comparison.pop('delta_log_loss_mlp_minus_elo')
                    results[key]['comparison_to_v1'] = comparison
                for sid, truth, probability in zip(ids, y, p):
                    prediction_rows.append({'model': key, 'scope': scope, 'id': sid, 'y': int(truth), 'p_a': float(probability)})
    y = np.array([int(old[sid]['y']) for sid in scopes['reused_412_series']])
    p = np.array([float(old[sid]['mlp_calibrated']) for sid in scopes['reused_412_series']])
    results['v1_mlp'] = {'reused_412_series': metrics(y, p)}
    summary = {'seed_policy': 'Equal probability weights for seeds 42, 43, 44; no best-test-seed selection.',
               'benchmark_status': 'The same 412 series have been inspected in earlier experiments; retrospective, not a new blind test.',
               'worlds_2026_sample_limit': {'first_map_examples': len(world), 'bo3_bo5_series': len(scopes['worlds_2026_series'])},
               'results': results, 'runs': {key: {'parameters': r['parameters'], 'choices': r['choices'],
                                                'results': r['results'], 'swap_max_error': r['swap_max_error']} for key, r in run_metrics.items()}}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (OUT/'ensemble_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    lines = ['# Dense + Transformer：联赛预训练与世界赛微调', '',
             '已使用 cxz 环境的 PyTorch 2.8.0 + CUDA 12.8，在 RTX 5060 Ti 16GB 上完成训练。', '',
             '## 模型', '', '| 模型 | 参数量 | 结构 |', '| --- | ---: | --- |',
             '| Dense | 1,656,273 | 1024 → 768 → 512 → 256，全连接、LayerNorm、GELU、Dropout |',
             '| Dense + Transformer | 2,486,225 | 同一 Dense 分支 + 4 层、8 头、128 维 Transformer |', '',
             'Transformer 输入双方最近 12 局的时序特征、五位置选手特征和比赛环境。模型在 Elo logit 上学习残差，并显式反对称化以保证交换双方概率互补。补齐历史位置带掩码。', '',
             '## 两阶段训练与时间边界', '',
             '- 原始队伍表共 60,140 行；联赛样本 12,940 个，世界赛样本 117 个。样本以系列赛分组，赛前冻结特征；单局记录只提供首局监督，不自动称作 BO1。',
             '- 联赛预训练使用地域联赛及发展联赛，风格统计按联赛做训练集归一化，Elo 保留共享尺度；世界赛样本不参与预训练的监督梯度。',
             '- 先用 2025 年世界赛之前的联赛做按时间划分的训练/验证来选择预训练轮数；用 2024 世界赛训练、2025 世界赛验证选择微调轮数。具体日期见 protocol.json。',
             '- 最终用 2026-04-01 之前的 10,248 个联赛样本预训练，再用 2024、2025 世界赛的 91 个分组样本微调。微调骨干学习率 2e-5、预测头 1e-4。',
             '- 2026-04-01 至 2026-05-23 的 1,178 个联赛样本只用于概率校准；训练与微调均未使用 2026 世界赛标签。',
             '- 2026 世界赛目前只有 26 个首局样本、其中 8 场可重建 BO3/BO5；不能把这 8 场的命中率当作稳定的整体水平。',
             '- 联赛预训练样本没有世界赛标签，但输入中的历史实力/状态可以包含预测时点以前公开的国际赛结果。', '',
             '## 与第一版在相同 412 场系列赛上的比较', '',
             '该集合此前已经查看，因此这里只做回顾性比较。三个种子等权平均，不选择测试集表现最高的一次运行。', '',
             '| 模型 | Accuracy ↑ | Log Loss ↓ | Brier ↓ |', '| --- | ---: | ---: | ---: |']
    labels = {'v1_mlp': '第一版 Elo + 小 MLP', 'dense_pretrained_ensemble': 'Dense：仅联赛预训练',
              'dense_worlds_finetuned_ensemble': 'Dense：世界赛微调后', 'dense_transformer_pretrained_ensemble': 'Dense + Transformer：仅预训练',
              'dense_transformer_worlds_finetuned_ensemble': 'Dense + Transformer：世界赛微调后'}
    for key in ['v1_mlp', 'dense_pretrained_ensemble', 'dense_worlds_finetuned_ensemble', 'dense_transformer_pretrained_ensemble', 'dense_transformer_worlds_finetuned_ensemble']:
        value = results[key]['reused_412_series']
        lines.append(f"| {labels[key]} | {value['accuracy']:.2%} | {value['log_loss']:.4f} | {value['brier']:.4f} |")
    lines += ['', '## 2026 世界赛专项结果', '', '| 模型 | 首局准确率（26） | BO3/BO5 准确率（8） | 系列赛 Log Loss ↓ |', '| --- | ---: | ---: | ---: |']
    for key in labels:
        if key == 'v1_mlp':
            continue
        first, series = results[key]['worlds_2026_first_map'], results[key]['worlds_2026_series']
        lines.append(f"| {labels[key]} | {first['accuracy']:.2%} | {series['accuracy']:.2%} | {series['log_loss']:.4f} |")
    lines += ['', '较大的参数量没有自动带来全面提升：需同时检查命中率、概率误差与微调前后的变化。首局与系列赛属于不同预测目标，不能直接混合比较。本轮还同时扩充联赛、引入历史时序并调整训练路线，差异不能单独归因于参数量或 Transformer。各随机种子和训练曲线保存在各子目录的 metrics.json。', '',
              '## 数据边界与来源', '',
              '赛制仍由小局序列和终场比分重建；并未逐场核验官方赛程。首发取首局身份记录并假定赛前已知，英雄选择与当场赛后统计不作为输入。', '',
              '世界赛标签按官方赛事日期窗口清理，剔除 12 个区域资格赛样本：', '',
              '- [Worlds 2024 Primer](https://lolesports.com/en-US/news/worlds-2024)',
              '- [Worlds 2025 Primer](https://lolesports.com/en-US/news/worlds-2025-primer)',
              '- [Worlds 2026 Primer](https://lolesports.com/en-GB/news/worlds-2026-primer)', '',
              '原始比赛数据：ChainCC. League of Legends esports match dataset. [开放数据主页](https://chaincc.lol/free/data)，CC BY 4.0。数据最晚记录为 2026-10-26。', '',
              '## 运行', '', '```powershell', r"$cxzPython = 'C:\Users\ASUS\.conda\envs\cxz\python.exe'", '& $cxzPython -m data_gen.deep_data',
              '& $cxzPython main.py train --seed 42', '& $cxzPython main.py train --seed 43', '& $cxzPython main.py train --seed 44',
              '& $cxzPython -m pipelines.summarize_deep', '& $cxzPython -m tests.validate_deep', '```', '',
              '未来预测使用 main.py test（模型与集成分支由 config.toml 选择，默认三种子等权集成）。可通过 --roster-a / --roster-b 提供五位置首发；未提供时明确使用最后已知阵容。日期必须晚于状态快照，防止用最新状态伪装历史预测。', '']
    (OUT/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({key: {scope: {k: v for k, v in values.items() if k != 'calibration_bins'} for scope, values in result.items() if scope in scopes} for key, result in results.items()}), flush=True)


if __name__ == '__main__':
    main()
