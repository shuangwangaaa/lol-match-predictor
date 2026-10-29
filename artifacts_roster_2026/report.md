# 2026 年数据限定实验

后续核对更新：下文的 8 场世界赛 BO3/BO5 包含 6 场入围赛，并非 8 场瑞士轮。瑞士轮应使用[完整逐场回放](swiss_replay/report.md)：截至赛事日期 10 月 28 日 26 场、命中 15 场（57.69%），BO1 11/20、BO3 4/6。下文两场初始预测误填版本 26.20；完整回放依据原始版本 16.20 已重算为 T1 77.67%、IG 60.24%。原实验数值保留用于追溯。

沿用个人历史 Transformer、五位置阵容 Transformer 与 Dense 架构。网络从随机权重开始，个人能力、阵容磨合、组织历史与赛区动态偏移从 2026-01-01 冷启动；2024/2025 比赛与旧 checkpoint 均未使用。人工指定六赛区先验保留。

没有可用的往年世界赛微调样本，本实验关闭世界赛微调；2026 世界赛标签不参与梯度训练或校准。个人近期状态仍按已完成比赛滚动更新。

有效历史单局 9,818 场，选手 handle 2,785 个。源文件保留多年份，但在构建目标与任何历史状态之前筛选 2026 年系列赛；选手行只保留这些单局。

| 实验 | 联赛训练时间 | 样本数 | 校准时间 | 评估 |
|---|---|---:|---|---|
| benchmark | 2026-01-01 至 2026-04-01（不含） | 1052 | 2026-04-01 至 2026-05-24（不含） | 412 场 |
| worlds | 2026-01-01 至 2026-09-01（不含） | 3395 | 2026-09-01 至 2026-10-15（不含） | 8 场 |

同一 412 场采用相同训练/校准截止日，衡量移除旧年份及历史世界赛微调后的整体变化；不是单独识别“年份”的因果实验。worlds 版本允许更多 2026 年训练样本，仅在训练与校准之后的世界赛上比较，不把早期已用于训练的比赛当作测试。

worlds 版本的校准样本只有 61 场：早段 58 场、后段前向验证 3 场。这一时间段赛事较少，校准可靠性有限；边界在读取该版本评估结果前固定，没有根据世界赛表现调参。

| 评估 | 模型 | Accuracy | Log Loss ↓ | Brier ↓ |
|---|---|---:|---:|---:|
| benchmark | roster_transformer_2026 | 66.75% | 0.604027 | 0.207560 |
| benchmark | multiyear_roster_same_matches | 67.72% | 0.595537 | 0.203817 |
| worlds | roster_transformer_2026 | 75.00% | 0.506858 | 0.169901 |
| worlds | multiyear_roster_same_matches | 62.50% | 0.531889 | 0.189974 |

## 10 月 28 日两场 BO3 回放

| 对阵 | 前者胜率 | 后者胜率 |
|---|---:|---:|
| Movistar KOI vs T1 | 22.25% | 77.75% |
| Invictus Gaming vs LYON | 60.29% | 39.71% |

这些赛果与原评估集合已查看，全部为回顾性实验。世界赛 BO3/BO5 样本很少，不能据此确定泛化收益。预测使用截至 10 月 26 日的历史和最后已知阵容，没有核对当天官方首发。

## 复现

```powershell
& 'C:\Users\ASUS\.conda\envs\cxz\python.exe' main.py train --model roster_transformer_2026 --max-pretrain-epochs 24
& 'C:\Users\ASUS\.conda\envs\cxz\python.exe' main.py test 'Invictus Gaming' LYON --model roster_transformer_2026 --date 2026-10-28 --patch 16.20 --best-of 3
& 'C:\Users\ASUS\.conda\envs\cxz\python.exe' -m tests.validate_roster_2026
```

默认配置与多年份权重保持原样；显式模型名 roster_transformer_2026 使用 worlds 版本。世界赛微调关闭，保存阶段为 pretrained；无需传 --stage，若指定则使用 --stage pretrained。
