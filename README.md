# LoL Match Predictor

英雄联盟职业比赛赛前胜率预测：以选手为核心，结合动态实力、赛区先验、Dense 与 Transformer，输出整场 BO3/BO5 的获胜概率。

当前默认使用 **2026 年底座 + 全瑞士轮加权微调**，模型名为 `roster_transformer_2026_swiss_all`。瑞士轮全部 **33 场系列赛、51 个小局** 已加入训练，选手状态更新至 **2026-10-31**。三个种子的最终权重、固定校准器和历史快照随仓库保存，克隆后可直接预测。

## 快速开始

使用 Python 3.12。已有合适的 PyTorch 环境时直接使用，无需重复安装。本地已验证环境为 Windows、Python 3.12.11、PyTorch 2.8.0+cu128、NumPy 2.1.2、SciPy 1.16.2、scikit-learn 1.7.2。

```powershell
git clone https://github.com/shuangwangaaa/lol-match-predictor.git
cd lol-match-predictor
python -m pip install -r requirements.txt
python main.py test 'Top Esports' 'Invictus Gaming' --date 2026-11-03 --patch 16.20 --best-of 5 --playoffs 1
```
```powershell
conda run -n cxz python main.py test 'Top Esports' 'Invictus Gaming' --date 2026-11-03 --patch 16.20 --best-of 5 --playoffs 1
```

上述命令预测八强 TES–IG，输出 TES 约 **52.72%**、IG 约 **47.28%**；日期采用数据中的 UTC 日期，对应北京时间 **11 月 4 日 06:00**。CPU 与 GPU 输出可能有轻微差异。默认模型已写入 `config.toml`，无需另传 `--model`。

**当前历史快照截止 2026-10-31。日期参数不会自动下载新比赛。** 未来对阵采用瑞士轮结束时最后已知首发；应优先传入实际公布的阵容。预测日期须晚于快照，历史日期会被拦截。旧 10 月 28 日回放可显式使用 `--model roster_transformer_2026_swiss_weighted`。

## 指定阵容与赛区

通过 `--roster-a` / `--roster-b` 提供包含 `top,jng,mid,bot,sup` 的 JSON；转会后的个人历史仍保留。新组织需提供五人阵容，必要时指定赛区。

```powershell
$lineupA = '{"top":"theshy","jng":"wei","mid":"rookie","bot":"jiaqi","sup":"meiko"}'
$lineupB = '{"top":"dhokla","jng":"inspired","mid":"saint","bot":"berserker","sup":"isles"}'
python main.py test 'Invictus Gaming' LYON --date 2026-11-02 --patch 16.20 --best-of 3 --roster-a $lineupA --roster-b $lineupB --region-a LPL --region-b LCS
```

上例只演示自定义首发，并不表示当天有该场比赛。输出 JSON 包括双方胜率、采用的五人阵容、赛区权重、快照日期与三个种子的预测组成。模型在 BP 前预测，不包含英雄选择、选边或 Fearless Draft。

## 当前结果

### 全瑞士轮训练后的淘汰赛预测

截至 **2026-11-02**，按 [官方赛程](https://lolesports.com/en-US/leagues/worlds) 和 [Riot 对阵图转载](https://gamedaily.com/games/lol-worlds-2026-knockouts-bracket) 的实际路径，精确传播七场 BO5 的全部可能结果：

| 北京时间 | 八强对阵 | 前队胜率 | 后队胜率 |
|---|---|---:|---:|
| 11 月 4 日 06:00 | TES vs IG | 52.72% | 47.28% |
| 11 月 5 日 06:00 | GEN vs LYON | 77.95% | 22.05% |
| 11 月 6 日 06:00 | DK vs BLG | 50.69% | 49.31% |
| 11 月 7 日 06:00 | T1 vs KC | 79.76% | 20.24% |

| 预测顺序 | 队伍 | 夺冠 | 进决赛 | 进四强 |
|---:|---|---:|---:|---:|
| 1 | GEN | 25.47% | 46.54% | 77.95% |
| 2 | T1 | 22.21% | 40.65% | 79.76% |
| 3 | DK | 15.09% | 28.14% | 50.69% |
| 4 | BLG | 14.37% | 26.86% | 49.31% |
| 5 | TES | 11.16% | 25.22% | 52.72% |
| 6 | IG | 9.09% | 21.72% | 47.28% |
| 7 | LYON | 1.43% | 6.52% | 22.05% |
| 8 | KC | 1.17% | 4.35% | 20.24% |

排序依据是当前赛程下的夺冠概率，不是确定的最终名次。TES/IG 胜者与 GEN/LYON 胜者进入同一半决赛；DK/BLG 胜者与 T1/KC 胜者进入另一半决赛。已淘汰的 HLE、MKOI、LOS 为并列 9–11 名，G2、CFO、TLAW 为 12–14 名，AL、TSW 为 15–16 名，夺冠概率均为 0。

**瑞士轮现在全部属于训练集，当前模型没有新的独立测试成绩。** 完整名次分布、十六队模型实力排序、来源和训练边界见 [全瑞士轮训练与排名报告](artifacts_roster_2026_swiss_all/report.md)。

### 旧 R1/R2 加权版的历史对照

R1/R2 的 16 场已用于微调，**只在 R3/R4 的 10 场上对照**：

| 版本  |       命中 |     Log Loss |       Brier  |
| --- | -------: | -----------: | -----------: |
| 模型  | **7/10** | **0.548813** | **0.186635** |

### 逐场预测

以下保留旧 `roster_transformer_2026_swiss_weighted` 对瑞士轮 R3/R4 的历史回放，不能作为全瑞士轮训练版的新测试成绩。胜率对应整场比赛，实际比分按对阵中的前后顺序列出。

| 轮次 | 赛制 | 对阵 | 前队胜率 | 后队胜率 | 预测胜者 | 实际比分 | 结果 |
|---|---|---|---:|---:|---|---|---|
| R3 | BO1 | IG vs HLE | 36.50% | 63.50% | HLE | 1:0 | ✗ |
| R3 | BO1 | G2 vs TES | 46.42% | 53.58% | TES | 0:1 | ✓ |
| R3 | BO3 | MKOI vs DK | 22.09% | 77.91% | DK | 0:2 | ✓ |
| R3 | BO3 | LYON vs GEN | 13.36% | 86.64% | GEN | 1:2 | ✓ |
| R3 | BO1 | T1 vs CFO | 77.64% | 22.36% | T1 | 1:0 | ✓ |
| R3 | BO1 | KC vs BLG | 27.99% | 72.01% | BLG | 0:1 | ✓ |
| R3 | BO3 | TSW vs LOS | 54.48% | 45.52% | TSW | 0:2 | ✗ |
| R3 | BO3 | AL vs TLAW | 71.97% | 28.03% | AL | 0:2 | ✗ |
| R4 | BO3 | MKOI vs T1 | 22.78% | 77.22% | T1 | 0:2 | ✓ |
| R4 | BO3 | IG vs LYON | 56.89% | 43.11% | IG | 2:1 | ✓ |

按轮次统计：R3 **5/8（62.5%）**，R4 **2/2（100%）**；按赛制统计：BO1 **3/4（75%）**，BO3 **4/6（66.7%）**。总计 **7/10（70%）**。

R1/R2 不计入上述成绩。这 10 场赛果此前已查看，属于小样本回顾性结果。

完整结构、训练时间边界与限制见 [模型说明](docs/MODEL_CARD.md)，逐场结果见 [加权微调报告](artifacts_roster_2026_swiss_weighted/report.md)。

## 复现微调与验证

仓库保留原 2026 底座及冻结特征，因此可离线复现当前微调阶段，不需要重新下载大型原始数据：

```powershell
python -m unittest discover -s tests -p 'test_*.py'
python -m tests.validate_release
python -m tests.validate_swiss_complete
python -m tests.validate_swiss_weighted
python main.py train
python -m pipelines.worlds_rank
```

`train` 会重写当前全瑞士轮模型目录，建议在独立克隆中复现。从同一原底座重新开始，全部 33 场加入训练，避免重复强化 R1/R2。损失为世界赛监督与联赛教师 KL 两组平均损失按 **10:1** 加权；监督项由 75% 系列赛胜负 BCE 与 25% 实际小局胜率 BCE 组成，按系列赛等权。沿用此前选出的 **24 轮** 和 3e-4 学习率，不用淘汰赛结果挑参数。Transformer 冻结，选手 embedding、Dense 与预测头更新。

`config.toml` 的 `[swiss_weighted]` 可调整训练系数；更改训练参数须重训才能生效。赛区或选手特征设置、保存阶段、种子集合与已保存模型不一致时，预测会拒绝混用。原校准器保持固定，未用 R3/R4 重新拟合。

全量训练核验会重新拟合首个种子，检查 33 场标签、51 局比分、22 局新增历史只更新一次、三个权重、冻结层与四场八强预测。52 项单元测试已通过，其中排名动态规划与 128 种完整赛果枚举逐项一致。旧加权版仍可单独核验。发布核验只加载当前最终权重；CPU 不保证与 CUDA 逐位一致。

## 文件组织

```text
main.py                                  统一预测/微调入口
config.toml                              默认模型、赛区先验、微调系数
models/roster_models.py                   个人与阵容 Transformer、Dense、预测头
data_gen/                                因果特征与历史状态构建
pipelines/swiss_complete.py               全瑞士轮因果输入与加权微调
pipelines/worlds_rank.py                  按实际赛程精确计算名次分布
artifacts_roster_2026_swiss_all/           当前三个最终权重、更新快照与排名报告
artifacts_roster_2026_swiss_weighted/      旧 R1/R2 加权版及历史预测对照
artifacts_roster_2026/                    复现所需的 2026 底座、特征、瑞士轮冻结输入
artifacts_roster_2026_swiss/              轻微调的轻量历史对照，已移除权重与重复快照
data/                                    来源元数据与赛程；本地原始数据不上传
docs/                                    模型说明、清理清单、发布校验记录
tests/                                   行为测试及产物核验
scripts/                                 清理与发布清单工具
```

2026 底座是重新微调必需的训练依赖，默认预测只加载最终加权版。完整从原始比赛重新预训练属于历史研究流程，本发布保证的是当前推理与基于已保存底座的微调复现。

清理记录见 [整理说明](docs/CLEANUP.md)。运行 `powershell -File scripts/cleanup_training.ps1` 只预览清理目标；没有 `-Apply` 不会删除文件。

数据来自 [ChainCC](https://chaincc.lol/free/data)，数据署名与许可记录见 [NOTICE](NOTICE.md)。
