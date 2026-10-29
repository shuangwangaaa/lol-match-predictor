# LoL Match Predictor

英雄联盟职业比赛赛前胜率预测：以选手为核心，结合动态实力、赛区先验、Dense 与 Transformer，输出整场 BO3/BO5 的获胜概率。

当前默认使用 **2026 年底座 + 瑞士轮 R1/R2 加权微调**，模型名为 `roster_transformer_2026_swiss_weighted`。三个种子的最终权重、校准器和历史快照已随仓库保存，克隆后可直接预测，无需先训练。

## 快速开始

使用 Python 3.12。已有合适的 PyTorch 环境时直接使用，无需重复安装。本地已验证环境为 Windows、Python 3.12.11、PyTorch 2.8.0+cu128、NumPy 2.1.2、SciPy 1.16.2、scikit-learn 1.7.2。

```powershell
git clone https://github.com/shuangwangaaa/lol-match-predictor.git
cd lol-match-predictor
python -m pip install -r requirements.txt
python main.py test 'Invictus Gaming' LYON --date 2026-10-28 --patch 16.20 --best-of 3
```
```powershell
conda run -n cxz python main.py test 'Invictus Gaming' LYON --date 2026-10-28 --patch 16.20 --best-of 3
```

上述命令是历史比赛的推理复核，输出 IG 约 **56.89%**、LYON 约 **43.11%**；CPU 与 GPU 输出可能有轻微差异。`model` 默认值已经写入 `config.toml`，无需另传 `--model`。

**当前历史快照截止 2026-10-26。日期参数不会自动更新数据。** 例如预测 10 月 29 日的比赛仍使用此快照。未显式提供首发时采用最后已知阵容；应优先使用已公布的实际首发。预测日期须晚于快照，历史日期会被拦截。

## 指定阵容与赛区

通过 `--roster-a` / `--roster-b` 提供包含 `top,jng,mid,bot,sup` 的 JSON；转会后的个人历史仍保留。新组织需提供五人阵容，必要时指定赛区。

```powershell
$lineupA = '{"top":"theshy","jng":"wei","mid":"rookie","bot":"jiaqi","sup":"meiko"}'
$lineupB = '{"top":"dhokla","jng":"inspired","mid":"saint","bot":"berserker","sup":"isles"}'
python main.py test 'Invictus Gaming' LYON --date 2026-10-28 --patch 16.20 --best-of 3 --roster-a $lineupA --roster-b $lineupB --region-a LPL --region-b LCS
```

输出 JSON 包括双方胜率、采用的五人阵容、赛区权重、快照日期与三个种子的预测组成。模型在 BP 前预测，不包含英雄选择、选边或 Fearless Draft。

## 当前结果

R1/R2 的 16 场已用于微调，**只在 R3/R4 的 10 场上对照**：

| 版本  |       命中 |     Log Loss |       Brier  |
| --- | -------: | -----------: | -----------: |
| 模型  | **7/10** | **0.548813** | **0.186635** |

### 逐场预测

以下为当前加权微调模型对瑞士轮 R3/R4 的历史回放；胜率对应整场比赛，实际比分按对阵中的前后顺序列出。

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
python -m tests.validate_swiss_weighted
python main.py train
```

`train` 会重写当前模型目录，建议在独立克隆中复现实验。它从原底座开始，以 R1 训练、R2 选轮次，再以 R1+R2 训练三个种子。损失为世界赛监督 BCE 与联赛教师 KL 两组平均损失按 **10:1** 加权；当前选中 24 轮，学习率 3e-4。Transformer 冻结，选手 embedding、Dense 与预测头更新。

`config.toml` 的 `[swiss_weighted]` 可调整训练系数；更改训练参数须重训才能生效。赛区或选手特征设置、保存阶段、种子集合与已保存模型不一致时，预测会拒绝混用。原校准器保持固定，未用 R3/R4 重新拟合。

训练产物核验会重新拟合首个种子，确认反转 R3/R4 赛果不改变训练结果。发布核验只加载最终权重，不需要底座。CUDA 训练与保存预测的严格重现建议使用上述本地环境；CPU 不保证逐位一致。

## 文件组织

```text
main.py                                  统一预测/微调入口
config.toml                              默认模型、赛区先验、微调系数
models/roster_models.py                   个人与阵容 Transformer、Dense、预测头
data_gen/                                因果特征与历史状态构建
pipelines/swiss_weighted.py               当前加权微调流程
artifacts_roster_2026_swiss_weighted/      实际使用的三个最终权重、快照、校准与报告
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
