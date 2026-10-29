# 发布整理记录

整理日期：2026-10-29。

## 保留

- 当前使用模型：`artifacts_roster_2026_swiss_weighted/seed42,43,44/finetuned.pt`，连同快照、校准、逐场预测、选择轨迹、训练协议和核验记录。
- 训练依赖：`artifacts_roster_2026/` 中世界赛前 2026 底座的三个 `pretrained.pt`、冻结特征、样本元数据与瑞士轮输入。重新微调与标签隔离核验需要这些文件，所以没有删除。
- 轻量历史对照：`artifacts_roster_2026_swiss/` 的表格、指标与协议；当前微调报告生成时需要读取旧预测，保留它们不会加载旧权重。
- 本地原始数据、源码和测试。大型原始表与下载缓存保留在本机，通过 `.gitignore` 排除上传。

## 删除

- 早期 Elo/MLP、V2、Dense/Transformer、树与融合模型、赛区实验、多年份选手模型的训练目录。
- 2026 早期 benchmark 的整套权重与重复数据。
- 轻微调模型的三个旧权重及重复快照。
- 项目内旧 `.deps` 依赖副本和 Python 缓存；实际使用的 `cxz` Conda 环境完整保留。

精确删除路径、文件数和字节数见 [cleanup_manifest.json](cleanup_manifest.json)。脚本在删除前验证绝对路径全部位于项目内，拒绝跨越 junction/symlink，保护最终模型目录。

## 默认与验证

`config.toml` 和缺省配置均指向 `roster_transformer_2026_swiss_weighted`。清理前后比较全部最终权重的 SHA-256，确认模型不变。整理后运行单元测试、最终模型独立推理核验和加权训练产物核验；详见 [release_verification.json](release_verification.json) 与模型目录的 `verification.json`。

原训练协议、选择轨迹及当时的源码哈希保留，避免把发布整理伪装成一次新训练。历史报告中的“旧权重保留”等表述描述训练当时状态；以本清单为准。旧实验的产物验证命令不再属于当前支持入口。

GitHub 上传代码、文档、最终权重及微调复现依赖，不上传 `.deps`、Conda 环境、原始 CSV、下载分块或凭据。
