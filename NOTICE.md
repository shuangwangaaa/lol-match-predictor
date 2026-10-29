# 数据来源与署名

本项目使用并转换 ChainCC League of Legends esports match dataset。

- 来源：[ChainCC Open Dataset](https://chaincc.lol/free/data)
- 原始元数据标注许可：CC BY 4.0。
- 访问与快照信息：`data/sources.json`、`data/player_sources.json`。
- 本项目改编内容：筛选、按时间构造选手/对手特征、冻结系列赛输入、历史状态快照和概率预测。
- 发布仓库包含用于复现的派生特征与快照；大型原始 CSV、下载分块及依赖缓存不上传。

瑞士轮赛程与实际比分的来源记录在 `data/swiss_2026_results.json` 中。赛果仅按训练协议规定的时间分区使用。

本项目为独立研究项目，未获得 Riot Games 的官方认证。英雄联盟及相关名称的权利归其各自权利人所有。
