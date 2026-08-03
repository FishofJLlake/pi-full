# PI0.5 与 STEAM 工作流

STEAM 是独立的 `policy.type="steam"` 时序进度模型。它不依赖 Value policy；生成的标签复用统一的 frame-index advantage bundle，并可直接供 PI0.5 的 advantage conditioning 使用。

## 数据约束

- STEAM 训练 pair 的两帧必须来自同一 episode。
- 数据集必须显式设置 `steam_source` 为 `expert` 或 `non_expert`。
- 默认使用 SigLIP 原生 `384x384` 输入；配置与输入分辨率不一致时直接报错，不做静默上采样。
- 时序偏移映射到带符号 bins；轨迹长度缩放由 `length_scale_enabled` 显式控制。
- Ensemble 支持任意 `N >= 1`，并按 RLinf 的逐帧 worst-of-N（最小 signed score）聚合。
- 独立成员可合并为一个 `members.N.*` checkpoint；单成员合并 checkpoint 仍保留 ensemble 轴。
- terminal frame 的 raw/effective advantage 都是 `0.0`，来源为 `steam_terminal_default`。

## 训练 STEAM ensemble

常规示例见 [steam_training_config.json](../configs/examples/steam_training_config.json)，与 RLinf
关键超参数对齐的参考配置见 [steam_rlinf_parity_config.json](../configs/examples/steam_rlinf_parity_config.json)：384 分辨率、32 bins、16k steps、两卡 global batch 512、AdamW `5e-5` 和 500-step warmup。
训练循环变更的本地验证只能使用仓库允许的 smoke 配置；不要直接运行生产步数。

```powershell
opentau-train --accelerate-config configs/examples/accelerate_ddp_config.yaml --config_path=configs/examples/steam_training_config.json --seed=0 --output_dir=outputs/steam/member_0
opentau-train --accelerate-config configs/examples/accelerate_ddp_config.yaml --config_path=configs/examples/steam_training_config.json --seed=1 --output_dir=outputs/steam/member_1
opentau-train --accelerate-config configs/examples/accelerate_ddp_config.yaml --config_path=configs/examples/steam_training_config.json --seed=2 --output_dir=outputs/steam/member_2
```

把任意数量的成员合并成一个 checkpoint：

```powershell
opentau-steam-merge-ensemble `
  --member outputs/steam/member_0/checkpoints/030000 `
  --member outputs/steam/member_1/checkpoints/030000 `
  --member outputs/steam/member_2/checkpoints/030000 `
  --output outputs/steam/ensemble_3
```

`--member PATH:idx` 还可以从已有 ensemble checkpoint 中抽取指定成员。

## 生成 advantage bundle

数据配置见 [steam_advantage_config.json](../configs/examples/steam_advantage_config.json)。推荐传入一个已合并的 ensemble checkpoint；也可以重复 `--checkpoint` 拼接多个单成员或 ensemble checkpoint。
`torchrun` 会对每个数据集做连续均衡分片，所有 rank 完成推理后仅由 rank 0 写入结果。

```powershell
torchrun --standalone --nproc-per-node=2 -m opentau.scripts.compute_steam_advantages `
  --checkpoint outputs/steam/ensemble_3 `
  --dataset-mixture configs/examples/steam_advantage_config.json `
  --score-mode rlinf_signed `
  --label-mode quantile `
  --expert-positive-fraction 0.8 `
  --non-expert-positive-fraction 0.3
```

每个数据集输出：

- `meta/advantages.json`
- `meta/raw_advantages.json`
- `meta/advantage_sources.json`
- `meta/advantage_report.json`
- `meta/advantages_<tag>.parquet`
- `meta/steam_advantage_diagnostics.json`

JSON bundle 保持现有消费者兼容；Parquet 使用 RLinf 风格字段，并额外保留 paper-baseline、signed-bin、成员熵与 provenance。diagnostics 文件保存每个成员曲线，供视频可视化读取。默认标签语义使用 RLinf signed score 和严格 `>`；旧行为可显式选择 `--score-mode paper_baseline --threshold-comparison inclusive`。

## 无推理重标 advantage

`opentau-steam-relabel` 只读取已有 `raw_advantages.json` 和
`advantages_<source_tag>.parquet`，不会加载 STEAM checkpoint，也不会执行模型推理。基础标签先按
`steam_source` 分开生成，再按以下优先级覆盖：

1. expert/non-expert 各自的 `all_positive`、`threshold` 或 `quantile` 基础规则；
2. 选定 source 的每条轨迹最后 N 帧强制为 positive；
3. 显式映射的人类干预帧强制为 positive（最高优先级）。

推荐先只验证统计：

```powershell
opentau-steam-relabel `
  --dataset-mixture DATASET_MIXTURE.json `
  --source-tag steam `
  --output-tag steam_recovery_v1 `
  --expert-mode all_positive `
  --non-expert-mode quantile `
  --non-expert-positive-fraction 0.3 `
  --quantile-grouping actual_lookahead `
  --tail-positive-frames 30 `
  --tail-positive-sources expert `
  --force-intervention-positive `
  --intervention-positive-sources non_expert `
  --dry-run
```

确认每个数据集和每个 `actual_lookahead` 桶的计数后，删除 `--dry-run` 才会写文件。
若 expert 仍需保留 top 80%，把 expert 参数改为：

```powershell
--expert-mode quantile --expert-positive-fraction 0.8 --quantile-grouping actual_lookahead
```

这里的 top 80% 是在相同 `steam_source + actual_lookahead` 内跨轨迹排名，不是整条轨迹共用一个
阈值。它能避免短 lookahead 的尾部帧与 `H=K` 的中段帧直接竞争。分位数采用确定性排序并精确取
`ceil(桶大小 * fraction)` 个样本；同分时按数据集和帧 key 打破平局。

尾部覆盖默认只允许 expert。只有确认 non-expert 轨迹末尾也是有效恢复/成功动作时，才显式传入
`--tail-positive-sources expert non_expert`；否则失败 rollout 的卡死、放弃和失败 terminal 也会被标成
positive。`--tail-positive-frames 30` 会包含每条轨迹的 terminal frame，与 STEAM 生成阶段默认将
terminal 设为 0 的规则不同。

人工干预覆盖要求数据配置通过 `data_features_name_mapping` 将标准角色 `intervention` 显式映射到
一个逐帧列，例如 `"intervention": "human_intervention"`。默认仅当值严格大于
`--intervention-value-threshold 0` 时置 1，不会从 `mistake` 或 episode success 猜测干预状态。

每个数据集新增 `meta/advantages_<output_tag>.parquet` 和
`meta/advantage_relabel_<output_tag>.json` 作为逐帧审计与汇总；同时原子更新 PI0.5 实际读取的
`meta/advantages.json` 与 `meta/advantage_sources.json`。原始 `raw_advantages.json`、
`advantage_report.json` 和 source-tag Parquet 保持不变。

## PI0.5 conditioning

示例配置见 [pi05_steam_training_config.json](../configs/examples/pi05_steam_training_config.json)。核心配置为：

```json
{
  "advantage": "use",
  "advantage_threshold": 0.5,
  "cfg_dropout": 0.1,
  "guidance_scale": 1.0
}
```

`advantage="ignore"`、`cfg_dropout=0.0`、`guidance_scale=1.0` 保持当前默认行为。推理启用 advantage 但未传值时按 positive 处理。

## 可视化

生成“上方视频、下方 advantage 曲线”的对齐 MP4：

```powershell
opentau-steam-visualize `
  --dataset-config configs/examples/steam_advantage_config.json `
  --dataset-index 0 `
  --episode 42 `
  --camera-key camera0 `
  --output outputs/steam/episode_42_advantage.mp4
```

输出固定为 800x880 H.264/yuv420p：上方 800x600 按比例 letterbox、绝不裁剪；下方 800x280 绘制浅色成员曲线、黑色 ensemble minimum 和红色当前帧标记。帧号读取真实 `frame_index`，FPS 默认取数据集 metadata。若 diagnostics 不存在，会警告并退化为仅显示聚合曲线。

原有 Streamlit 浏览器仍可用于交互检查 JSON bundle：

```powershell
streamlit run src/opentau/scripts/value_visualizer_app.py -- `
  --dataset-config DATASET_CONFIG.json `
  --values DATASET_ROOT/meta/raw_advantages.json `
  --effective-advantages DATASET_ROOT/meta/advantages.json `
  --advantage-sources DATASET_ROOT/meta/advantage_sources.json
```
