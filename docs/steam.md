# PI0.5 与 STEAM 工作流

STEAM 是独立的 `policy.type="steam"` 时序进度模型。它不依赖 Value policy；生成的标签复用统一的 frame-index advantage bundle，并可直接供 PI0.5 的 advantage conditioning 使用。

## 数据约束

- STEAM 训练 pair 的两帧必须来自同一 episode。
- 数据集必须显式设置 `steam_source` 为 `expert` 或 `non_expert`。
- 时序偏移按全局 episode 长度参考缩放后映射到带符号 bins。
- Advantage 生成固定加载三个独立 checkpoint，并对三个进度差取最小值。
- terminal frame 的 raw/effective advantage 都是 `0.0`，来源为 `steam_terminal_default`。

## 训练 STEAM ensemble

示例配置见 [steam_training_config.json](../configs/examples/steam_training_config.json)。训练循环变更的本地验证只能使用仓库允许的 smoke 配置；不要直接运行示例中的生产步数。

```powershell
opentau-train --accelerate-config configs/examples/accelerate_ddp_config.yaml --config_path=configs/examples/steam_training_config.json --seed=0 --output_dir=outputs/steam/member_0
opentau-train --accelerate-config configs/examples/accelerate_ddp_config.yaml --config_path=configs/examples/steam_training_config.json --seed=1 --output_dir=outputs/steam/member_1
opentau-train --accelerate-config configs/examples/accelerate_ddp_config.yaml --config_path=configs/examples/steam_training_config.json --seed=2 --output_dir=outputs/steam/member_2
```

## 生成 advantage bundle

数据配置见 [steam_advantage_config.json](../configs/examples/steam_advantage_config.json)。三个 checkpoint 的数量是强约束。

```powershell
opentau-steam-advantages `
  --checkpoint outputs/steam/member_0/checkpoints/030000 `
  --checkpoint outputs/steam/member_1/checkpoints/030000 `
  --checkpoint outputs/steam/member_2/checkpoints/030000 `
  --dataset-mixture configs/examples/steam_advantage_config.json
```

每个数据集输出：

- `meta/advantages.json`
- `meta/raw_advantages.json`
- `meta/advantage_sources.json`
- `meta/advantage_report.json`

四份文件使用 `episode_index,frame_index` 键并要求选中帧覆盖率为 `1.0`。

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

```powershell
streamlit run src/opentau/scripts/value_visualizer_app.py -- `
  --dataset-config DATASET_CONFIG.json `
  --values DATASET_ROOT/meta/raw_advantages.json `
  --effective-advantages DATASET_ROOT/meta/advantages.json `
  --advantage-sources DATASET_ROOT/meta/advantage_sources.json `
  --series-label "STEAM Advantage"
```

可视化器只用 frame index 查找；timestamp 仅用于展示。
