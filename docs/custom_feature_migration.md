# 自有功能迁移说明

本分支以当前 OpenTau 实现为基线，迁移旧工程中的 Value/Advantage、PI0.5、EMA、STEAM、可视化和通用 gRPC 服务端能力。所有新增训练行为默认关闭；RTC 客户端、RealMan adapter、RTC 配置和 RTC 依赖均不在迁移范围内。

## Value 与 Advantage

Value 和 advantage 制品统一使用 `episode_index,frame_index` 键。旧 timestamp 键、负索引、重复键、非有限数值会被拒绝。完整 advantage bundle 包含：

- `meta/advantages.json`
- `meta/raw_advantages.json`
- `meta/advantage_sources.json`
- `meta/advantage_report.json`

数据集可以通过 feature mapping 将逐帧人工干预字段映射为标准 `intervention` 角色：

```json
{
  "data_features_name_mapping": {
    "intervention": "human_intervention"
  }
}
```

只有显式字段值大于零时才把 effective advantage 覆盖为 `1.0`；原始 TD residual 保存在 `raw_advantages.json`，来源记为 `human_intervention_override`。`mistake` 和失败 episode 不会触发该覆盖。

Value held-out 评估入口为：

```powershell
python -m opentau.scripts.evaluate_value `
  --train_config CONFIG.json `
  --dataset_mixture HELD_OUT.json `
  --output_dir outputs/value_eval `
  --batch_size 16
```

held-out mixture 必须显式列出 episodes；checkpoint 和数据集均以 local-only 方式打开。报告包含 overall、by-dataset 和 by-task 的 MAE、NLL、Spearman 与 ordinal CDF calibration。

## Quantile sidecar

`compute_quantiles` 支持 exact 和固定 seed 的 reservoir 模式，输出 `meta/stats_quantiles.json`。加载时 sidecar 只补充缺失的 `q01`/`q99`，不覆盖现有 stats 或显式配置。

## PI0.5

新增配置均保持向后兼容：

```json
{
  "advantage": "ignore",
  "advantage_threshold": 0.0,
  "cfg_dropout": 0.0,
  "guidance_scale": 1.0,
  "delay_sampling": "uniform",
  "delay_exponential_decay": 1.0
}
```

离散动作的第一个 token 继续由有效的 `"Action: "` indicator 上下文预测，没有迁移旧版逐样本动态查找逻辑。

## EMA

设置 `ema_decay` 可启用仓库内置的具名参数 EMA；`None` 为关闭。EMA 只在完成一个同步 optimizer step 后更新，验证时临时切换，checkpoint/resume 保存 shadow state 和 update count。首版只支持单卡和 DDP；DeepSpeed/FSDP 会在配置校验阶段被拒绝。

## STEAM 与可视化

STEAM 的完整流程见 [steam.md](steam.md)。Value/STEAM 可视化器通过浏览器 `input` 事件实时拖动，并显示 raw/effective advantage 和来源。

## gRPC 图像处理

通用 gRPC 服务端不再把输入无条件缩放到训练分辨率。JPEG/PNG 保留解码后的原始宽高；现有 raw wire contract 仍是 `float32 HWC RGB` 方形图像，并严格校验 payload。resize、letterbox 和归一化由 policy preprocessing 负责。
