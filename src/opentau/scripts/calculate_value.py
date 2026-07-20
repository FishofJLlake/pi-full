#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compute value function outputs for a value policy over a configured dataset.

Loads a value policy from a checkpoint and runs predict_value on each batch from
the dataset mixture in the config. Saves (episode_index, frame_index) -> value to JSON.

Usage:
  # From checkpoint dir (has train_config.json and policy weights)
  python -m opentau.scripts.calculate_value \\
    --config_path /path/to/checkpoints/00520000 \\
    [--batch_size 20] [--output_file values.json]

  # Override dataset (config + policy from checkpoint, data from dataset_mixture)
  python -m opentau.scripts.calculate_value \\
    --config_path /path/to/checkpoints/00520000 \\
    --dataset_mixture examples/my_datasets.json \\
    [--output_file values.json]

  # From full train config file (policy.pretrained_path must point to checkpoint)
  python -m opentau.scripts.calculate_value \\
    --train_config configs/train/value_config.json \\
    [--output_file values.json]

  # Override checkpoint when using train config
  python -m opentau.scripts.calculate_value --train_config configs/train/value_config.json \\
    --checkpoint_path /path/to/checkpoints/00520000

  # Override dataset mixture (works with either --config_path or --train_config)
  python -m opentau.scripts.calculate_value --config_path /path/to/ckpt --dataset_mixture examples/advantage_config.json
"""

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader

from opentau.configs.default import DatasetMixtureConfig
from opentau.configs.refs import resolve_refs_to_tempfile
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset
from opentau.policies.factory import get_policy_class
from opentau.policies.value.configuration_value import ValueConfig
from opentau.scripts.value_artifacts import serialize_value_key
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import auto_torch_device, init_logging


def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.numpy()
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.flat[0])
        return x
    return x


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute value function outputs for a value policy over a configured dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Path to checkpoint directory (contains train_config.json) or config file.",
    )
    group.add_argument(
        "--train_config",
        type=Path,
        default=None,
        help="Path to full train config JSON (policy.pretrained_path must point to checkpoint).",
    )
    parser.add_argument(
        "--dataset_mixture",
        type=Path,
        default=None,
        help="Override dataset: path to dataset mixture JSON (e.g. dataset_config.json).",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("values.json"),
        help="Output JSON file for (episode_index,frame_index) -> value.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help="Override checkpoint directory when using --train_config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Dataloader batch size (overrides config).",
    )
    return parser.parse_args()


def main(cfg: TrainPipelineConfig, args: argparse.Namespace):
    output_file = args.output_file
    dataset_mixture_path = args.dataset_mixture
    if not isinstance(cfg.policy, ValueConfig):
        raise ValueError(
            f"calculate_value requires policy.type='value'; got {cfg.policy.type!r}"
        )
    if args.train_config:
        logging.info("Using full train config: %s", args.train_config)

    if dataset_mixture_path:
        logging.info(f"Overriding dataset: loading mixture from {dataset_mixture_path}")
        tmp_mixture = resolve_refs_to_tempfile(dataset_mixture_path)
        try:
            mixture_cfg = draccus.parse(
                config_class=DatasetMixtureConfig,
                config_path=str(tmp_mixture),
                args=[],
            )
        finally:
            tmp_mixture.unlink(missing_ok=True)
    else:
        logging.info("Using dataset mixture from train config")
        mixture_cfg = cfg.dataset_mixture

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = auto_torch_device()
    checkpoint_path = args.checkpoint_path or cfg.policy.pretrained_path
    if checkpoint_path is None:
        raise ValueError("A Value checkpoint must be provided")
    logging.info("Loading value policy from checkpoint: %s", checkpoint_path)
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(
        checkpoint_path,
        config=cfg.policy,
        local_files_only=True,
        backbone_local_files_only=True,
    )
    policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()

    # (episode_index, frame_index) -> value (float)
    all_values = {}

    for dataset_idx, dataset_cfg in enumerate(mixture_cfg.datasets):
        logging.info(f"Creating dataset {dataset_idx}")
        result = make_dataset(dataset_cfg, cfg, return_advantage_input=True)
        dataset = result[0] if isinstance(result, tuple) else result

        batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
        dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": cfg.num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if cfg.num_workers > 0 and cfg.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        dataloader = DataLoader(dataset, **dataloader_kwargs)

        with torch.inference_mode():
            for batch in dataloader:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                values_tensor = policy.predict_value(batch)
                for ep_idx, frame_idx, val in zip(
                    batch["episode_index"],
                    batch["frame_index"],
                    values_tensor,
                    strict=True,
                ):
                    ep_idx = _to_scalar(ep_idx)
                    frame_idx = _to_scalar(frame_idx)
                    val = _to_scalar(val)
                    if isinstance(ep_idx, np.ndarray):
                        ep_idx = int(ep_idx.flat[0])
                    if isinstance(frame_idx, np.ndarray):
                        frame_idx = int(frame_idx.flat[0])
                    if isinstance(val, np.ndarray):
                        val = float(val.flat[0])
                    key = serialize_value_key(int(ep_idx), int(frame_idx))
                    value = float(val)
                    if not math.isfinite(value):
                        raise ValueError(f"Value for frame key {key!r} is not finite: {value!r}")
                    if key in all_values:
                        raise ValueError(
                            f"Duplicate frame key {key!r} across the configured value dataset(s). "
                            "Generate separate artifacts or remap episode indices."
                        )
                    all_values[key] = value

    values_list = list(all_values.values())
    n = len(values_list)
    logging.info(f"Computed {n} values")

    if n == 0:
        logging.warning("No values computed (no LeRobotDataset in mixture or all skipped).")
        return

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_values, f, indent=2)
    logging.info(f"Saved values to {out_path}")

    arr = np.array(values_list)
    logging.info(f"Value stats: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, count={n}")

    # Plot value over frame index
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed; skipping value-over-frame-index plot.")
    else:
        frame_indices = []
        values_plot = []
        for key, val in all_values.items():
            _, frame_idx_str = key.split(",", 1)
            frame_indices.append(int(frame_idx_str))
            values_plot.append(val)

        frame_indices = np.array(frame_indices)
        values_plot = np.array(values_plot)
        out_dir = out_path.parent
        plot_path = out_dir / "value_over_frame_index.png"

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

        # 1) Scatter: all (frame index, value) points
        axes[0].scatter(frame_indices, values_plot, alpha=0.25, s=4, c="steelblue")
        axes[0].set_xlabel("Frame index")
        axes[0].set_ylabel("Value")
        axes[0].set_title("Value vs frame index (all points)")
        axes[0].grid(True, alpha=0.3)

        # 2) Lines: value over frame index for first N episodes
        by_episode = defaultdict(list)
        for key, val in all_values.items():
            ep_idx_str, frame_idx_str = key.split(",", 1)
            by_episode[int(ep_idx_str)].append((int(frame_idx_str), val))
        max_episodes_plot = 10
        for ep_idx, points in sorted(by_episode.items())[:max_episodes_plot]:
            points.sort(key=lambda p: p[0])
            frame_idx_ep = np.array([p[0] for p in points])
            val_ep = np.array([p[1] for p in points])
            axes[1].plot(frame_idx_ep, val_ep, alpha=0.8, label=f"Episode {ep_idx}")
        axes[1].set_xlabel("Frame index")
        axes[1].set_ylabel("Value")
        axes[1].set_title(f"Value vs frame index (first {min(max_episodes_plot, len(by_episode))} episodes)")
        axes[1].legend(loc="best", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logging.info(f"Saved value-over-frame-index plot to {plot_path}")
    # end plot


if __name__ == "__main__":
    init_logging()
    args = _parse_args()

    if args.train_config is not None:
        cfg = TrainPipelineConfig.from_pretrained(args.train_config, local_files_only=True)
    else:
        cfg = TrainPipelineConfig.from_pretrained(args.config_path, local_files_only=True)

    if args.checkpoint_path is not None:
        cfg.policy.pretrained_path = args.checkpoint_path

    main(cfg, args)
