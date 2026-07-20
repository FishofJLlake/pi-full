#!/usr/bin/env python
# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streamlit app to visualize VLA value function with camera images from a LeRobot dataset.

Reads dataset_config.json and values.json, loads a single episode via LeRobotDataset,
and provides a timeline scrubber + value curve. values.json is a dict with keys
(episode_index, frame_index) and values as the value function outputs.

Usage:
    streamlit run src/opentau/scripts/value_visualizer_app.py -- \\
        --dataset-config path/to/dataset_config.json \\
        --values path/to/values.json \\
        [--train-config path/to/dir_or_train_config.json] \\
        [--episode 0]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import draccus
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.constants import HF_OPENTAU_HOME
from opentau.datasets.factory import make_dataset
from opentau.scripts.value_artifacts import load_value_labels
from opentau.scripts.value_artifacts import load_values
from opentau.scripts.value_visualizer_frontend import build_scrubber_frames, build_value_scrubber_html

# Hardcoded path to logo image shown in the header.
LOGO_PATH = Path("assets/logo.png")


def load_dataset_config(path: Path) -> tuple[DatasetConfig, list[int]]:
    """Load the first complete dataset entry and its configured episodes."""
    mixture = draccus.parse(
        config_class=DatasetMixtureConfig,
        config_path=path,
        args=[],
    )
    if not mixture.datasets:
        raise ValueError(f"No 'datasets' in {path}")
    first = copy.deepcopy(mixture.datasets[0])
    if first.repo_id is None:
        raise ValueError("Value visualization requires a LeRobot repo_id dataset")
    root = Path(first.root) if first.root is not None else HF_OPENTAU_HOME / first.repo_id
    first.root = str(root.resolve())
    episodes = first.episodes if first.episodes is not None else [0]
    episodes = sorted(episodes) if isinstance(episodes, list) else list(range(episodes))
    return first, episodes


def _tensor_or_array_to_pil(x) -> Image.Image | None:
    """Convert a tensor (C,H,W) or numpy array to PIL Image for display."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):
            x = np.transpose(x, (1, 2, 0))
        if np.issubdtype(x.dtype, np.floating):
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(x)
    return None


@st.cache_data(show_spinner=True)
def load_frames_lerobot_cached(
    dataset_config_path: Path,
    train_config_path: Path | None,
    values_path: Path,
    episode_index: int,
    camera_key: str,
    raw_advantages_path: Path | None = None,
    effective_advantages_path: Path | None = None,
    advantage_sources_path: Path | None = None,
) -> pd.DataFrame:
    """Load one episode and attach frame-indexed values to its display rows."""
    dataset_cfg, _ = load_dataset_config(dataset_config_path)
    value_lookup = load_values(values_path)

    raw_lookup = load_values(raw_advantages_path) if raw_advantages_path is not None else {}
    effective_lookup = (
        load_values(effective_advantages_path) if effective_advantages_path is not None else {}
    )
    source_lookup = (
        load_value_labels(advantage_sources_path) if advantage_sources_path is not None else {}
    )

    def advantage_columns(key: tuple[int, int]) -> dict[str, object]:
        return {
            "raw_advantage": raw_lookup.get(key, np.nan),
            "effective_advantage": effective_lookup.get(key, np.nan),
            "advantage_source": source_lookup.get(key, ""),
        }

    if train_config_path is None:
        train_config_path = dataset_config_path.parent / "train_config.json"
        if not train_config_path.is_file():
            train_config_path = dataset_config_path.parent
    path = Path(train_config_path).resolve()
    train_cfg = TrainPipelineConfig.from_pretrained(path, local_files_only=True)
    train_cfg.val_freq = 0

    dataset_cfg.episodes = [episode_index]
    dataset_cfg.prompt_substitutions = None
    dataset_cfg.image_transforms.enable = False
    res = make_dataset(
        dataset_cfg,
        train_cfg,
        return_advantage_input=True,
        local_files_only=True,
    )
    dataset = res[0] if isinstance(res, tuple) else res
    camera_key = camera_key if camera_key is not None else "camera0"

    # DataLoader calls __getitem__; each batch contains episode, frame, and timestamp metadata.
    # With batch_size=1, each iteration visits one datapoint (one frame) exactly once.
    batch_size = 1
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    all_rows = []
    for batch in dataloader:
        if any(key not in batch for key in ("episode_index", "frame_index", "timestamp")):
            # Fallback: batch from older dataset; build from per-sample __getitem__
            start_idx = len(all_rows)
            n_in_batch = next(
                v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor) and v.dim() > 0
            )
            for i in range(n_in_batch):
                item = dataset[start_idx + i]
                ep = item.get("episode_index", episode_index)
                ep = int(ep.item() if hasattr(ep, "item") else ep) if ep is not None else episode_index
                frame_idx = item.get("frame_index")
                if frame_idx is None:
                    raise KeyError("Value visualization requires frame_index metadata")
                frame_idx = int(frame_idx.item() if hasattr(frame_idx, "item") else frame_idx)
                ts = item.get("timestamp")
                ts = float(ts.item() if hasattr(ts, "item") else ts) if ts is not None else 0.0
                value = value_lookup.get((ep, frame_idx), np.nan)
                pil_img = _tensor_or_array_to_pil(item.get(camera_key))
                all_rows.append(
                    {
                        "step": len(all_rows),
                        "episode_index": ep,
                        "frame_index": frame_idx,
                        "timestamp": ts,
                        "value": value,
                        "image": pil_img,
                        **advantage_columns((ep, frame_idx)),
                    }
                )
            continue
        ep_b = batch["episode_index"]
        frame_b = batch["frame_index"]
        ts_b = batch["timestamp"]
        if ep_b.dim() > 1:
            ep_b = ep_b.squeeze(-1)
        if frame_b.dim() > 1:
            frame_b = frame_b.squeeze(-1)
        if ts_b.dim() > 1:
            ts_b = ts_b.squeeze(-1)
        imgs_b = batch[camera_key]
        if imgs_b.dim() == 5:
            imgs_b = imgs_b[:, -1]
        for i in range(ep_b.shape[0]):
            ep = int(ep_b[i].item())
            frame_idx = int(frame_b[i].item())
            ts = float(ts_b[i].item())
            value = value_lookup.get((ep, frame_idx), np.nan)
            pil_img = _tensor_or_array_to_pil(imgs_b[i])
            all_rows.append(
                {
                    "step": len(all_rows),
                    "episode_index": ep,
                    "frame_index": frame_idx,
                    "timestamp": ts,
                    "value": value,
                    "image": pil_img,
                    **advantage_columns((ep, frame_idx)),
                }
            )
    df = pd.DataFrame(all_rows)
    df = df.sort_values("frame_index").reset_index(drop=True)
    df["step"] = np.arange(len(df))
    return df


def run_app(df: pd.DataFrame, series_label: str = "Value") -> None:
    """Run the Streamlit UI with the prepared DataFrame (step, value, image)."""
    series_label = series_label.strip() or "Value"
    st.set_page_config(page_title=f"VLA {series_label} Visualizer", layout="wide")
    # Header: logo (if file exists) + title
    logo_col, title_col = st.columns([1, 5])
    with logo_col:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=380)
    with title_col:
        st.title(f"VLA {series_label} Analysis")
    st.markdown(f"Synchronize robot states with predicted {series_label} (single episode).")

    st.sidebar.header("Settings")
    smoothing = st.sidebar.slider("Graph Smoothing (Window)", 1, 10, 3)
    if smoothing > 1:
        df = df.copy()
        df["display_value"] = df["value"].rolling(window=smoothing, center=True).mean()
    else:
        df = df.copy()
        df["display_value"] = df["value"]

    df["display_value"] = df["display_value"].ffill().bfill().fillna(0)

    n = len(df)
    if n == 0:
        st.warning("No frames loaded.")
        return

    # Render the timeline in the browser so dragging updates without waiting for a Streamlit rerun.
    current_step = st.session_state.get("timeline_scrubber", 0)
    current_step = max(0, min(current_step, n - 1))
    frames = build_scrubber_frames(df)
    components.html(
        build_value_scrubber_html(
            frames,
            initial_step=current_step,
            series_label=series_label,
        ),
        height=940,
        scrolling=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLA Value Function Visualizer: single episode via LeRobotDataset, values from values.json"
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        required=True,
        help="Path to dataset_config.json (contains root, repo_id, episodes)",
    )
    parser.add_argument(
        "--values",
        type=Path,
        required=True,
        help="Path to values.json (dict: keys (episode_index, frame_index), values: float)",
    )
    parser.add_argument(
        "--series-label",
        type=str,
        default="Value",
        help="Displayed series name, for example 'Raw STEAM Advantage'.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Path to train_config.json or directory containing it (default: same dir as dataset-config)",
    )
    parser.add_argument(
        "--raw-advantages",
        type=Path,
        default=None,
        help="Optional frame-keyed raw_advantages.json shown in frame details.",
    )
    parser.add_argument(
        "--effective-advantages",
        type=Path,
        default=None,
        help="Optional frame-keyed advantages.json shown in frame details.",
    )
    parser.add_argument(
        "--advantage-sources",
        type=Path,
        default=None,
        help="Optional frame-keyed advantage_sources.json shown in frame details.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to load (default: first episode from dataset_config)",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default=None,
        help="Camera key to load (default: camera0)",
    )
    args = parser.parse_args()

    dataset_config_path = args.dataset_config.resolve()
    values_path = args.values.resolve()
    if not dataset_config_path.is_file():
        raise SystemExit(f"Dataset config not found: {dataset_config_path}")
    if not values_path.is_file():
        raise SystemExit(f"Values file not found: {values_path}")

    _, episodes = load_dataset_config(dataset_config_path)
    optional_paths = {
        "raw advantages": args.raw_advantages,
        "effective advantages": args.effective_advantages,
        "advantage sources": args.advantage_sources,
    }
    for label, optional_path in optional_paths.items():
        if optional_path is not None and not optional_path.resolve().is_file():
            raise SystemExit(f"{label.title()} file not found: {optional_path.resolve()}")

    episode_index = args.episode if args.episode is not None else episodes[0]

    df = load_frames_lerobot_cached(
        dataset_config_path,
        args.train_config,
        values_path,
        episode_index,
        args.camera_key,
        raw_advantages_path=args.raw_advantages.resolve() if args.raw_advantages else None,
        effective_advantages_path=args.effective_advantages.resolve() if args.effective_advantages else None,
        advantage_sources_path=args.advantage_sources.resolve() if args.advantage_sources else None,
    )
    run_app(df, series_label=args.series_label)


if __name__ == "__main__":
    main()
