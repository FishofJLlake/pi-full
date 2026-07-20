r"python src/opentau/scripts/get_advantage_and_percentiles.py  \
--config_path=outputs/train/2025-11-29/00-38-59_value/checkpoints/00520000 \
--batch_size=20 \
--dataloader_batch_size=20 \
--dataset_mixture=examples/advantage_config.json"

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
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader

from opentau.configs import parser
from opentau.configs.default import DatasetMixtureConfig
from opentau.configs.refs import resolve_refs_to_tempfile
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.advantage_metadata import persist_advantage_bundle
from opentau.datasets.factory import make_dataset
from opentau.policies.factory import get_policy_class
from opentau.policies.value.configuration_value import ValueConfig
from opentau.policies.value.reward import calculate_n_step_return
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import (
    auto_torch_device,
    init_logging,
)


def ensure_primitive(maybe_tensor):
    """Convert single-element tensors/arrays to Python scalars so they can be used as stable dict keys."""
    if isinstance(maybe_tensor, np.ndarray):
        return ensure_primitive(torch.from_numpy(maybe_tensor))
    if isinstance(maybe_tensor, torch.Tensor):
        assert maybe_tensor.numel() == 1, f"Tensor must be a single value, got shape={maybe_tensor.numel()}"
        return maybe_tensor.item()
    return maybe_tensor


_default0 = defaultdict(int)
POSITIVE_ADVANTAGE_OVERRIDE = 1.0
SOURCE_TD = "td"
SOURCE_HUMAN_INTERVENTION_OVERRIDE = "human_intervention_override"


def apply_intervention_override(raw_advantage: float, intervention: float) -> tuple[float, str]:
    """Return the effective advantage without conflating failure and intervention."""
    if intervention > 0:
        return POSITIVE_ADVANTAGE_OVERRIDE, SOURCE_HUMAN_INTERVENTION_OVERRIDE
    return raw_advantage, SOURCE_TD


def _synchronize_device_for_timing(device):
    """Synchronize accelerator work before reading a wall-clock timestamp."""
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _timed_now(device=None):
    if device is not None:
        _synchronize_device_for_timing(device)
    return time.perf_counter()


def _elapsed_since(start, device=None):
    if device is not None:
        _synchronize_device_for_timing(device)
    return time.perf_counter() - start


def _safe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return "unknown"


def _get_batch_size(batch):
    if "current_idx" in batch:
        return len(batch["current_idx"])
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return value.shape[0]
        try:
            return len(value)
        except TypeError:
            continue
    return 0


def _dataset_log_summary(dataset):
    meta = getattr(dataset, "meta", None)
    return {
        "type": type(dataset).__name__,
        "length": _safe_len(dataset),
        "root": getattr(dataset, "root", None),
        "repo_id": getattr(dataset, "repo_id", None),
        "episodes": getattr(meta, "total_episodes", None) if meta is not None else None,
    }

# Store dataset_mixture_path before filtering (needed for parsing inside main)
# Handle both --dataset_mixture_path=<path> and --dataset_mixture=<path> (without nested fields)
_dataset_mixture_path_value = None
for arg in sys.argv:
    if arg.startswith("--dataset_mixture_path="):
        _dataset_mixture_path_value = arg.split("=", 1)[1]
        break
    elif arg.startswith("--dataset_mixture=") and "." not in arg.split("=", 1)[0]:
        # --dataset_mixture=<path> without nested fields (e.g., not --dataset_mixture.datasets.0.repo_id=...)
        _dataset_mixture_path_value = arg.split("=", 1)[1]
        break

# Create a wrapper that filters dataset_mixture path arguments before draccus parsing
_original_wrap = parser.wrap()


def _filter_dataset_mixture_path(fn):
    """Filter dataset mixture path args from sys.argv before draccus sees them."""
    wrapped_fn = _original_wrap(fn)

    def filtered_wrapper(*args, **kwargs):
        # If config is already provided, just call the function
        if len(args) > 0:
            return wrapped_fn(*args, **kwargs)

        # Otherwise, filter dataset_mixture path arguments from sys.argv before draccus parses
        original_argv = sys.argv.copy()
        try:
            filtered_args = []
            for arg in sys.argv:
                # Filter --dataset_mixture_path=<path>
                if (
                    arg.startswith("--dataset_mixture_path=")
                    or arg.startswith("--dataset_mixture=")
                    and "." not in arg.split("=", 1)[0]
                ):
                    continue
                else:
                    filtered_args.append(arg)
            sys.argv = filtered_args
            return wrapped_fn(*args, **kwargs)
        finally:
            sys.argv = original_argv

    return filtered_wrapper


@_filter_dataset_mixture_path
def main(cfg: TrainPipelineConfig):
    dataset_mixture_path = _dataset_mixture_path_value

    if not isinstance(cfg.policy, ValueConfig):
        raise ValueError(
            "get_advantage_and_percentiles requires policy.type='value'; "
            f"got {cfg.policy.type!r}"
        )
    if cfg.policy.pretrained_path is None:
        raise ValueError("policy.pretrained_path must name the Value checkpoint")

    if dataset_mixture_path:
        logging.info(f"Loading dataset config from separate file: {dataset_mixture_path}")
        tmp_mixture = resolve_refs_to_tempfile(dataset_mixture_path)
        try:
            mixture_cfg = draccus.parse(
                config_class=DatasetMixtureConfig, config_path=str(tmp_mixture), args=[]
            )
        finally:
            tmp_mixture.unlink(missing_ok=True)
    else:
        logging.info("Using the dataset mixture config from the TrainPipelineConfig")
        mixture_cfg = cfg.dataset_mixture

    script_start_time = time.perf_counter()
    device = auto_torch_device()
    logging.info(
        "Advantage script config: device=%s, batch_size=%s, dataloader_batch_size=%s, "
        "num_workers=%s, prefetch_factor=%s, pin_memory=%s, datasets=%s",
        device,
        cfg.batch_size,
        cfg.dataloader_batch_size,
        cfg.num_workers,
        cfg.prefetch_factor,
        torch.cuda.is_available(),
        len(mixture_cfg.datasets),
    )
    # torch.autograd.set_detect_anomaly(True)

    # TODO(shuheng): Do we need the random seed here?
    if cfg.seed is not None:
        set_seed(cfg.seed)

    logging.info("Creating policy")
    policy_load_start = time.perf_counter()
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(
        cfg.policy.pretrained_path,
        config=cfg.policy,
        local_files_only=True,
        backbone_local_files_only=True,
    )
    policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()
    logging.info(
        "Policy ready: type=%s, load_and_to_device=%.3fs",
        cfg.policy.type,
        time.perf_counter() - policy_load_start,
    )

    # Effective advantages are used for policy conditioning; raw advantages
    # keep the TD residual before overrides.
    advantages = []
    raw_advantages = []

    for dataset_idx, dataset_cfg in enumerate(mixture_cfg.datasets):
        dataset_start_time = time.perf_counter()
        logging.info("Creating dataset %s: cfg=%s", dataset_idx, dataset_cfg)
        ds_res = make_dataset(dataset_cfg, cfg, return_advantage_input=True)
        dataset = ds_res[0] if isinstance(ds_res, tuple) else ds_res
        dataset_summary = _dataset_log_summary(dataset)
        logging.info(
            "Dataset %s ready: summary=%s, create_time=%.3fs",
            dataset_idx,
            dataset_summary,
            time.perf_counter() - dataset_start_time,
        )
        dataloader_start_time = time.perf_counter()
        dataloader_kwargs = {
            "batch_size": cfg.batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": cfg.num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": cfg.num_workers > 0,
        }
        if cfg.num_workers > 0 and cfg.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        logging.info(
            "Dataloader %s ready: batch_size=%s, num_workers=%s, prefetch_factor=%s, "
            "pin_memory=%s, persistent_workers=%s, create_time=%.3fs",
            dataset_idx,
            cfg.batch_size,
            cfg.num_workers,
            cfg.prefetch_factor,
            torch.cuda.is_available(),
            cfg.num_workers > 0,
            time.perf_counter() - dataloader_start_time,
        )

        values = {}
        advantage_records = []
        ds_advantage = {}  # per-dataset advantages
        ds_raw_advantage = {}
        ds_advantage_source = {}
        processed_keys = []
        key_to_task = {}
        log_every = max(1, cfg.log_freq or 1)
        with torch.inference_mode():
            # First pass to get the values
            first_pass_completed_episodes = set()
            first_pass_start = time.perf_counter()
            first_pass_wait_start = time.perf_counter()
            first_pass_batches = 0
            first_pass_samples = 0
            first_pass_data_wait_s = 0.0
            first_pass_h2d_s = 0.0
            first_pass_inference_s = 0.0
            first_pass_postprocess_s = 0.0
            for batch_idx, batch in enumerate(dataloader):
                data_wait_s = time.perf_counter() - first_pass_wait_start
                batch_size = _get_batch_size(batch)
                first_pass_batches += 1
                first_pass_samples += batch_size
                first_pass_data_wait_s += data_wait_s

                h2d_start = _timed_now(device)
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
                h2d_s = _elapsed_since(h2d_start, device)
                first_pass_h2d_s += h2d_s

                if "reward_normalizer" in batch:
                    reward_normalizers = batch["reward_normalizer"]
                else:
                    reward_normalizers = [
                        cfg.policy.reward_config.reward_normalizer
                    ] * len(batch["current_idx"])
                if "intervention" in batch:
                    interventions = batch["intervention"]
                else:
                    interventions = [0] * len(batch["current_idx"])

                inference_start = _timed_now(device)
                predicted_values = policy.predict_value(batch)
                inference_s = _elapsed_since(inference_start, device)
                first_pass_inference_s += inference_s

                postprocess_start = time.perf_counter()
                for sample_idx, (
                    success,
                    episode_index,
                    episode_end_idx,
                    current_idx,
                    reward_normalizer,
                    v0,
                    timestamp,
                    intervention,
                ) in enumerate(
                    zip(
                        batch["success"],
                        batch["episode_index"],
                        batch["episode_end_idx"],
                        batch["current_idx"],
                        reward_normalizers,
                        predicted_values,
                        batch["timestamp"],
                        interventions,
                        strict=True,
                    )
                ):
                    (
                        success,
                        episode_index,
                        episode_end_idx,
                        current_idx,
                        v0,
                        reward_normalizer,
                        timestamp,
                        intervention,
                    ) = map(
                        ensure_primitive,
                        (
                            success,
                            episode_index,
                            episode_end_idx,
                            current_idx,
                            v0,
                            reward_normalizer,
                            timestamp,
                            intervention,
                        ),
                    )
                    episode_index = int(episode_index)
                    current_idx = int(current_idx)
                    frame_index = int(batch["frame_index"][sample_idx])
                    task = batch["prompt"][sample_idx]
                    key = (episode_index, frame_index)
                    processed_keys.append(key)
                    key_to_task[key] = task
                    reward = calculate_n_step_return(
                        success=success,
                        n_steps_look_ahead=cfg.policy.reward_config.N_steps_look_ahead,
                        episode_end_idx=episode_end_idx,
                        reward_normalizer=reward_normalizer,
                        current_idx=current_idx,
                        c_neg=cfg.policy.reward_config.C_neg,
                    )

                    values[(episode_index, current_idx)] = {"v0": v0, "reward": reward}
                    advantage_records.append((episode_index, current_idx, key, timestamp, intervention))
                    if current_idx == episode_end_idx and episode_index not in first_pass_completed_episodes:
                        first_pass_completed_episodes.add(episode_index)
                        logging.info(
                            f"[trajectory value-inference done] dataset={dataset_idx}, "
                            f"episode={episode_index}, terminal_idx={episode_end_idx}"
                        )
                postprocess_s = time.perf_counter() - postprocess_start
                first_pass_postprocess_s += postprocess_s

                if batch_idx == 0 or first_pass_batches % log_every == 0:
                    elapsed_s = time.perf_counter() - first_pass_start
                    logging.info(
                        "[advantage first_pass] dataset=%s batch=%s samples=%s elapsed=%.3fs "
                        "samples_per_sec=%.2f data_wait=%.3fs h2d=%.3fs inference=%.3fs postprocess=%.3fs "
                        "totals(data_wait=%.3fs,h2d=%.3fs,inference=%.3fs,postprocess=%.3fs)",
                        dataset_idx,
                        first_pass_batches,
                        first_pass_samples,
                        elapsed_s,
                        first_pass_samples / elapsed_s if elapsed_s > 0 else 0.0,
                        data_wait_s,
                        h2d_s,
                        inference_s,
                        postprocess_s,
                        first_pass_data_wait_s,
                        first_pass_h2d_s,
                        first_pass_inference_s,
                        first_pass_postprocess_s,
                    )
                first_pass_wait_start = time.perf_counter()

            first_pass_total_s = time.perf_counter() - first_pass_start
            logging.info(
                "[advantage first_pass done] dataset=%s batches=%s samples=%s values=%s elapsed=%.3fs "
                "samples_per_sec=%.2f totals(data_wait=%.3fs,h2d=%.3fs,inference=%.3fs,postprocess=%.3fs)",
                dataset_idx,
                first_pass_batches,
                first_pass_samples,
                len(values),
                first_pass_total_s,
                first_pass_samples / first_pass_total_s if first_pass_total_s > 0 else 0.0,
                first_pass_data_wait_s,
                first_pass_h2d_s,
                first_pass_inference_s,
                first_pass_postprocess_s,
            )

            # Track progress per trajectory (episode): print once when each trajectory is fully processed.
            episode_remaining = defaultdict(int)
            for episode_index, _, _, _, _ in advantage_records:
                episode_remaining[episode_index] += 1
            episode_lengths = dict(episode_remaining)
            total_episodes = len(episode_remaining)
            completed_episodes = 0
            # Second pass uses cached records so it does not decode or run the
            # value model again.
            second_pass_start = time.perf_counter()
            second_pass_batches = 0
            second_pass_samples = 0
            second_pass_data_wait_s = 0.0
            second_pass_postprocess_s = 0.0
            cached_batch_size = max(1, cfg.batch_size)
            for batch_idx, record_start in enumerate(range(0, len(advantage_records), cached_batch_size)):
                record_batch = advantage_records[record_start : record_start + cached_batch_size]
                data_wait_s = 0.0
                batch_size = len(record_batch)
                second_pass_batches += 1
                second_pass_samples += batch_size

                postprocess_start = time.perf_counter()
                for episode_index, current_idx, key, timestamp, intervention in record_batch:
                    # check if the value for the next n_steps_look_ahead steps is available, else set it to 0
                    look_ahead_idx = current_idx + cfg.policy.reward_config.N_steps_look_ahead
                    vn = values.get((episode_index, look_ahead_idx), _default0)["v0"]
                    reward = values.get((episode_index, current_idx), _default0)["reward"]
                    v0 = values.get((episode_index, current_idx), _default0)["v0"]
                    raw_advantage = ensure_primitive(reward + vn - v0)
                    advantage, advantage_source = apply_intervention_override(raw_advantage, intervention)
                    raw_advantages.append(raw_advantage)
                    advantages.append(advantage)
                    if len(advantages) < 5:
                        logging.info(
                            "Debug: reward=%s, vn=%s, v0=%s, raw_advantage=%s, "
                            "intervention=%s, advantage=%s, source=%s",
                            reward,
                            vn,
                            v0,
                            raw_advantage,
                            intervention,
                            advantage,
                            advantage_source,
                        )
                    ds_advantage[key] = advantage
                    ds_raw_advantage[key] = raw_advantage
                    ds_advantage_source[key] = advantage_source
                    episode_remaining[episode_index] -= 1
                    if episode_remaining[episode_index] == 0:
                        completed_episodes += 1
                        logging.info(
                            f"[trajectory done] dataset={dataset_idx}, episode={episode_index}, "
                            f"steps={episode_lengths[episode_index]}, "
                            f"progress={completed_episodes}/{total_episodes}"
                        )
                postprocess_s = time.perf_counter() - postprocess_start
                second_pass_postprocess_s += postprocess_s

                if batch_idx == 0 or second_pass_batches % log_every == 0:
                    elapsed_s = time.perf_counter() - second_pass_start
                    logging.info(
                        "[advantage second_pass] dataset=%s batch=%s samples=%s elapsed=%.3fs "
                        "samples_per_sec=%.2f data_wait=%.3fs postprocess=%.3fs "
                        "totals(data_wait=%.3fs,postprocess=%.3fs)",
                        dataset_idx,
                        second_pass_batches,
                        second_pass_samples,
                        elapsed_s,
                        second_pass_samples / elapsed_s if elapsed_s > 0 else 0.0,
                        data_wait_s,
                        postprocess_s,
                        second_pass_data_wait_s,
                        second_pass_postprocess_s,
                    )

            second_pass_total_s = time.perf_counter() - second_pass_start
            logging.info(
                "[advantage second_pass done] dataset=%s batches=%s samples=%s advantages=%s episodes=%s "
                "elapsed=%.3fs samples_per_sec=%.2f totals(data_wait=%.3fs,postprocess=%.3fs)",
                dataset_idx,
                second_pass_batches,
                second_pass_samples,
                len(ds_advantage),
                total_episodes,
                second_pass_total_s,
                second_pass_samples / second_pass_total_s if second_pass_total_s > 0 else 0.0,
                second_pass_data_wait_s,
                second_pass_postprocess_s,
            )

        json_start = time.perf_counter()
        dataset_root = Path(dataset.root)
        report = persist_advantage_bundle(
            dataset_root,
            processed_keys=processed_keys,
            advantages=ds_advantage,
            raw_advantages=ds_raw_advantage,
            advantage_sources=ds_advantage_source,
            key_to_task=key_to_task,
        )
        if report["coverage"] < 1.0:
            raise ValueError(f"Advantage coverage must be 1.0; got {report['coverage']!r}")
        logging.info(
            "[advantage bundle_write done] dataset=%s root=%s records=%s coverage=%.6f elapsed=%.3fs",
            dataset_idx,
            dataset_root,
            report["written_count"],
            report["coverage"],
            time.perf_counter() - json_start,
        )

    # Calculate percentiles of advantages: 0th, 5th, 10th, ..., 100th
    percentile_start = time.perf_counter()
    percentiles = list(range(0, 101, 5))  # [0, 5, 10, 15, ..., 100]
    advantage_percentiles = np.percentile(np.array(advantages), percentiles)
    raw_advantage_percentiles = np.percentile(np.array(raw_advantages), percentiles)
    logging.info(
        "[advantage percentile done] records=%s raw_records=%s percentiles=%s "
        "elapsed=%.3fs total_elapsed=%.3fs",
        len(advantages),
        len(raw_advantages),
        len(percentiles),
        time.perf_counter() - percentile_start,
        time.perf_counter() - script_start_time,
    )

    print("Effective advantage percentiles for policy conditioning:")
    for p, val in zip(percentiles, advantage_percentiles, strict=False):
        print(f"  {p:03d}th percentile: {val:.6f}")
    print("Raw TD advantage percentiles for deciding epsilon threshold:")
    for p, val in zip(percentiles, raw_advantage_percentiles, strict=False):
        print(f"  {p:03d}th percentile: {val:.6f}")


if __name__ == "__main__":
    init_logging()
    main()
