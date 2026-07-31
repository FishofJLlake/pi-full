#!/usr/bin/env python
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

"""Merge independent STEAM checkpoints into one inference ensemble."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file, save_file

from opentau.configs.train import TrainPipelineConfig
from opentau.policies.steam.configuration_steam import SteamConfig


@dataclass(frozen=True)
class MemberSpec:
    checkpoint: Path
    member_index: int | None


def parse_member_spec(raw: str) -> MemberSpec:
    """Parse ``PATH`` or RLinf-compatible ``PATH:member_index``."""
    direct = Path(raw).expanduser()
    if direct.exists():
        return MemberSpec(direct.resolve(), None)
    if ":" in raw:
        maybe_path, maybe_index = raw.rsplit(":", 1)
        if maybe_index.isdigit() and Path(maybe_path).expanduser().exists():
            return MemberSpec(Path(maybe_path).expanduser().resolve(), int(maybe_index))
    return MemberSpec(direct.resolve(), None)


def _load_source(spec: MemberSpec) -> tuple[TrainPipelineConfig, dict[str, object]]:
    cfg = TrainPipelineConfig.from_pretrained(spec.checkpoint, local_files_only=True)
    if not isinstance(cfg.policy, SteamConfig):
        raise TypeError(f"Checkpoint {spec.checkpoint} is not a STEAM checkpoint.")
    weights_path = spec.checkpoint / SAFETENSORS_SINGLE_FILE
    if not weights_path.is_file():
        raise FileNotFoundError(f"STEAM weights not found: {weights_path}")
    state = load_file(str(weights_path), device="cpu")
    ensemble_keys = any(key.startswith("members.") for key in state)
    if ensemble_keys:
        if spec.member_index is None:
            raise ValueError(
                f"{spec.checkpoint} is already an ensemble; select one member as "
                f"'{spec.checkpoint}:member_index'."
            )
        prefix = f"members.{spec.member_index}."
        extracted = {
            key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
        }
        if not extracted:
            raise ValueError(
                f"Member {spec.member_index} does not exist in ensemble checkpoint {spec.checkpoint}."
            )
        return cfg, extracted
    if spec.member_index not in (None, 0):
        raise ValueError(f"Single-member checkpoint {spec.checkpoint} only supports member index 0.")
    return cfg, state


def _compatibility_signature(config: SteamConfig) -> tuple:
    return (
        config.num_bins,
        config.max_temporal_offset,
        config.fusion_hidden_dim,
        config.vision_pretrained_path,
        config.language_pretrained_path,
        config.tokenizer_path,
        tuple(config.image_resolution),
    )


def merge_checkpoints(member_specs: list[str], output: Path) -> Path:
    """Write a ``members.N.*`` safetensors checkpoint and provenance manifest."""
    if not member_specs:
        raise ValueError("At least one --member is required.")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")

    sources: list[tuple[MemberSpec, TrainPipelineConfig, dict[str, object]]] = []
    reference_keys: set[str] | None = None
    signature = None
    for raw in member_specs:
        spec = parse_member_spec(raw)
        cfg, state = _load_source(spec)
        current_signature = _compatibility_signature(cfg.policy)
        if signature is not None and current_signature != signature:
            raise ValueError(f"Incompatible STEAM member checkpoint: {spec.checkpoint}")
        signature = current_signature
        sources.append((spec, cfg, state))
        current_keys = set(state)
        if reference_keys is not None and current_keys != reference_keys:
            raise ValueError(
                f"STEAM member state keys differ for {spec.checkpoint}: "
                f"missing={sorted(reference_keys - current_keys)[:10]}, "
                f"unexpected={sorted(current_keys - reference_keys)[:10]}."
            )
        reference_keys = current_keys

    output.mkdir(parents=True)
    merged_state = {
        f"members.{member_index}.{key}": value
        for member_index, (_spec, _cfg, state) in enumerate(sources)
        for key, value in state.items()
    }
    save_file(merged_state, str(output / SAFETENSORS_SINGLE_FILE))

    merged_cfg = sources[0][1]
    merged_cfg.policy.ensemble_size = len(sources)
    merged_cfg.policy.pretrained_path = output
    merged_cfg.save_pretrained(output)
    manifest = {
        "schema_version": 1,
        "ensemble_size": len(sources),
        "members": [
            {
                "output_member_index": index,
                "checkpoint": str(spec.checkpoint),
                "source_member_index": spec.member_index,
            }
            for index, (spec, _cfg, _state) in enumerate(sources)
        ],
    }
    with open(output / "merge_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="Checkpoint PATH or PATH:member_index. Repeat in output member order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cli() -> None:
    args = _parse_args()
    output = merge_checkpoints(args.member, args.output)
    print(f"Wrote merged STEAM ensemble checkpoint: {output}")


if __name__ == "__main__":
    cli()
