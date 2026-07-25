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

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from opentau.datasets.utils import (
    ADVANTAGE_REPORT_PATH,
    ADVANTAGE_SOURCES_PATH,
    ADVANTAGES_PATH,
    RAW_ADVANTAGES_PATH,
    AdvantageKey,
    serialize_advantage_key,
)


def _validate_key(key: AdvantageKey) -> AdvantageKey:
    try:
        episode_index, frame_index = key
        serialize_advantage_key(episode_index, frame_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid advantage frame key: {key!r}") from exc
    return episode_index, frame_index


def _validate_bundle(
    processed_keys: Sequence[AdvantageKey],
    advantages: Mapping[AdvantageKey, float],
    raw_advantages: Mapping[AdvantageKey, float],
    advantage_sources: Mapping[AdvantageKey, str],
    key_to_task: Mapping[AdvantageKey, str],
) -> tuple[list[AdvantageKey], dict[str, object]]:
    validated_processed_keys: list[AdvantageKey] = []
    processed_set: set[AdvantageKey] = set()
    for key in processed_keys:
        validated_key = _validate_key(key)
        if validated_key in processed_set:
            raise ValueError(f"Found duplicate processed frame key: {validated_key!r}")
        processed_set.add(validated_key)
        validated_processed_keys.append(validated_key)

    mappings = {
        "advantages": advantages,
        "raw_advantages": raw_advantages,
        "advantage_sources": advantage_sources,
        "key_to_task": key_to_task,
    }
    for name, mapping in mappings.items():
        mapping_keys = {_validate_key(key) for key in mapping}
        if mapping_keys != processed_set:
            missing = sorted(processed_set - mapping_keys)
            unexpected = sorted(mapping_keys - processed_set)
            raise ValueError(
                f"Processed and {name} key sets differ: missing={missing!r}, unexpected={unexpected!r}"
            )

    for name, mapping in (("advantages", advantages), ("raw_advantages", raw_advantages)):
        for key, value in mapping.items():
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} value for {key!r} must be finite; got {value!r}") from exc
            if not finite:
                raise ValueError(f"{name} value for {key!r} must be finite; got {value!r}")

    for key, source in advantage_sources.items():
        if not isinstance(source, str) or not source:
            raise ValueError(f"Advantage source for {key!r} must be a non-empty string; got {source!r}")
    for key, task in key_to_task.items():
        if not isinstance(task, str):
            raise ValueError(f"Task for {key!r} must be a string; got {task!r}")

    task_counts = Counter(key_to_task[key] for key in validated_processed_keys)
    by_task = {
        task: {
            "processed_count": count,
            "written_count": count,
            "coverage": 1.0,
        }
        for task, count in sorted(task_counts.items())
    }
    count = len(validated_processed_keys)
    report: dict[str, object] = {
        "schema_version": 1,
        "key_format": "episode_index,frame_index",
        "processed_count": count,
        "written_count": count,
        "coverage": 1.0,
        "duplicate_count": 0,
        "missing_count": 0,
        "unexpected_count": 0,
        "by_task": by_task,
    }
    return validated_processed_keys, report


def _write_json_temp(final_path: Path, payload: object) -> Path:
    fd, temporary_name = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=4, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def persist_advantage_bundle(
    root: Path,
    processed_keys: Sequence[AdvantageKey],
    advantages: Mapping[AdvantageKey, float],
    raw_advantages: Mapping[AdvantageKey, float],
    advantage_sources: Mapping[AdvantageKey, str],
    key_to_task: Mapping[AdvantageKey, str],
) -> dict[str, object]:
    """Validate and persist a complete set of frame-keyed advantage metadata."""
    validated_keys, report = _validate_bundle(
        processed_keys,
        advantages,
        raw_advantages,
        advantage_sources,
        key_to_task,
    )

    serialized_keys = {key: serialize_advantage_key(*key) for key in validated_keys}
    advantage_payload = {serialized_keys[key]: f"{float(advantages[key]):.6f}" for key in validated_keys}
    raw_advantage_payload = {
        serialized_keys[key]: f"{float(raw_advantages[key]):.6f}" for key in validated_keys
    }
    source_payload = {serialized_keys[key]: advantage_sources[key] for key in validated_keys}

    root = Path(root)
    final_payloads = (
        (root / ADVANTAGES_PATH, advantage_payload),
        (root / RAW_ADVANTAGES_PATH, raw_advantage_payload),
        (root / ADVANTAGE_SOURCES_PATH, source_payload),
        (root / ADVANTAGE_REPORT_PATH, report),
    )
    final_payloads[0][0].parent.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path] = {}
    try:
        for final_path, payload in final_payloads:
            staged.append((final_path, _write_json_temp(final_path, payload)))

        originally_present = {final_path for final_path, _ in final_payloads if final_path.exists()}
        try:
            for final_path, _ in final_payloads:
                if final_path not in originally_present:
                    continue
                fd, backup_name = tempfile.mkstemp(
                    dir=final_path.parent,
                    prefix=f".{final_path.name}.",
                    suffix=".backup",
                )
                os.close(fd)
                backup_path = Path(backup_name)
                try:
                    final_path.replace(backup_path)
                except BaseException:
                    backup_path.unlink(missing_ok=True)
                    raise
                backups[final_path] = backup_path

            for final_path, temporary_path in staged:
                temporary_path.replace(final_path)
        except BaseException as transaction_error:
            rollback_errors = []
            for final_path, backup_path in reversed(backups.items()):
                try:
                    backup_path.replace(final_path)
                except BaseException as rollback_error:
                    rollback_errors.append((final_path, rollback_error))
            for final_path, _ in final_payloads:
                if final_path not in originally_present:
                    try:
                        final_path.unlink(missing_ok=True)
                    except BaseException as rollback_error:
                        rollback_errors.append((final_path, rollback_error))
            if rollback_errors:
                details = ", ".join(f"{path}: {error}" for path, error in rollback_errors)
                raise RuntimeError(f"Advantage bundle rollback failed: {details}") from transaction_error
            raise
        else:
            for backup_path in backups.values():
                backup_path.unlink(missing_ok=True)
    finally:
        for _, temporary_path in staged:
            temporary_path.unlink(missing_ok=True)

    return report
