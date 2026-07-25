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
"""Serialization helpers for value outputs keyed by episode and frame index."""

from __future__ import annotations

import json
import math
import operator
from pathlib import Path

ValueKey = tuple[int, int]


def serialize_value_key(episode_index: int, frame_index: int) -> str:
    """Serialize a canonical non-negative ``(episode_index, frame_index)`` key."""
    episode_index = operator.index(episode_index)
    frame_index = operator.index(frame_index)
    if episode_index < 0 or frame_index < 0:
        raise ValueError("Value indices must be non-negative")
    return f"{episode_index},{frame_index}"


def _parse_value_key(serialized: str) -> ValueKey:
    parts = serialized.split(",")
    if len(parts) != 2 or any(not part.isdecimal() for part in parts):
        raise ValueError(
            f"Expected integer frame key 'episode_index,frame_index'; got {serialized!r}. "
            "Regenerate timestamp-keyed value files."
        )

    key = (int(parts[0]), int(parts[1]))
    if serialize_value_key(*key) != serialized:
        raise ValueError(f"Noncanonical integer frame key: {serialized!r}")
    return key


def load_values(path: Path) -> dict[ValueKey, float]:
    """Load finite values keyed by canonical integer frame keys from JSON."""

    class JSONObjectPairs(list):
        pass

    with open(path, encoding="utf-8") as f:
        serialized_values = json.loads(f.read(), object_pairs_hook=JSONObjectPairs)

    if not isinstance(serialized_values, JSONObjectPairs):
        raise ValueError(
            "Expected values file to contain a JSON object keyed by canonical integer frame keys; "
            f"got {type(serialized_values).__name__}."
        )

    values: dict[ValueKey, float] = {}
    serialized_keys: set[str] = set()
    for serialized, value in serialized_values:
        if serialized in serialized_keys:
            raise ValueError(f"Duplicate value key: {serialized!r}")
        serialized_keys.add(serialized)

        key = _parse_value_key(serialized)
        try:
            parsed_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Value for {serialized!r} must be finite; got {value!r}") from exc
        if not math.isfinite(parsed_value):
            raise ValueError(f"Value for {serialized!r} must be finite; got {value!r}")
        values[key] = parsed_value

    return values


def load_value_labels(path: Path) -> dict[ValueKey, str]:
    """Load string labels keyed by the same strict frame-index contract."""

    class JSONObjectPairs(list):
        pass

    with open(path, encoding="utf-8") as stream:
        serialized_labels = json.loads(stream.read(), object_pairs_hook=JSONObjectPairs)
    if not isinstance(serialized_labels, JSONObjectPairs):
        raise ValueError(
            "Expected label file to contain a JSON object keyed by canonical integer frame keys."
        )

    labels: dict[ValueKey, str] = {}
    serialized_keys: set[str] = set()
    for serialized, value in serialized_labels:
        if serialized in serialized_keys:
            raise ValueError(f"Duplicate value key: {serialized!r}")
        serialized_keys.add(serialized)
        key = _parse_value_key(serialized)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Label for {serialized!r} must be a non-empty string; got {value!r}")
        labels[key] = value
    return labels
