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
from pathlib import Path

import pytest

from opentau.datasets.advantage_metadata import persist_advantage_bundle
from opentau.datasets.utils import load_advantages, validate_advantage_bundle_files

FINAL_FILENAMES = (
    "advantages.json",
    "raw_advantages.json",
    "advantage_sources.json",
    "advantage_report.json",
)


def _valid_bundle():
    keys = [(0, 0), (0, 1)]
    return {
        "processed_keys": keys,
        "advantages": {keys[0]: 0.1, keys[1]: -0.25},
        "raw_advantages": {keys[0]: 0.2, keys[1]: -0.5},
        "advantage_sources": {keys[0]: "td", keys[1]: "human_intervention_override"},
        "key_to_task": {keys[0]: "pick", keys[1]: "pick"},
    }


def _write_previous_files(root):
    meta = root / "meta"
    meta.mkdir()
    previous = {}
    for filename in FINAL_FILENAMES:
        path = meta / filename
        content = f"previous-{filename}"
        path.write_text(content, encoding="utf-8")
        previous[path] = content
    return previous


def _assert_previous_files_unchanged(previous):
    assert {path: path.read_text(encoding="utf-8") for path in previous} == previous


def test_bundle_rejects_duplicate_processed_frames_before_writing(tmp_path):
    previous = _write_previous_files(tmp_path)
    bundle = _valid_bundle()
    bundle["processed_keys"] = [(0, 0), (0, 0), (0, 1)]

    with pytest.raises(ValueError, match="duplicate processed frame key"):
        persist_advantage_bundle(tmp_path, **bundle)

    _assert_previous_files_unchanged(previous)


@pytest.mark.parametrize(
    "mapping_name",
    ["advantages", "raw_advantages", "advantage_sources", "key_to_task"],
)
def test_bundle_rejects_key_mismatch_before_writing(tmp_path, mapping_name):
    previous = _write_previous_files(tmp_path)
    bundle = _valid_bundle()
    del bundle[mapping_name][(0, 1)]

    with pytest.raises(ValueError, match="key sets differ"):
        persist_advantage_bundle(tmp_path, **bundle)

    _assert_previous_files_unchanged(previous)


def test_bundle_rejects_non_string_sources_before_writing(tmp_path):
    previous = _write_previous_files(tmp_path)
    bundle = _valid_bundle()
    bundle["advantage_sources"][(0, 0)] = 1

    with pytest.raises(ValueError, match="source.*must be a string"):
        persist_advantage_bundle(tmp_path, **bundle)

    _assert_previous_files_unchanged(previous)


@pytest.mark.parametrize("mapping_name", ["advantages", "raw_advantages"])
@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_bundle_rejects_non_finite_values_before_writing(tmp_path, mapping_name, value):
    previous = _write_previous_files(tmp_path)
    bundle = _valid_bundle()
    bundle[mapping_name][(0, 0)] = value

    with pytest.raises(ValueError, match="must be finite"):
        persist_advantage_bundle(tmp_path, **bundle)

    _assert_previous_files_unchanged(previous)


def test_bundle_writes_complete_files_and_report(tmp_path):
    bundle = _valid_bundle()
    report = persist_advantage_bundle(tmp_path, **bundle)

    meta = tmp_path / "meta"
    assert json.loads((meta / "advantages.json").read_text(encoding="utf-8")) == {
        "0,0": "0.100000",
        "0,1": "-0.250000",
    }
    assert json.loads((meta / "raw_advantages.json").read_text(encoding="utf-8")) == {
        "0,0": "0.200000",
        "0,1": "-0.500000",
    }
    assert json.loads((meta / "advantage_sources.json").read_text(encoding="utf-8")) == {
        "0,0": "td",
        "0,1": "human_intervention_override",
    }
    assert report == {
        "schema_version": 1,
        "key_format": "episode_index,frame_index",
        "processed_count": 2,
        "written_count": 2,
        "coverage": 1.0,
        "duplicate_count": 0,
        "missing_count": 0,
        "unexpected_count": 0,
        "by_task": {"pick": {"processed_count": 2, "written_count": 2, "coverage": 1.0}},
    }
    assert json.loads((meta / "advantage_report.json").read_text(encoding="utf-8")) == report
    assert not list(meta.glob("*.tmp"))


def test_persisted_bundle_passes_runtime_validation(tmp_path):
    persist_advantage_bundle(tmp_path, **_valid_bundle())

    validate_advantage_bundle_files(tmp_path, load_advantages(tmp_path))


def test_runtime_validation_rejects_companion_key_mismatch(tmp_path):
    persist_advantage_bundle(tmp_path, **_valid_bundle())
    raw_path = tmp_path / "meta" / "raw_advantages.json"
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    del raw_payload["0,1"]
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="key sets differ"):
        validate_advantage_bundle_files(tmp_path, load_advantages(tmp_path))


def test_runtime_validation_requires_complete_bundle(tmp_path):
    persist_advantage_bundle(tmp_path, **_valid_bundle())
    (tmp_path / "meta" / "advantage_sources.json").unlink()

    with pytest.raises(ValueError, match="complete four-file bundle"):
        validate_advantage_bundle_files(tmp_path, load_advantages(tmp_path))


def test_bundle_restores_all_existing_files_when_final_replace_fails(tmp_path, monkeypatch):
    previous = _write_previous_files(tmp_path)
    original_replace = Path.replace
    final_replace_count = 0

    def fail_third_final_replace(source, target):
        nonlocal final_replace_count
        target = Path(target)
        if source.name.endswith(".tmp") and target.name in FINAL_FILENAMES:
            final_replace_count += 1
            if final_replace_count == 3:
                raise OSError("injected final replacement failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_third_final_replace)

    with pytest.raises(OSError, match="injected final replacement failure"):
        persist_advantage_bundle(tmp_path, **_valid_bundle())

    _assert_previous_files_unchanged(previous)
    assert {path.name for path in (tmp_path / "meta").iterdir()} == set(FINAL_FILENAMES)


def test_bundle_removes_partial_new_files_when_final_replace_fails(tmp_path, monkeypatch):
    original_replace = Path.replace
    final_replace_count = 0

    def fail_second_final_replace(source, target):
        nonlocal final_replace_count
        target = Path(target)
        if source.name.endswith(".tmp") and target.name in FINAL_FILENAMES:
            final_replace_count += 1
            if final_replace_count == 2:
                raise OSError("injected final replacement failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_final_replace)

    with pytest.raises(OSError, match="injected final replacement failure"):
        persist_advantage_bundle(tmp_path, **_valid_bundle())

    assert not list((tmp_path / "meta").iterdir())


def test_generation_uses_frame_identity_and_bundle_persistence():
    script = Path("src/opentau/scripts/get_advantage_and_percentiles.py").read_text(encoding="utf-8")

    assert "key = (episode_index, frame_index)" in script
    assert "processed_keys.append(key)" in script
    assert 'task = batch["prompt"][sample_idx]' in script
    assert 'task = batch["task"][sample_idx]' not in script
    assert "persist_advantage_bundle(" in script
    assert "json.dump(" not in script
