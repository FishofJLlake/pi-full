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

from types import SimpleNamespace

import pytest
import torch

from opentau.datasets.lerobot_dataset import BaseDataset, LeRobotDataset


def test_intervention_reads_only_an_explicitly_mapped_frame_field():
    mapped = SimpleNamespace(_get_name_map=lambda strict=False: {"intervention": "human_intervention"})
    item = {
        "human_intervention": torch.tensor(0.25),
        "mistake": torch.tensor(True),
        "success": torch.tensor(False),
    }

    LeRobotDataset._attach_intervention_raw(mapped, item)

    assert item["intervention_raw"] == pytest.approx(0.25)


def test_intervention_is_not_inferred_from_mistake_or_failure():
    unmapped = SimpleNamespace(_get_name_map=lambda strict=False: {})
    item = {"mistake": torch.tensor(True), "success": torch.tensor(False)}

    LeRobotDataset._attach_intervention_raw(unmapped, item)

    assert "intervention_raw" not in item


def test_missing_intervention_emits_false_with_pad_mask():
    output = {}

    BaseDataset._emit_intervention({}, output)

    assert output["intervention"].item() is False
    assert output["intervention_is_pad"].item() is True


def test_present_intervention_emits_value_without_pad_mask():
    output = {}

    BaseDataset._emit_intervention({"intervention_raw": 1}, output)

    assert output["intervention"].item() is True
    assert output["intervention_is_pad"].item() is False


@pytest.mark.parametrize("raw_value", [0, -1, -0.5])
def test_nonpositive_intervention_does_not_trigger_override(raw_value):
    output = {}

    BaseDataset._emit_intervention(
        {"intervention_raw": raw_value},
        output,
    )

    assert output["intervention"].item() is False
    assert output["intervention_is_pad"].item() is False
