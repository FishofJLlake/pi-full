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

from opentau.scripts.get_advantage_and_percentiles import (
    apply_intervention_override,
)


def test_positive_intervention_overrides_effective_advantage_and_preserves_raw_input():
    raw_advantage = -0.375

    effective, source = apply_intervention_override(raw_advantage, intervention=1)

    assert raw_advantage == -0.375
    assert effective == 1.0
    assert source == "human_intervention_override"


def test_non_intervention_keeps_td_advantage():
    effective, source = apply_intervention_override(-0.375, intervention=0)

    assert effective == -0.375
    assert source == "td"
