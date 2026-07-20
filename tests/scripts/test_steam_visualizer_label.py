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

from opentau.scripts.value_visualizer_frontend import build_value_scrubber_html


def test_visualizer_supports_a_custom_steam_series_label():
    frames = [
        {
            "step": 0,
            "episode_index": 0,
            "timestamp": 0.0,
            "value": -0.2,
            "display_value": -0.2,
            "value_label": "-0.200",
            "display_value_label": "-0.200",
            "timestamp_label": "0.00",
            "image_data_uri": None,
        }
    ]
    html = build_value_scrubber_html(
        frames,
        series_label="Raw STEAM Advantage",
    )
    assert "Raw STEAM Advantage Curve" in html
    assert "Predicted Raw STEAM Advantage" in html
