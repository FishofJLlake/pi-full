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

import unittest
from importlib import util
from pathlib import Path


def load_frontend_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "src" / "opentau" / "scripts" / "value_visualizer_frontend.py"
    spec = util.spec_from_file_location("value_visualizer_frontend", module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeImage:
    def convert(self, mode):
        self.mode = mode
        return self

    def save(self, output, format=None, quality=None, optimize=None):
        output.write(b"fake-image-bytes")


class FakeDataFrame:
    def __init__(self, rows):
        self._rows = rows

    def iterrows(self):
        yield from enumerate(self._rows)

    def __len__(self):
        return len(self._rows)


class ValueVisualizerFrontendTest(unittest.TestCase):
    def test_build_scrubber_frames_serializes_images_and_metrics(self):
        frontend = load_frontend_module()

        frames = frontend.build_scrubber_frames(
            FakeDataFrame(
                [
                    {
                        "step": 0,
                        "advantage_source": "td",
                        "episode_index": 7,
                        "timestamp": 1.25,
                        "value": 0.5,
                        "display_value": 0.4,
                        "image": FakeImage(),
                    },
                    {
                        "advantage_source": float("nan"),
                        "step": 1,
                        "episode_index": 7,
                        "timestamp": 1.5,
                        "value": float("nan"),
                        "display_value": 0.8,
                        "image": None,
                    },
                ]
            )
        )

        self.assertEqual(frames[0]["advantage_source"], "td")
        self.assertEqual(frames[1]["advantage_source"], "")
        self.assertEqual(frames[0]["step"], 0)
        self.assertEqual(frames[0]["episode_index"], 7)
        self.assertEqual(frames[0]["value_label"], "0.500")
        self.assertEqual(frames[1]["value"], None)
        self.assertEqual(frames[1]["value_label"], "N/A")
        self.assertTrue(frames[0]["image_data_uri"].startswith("data:image/jpeg;base64,"))
        self.assertIsNone(frames[1]["image_data_uri"])

    def test_value_scrubber_html_updates_on_browser_input_event(self):
        frontend = load_frontend_module()

        html = frontend.build_value_scrubber_html(
            [
                {
                    "step": 0,
                    "episode_index": 0,
                    "timestamp": 0.0,
                    "value": 0.1,
                    "display_value": 0.1,
                    "value_label": "0.100",
                    "image_data_uri": "data:image/jpeg;base64,AAA=",
                },
                {
                    "step": 1,
                    "episode_index": 0,
                    "timestamp": 0.1,
                    "value": 0.2,
                    "display_value": 0.2,
                    "value_label": "0.200",
                    "image_data_uri": "data:image/jpeg;base64,BBB=",
                },
            ],
            initial_step=1,
        )

        self.assertIn('type="range"', html)
        self.assertIn('id="timeline-scrubber"', html)
        self.assertIn('addEventListener("input"', html)
        self.assertIn("renderFrame(Number(event.target.value))", html)
        self.assertIn("data:image/jpeg;base64,BBB=", html)
        self.assertNotIn("st.slider", html)

    def test_camera_feed_gets_larger_desktop_viewport(self):
        frontend = load_frontend_module()

        html = frontend.build_value_scrubber_html(
            [
                {
                    "step": 0,
                    "episode_index": 0,
                    "timestamp": 0.0,
                    "value": 0.1,
                    "display_value": 0.1,
                    "value_label": "0.100",
                    "image_data_uri": "data:image/jpeg;base64,AAA=",
                }
            ],
            initial_step=0,
        )

        self.assertIn("grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);", html)
        self.assertIn("min-height: 560px;", html)
        self.assertIn("max-height: 720px;", html)

    def test_camera_image_scales_up_without_cropping(self):
        frontend = load_frontend_module()

        html = frontend.build_value_scrubber_html(
            [
                {
                    "step": 0,
                    "episode_index": 0,
                    "timestamp": 0.0,
                    "value": 0.1,
                    "display_value": 0.1,
                    "value_label": "0.100",
                    "image_data_uri": "data:image/jpeg;base64,AAA=",
                }
            ],
            initial_step=0,
        )

        self.assertIn("width: 100%;", html)
        self.assertIn("height: 100%;", html)
        self.assertIn("object-fit: contain;", html)
        self.assertNotIn("object-fit: cover;", html)


if __name__ == "__main__":
    unittest.main()
