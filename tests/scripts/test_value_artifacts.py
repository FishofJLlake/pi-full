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

import ast
import json
import unittest
from importlib import util
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "src" / "opentau" / "scripts" / "value_artifacts.py"
CALCULATE_PATH = ROOT / "src" / "opentau" / "scripts" / "calculate_value.py"
VISUALIZER_PATH = ROOT / "src" / "opentau" / "scripts" / "value_visualizer_app.py"


def load_value_artifacts_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"Missing value artifact module: {MODULE_PATH}")
    spec = util.spec_from_file_location("value_artifacts", MODULE_PATH)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValueArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.module = load_value_artifacts_module()
        self.tmpdir = TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def write_json_text(self, content: str) -> Path:
        path = Path(self.tmpdir.name) / "values.json"
        path.write_text(content, encoding="utf-8")
        return path

    def test_canonical_frame_keys_round_trip(self):
        path = self.write_json_text('{"2,7": 0.25, "2,8": -0.5}')

        self.assertEqual(self.module.serialize_value_key(2, 7), "2,7")
        self.assertEqual(self.module.load_values(path), {(2, 7): 0.25, (2, 8): -0.5})

    def test_timestamp_key_is_rejected(self):
        path = self.write_json_text('{"2,0.04": 0.25}')

        with self.assertRaisesRegex(ValueError, "integer frame key"):
            self.module.load_values(path)

    def test_noncanonical_and_negative_indices_are_rejected(self):
        for serialized in ("02,7", "2,-1", "2,+1", "2, 1"):
            with self.subTest(serialized=serialized):
                path = self.write_json_text(json.dumps({serialized: 0.25}))
                with self.assertRaises(ValueError):
                    self.module.load_values(path)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.module.serialize_value_key(2, -1)

    def test_duplicate_keys_are_rejected(self):
        path = self.write_json_text('{"2,7": 0.25, "2,7": 0.5}')

        with self.assertRaisesRegex(ValueError, "Duplicate value key"):
            self.module.load_values(path)

    def test_non_finite_values_are_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                path = self.write_json_text(f'{{"2,7": {value}}}')
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.module.load_values(path)

    def test_value_scripts_use_shared_frame_index_contract(self):
        calculate_source = CALCULATE_PATH.read_text(encoding="utf-8")
        visualizer_source = VISUALIZER_PATH.read_text(encoding="utf-8")
        ast.parse(calculate_source)
        ast.parse(visualizer_source)

        self.assertIn('batch["frame_index"]', calculate_source)
        self.assertIn(
            "from opentau.scripts.value_artifacts import serialize_value_key",
            calculate_source,
        )
        self.assertIn(
            "from opentau.scripts.value_artifacts import load_values",
            visualizer_source,
        )
        self.assertNotIn("def load_values(", visualizer_source)
        self.assertIn("value_lookup.get((ep, frame_idx), np.nan)", visualizer_source)
        self.assertIn('sort_values("frame_index")', visualizer_source)
        self.assertNotIn('batch["timestamp"]', calculate_source)
        self.assertNotIn("(episode_index, timestamp) -> value", calculate_source)


if __name__ == "__main__":
    unittest.main()
