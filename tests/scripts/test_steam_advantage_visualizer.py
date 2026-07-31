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

import av
import numpy as np
import pytest

from opentau.scripts.steam_advantage_visualizer import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    compose_visualization_frame,
    episode_frame_indices_from_advantages,
    letterbox_frame,
    load_episode_advantages,
    render_curve_background,
    write_advantage_video,
)


def test_letterbox_preserves_both_edges_without_cropping():
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    frame[:, :20] = (255, 0, 0)
    frame[:, -20:] = (0, 0, 255)
    result = letterbox_frame(frame, width=800, height=600)
    non_black_rows = np.flatnonzero(result.max(axis=(1, 2)) > 0)
    content = result[non_black_rows[0] : non_black_rows[-1] + 1]
    assert np.any(np.all(content == (255, 0, 0), axis=-1))
    assert np.any(np.all(content == (0, 0, 255), axis=-1))


def test_red_marker_uses_actual_frame_index_coordinates():
    frame_indices = [10, 20, 40]
    members = [[-0.8, -0.6], [0.1, 0.2], [0.7, 0.9]]
    aggregate = [-0.8, 0.1, 0.7]
    background, geometry = render_curve_background(frame_indices, members, aggregate)
    composed = compose_visualization_frame(
        np.zeros((20, 20, 3), dtype=np.uint8),
        background,
        geometry,
        frame_index=20,
        aggregate_score=0.1,
    )
    x, y = geometry.point(20, 0.1)
    np.testing.assert_array_equal(composed[600 + y, x], np.array([220, 35, 45], dtype=np.uint8))
    assert geometry.point(20, 0.1)[0] < (geometry.x_left + geometry.x_right) // 2


def test_three_frame_video_has_expected_count_fps_and_resolution(tmp_path):
    frames = [np.full((90, 160, 3), value, dtype=np.uint8) for value in (20, 80, 140)]
    output = tmp_path / "comparison.mp4"
    write_advantage_video(
        frames,
        frame_indices=[0, 1, 2],
        member_scores=[[-0.8, -0.7], [0.1, 0.2], [0.6, 0.8]],
        aggregate_scores=[-0.8, 0.1, 0.6],
        output=output,
        fps=5.0,
    )

    container = av.open(str(output))
    try:
        stream = container.streams.video[0]
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        assert float(stream.average_rate) == pytest.approx(5.0)
    finally:
        container.close()
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.format.name == "yuv420p"
    assert len(decoded) == 3
    assert decoded[0].shape == (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_advantage_video(
            frames,
            frame_indices=[0, 1, 2],
            member_scores=[[-0.8], [0.1], [0.6]],
            aggregate_scores=[-0.8, 0.1, 0.6],
            output=output,
            fps=5.0,
        )


def test_diagnostics_absence_falls_back_to_aggregate_curve(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "raw_advantages.json").write_text(
        json.dumps({"3,0": "-0.25", "3,1": "0.50"}),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="aggregate curve only"):
        members, aggregate, diagnostics = load_episode_advantages(
            tmp_path,
            episode_index=3,
            expected_frame_indices=[0, 1],
        )
    assert members == [[-0.25], [0.5]]
    assert aggregate == [-0.25, 0.5]
    assert diagnostics["aggregate_only"] is True


def test_diagnostics_mismatch_is_a_hard_failure(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "raw_advantages.json").write_text(json.dumps({"0,0": "-0.5"}), encoding="utf-8")
    (meta / "steam_advantage_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key_format": "episode_index,frame_index",
                "aggregation": "minimum",
                "ensemble_size": 2,
                "member_scores": {"0,0": [0.1, 0.2]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Diagnostics/raw mismatch"):
        load_episode_advantages(tmp_path, episode_index=0, expected_frame_indices=[0])


def test_actual_frame_indices_are_loaded_from_advantage_metadata(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "raw_advantages.json").write_text(
        json.dumps({"2,10": -0.5, "2,30": 0.5, "2,20": 0.0, "3,0": 1.0}),
        encoding="utf-8",
    )
    assert episode_frame_indices_from_advantages(tmp_path, 2) == [10, 20, 30]
    with pytest.raises(ValueError, match="no frames"):
        episode_frame_indices_from_advantages(tmp_path, 99)
