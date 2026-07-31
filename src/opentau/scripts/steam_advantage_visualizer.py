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

"""Render an RLinf-style STEAM trajectory video with aligned advantage curves."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import draccus
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.refs import resolve_refs_to_tempfile
from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
from opentau.datasets.utils import DEFAULT_IMAGE_PATH, RAW_ADVANTAGES_PATH

OUTPUT_WIDTH = 800
VIDEO_HEIGHT = 600
PLOT_HEIGHT = 280
OUTPUT_HEIGHT = VIDEO_HEIGHT + PLOT_HEIGHT
_DIAGNOSTICS_PATH = Path("meta/steam_advantage_diagnostics.json")
_MEMBER_COLORS = (
    (112, 173, 255),
    (255, 169, 96),
    (99, 201, 146),
    (184, 151, 255),
    (244, 137, 181),
    (112, 206, 213),
)


@dataclass(frozen=True)
class CurveGeometry:
    frame_indices: tuple[int, ...]
    x_left: int
    x_right: int
    y_top: int
    y_bottom: int
    y_min: float
    y_max: float

    def point(self, frame_index: int, value: float) -> tuple[int, int]:
        if frame_index not in self.frame_indices:
            raise KeyError(f"Unknown frame_index {frame_index}.")
        x_min = self.frame_indices[0]
        x_max = self.frame_indices[-1]
        x_fraction = 0.5 if x_max == x_min else (frame_index - x_min) / (x_max - x_min)
        y_fraction = (self.y_max - value) / (self.y_max - self.y_min)
        x = round(self.x_left + x_fraction * (self.x_right - self.x_left))
        y = round(self.y_top + y_fraction * (self.y_bottom - self.y_top))
        return x, y


def _parse_frame_key(serialized: str) -> tuple[int, int]:
    try:
        episode, frame = serialized.split(",", 1)
        return int(episode), int(frame)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid advantage key {serialized!r}; expected 'episode_index,frame_index'."
        ) from exc


def episode_frame_indices_from_advantages(dataset_root: Path, episode_index: int) -> list[int]:
    """Return the selected episode's actual frame indices from persisted metadata."""
    raw_path = dataset_root / RAW_ADVANTAGES_PATH
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw advantage metadata not found: {raw_path}")
    with open(raw_path, encoding="utf-8") as handle:
        raw_payload = json.load(handle)
    frame_indices = sorted(
        frame_index
        for serialized in raw_payload
        for parsed_episode, frame_index in [_parse_frame_key(serialized)]
        if parsed_episode == episode_index
    )
    if not frame_indices:
        raise ValueError(f"Raw advantage metadata contains no frames for episode {episode_index}.")
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError(f"Raw advantage metadata contains duplicate frames for episode {episode_index}.")
    return frame_indices


def load_episode_advantages(
    dataset_root: Path,
    *,
    episode_index: int,
    expected_frame_indices: Sequence[int],
) -> tuple[list[list[float]], list[float], dict]:
    """Load member curves and validate coverage/minimum aggregation for one episode."""
    raw_path = dataset_root / RAW_ADVANTAGES_PATH
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw advantage metadata not found: {raw_path}")
    with open(raw_path, encoding="utf-8") as handle:
        raw_payload = json.load(handle)
    raw_by_key = {_parse_frame_key(key): float(value) for key, value in raw_payload.items()}
    expected_keys = [(episode_index, int(frame)) for frame in expected_frame_indices]
    missing_raw = [key for key in expected_keys if key not in raw_by_key]
    if missing_raw:
        raise ValueError(
            f"Episode {episode_index} is missing {len(missing_raw)} raw advantage frame(s): "
            f"{missing_raw[:10]}."
        )
    aggregate = [raw_by_key[key] for key in expected_keys]

    diagnostics_path = dataset_root / _DIAGNOSTICS_PATH
    if not diagnostics_path.is_file():
        warnings.warn(
            f"{diagnostics_path} is absent; rendering the aggregate curve only.",
            stacklevel=2,
        )
        return (
            [[value] for value in aggregate],
            aggregate,
            {
                "schema_version": 0,
                "aggregation": "minimum",
                "ensemble_size": 1,
                "aggregate_only": True,
            },
        )

    with open(diagnostics_path, encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    if diagnostics.get("schema_version") != 1:
        raise ValueError(f"Unsupported diagnostics schema: {diagnostics.get('schema_version')!r}.")
    if diagnostics.get("key_format") != "episode_index,frame_index":
        raise ValueError(f"Unsupported diagnostics key format: {diagnostics.get('key_format')!r}.")
    if diagnostics.get("aggregation") != "minimum":
        raise ValueError(f"Unsupported diagnostics aggregation: {diagnostics.get('aggregation')!r}.")
    ensemble_size = int(diagnostics.get("ensemble_size", 0))
    if ensemble_size < 1:
        raise ValueError(f"Invalid diagnostics ensemble_size={ensemble_size}.")
    member_payload = diagnostics.get("member_scores")
    if not isinstance(member_payload, dict):
        raise ValueError("Diagnostics member_scores must be an object.")

    member_scores: list[list[float]] = []
    for key, expected_minimum in zip(expected_keys, aggregate, strict=True):
        serialized = f"{key[0]},{key[1]}"
        if serialized not in member_payload:
            raise ValueError(f"Diagnostics do not cover selected episode frame {serialized}.")
        values = [float(value) for value in member_payload[serialized]]
        if len(values) != ensemble_size:
            raise ValueError(
                f"Diagnostics frame {serialized} has {len(values)} members, expected {ensemble_size}."
            )
        if not math.isclose(min(values), expected_minimum, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(
                f"Diagnostics/raw mismatch at {serialized}: min(member_scores)={min(values):.8f}, "
                f"raw_advantage={expected_minimum:.8f}."
            )
        member_scores.append(values)
    return member_scores, aggregate, diagnostics


def letterbox_frame(
    frame: np.ndarray,
    *,
    width: int = OUTPUT_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> np.ndarray:
    """Aspect-preserving resize with centered black padding and no crop."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected an RGB/RGBA frame, got shape={array.shape}.")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    source_height, source_width = array.shape[:2]
    if source_height < 1 or source_width < 1:
        raise ValueError(f"Frame dimensions must be positive, got {array.shape}.")
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(array, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _polyline_points(
    geometry: CurveGeometry,
    frame_indices: Sequence[int],
    values: Sequence[float],
) -> np.ndarray:
    return np.asarray(
        [
            geometry.point(int(frame_index), float(value))
            for frame_index, value in zip(frame_indices, values, strict=True)
        ],
        dtype=np.int32,
    )


def render_curve_background(
    frame_indices: Sequence[int],
    member_scores: Sequence[Sequence[float]],
    aggregate_scores: Sequence[float],
) -> tuple[np.ndarray, CurveGeometry]:
    """Render the static 800x280 member/minimum plot once."""
    if not frame_indices:
        raise ValueError("At least one frame is required for an advantage plot.")
    if len(frame_indices) != len(member_scores) or len(frame_indices) != len(aggregate_scores):
        raise ValueError("Frame indices, member scores, and aggregate scores must have equal lengths.")
    normalized_members = [[float(value) for value in row] for row in member_scores]
    member_count = len(normalized_members[0])
    if member_count < 1 or any(len(row) != member_count for row in normalized_members):
        raise ValueError("Every frame must contain the same non-zero member count.")
    if any(right <= left for left, right in zip(frame_indices, frame_indices[1:], strict=False)):
        raise ValueError("frame_indices must be strictly increasing.")

    all_values = [value for row in normalized_members for value in row]
    all_values.extend(float(value) for value in aggregate_scores)
    data_min = min(-1.0, min(all_values))
    data_max = max(1.0, max(all_values))
    span = max(data_max - data_min, 1e-6)
    y_min = data_min - 0.06 * span
    y_max = data_max + 0.06 * span
    geometry = CurveGeometry(
        frame_indices=tuple(int(value) for value in frame_indices),
        x_left=58,
        x_right=OUTPUT_WIDTH - 18,
        y_top=28,
        y_bottom=PLOT_HEIGHT - 42,
        y_min=y_min,
        y_max=y_max,
    )

    canvas = np.full((PLOT_HEIGHT, OUTPUT_WIDTH, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for value in np.linspace(y_min, y_max, 5):
        _, y = geometry.point(frame_indices[0], float(value))
        cv2.line(canvas, (geometry.x_left, y), (geometry.x_right, y), (224, 224, 224), 1)
        cv2.putText(
            canvas,
            f"{value:.1f}",
            (6, y + 4),
            font,
            0.42,
            (82, 82, 82),
            1,
            cv2.LINE_AA,
        )
    zero_y = geometry.point(frame_indices[0], 0.0)[1]
    cv2.line(canvas, (geometry.x_left, zero_y), (geometry.x_right, zero_y), (150, 150, 150), 1)
    cv2.line(
        canvas,
        (geometry.x_left, geometry.y_top),
        (geometry.x_left, geometry.y_bottom),
        (70, 70, 70),
        1,
    )
    cv2.line(
        canvas,
        (geometry.x_left, geometry.y_bottom),
        (geometry.x_right, geometry.y_bottom),
        (70, 70, 70),
        1,
    )

    tick_positions = np.linspace(0, len(frame_indices) - 1, min(6, len(frame_indices))).round().astype(int)
    for position in np.unique(tick_positions):
        frame_index = int(frame_indices[position])
        x, _ = geometry.point(frame_index, 0.0)
        cv2.line(canvas, (x, geometry.y_bottom), (x, geometry.y_bottom + 5), (70, 70, 70), 1)
        label = str(frame_index)
        text_width = cv2.getTextSize(label, font, 0.42, 1)[0][0]
        cv2.putText(
            canvas,
            label,
            (x - text_width // 2, geometry.y_bottom + 22),
            font,
            0.42,
            (82, 82, 82),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, "frame_index", (OUTPUT_WIDTH - 105, PLOT_HEIGHT - 9), font, 0.42, (82, 82, 82), 1)
    cv2.putText(canvas, "Advantage", (6, 17), font, 0.48, (25, 25, 25), 1, cv2.LINE_AA)

    matrix = np.asarray(normalized_members, dtype=np.float64)
    for member_index in range(member_count):
        points = _polyline_points(geometry, frame_indices, matrix[:, member_index])
        color = _MEMBER_COLORS[member_index % len(_MEMBER_COLORS)]
        cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
    aggregate_points = _polyline_points(geometry, frame_indices, aggregate_scores)
    cv2.polylines(canvas, [aggregate_points], False, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Members ({member_count})",
        (OUTPUT_WIDTH - 200, 17),
        font,
        0.42,
        (105, 105, 105),
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (OUTPUT_WIDTH - 91, 12), (OUTPUT_WIDTH - 65, 12), (20, 20, 20), 3)
    cv2.putText(canvas, "Minimum", (OUTPUT_WIDTH - 60, 17), font, 0.42, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas, geometry


def compose_visualization_frame(
    video_frame: np.ndarray,
    curve_background: np.ndarray,
    geometry: CurveGeometry,
    *,
    frame_index: int,
    aggregate_score: float,
) -> np.ndarray:
    """Compose one 800x880 RGB frame and draw the moving red marker."""
    video_panel = letterbox_frame(video_frame)
    curve_panel = curve_background.copy()
    x, y = geometry.point(frame_index, aggregate_score)
    cv2.circle(curve_panel, (x, y), 7, (220, 35, 45), -1, cv2.LINE_AA)
    cv2.circle(curve_panel, (x, y), 7, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([video_panel, curve_panel], axis=0)


def write_advantage_video(
    frames: Iterable[np.ndarray],
    *,
    frame_indices: Sequence[int],
    member_scores: Sequence[Sequence[float]],
    aggregate_scores: Sequence[float],
    output: Path,
    fps: float,
    overwrite: bool = False,
) -> Path:
    """Stream frames into a precomposed H.264/yuv420p comparison MP4."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing video: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    curve_background, geometry = render_curve_background(
        frame_indices,
        member_scores,
        aggregate_scores,
    )

    writer = imageio.get_writer(
        output,
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-movflags", "+faststart"],
    )
    written = 0
    try:
        for position, frame in enumerate(frames):
            if position >= len(frame_indices):
                raise ValueError("Video source contains more frames than the selected episode metadata.")
            composed = compose_visualization_frame(
                frame,
                curve_background,
                geometry,
                frame_index=int(frame_indices[position]),
                aggregate_score=float(aggregate_scores[position]),
            )
            writer.append_data(composed)
            written += 1
    except BaseException:
        writer.close()
        output.unlink(missing_ok=True)
        raise
    else:
        writer.close()
    if written != len(frame_indices):
        output.unlink(missing_ok=True)
        raise ValueError(
            f"Video source yielded {written} frames, but episode metadata requires {len(frame_indices)}."
        )
    return output


def _parse_dataset_mixture(path: Path) -> DatasetMixtureConfig:
    resolved = resolve_refs_to_tempfile(path)
    try:
        return draccus.parse(config_class=DatasetMixtureConfig, config_path=resolved, args=[])
    finally:
        resolved.unlink(missing_ok=True)


def _resolve_camera_key(dataset_config: DatasetConfig, meta: LeRobotDatasetMetadata, requested: str) -> str:
    mapping = dataset_config.data_features_name_mapping or {}
    actual = mapping.get(requested, requested)
    if actual not in meta.camera_keys:
        standardized = {
            standard: source for standard, source in mapping.items() if standard.startswith("camera")
        }
        raise ValueError(
            f"Camera {requested!r} resolves to {actual!r}, which is unavailable. "
            f"Dataset cameras={meta.camera_keys}; standardized mapping={standardized}."
        )
    return actual


def _video_frames(
    video_path: Path,
    *,
    start_frame: int,
    frame_count: int,
) -> Iterator[np.ndarray]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Camera video not found: {video_path}")
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        yielded = 0
        for decoded_index, frame in enumerate(container.decode(stream)):
            if decoded_index < start_frame:
                continue
            if yielded >= frame_count:
                break
            yield frame.to_ndarray(format="rgb24")
            yielded += 1
    finally:
        container.close()


def _image_frames(
    dataset_root: Path,
    *,
    camera_key: str,
    episode_index: int,
    frame_indices: Sequence[int],
) -> Iterator[np.ndarray]:
    for frame_index in frame_indices:
        relative = DEFAULT_IMAGE_PATH.format(
            image_key=camera_key,
            episode_index=episode_index,
            frame_index=int(frame_index),
        )
        image_path = dataset_root / relative
        if not image_path.is_file():
            raise FileNotFoundError(f"Camera image not found: {image_path}")
        with Image.open(image_path) as image:
            yield np.asarray(image.convert("RGB"))


def episode_frame_source(
    dataset_root: Path,
    meta: LeRobotDatasetMetadata,
    *,
    camera_key: str,
    episode_index: int,
    frame_indices: Sequence[int],
) -> Iterable[np.ndarray]:
    """Return a streaming video source, falling back to per-frame images."""
    if camera_key in meta.video_keys:
        video_path = dataset_root / meta.get_video_file_path(episode_index, camera_key)
        offset_seconds = float(
            meta.episodes[episode_index].get(f"videos/{camera_key}/from_timestamp", 0.0) or 0.0
        )
        start_frame = round(offset_seconds * float(meta.fps))
        return _video_frames(video_path, start_frame=start_frame, frame_count=len(frame_indices))
    if camera_key in meta.image_keys:
        return _image_frames(
            dataset_root,
            camera_key=camera_key,
            episode_index=episode_index,
            frame_indices=frame_indices,
        )
    raise ValueError(f"Camera {camera_key!r} is neither a video nor image feature.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--camera-key", default="camera0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> Path:
    from opentau.scripts.steam_advantage_pipeline import normalized_dataset_root

    mixture = _parse_dataset_mixture(args.dataset_config)
    if not 0 <= args.dataset_index < len(mixture.datasets):
        raise IndexError(f"dataset-index {args.dataset_index} is outside [0, {len(mixture.datasets)}).")
    dataset_config = mixture.datasets[args.dataset_index]
    dataset_root = normalized_dataset_root(dataset_config)
    meta = LeRobotDatasetMetadata(
        str(dataset_config.repo_id),
        root=dataset_root,
        revision=dataset_config.revision,
        local_files_only=True,
    )
    if args.episode not in meta.episodes:
        raise KeyError(
            f"Episode {args.episode} does not exist. Available episodes: {sorted(meta.episodes)[:20]}"
        )
    episode_length = int(meta.episodes[args.episode].get("length", 0))
    if episode_length < 1:
        raise ValueError(f"Episode {args.episode} has invalid length={episode_length}.")
    frame_indices = episode_frame_indices_from_advantages(dataset_root, args.episode)
    if len(frame_indices) != episode_length:
        raise ValueError(
            f"Episode {args.episode} metadata length={episode_length}, but advantage metadata "
            f"contains {len(frame_indices)} frames."
        )
    member_scores, aggregate_scores, _diagnostics = load_episode_advantages(
        dataset_root,
        episode_index=args.episode,
        expected_frame_indices=frame_indices,
    )
    camera_key = _resolve_camera_key(dataset_config, meta, args.camera_key)
    frames = episode_frame_source(
        dataset_root,
        meta,
        camera_key=camera_key,
        episode_index=args.episode,
        frame_indices=frame_indices,
    )
    output = write_advantage_video(
        frames,
        frame_indices=frame_indices,
        member_scores=member_scores,
        aggregate_scores=aggregate_scores,
        output=args.output,
        fps=float(args.fps if args.fps is not None else meta.fps),
        overwrite=args.overwrite,
    )
    print(
        f"Wrote STEAM advantage video: {output} "
        f"({OUTPUT_WIDTH}x{OUTPUT_HEIGHT}, frames={episode_length}, camera={camera_key})"
    )
    return output


def cli() -> None:
    main(_parse_args())


if __name__ == "__main__":
    cli()
