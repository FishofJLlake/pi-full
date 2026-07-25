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

import io

import numpy as np
import pytest
import torch
from PIL import Image

from opentau.scripts.grpc import robot_inference_pb2
from opentau.scripts.grpc.server import RobotPolicyServicer


def _servicer() -> RobotPolicyServicer:
    servicer = RobotPolicyServicer.__new__(RobotPolicyServicer)
    servicer.device = torch.device("cpu")
    servicer.dtype = torch.float32
    return servicer


def _compressed_message(array: np.ndarray, encoding: str):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=encoding.upper())
    return robot_inference_pb2.CameraImage(image_data=buffer.getvalue(), encoding=encoding)


@pytest.mark.parametrize("encoding", ["jpeg", "png"])
def test_compressed_image_keeps_native_resolution(encoding):
    message = _compressed_message(np.full((19, 31, 3), 127, dtype=np.uint8), encoding)

    decoded = _servicer()._decode_image(message)

    assert decoded.shape == (1, 3, 19, 31)
    assert decoded.dtype == torch.float32
    assert decoded.min() >= 0
    assert decoded.max() <= 1


@pytest.mark.parametrize("scale", [1.0, 255.0])
def test_raw_float32_image_keeps_native_resolution_and_normalizes(scale):
    array = np.full((7, 7, 3), 0.5 * scale, dtype=np.float32)
    message = robot_inference_pb2.CameraImage(image_data=array.tobytes(), encoding="raw")

    decoded = _servicer()._decode_image(message)

    assert decoded.shape == (1, 3, 7, 7)
    torch.testing.assert_close(decoded, torch.full_like(decoded, 0.5), atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"", "empty"),
        (b"12345", "byte length"),
        (np.zeros((6, 3), dtype=np.float32).tobytes(), "square"),
        (np.full((2, 2, 3), np.nan, dtype=np.float32).tobytes(), "non-finite"),
        (np.full((2, 2, 3), 256.0, dtype=np.float32).tobytes(), r"\[0, 1\]"),
    ],
)
def test_raw_payload_validation(payload, match):
    message = robot_inference_pb2.CameraImage(image_data=payload, encoding="raw")

    with pytest.raises(ValueError, match=match):
        _servicer()._decode_image(message)


def test_compressed_encoding_must_match_payload():
    message = _compressed_message(np.zeros((4, 5, 3), dtype=np.uint8), "png")
    message.encoding = "jpeg"

    with pytest.raises(ValueError, match="payload is 'png'"):
        _servicer()._decode_image(message)


def test_unknown_encoding_is_rejected():
    message = robot_inference_pb2.CameraImage(image_data=b"payload", encoding="webp")

    with pytest.raises(ValueError, match="Unknown image encoding"):
        _servicer()._decode_image(message)
