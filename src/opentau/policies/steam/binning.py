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

"""Signed temporal-offset binning shared by STEAM data and model code."""

from __future__ import annotations

import torch
from einops import reduce
from torch import Tensor


def validate_binning(max_temporal_offset: int, num_bins: int) -> None:
    """Validate the uniform sign-preserving bin layout."""
    if max_temporal_offset < 1:
        raise ValueError(f"max_temporal_offset must be >= 1, got {max_temporal_offset}.")
    if num_bins < 2 or num_bins % 2:
        raise ValueError(f"num_bins must be >= 2 and even, got {num_bins}.")
    if (2 * max_temporal_offset) % num_bins:
        raise ValueError(
            "2 * max_temporal_offset must be divisible by num_bins; "
            f"got max_temporal_offset={max_temporal_offset}, num_bins={num_bins}."
        )


def signed_offset_to_bin(offset: int, max_temporal_offset: int, num_bins: int) -> int:
    """Map an integer offset in ``[-K, -1] U [1, K]`` to a bin index."""
    validate_binning(max_temporal_offset, num_bins)
    if offset == 0 or abs(offset) > max_temporal_offset:
        raise ValueError(
            "offset must be non-zero and within max_temporal_offset; "
            f"got offset={offset}, max_temporal_offset={max_temporal_offset}."
        )
    position = offset + max_temporal_offset if offset < 0 else offset + max_temporal_offset - 1
    return int((position * num_bins) // (2 * max_temporal_offset))


def scaled_signed_offset_to_bin(scaled_offset: float, max_temporal_offset: int, num_bins: int) -> int:
    """Round, sign-preserve and clamp a length-normalized temporal offset."""
    if scaled_offset == 0:
        raise ValueError("scaled_offset must be non-zero.")
    rounded = int(round(float(scaled_offset)))
    if rounded == 0:
        rounded = 1 if scaled_offset > 0 else -1
    rounded = max(-max_temporal_offset, min(max_temporal_offset, rounded))
    return signed_offset_to_bin(rounded, max_temporal_offset, num_bins)


def bin_centers(
    max_temporal_offset: int,
    num_bins: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return signed-offset centers for all categorical bins."""
    validate_binning(max_temporal_offset, num_bins)
    strides_per_bin = (2 * max_temporal_offset) // num_bins
    half = num_bins // 2
    centers = []
    for bin_index in range(num_bins):
        if bin_index < half:
            low = -max_temporal_offset + bin_index * strides_per_bin
        else:
            low = 1 + (bin_index - half) * strides_per_bin
        high = low + strides_per_bin - 1
        centers.append((low + high) / 2.0)
    return torch.tensor(centers, device=device, dtype=dtype)


def expected_signed_offset(probabilities: Tensor, max_temporal_offset: int, num_bins: int) -> Tensor:
    """Decode categorical probabilities to an expected signed offset."""
    if probabilities.shape[-1] != num_bins:
        raise ValueError(f"Expected probabilities[..., {num_bins}], got {tuple(probabilities.shape)}.")
    centers = bin_centers(
        max_temporal_offset,
        num_bins,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return reduce(probabilities * centers, "... bin -> ...", "sum")
