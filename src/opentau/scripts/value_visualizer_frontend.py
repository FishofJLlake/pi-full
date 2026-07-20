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

from __future__ import annotations

import base64
import html
import io
import json
import math
from typing import Any


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "" if not math.isfinite(numeric) else str(value)


def _format_float(value: float | None, digits: int) -> str:
    return f"{value:.{digits}f}" if value is not None else "N/A"


def image_to_data_uri(image: Any) -> str | None:
    """Encode a PIL-like image as a browser-displayable data URI."""
    if image is None:
        return None

    output = io.BytesIO()
    try:
        image_for_jpeg = image.convert("RGB") if hasattr(image, "convert") else image
        image_for_jpeg.save(output, format="JPEG", quality=85, optimize=True)
        mime = "image/jpeg"
    except Exception:
        output = io.BytesIO()
        try:
            image.save(output, format="PNG")
        except Exception:
            return None
        mime = "image/png"

    payload = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def build_scrubber_frames(df: Any) -> list[dict[str, Any]]:
    """Prepare all frame data for browser-side timeline scrubbing."""
    frames: list[dict[str, Any]] = []
    for fallback_step, (_, row) in enumerate(df.iterrows()):
        step = int(row.get("step", fallback_step))
        episode_index = int(row.get("episode_index", 0))
        frame_index = int(row.get("frame_index", step))
        raw_advantage = _finite_float_or_none(row.get("raw_advantage"))
        effective_advantage = _finite_float_or_none(row.get("effective_advantage"))
        advantage_source = _text_or_empty(row.get("advantage_source"))
        timestamp = _finite_float_or_none(row.get("timestamp"))
        value = _finite_float_or_none(row.get("value"))
        display_value = _finite_float_or_none(row.get("display_value"))
        if display_value is None:
            display_value = value if value is not None else 0.0

        frames.append(
            {
                "step": step,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "raw_advantage": raw_advantage,
                "effective_advantage": effective_advantage,
                "raw_advantage_label": _format_float(raw_advantage, 3),
                "effective_advantage_label": _format_float(effective_advantage, 3),
                "timestamp": timestamp,
                "value": value,
                "advantage_source": advantage_source,
                "display_value": display_value,
                "value_label": _format_float(value, 3),
                "display_value_label": _format_float(display_value, 3),
                "timestamp_label": _format_float(timestamp, 2),
                "image_data_uri": image_to_data_uri(row.get("image")),
            }
        )
    return frames


def build_value_scrubber_html(
    frames: list[dict[str, Any]],
    initial_step: int = 0,
    series_label: str = "Value",
) -> str:
    """Build a self-contained HTML timeline that updates on browser input events."""
    if not frames:
        return "<p>No frames loaded.</p>"

    initial_step = max(0, min(initial_step, len(frames) - 1))
    values = [frame.get("value") for frame in frames if isinstance(frame.get("value"), (int, float))]
    max_value = max(values) if values else None
    frames_json = json.dumps(frames, ensure_ascii=False, allow_nan=False)
    max_value_label = _format_float(_finite_float_or_none(max_value), 3)
    escaped_label = html.escape(series_label.strip() or "Value")
    curve_title = (
        "Value Function V(s)" if escaped_label == "Value" else f"{escaped_label} Curve"
    )

    return f"""
<style>
  :root {{
    color-scheme: light;
  }}

  .vv-root {{
    box-sizing: border-box;
    width: 100%;
    min-height: 860px;
    font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #1f2937;
  }}

  .vv-main {{
    display: grid;
    grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
    gap: 1rem;
    align-items: start;
  }}

  .vv-panel {{
    border: 1px solid #d9e1ea;
    border-radius: 8px;
    background: #ffffff;
    overflow: hidden;
  }}

  .vv-panel h3 {{
    margin: 0;
    padding: 0.85rem 1rem;
    font-size: 1.18rem;
    font-weight: 650;
    line-height: 1.25;
  }}

  .vv-image-shell {{
    display: grid;
    place-items: center;
    width: 100%;
    height: 100%;
    min-height: 560px;
    background: #f6f8fb;
  }}

  .vv-image-shell img {{
    display: block;
    width: 100%;
    height: 100%;
    max-width: 100%;
    max-height: 720px;
    object-fit: contain;
  }}

  .vv-empty-image {{
    display: none;
    color: #5f6f83;
    font-size: 0.95rem;
  }}

  .vv-metric-row {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    border-top: 1px solid #e6edf5;
  }}

  .vv-metric {{
    min-width: 0;
    padding: 0.75rem 1rem;
    border-right: 1px solid #e6edf5;
  }}

  .vv-metric:last-child {{
    border-right: 0;
  }}

  .vv-metric-label {{
    display: block;
    color: #687789;
    font-size: 0.76rem;
    line-height: 1.2;
  }}

  .vv-metric-value {{
    display: block;
    margin-top: 0.18rem;
    font-size: 1.08rem;
    font-weight: 650;
    line-height: 1.2;
    overflow-wrap: anywhere;
  }}

  .vv-caption {{
    padding: 0.55rem 1rem 0.9rem;
    color: #5f6f83;
    font-size: 0.88rem;
  }}

  .vv-chart-shell {{
    padding: 0 0.9rem 0.8rem;
  }}

  .vv-chart {{
    width: 100%;
    height: 420px;
    display: block;
    background: #ffffff;
  }}

  .vv-axis-label {{
    fill: #687789;
    font-size: 13px;
  }}

  .vv-slider-panel {{
    margin-top: 1rem;
    border: 1px solid #d9e1ea;
    border-radius: 8px;
    background: #ffffff;
    padding: 0.85rem 1rem 0.95rem;
  }}

  .vv-slider-heading {{
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.55rem;
    color: #334155;
    font-size: 0.92rem;
    font-weight: 600;
  }}

  #timeline-scrubber {{
    width: 100%;
    accent-color: #d62728;
  }}

  .vv-slider-scale {{
    display: flex;
    justify-content: space-between;
    color: #687789;
    font-size: 0.78rem;
    margin-top: 0.28rem;
  }}

  @media (max-width: 760px) {{
    .vv-root {{
      min-height: 1120px;
    }}

    .vv-main {{
      grid-template-columns: 1fr;
    }}

    .vv-image-shell {{
      min-height: 300px;
    }}

    .vv-metric-row {{
      grid-template-columns: 1fr;
    }}

    .vv-metric {{
      border-right: 0;
      border-bottom: 1px solid #e6edf5;
    }}

    .vv-metric:last-child {{
      border-bottom: 0;
    }}
  }}
</style>

<div class="vv-root">
  <div class="vv-main">
    <section class="vv-panel">
      <h3 id="camera-title">Camera Feed</h3>
      <div class="vv-image-shell">
        <img id="frame-image" alt="Camera frame" />
        <div id="empty-image" class="vv-empty-image">No image for this frame.</div>
      </div>
      <div class="vv-metric-row">
        <div class="vv-metric">
          <span class="vv-metric-label">Predicted {escaped_label}</span>
          <span class="vv-metric-value" id="predicted-value">N/A</span>
        </div>
        <div class="vv-metric">
          <span class="vv-metric-label">Current {escaped_label}</span>
          <span class="vv-metric-value" id="current-value">N/A</span>
        </div>
        <div class="vv-metric">
          <span class="vv-metric-label">Step Delta</span>
          <span class="vv-metric-value" id="step-delta">N/A</span>
        </div>
      </div>
      <div class="vv-caption" id="frame-caption"></div>
    </section>

    <section class="vv-panel">
      <h3>{curve_title}</h3>
      <div class="vv-chart-shell">
        <svg id="value-chart" class="vv-chart" viewBox="0 0 840 420" preserveAspectRatio="none">
          <line x1="58" y1="28" x2="58" y2="358" stroke="#d9e1ea" />
          <line x1="58" y1="358" x2="812" y2="358" stroke="#d9e1ea" />
          <g id="grid-lines"></g>
          <polyline id="value-line" fill="none" stroke="#1f77b4" stroke-width="3" points="" />
          <line id="current-line" y1="28" y2="358" stroke="#d62728" stroke-width="2" stroke-dasharray="6 5" />
          <circle id="current-marker" r="6" fill="#d62728" stroke="#ffffff" stroke-width="2" />
          <text x="58" y="392" class="vv-axis-label">Timestep (t)</text>
          <text id="y-max-label" x="12" y="34" class="vv-axis-label"></text>
          <text id="y-min-label" x="12" y="360" class="vv-axis-label"></text>
        </svg>
      </div>
      <div class="vv-metric-row">
        <div class="vv-metric">
          <span class="vv-metric-label">Max {escaped_label}</span>
          <span class="vv-metric-value">{max_value_label}</span>
        </div>
        <div class="vv-metric">
          <span class="vv-metric-label">Displayed {escaped_label}</span>
          <span class="vv-metric-value" id="display-value">N/A</span>
        </div>
        <div class="vv-metric">
          <span class="vv-metric-label">Step</span>
          <span class="vv-metric-value" id="step-value">0</span>
        </div>
      </div>
    </section>
  </div>

  <div class="vv-slider-panel">
    <div class="vv-slider-heading">
      <span>Timeline Scrubber</span>
      <span id="slider-readout">Step 0</span>
    </div>
    <input
      id="timeline-scrubber"
      type="range"
      min="0"
      max="{len(frames) - 1}"
      value="{initial_step}"
      step="1"
    />
    <div class="vv-slider-scale">
      <span>0</span>
      <span>{len(frames) - 1}</span>
    </div>
  </div>
</div>

<script>
  const frames = {frames_json};
  const slider = document.getElementById("timeline-scrubber");
  const cameraTitle = document.getElementById("camera-title");
  const frameImage = document.getElementById("frame-image");
  const emptyImage = document.getElementById("empty-image");
  const predictedValue = document.getElementById("predicted-value");
  const currentValue = document.getElementById("current-value");
  const displayValue = document.getElementById("display-value");
  const stepDelta = document.getElementById("step-delta");
  const stepValue = document.getElementById("step-value");
  const frameCaption = document.getElementById("frame-caption");
  const sliderReadout = document.getElementById("slider-readout");
  const valueLine = document.getElementById("value-line");
  const currentLine = document.getElementById("current-line");
  const currentMarker = document.getElementById("current-marker");
  const yMaxLabel = document.getElementById("y-max-label");
  const yMinLabel = document.getElementById("y-min-label");
  const gridLines = document.getElementById("grid-lines");
  const chart = {{
    left: 58,
    right: 28,
    top: 28,
    bottom: 62,
    width: 840,
    height: 420,
  }};
  chart.plotWidth = chart.width - chart.left - chart.right;
  chart.plotHeight = chart.height - chart.top - chart.bottom;

  const finiteValues = frames
    .map((frame) => Number(frame.display_value))
    .filter((value) => Number.isFinite(value));
  let yMin = finiteValues.length ? Math.min(...finiteValues) : 0;
  let yMax = finiteValues.length ? Math.max(...finiteValues) : 1;
  const yPad = Math.max((yMax - yMin) * 0.05, 1e-6);
  yMin -= yPad;
  yMax += yPad;

  function chartX(index) {{
    if (frames.length <= 1) {{
      return chart.left;
    }}
    return chart.left + (index / (frames.length - 1)) * chart.plotWidth;
  }}

  function chartY(value) {{
    const numericValue = Number.isFinite(Number(value)) ? Number(value) : 0;
    return chart.top + ((yMax - numericValue) / (yMax - yMin)) * chart.plotHeight;
  }}

  function setupChart() {{
    valueLine.setAttribute(
      "points",
      frames.map((frame, index) => `${{chartX(index)}},${{chartY(frame.display_value)}}`).join(" ")
    );
    yMaxLabel.textContent = yMax.toFixed(3);
    yMinLabel.textContent = yMin.toFixed(3);
    for (let i = 1; i <= 3; i += 1) {{
      const y = chart.top + (i / 4) * chart.plotHeight;
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", chart.left);
      line.setAttribute("x2", chart.width - chart.right);
      line.setAttribute("y1", y);
      line.setAttribute("y2", y);
      line.setAttribute("stroke", "#eef3f8");
      gridLines.appendChild(line);
    }}
  }}

  function labelOrNA(value, digits = 3) {{
    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue.toFixed(digits) : "N/A";
  }}

  function renderFrame(step) {{
    const boundedStep = Math.max(0, Math.min(Number(step), frames.length - 1));
    const frame = frames[boundedStep];
    const previousFrame = frames[Math.max(0, boundedStep - 1)];
    const x = chartX(boundedStep);
    const y = chartY(frame.display_value);
    const delta =
      Number.isFinite(Number(frame.value)) && Number.isFinite(Number(previousFrame.value))
        ? Number(frame.value) - Number(previousFrame.value)
        : null;

    slider.value = String(boundedStep);
    cameraTitle.textContent = `Camera Feed (Step ${{frame.step}})`;
    predictedValue.textContent = frame.value_label || labelOrNA(frame.value);
    currentValue.textContent = frame.value_label || labelOrNA(frame.value);
    displayValue.textContent = frame.display_value_label || labelOrNA(frame.display_value);
    stepDelta.textContent = delta === null ? "N/A" : delta.toFixed(4);
    stepValue.textContent = String(frame.step);
    sliderReadout.textContent = `Step ${{frame.step}}`;
    const timestampLabel = frame.timestamp_label || labelOrNA(frame.timestamp, 2);
    const details = [
      `Episode ${{frame.episode_index}}`,
      `frame ${{frame.frame_index}}`,
      `t = ${{timestampLabel}}s`,
    ];
    if (frame.raw_advantage_label && frame.raw_advantage_label !== "N/A") {{
      details.push(`raw advantage = ${{frame.raw_advantage_label}}`);
    }}
    if (frame.effective_advantage_label && frame.effective_advantage_label !== "N/A") {{
      details.push(`effective advantage = ${{frame.effective_advantage_label}}`);
    }}
    if (frame.advantage_source) details.push(`source = ${{frame.advantage_source}}`);

    if (frame.image_data_uri) {{
      frameImage.src = frame.image_data_uri;
      frameImage.style.display = "block";
      emptyImage.style.display = "none";
    }} else {{
      frameImage.removeAttribute("src");
      frameImage.style.display = "none";
      emptyImage.style.display = "block";
    }}

    currentLine.setAttribute("x1", x);
    currentLine.setAttribute("x2", x);
    currentMarker.setAttribute("cx", x);
    currentMarker.setAttribute("cy", y);
  }}

  setupChart();
  slider.addEventListener("input", (event) => renderFrame(Number(event.target.value)));
  renderFrame({initial_step});
</script>
"""
