# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def _rerun_scalar(value: float):
    scalar_type = getattr(rr, "Scalar", None)
    return scalar_type(value) if scalar_type is not None else rr.Scalars(value)


def _tactile_entity_path(name: str) -> str:
    parts = name.split("_", maxsplit=2)
    if len(parts) == 3:
        usb_name, route_name, channel_index = parts
        return f"tactile.{usb_name}.{route_name}.channel_{channel_index}"
    return f"tactile.{name}"


def _recording_blueprint(
    observation_names: Sequence[str],
    action_names: Sequence[str],
    tactile_names: Sequence[str],
    image_names: Sequence[str],
) -> rrb.Blueprint:
    time_series = rrb.Vertical(
        rrb.TimeSeriesView(
            name="State",
            origin="/",
            contents=[f"/observation.{name}" for name in observation_names],
        ),
        rrb.TimeSeriesView(
            name="Action",
            origin="/",
            contents=[f"/action.{name}" for name in action_names],
        ),
        rrb.TimeSeriesView(
            name="Tactile",
            origin="/",
            contents=[f"/{_tactile_entity_path(name)}" for name in tactile_names],
        ),
        row_shares=[1, 1, 2],
    )

    image_views = [
        rrb.Spatial2DView(
            name=f"Camera: {name}",
            origin=f"/observation.{name}",
            contents=[f"/observation.{name}"],
        )
        for name in image_names
    ]
    layout = (
        rrb.Horizontal(
            time_series,
            rrb.Vertical(*image_views),
            column_shares=[2, 1],
        )
        if image_views
        else time_series
    )
    return rrb.Blueprint(layout, collapse_panels=True)


def _init_rerun(
    session_name: str = "lerobot_control_loop",
    *,
    observation_names: Sequence[str] | None = None,
    action_names: Sequence[str] | None = None,
    tactile_names: Sequence[str] | None = None,
    image_names: Sequence[str] = (),
) -> None:
    """Initializes the Rerun SDK for visualizing the control loop."""
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)
    if observation_names is not None and action_names is not None and tactile_names is not None:
        rr.send_blueprint(
            _recording_blueprint(observation_names, action_names, tactile_names, image_names)
        )


def set_rerun_time(frame_index: int, elapsed_time_s: float) -> None:
    """Set common timelines for every value logged in one control-loop iteration."""
    if hasattr(rr, "set_time"):
        rr.set_time("frame_index", sequence=frame_index)
        rr.set_time("recording_time", duration=elapsed_time_s)
    else:
        rr.set_time_sequence("frame_index", frame_index)
        rr.set_time_seconds("recording_time", elapsed_time_s)


def log_rerun_data(observation: dict[str | Any], action: dict[str | Any]):
    for obs, val in observation.items():
        if isinstance(val, float):
            rr.log(f"observation.{obs}", _rerun_scalar(val))
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                for i, v in enumerate(val):
                    rr.log(f"observation.{obs}_{i}", _rerun_scalar(float(v)))
            else:
                rr.log(f"observation.{obs}", rr.Image(val))
    for act, val in action.items():
        if isinstance(val, float):
            rr.log(f"action.{act}", _rerun_scalar(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action.{act}_{i}", _rerun_scalar(float(v)))


def log_rerun_tactile_data(tactile: np.ndarray, names: Sequence[str]) -> None:
    values = np.asarray(tactile, dtype=np.float32).reshape(-1)
    if len(values) != len(names):
        raise ValueError(f"Tactile value/name count mismatch: {len(values)} != {len(names)}.")

    for name, value in zip(names, values, strict=True):
        rr.log(
            _tactile_entity_path(name),
            _rerun_scalar(float(value)),
        )
