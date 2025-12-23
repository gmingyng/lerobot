#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


from ..config import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    port: str = "can0"

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_right_wrist": OpenCVCameraConfig(
                index_or_path=10,
                fps=30,
                width=640,
                height=480,
            ),
            # "cam_left_wrist": OpenCVCameraConfig(
            #     index_or_path=5,
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),    
            "cam_high": OpenCVCameraConfig(
                index_or_path=4,
                fps=30,
                width=640,
                height=480,
            ),    

        }
    )

class BiPiperConfig(RobotConfig):
    left_arm_port: str = "can0"
    right_arm_port: str = "can1"

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_right_wrist": OpenCVCameraConfig(
                index_or_path=10,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": OpenCVCameraConfig(
                index_or_path=5,
                fps=30,
                width=640,
                height=480,
            ),    
            "cam_high": OpenCVCameraConfig(
                index_or_path=4,
                fps=30,
                width=640,
                height=480,
            ),    

        }
    )

    # Set to `True` for backward compatibility with previous policies/dataset
    # use_degrees: bool = False
