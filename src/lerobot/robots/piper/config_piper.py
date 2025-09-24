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
# from lerobot.motors.configs import PiperMotorsBusConfig
# from lerobot.motors.configs import MotorsBusConfig

# @RobotConfig.register_subclass("piper_old")
# @dataclass
# class PiperRobotConfig(RobotConfig):
#     inference_time: bool
    
#     follower_arm: dict[str, MotorsBusConfig] = field(
#         default_factory=lambda: {
#             "main": PiperMotorsBusConfig(
#                 can_name="can0",
#                 motors={
#                     # name: (index, model)
#                     "joint_1": [1, "agilex_piper"],
#                     "joint_2": [2, "agilex_piper"],
#                     "joint_3": [3, "agilex_piper"],
#                     "joint_4": [4, "agilex_piper"],
#                     "joint_5": [5, "agilex_piper"],
#                     "joint_6": [6, "agilex_piper"],
#                     "gripper": (7, "agilex_piper"),
#                 },
#             ),
#         }
#     )

#     cameras: dict[str, CameraConfig] = field(
#         default_factory=lambda: {
#             "one": OpenCVCameraConfig(
#                 camera_index=0,
#                 fps=30,
#                 width=640,
#                 height=480,
#             ),
#             "two": OpenCVCameraConfig(
#                 camera_index=2,
#                 fps=30,
#                 width=640,
#                 height=480,
#             ),
#             # "cam_left_wrist": IntelRealSenseCameraConfig(
#             #     serial_number=218622272670,
#             #     fps=30,
#             #     width=640,
#             #     height=480,
#             # ),
#             # "cam_right_wrist": IntelRealSenseCameraConfig(
#             #     serial_number=130322272300,
#             #     fps=30,
#             #     width=640,
#             #     height=480,
#             # ),

#         }
#     )



@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    port: str = "can0"

    # disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    # max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist_camera": OpenCVCameraConfig(
                index_or_path=6,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
    # Set to `True` for backward compatibility with previous policies/dataset
    # use_degrees: bool = False
