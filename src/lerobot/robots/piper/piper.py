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

import logging
from functools import cached_property
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from piper_sdk import *
from lerobot.cameras import make_cameras_from_configs
# from lerobot.motors import Motor, MotorNormMode
# from lerobot.motors.feetech import FeetechMotorsBus
# from lerobot.motors.piper_motor import PiperMotorsBus

from .config_piper import PiperConfig

logger = logging.getLogger(__name__)


class Piper(Robot):
    """
    TODO: Add a description of your robot class here.
    This class should handle the connection, control, and observation of your robot.
    """

    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.config = config
        # TODO: Initialize your robot's hardware interface here.
        # Example: self.bus = MyRobotBus(port=self.config.port)
        # Example: self.cameras = make_cameras_from_configs(config.cameras)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._arm_connected = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Define the motor features for the arm"""
        motor_features = {}
        # Arm: joint_1 to joint_6 and gripper
        for i in range(1, 7):
            motor_features[f"joint_{i}"] = float
        motor_features["gripper"] = float

        return motor_features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def is_connected(self) -> bool:
        arm_connected = self._arm_connected is not None
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return arm_connected and cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        try:
            # Connect left arm
            logger.info(f"Connecting to arm on CAN port: {self.config.port}")
            self._arm_connected = C_PiperInterface_V2(self.config.port)
            self._arm_connected.ConnectPort(True)

            # Connect cameras
            for cam in self.cameras.values():
                cam.connect()

            logger.info("Piper robot connected successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Piper robot: {e}")
            raise

    def disconnect(self) -> None:
        """
        Disconnect from the Piper arm and cameras
        """
        try:
            if self._arm_connected is not None:
                try:
                    self._arm_connected.DisconnectPort()
                    logger.info("Arm disconnected from CAN port")
                except Exception as e:
                    logger.warning(f"Error disconnecting arm: {e}")
                self._arm_connected = None

            for cam in self.cameras.values():
                cam.disconnect()

            logger.info("Piper robot disconnected successfully")

        except Exception as e:
            logger.error(f"Error during Piper robot disconnect: {e}")

    def get_observation(self) -> dict:
        """
        Capture current joint positions and camera images
        """
        if not self.is_connected:
            raise RuntimeError("Piper robot is not connected")

        observation = {}

        try:
            # Get arm joint positions
            joint_msgs = self._arm_connected.GetArmJointMsgs()
            gripper_msgs = self._arm_connected.GetArmGripperMsgs()

            # Parse joint positions for the arm
            # Based on the format: ArmMsgFeedBackJointStates with Joint 1-6 values
            self._parse_joint_messages(joint_msgs, observation)

            # Parse gripper position for the arm
            # Based on the format: ArmMsgFeedBackGripper with grippers_angle
            self._parse_gripper_messages(gripper_msgs, observation)

            # Capture camera images
            for cam_name, cam in self.cameras.items():
                observation[cam_name] = cam.async_read()

        except Exception as e:
            logger.error(f"Error capturing observation: {e}")
            raise

        return observation

    def _parse_joint_messages(self, joint_msgs, observation: dict) -> None:
        """
        Parse joint messages from piper SDK format.
        Expected format: joint_msgs.joint_state.joint_1, joint_msgs.joint_state.joint_2, etc.
        """

        try:
            # Extract joint values using direct attribute access
            for i in range(1, 7):
                joint_key = f"joint_{i}"
                try:
                    # Access joint value using joint_msgs.joint_state.joint_{i}
                    joint_attr = f"joint_{i}"
                    joint_value = getattr(joint_msgs.joint_state, joint_attr)
                    observation[joint_key] = float(joint_value)
                except AttributeError:
                    logger.warning(f"Joint {i} attribute not found in joint_state")
                    observation[joint_key] = 0.0
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse joint {i} value")
                    observation[joint_key] = 0.0


        except Exception as e:
            logger.error(f"Error parsing joint messages: {e}")
            # Fallback: set all joints to 0
            for i in range(1, 7):
                observation[f"joint_{i}"] = 0.0

    def _parse_gripper_messages(self, gripper_msgs, observation: dict) -> None:
        """
        Parse gripper messages from piper SDK format.
        Expected format: ArmGripper object with gripper_state.grippers_angle attribute
        grippers_angle is in 0.001mm units, needs conversion to mm.
        """
        try:
            # Access gripper_state.grippers_angle - convert from 0.001mm to mm
            angle_raw = gripper_msgs.gripper_state.grippers_angle
            angle_mm = float(angle_raw) / 1000.0
            observation[f"gripper"] = angle_mm
            logger.debug(f"gripper position: {angle_mm}mm (raw: {angle_raw})")

        except Exception as e:
            logger.error(f"Error parsing gripper messages: {e}")
            logger.error(f"Gripper message type: {type(gripper_msgs)}")
            logger.error(f"Gripper message content: {gripper_msgs}")
            observation[f"gripper"] = 0.0

    def send_action(self, action: dict) -> None:
        """
        Send action to the Piper arm
        Note: This is a placeholder since you mentioned the robot movement is handled separately
        """
        if not self.is_connected:
            raise RuntimeError("Piper robot is not connected")

        # Since you mentioned that robot movement is handled separately,
        # we don't need to implement actual movement commands here
        # This method is required by the Robot interface but can be a no-op
        logger.debug("send_action called - movement handled separately")
        pass

    @property
    def is_calibrated(self) -> bool:
        """Piper robots are assumed to be always calibrated"""
        return True

    def calibrate(self) -> None:
        """Piper robots don't require manual calibration"""
        logger.info("Piper robot calibration - no action needed, assumed calibrated")
        pass

    def configure(self) -> None:
        """Configure the Piper robot - no specific configuration needed"""
        logger.info("Piper robot configuration - no action needed")
        pass
