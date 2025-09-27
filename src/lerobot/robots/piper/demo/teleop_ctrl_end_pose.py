"""
This script allows direct teleoperation of the Piper robot using a gamepad.
It prints real-time status and action information to the console.

To run this script, execute the following command from the root of the project:
python src/lerobot/robots/piper/ctrl_end_pose.py
"""

import logging
from pprint import pformat

# The main teleoperation function and its configuration
from lerobot.teleoperate import teleoperate, TeleoperateConfig

# Import the specific configurations for our robot and controller.
# We use relative import for PiperConfig because this script is inside the piper directory.
from ..config_piper import PiperConfig
from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig

from lerobot.utils.utils import init_logging

def main():
    """
    Main function to configure and start the teleoperation.
    """
    init_logging()

    # 1. Configure the Piper robot.
    # TODO: IMPORTANT! Verify that the `port` is correct for your hardware setup.
    robot_config = PiperConfig() 

    # 2. Configure the teleoperator to use a gamepad.
    # This uses the default gamepad configuration.
    teleop_config = GamepadTeleopConfig()

    teleop_main_config = TeleoperateConfig(
        teleop=teleop_config,
        robot=robot_config,
        display_data=True,
        fps=30,
    )

    logging.info("Starting Piper teleoperation with the following configuration:")
    logging.info(pformat(teleop_main_config))

    # This function starts the control loop that reads from the gamepad,
    # sends actions to the robot, and prints the data.
    teleoperate(teleop_main_config)

if __name__ == "__main__":
    main()
