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

"""
Records robot data together with one or two dual-route USB tactile sensors.
"""

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    piper,
    so100_follower,
    so101_follower,
)
from lerobot.tactile import TactileConfig, TactileReader, tactile_feature
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    so100_leader,
    so101_leader,
)
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import (
    _init_rerun,
    log_rerun_data,
    log_rerun_tactile_data,
    set_rerun_time,
)


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 15
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = False
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 1

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    tactile: TactileConfig = field(default_factory=TactileConfig)
    # Teleop and Policy are optional and default to None, kept for compatibility but ignored in logic
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def discard_current_episode(dataset: LeRobotDataset) -> None:
    if dataset.image_writer is not None:
        dataset.image_writer.wait_until_done()
    dataset.clear_episode_buffer()


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    key_state: dict,  # Added key_state for manual control
    tactile_reader: TactileReader,
    dataset: LeRobotDataset | None = None,
    # Removed teleop and policy arguments
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    loop_type: str = "episode",
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 80  # Fallback width

    start_episode_t = time.perf_counter()

    # Reset flags when entering the loop
    if loop_type == "episode":
        key_state["stop_episode"] = False
        key_state["discard_episode"] = False
    elif loop_type == "reset":
        key_state["start_next"] = False

    loop_result = "finished"

    # Modified Loop Condition: Loop until manual break or exit_early
    while True:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Discard this episode and immediately begin recording it again.
        if loop_type == "episode" and key_state["discard_episode"]:
            print("\nDiscard current episode (m pressed).")
            key_state["discard_episode"] = False
            loop_result = "discard"
            break

        # Check for manual 'p' stop during episode recording
        if loop_type == "episode" and key_state["stop_episode"]:
            print("\nManual stop triggered (p pressed).")
            key_state["stop_episode"] = False  # Reset flag
            break

        # Check for manual 'a' start during reset/wait period
        if loop_type == "reset" and key_state["start_next"]:
            print("\nManual start triggered (a pressed).")
            key_state["start_next"] = False  # Reset flag
            break

        observation = robot.get_observation()

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        # Simplified Logic: Directly get action from Piper (Lead-through teaching)
        # Assuming we are just recording the robot's current state/control as the action
        action = robot.get_piper_ctrl()

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        sent_action = robot.send_action(action)
        if sent_action is None:
            sent_action = action

        tactile_values = None
        if (dataset is not None and loop_type == "episode") or display_data:
            tactile_values = tactile_reader.read()

        if dataset is not None and loop_type == "episode":
            # Prepare action for dataset: append task_completion
            dataset_action = sent_action.copy()
            dataset_action["task_completion"] = 0.0  # Default to 0, will update later if needed

            action_frame = build_dataset_frame(dataset.features, dataset_action, prefix="action")

            frame = {**observation_frame, **action_frame, "tactile": tactile_values}
            dataset.add_frame(frame, task=single_task)

        if display_data:
            set_rerun_time(
                key_state["rerun_frame_index"],
                time.perf_counter() - key_state["rerun_start_time"],
            )
            log_rerun_data(observation, action)
            log_rerun_tactile_data(tactile_values, dataset.features["tactile"]["names"])
            key_state["rerun_frame_index"] += 1

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t

        # Modified UI Feedback
        if loop_type == "episode":
            text = f"Recording... {timestamp:.1f}s | 'p': finish | 'm': discard and restart"
            color_code = "\033[92m"  # Green
        else:  # reset
            text = f"Waiting... Time: {timestamp:.1f}s | Press 'a' to start next episode"
            color_code = "\033[94m"  # Blue

        padding = " " * ((terminal_width - len(text)) // 2)

        # Single-line, centered, colored countdown
        print(f"{padding}{color_code}{text}\033[0m", end="\r", flush=True)

    # Clear the status line
    print(" " * terminal_width, end="\r")
    return loop_result


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    # Removed teleop and policy initialization logic as they are not used in the command
    # teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    # policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    # 1. MODIFY ACTION FEATURES: Add 1 dimension for task_completion
    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)

    # Extend the action shape and names
    act_feat_config = action_features["action"]
    original_shape = act_feat_config["shape"]
    act_feat_config["shape"] = (original_shape[0] + 1,)
    act_feat_config["names"].append("task_completion")

    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    tactile_features = tactile_feature(len(cfg.tactile.ports))
    dataset_features = {**action_features, **obs_features, **tactile_features}

    if cfg.display_data:
        observation_names = [
            name for name, feature_type in robot.observation_features.items() if feature_type is float
        ]
        image_names = [
            name
            for name, feature_type in robot.observation_features.items()
            if isinstance(feature_type, tuple)
        ]
        _init_rerun(
            session_name="recording",
            observation_names=observation_names,
            action_names=list(robot.action_features),
            tactile_names=tactile_features["tactile"]["names"],
            image_names=image_names,
        )

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        # Removed sanity_check_dataset_name since policy is None
        # sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    tactile_reader = TactileReader(cfg.tactile)
    key_state = {
        "stop_episode": False,
        "start_next": False,
        "discard_episode": False,
        "rerun_frame_index": 0,
        "rerun_start_time": time.perf_counter(),
    }

    def on_press(key):
        try:
            if hasattr(key, "char"):
                if key.char == "p":
                    key_state["stop_episode"] = True
                elif key.char == "a":
                    key_state["start_next"] = True
                elif key.char == "m":
                    key_state["discard_episode"] = True
        except AttributeError:
            pass

    listener = None
    input_listener = None
    robot.connect()
    try:
        tactile_reader.connect()
        listener, events = init_keyboard_listener()
        from pynput import keyboard

        input_listener = keyboard.Listener(on_press=on_press)
        input_listener.start()

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)

                # RECORDING LOOP
                episode_result = record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    key_state=key_state,  # Pass key state
                    tactile_reader=tactile_reader,
                    dataset=dataset,
                    # Removed teleop and policy arguments
                    control_time_s=None,  # Deprecated in favor of manual control
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    loop_type="episode",
                )

                if episode_result == "discard":
                    logging.info("Discarding the current episode and restarting immediately.")
                    discard_current_episode(dataset)
                    continue

                # 3. POST-PROCESSING: Mark last 10 frames as completed in the ACTION tensor
                if not events["exit_early"] and not events["rerecord_episode"]:
                    # Access the action buffer directly
                    if "action" in dataset.episode_buffer:
                        buffer_len = len(dataset.episode_buffer["action"])
                        start_idx = max(0, buffer_len - 10)

                        logging.info(
                            f"Marking frames {start_idx} to {buffer_len} task_completion bit (last dim) to 1."
                        )

                        for action_values in dataset.episode_buffer["action"][start_idx:]:
                            action_values[-1] = 1.0

                # Execute wait loop (Reset phase)
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Waiting for next episode. Press 'a' to start.", cfg.play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        key_state=key_state,  # Pass key state
                        tactile_reader=tactile_reader,
                        dataset=dataset,
                        # Removed teleop argument
                        control_time_s=None,  # Deprecated
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        loop_type="reset",
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

        log_say("Stop recording", cfg.play_sounds, blocking=True)

    finally:
        if input_listener is not None:
            input_listener.stop()
        tactile_reader.disconnect()
        robot.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
