
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from datasets import Dataset, Features
import shutil
from tqdm import tqdm
import os

def main():
    """
    这个演示脚本展示了如何从多个 LeRobot 数据集中选取指定的 episodes，
    然后将它们组合成一个虚拟数据集用于训练，并最终保存为一个全新的物理数据集。
    """
    print("开始从多个数据集中拼接指定的 episodes...")

    # 1. 定义源数据集在 Hugging Face Hub 上的仓库 ID。
    #    这些是根据你提供的列表生成的。
    repo_ids = [
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-1",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-2",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-3",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-4",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-5",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-7",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-8",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-9",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-10",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-11",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-12",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-13",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-14",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-15",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-16",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-17",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-18",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-19",
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-20",
    ]

    # 2. 定义一个字典，来指定你想从每个数据集中加载哪些 episodes。
    #    '1' 代表加载该 episode, '0' 代表跳过。
    episodes_to_load = {
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-1": [0, 1],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-2": [0],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-3": [0, 1],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-4": [0, 1],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-5": [0, 1],
        # piper-ds-6 被跳过，因为其可用 episode 为 '00'
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-7": [0, 1],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-8": [0, 1],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-9": [0, 1],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-10": [0, 1],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-11": [0, 1],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-12": [0, 1],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-13": [0, 1],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-14": [0, 1, 2, 3],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-15": [0, 1, 2, 3],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-16": [0, 1, 2],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-17": [0, 1, 2, 3],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-18": [0, 1, 2, 3],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-19": [0, 1, 2, 3],
        "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-20": [0, 1, 2, 3],
    }

    print(f"\n准备从以下数据集中加载: {repo_ids}")
    print(f"指定的 Episodes: {episodes_to_load}")

    print("\n正在创建并加载虚拟数据集...")
    combined_dataset = MultiLeRobotDataset(
        repo_ids=repo_ids,
        episodes=episodes_to_load,
    )

    print("\n虚拟数据集创建成功!")
    print(f"  - 拼接后的总 episodes 数: {combined_dataset.num_episodes}")
    print(f"  - 拼接后的总帧数: {combined_dataset.num_frames}")

    
    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id="gmingyng/piper-ds-tt",
        root="/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-tt",
        robot_type="piper",
        fps=30,
        features={
        "action": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": [[
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper"
            ]]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": [ [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper"
             ] ]
        },
        "observation.images.wrist_camera": {
            "dtype": "image",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ]
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    },
        image_writer_threads=0,
        image_writer_processes=8,
        use_videos=False,
    )

    print(combined_dataset)
    # # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # # You can modify this for your own data format
    # for raw_dataset_name in RAW_DATASET_NAMES:
    #     raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    #     for episode in raw_dataset:
    #         for step in episode["steps"].as_numpy_iterator():
    #             dataset.add_frame(
    #                 {
    #                     "image": step["observation"]["image"],
    #                     "wrist_image": step["observation"]["wrist_image"],
    #                     "state": step["observation"]["state"],
    #                     "actions": step["action"],
    #                     "task": step["language_instruction"].decode(),
    #                 }
    #             )
    #         dataset.save_episode()

if __name__ == "__main__":
    main()
