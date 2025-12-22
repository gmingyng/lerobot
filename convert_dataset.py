
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

    try:
        # =====================================================================
        # 步骤 1: 加载虚拟拼接数据集
        # =====================================================================
        print("\n正在创建并加载虚拟数据集...")
        combined_dataset = MultiLeRobotDataset(
            repo_ids=repo_ids,
            episodes=episodes_to_load,
        )

        print("\n虚拟数据集创建成功!")
        print(f"  - 拼接后的总 episodes 数: {combined_dataset.num_episodes}")
        print(f"  - 拼接后的总帧数: {combined_dataset.num_frames}")

        # =====================================================================
        # 步骤 2: 将虚拟数据集保存为新的物理数据集
        # =====================================================================
        # ==================================================================
        # 步骤 3: 定义保存路径和数据生成器
        # ==================================================================
        # 这是你新数据集在本地的保存路径，它也将成为新数据集的本地 repo_id
        new_dataset_path = "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds" 
        os.makedirs(new_dataset_path, exist_ok=True)
        
        # 定义一个简单的数据生成器，它会逐帧地从 combined_dataset 中产出数据
        def data_generator():
            for frame in tqdm(combined_dataset, desc="Streaming data frames"):
                yield frame

        # ==================================================================
        # 步骤 4: 从生成器创建并保存物理数据集
        # ==================================================================
        print(f"\n--- 2. 开始将数据集物理保存在: {new_dataset_path} ---")
        
        try:
            # 关键步骤：调用 from_generator 但 **不传入 features 参数**
            # `datasets` 库会自动检查生成的第一条数据，并推断出整个数据集的结构
            print("正在从数据流创建新的数据集对象 (自动推断数据结构)...")
            new_hf_dataset = Dataset.from_generator(data_generator)
            
            print("\n新的数据集对象已在内存中创建，自动推断出的结构为:")
            print(new_hf_dataset.features)

            # 将内存中的新数据集对象以 Arrow 格式保存到磁盘
            print("\n开始写入磁盘...")
            new_hf_dataset.save_to_disk(new_dataset_path)

            print(f"\n✅ 新的组合数据集已成功保存在: {new_dataset_path}")

        except Exception as e:
            import traceback
            print(f"\n在保存过程中发生错误: {e}")
            traceback.print_exc()

        # ==================================================================
        # 步骤 5: (可选) 验证并使用你新保存的数据集
        # ==================================================================
        print(f"\n--- 3. 验证新保存的本地数据集 ---")
        try:
            local_dataset = LeRobotDataset(new_dataset_path)
            print("成功加载本地新数据集！")
            print(local_dataset)
            print("\n可以像这样将其用于 DataLoader:")
            local_dataloader = DataLoader(local_dataset, batch_size=4)
            print(f"DataLoader 创建成功，包含 {len(local_dataloader)} 个批次。")
        except Exception as e:
            print(f"加载新创建的数据集时出错: {e}")

    except Exception as e:
        print(f"\n发生错误: {e}")
        print("\n请确保：")
        print("1. 你有正常的网络连接，以便从 Hugging Face Hub 下载数据集。")
        print("2. `repo_ids` 中指定的数据集存在且你有权限访问。")
        print("3. `episodes_to_load` 中指定的 episode 索引号对于每个数据集都是有效的。")

if __name__ == "__main__":
    main()
        # print("\n虚拟数据集创建成功!")
        # print(combined_dataset)

        # # --- 第4步：像使用普通数据集一样使用它 ---

        # # 获取拼接后的总帧数
        # total_frames = len(combined_dataset)
        # print(f"\n拼接后的总帧数: {total_frames}")

        # # 获取拼接后数据集的特征 (features)
        # # 注意：这会是所有子数据集特征的交集
        # print(f"拼接后数据集的共同特征: {list(combined_dataset.features.keys())}")

        # # 索引一个样本 (它会自动从正确的子数据集中提取)
        # sample_frame = combined_dataset[46]
        # print("\n获取第一个样本帧 (来自第一个数据集):")
        # # 打印样本中部分键的信息
        # print({k: v.shape if hasattr(v, 'shape') else v for k, v in sample_frame.items() if 'image' not in k})
        # print(sample_frame)


        # # 将组合后的数据集送入 PyTorch DataLoader 进行批量加载
        # data_loader = DataLoader(combined_dataset, batch_size=4, shuffle=True, num_workers=2)

        # print("\n成功创建 DataLoader。可以开始训练了。")
