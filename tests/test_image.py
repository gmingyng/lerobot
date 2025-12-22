import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from datasets import Dataset, Features

# 使用你配置文件中的 repo_id
# 注意：这里我们使用本地路径，因为数据已经下载到缓存了
dataset_path = "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds-1"
dataset_path2 = "/home/gming/.cache/huggingface/lerobot/gmingyng/piper-ds"

print(f"--- 正在加载数据集: {dataset_path} ---")
dataset = LeRobotDataset(dataset_path)
ds2 = LeRobotDataset(dataset_path2)
print("\n--- 数据集 Features (结构定义) ---")
print(dataset.features)

# (可选) 取出第一条数据，检查每一项的实际类型
print("\n--- 第一条数据的各项类型 ---")
print(ds2.features)
first_item = ds2[0]
# print("ds fs: ", {ds2[0]})
for key, value in first_item.items():
    print(f"Key: '{key}', Type: {type(value)}")
