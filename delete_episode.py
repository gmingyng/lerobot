import argparse
import shutil
from pathlib import Path
import torch
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.constants import HF_LEROBOT_HOME

def delete_episode(repo_id, episode_index, root=None, output_dir=None):
    if root is None:
        root = HF_LEROBOT_HOME / repo_id
    else:
        root = Path(root)

    if not root.exists():
        print(f"Dataset root not found: {root}")
        return

    print(f"Loading dataset from {root}")
    dataset = LeRobotDataset(repo_id, root=root)
    
    print(f"Dataset has {dataset.num_episodes} episodes.")
    if episode_index < 0 or episode_index >= dataset.num_episodes:
        print(f"Invalid episode index: {episode_index}")
        return

    # Create new dataset path
    if output_dir:
        new_root = Path(output_dir)
        new_repo_id = new_root.name 
    else:
        new_repo_id = f"{repo_id}_cleaned"
        new_root = root.parent / f"{root.name}_cleaned"

    if new_root.exists():
        print(f"Removing existing temporary directory: {new_root}")
        shutil.rmtree(new_root)

    print(f"Creating new dataset at {new_root}")
    
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=dataset.fps,
        features=dataset.features,
        robot_type=dataset.robot_type,
        root=new_root,
        use_videos=dataset.video
    )
    
    for ep_idx in range(dataset.num_episodes):
        if ep_idx == episode_index:
            print(f"Skipping episode {ep_idx}")
            continue
        
        # Get start and end frame indices
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()
        
        print(f"Copying episode {ep_idx} (frames {from_idx} to {to_idx})")
        
        # Get task
        tasks = dataset.episodes[ep_idx]["tasks"]
        task = tasks[0] if isinstance(tasks, list) else tasks
        
        for frame_idx in tqdm(range(from_idx, to_idx), leave=False):
            frame = dataset[frame_idx]
            
            # Filter frame to only include features
            filtered_frame = {k: v for k, v in frame.items() if k in dataset.features}
            
            new_dataset.add_frame(filtered_frame, task=task)
            
        new_dataset.save_episode()
        
    print(f"New dataset created at {new_root}")
    print("You can now verify the new dataset and replace the old one if satisfied.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete an episode from a LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, required=True, help="Repo ID of the dataset")
    parser.add_argument("--episode-index", type=int, required=True, help="Index of the episode to delete")
    parser.add_argument("--root", type=str, default=None, help="Root directory of the dataset (optional)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for the new dataset (optional)")
    
    args = parser.parse_args()
    
    delete_episode(args.repo_id, args.episode_index, args.root, args.output_dir)
