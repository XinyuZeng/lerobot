from dataset_tools import merge_datasets
from lerobot_dataset import LeRobotDataset
from pathlib import Path
import argparse


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets into one.")
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        required=True,
        help="Dataset directories to merge, separated by commas (e.g., /dir1,/dir2,/dir3).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_dataset",
        help="Directory to save the merged dataset (default: ./merged_dataset).",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default="Galaxea/merged_dataset",
        help="Repo ID for the merged dataset (default: Galaxea/merged_dataset).",
    )

    args = parser.parse_args()

    # 1. 解析逗号分隔的数据集目录，转换为Path对象
    dataset_dirs = [Path(dir_str.strip()) for dir_str in args.dataset_dirs.split()]
    
    # 2. 验证目录是否存在，并创建LeRobotDataset实例列表
    datasets = []
    for idx, dir_path in enumerate(dataset_dirs):
        if not dir_path.exists():
            raise FileNotFoundError(f"Dataset directory {dir_path} does not exist!")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"{dir_path} is not a valid directory!")
        
        # 按你的示例创建LeRobotDataset（repo_id可自定义，这里用目录名+序号）
        repo_id = f"dataset_{idx}_{dir_path.name}"
        ds = LeRobotDataset(repo_id=repo_id, root=str(dir_path))
        datasets.append(ds)
        print(f"Successfully loaded dataset from {dir_path}")

    # 3. 调用合并函数
    print(f"Starting to merge {len(datasets)} datasets...")
    merged_dataset = merge_datasets(
        datasets=datasets,          # 传入LeRobotDataset实例列表
        output_repo_id=args.output_repo_id,
        output_dir=Path(args.output_dir)
    )

    print(f"Merge completed! Merged dataset saved to {args.output_dir}")
    return merged_dataset


if __name__ == "__main__":
    main()