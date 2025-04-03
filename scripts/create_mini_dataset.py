#!/usr/bin/env python3
import os
import glob
from pathlib import Path
import random
import argparse

def create_vox2_lists(vox2_dir, output_dir, tiny_size=100):
    """
    创建VoxCeleb2数据集的训练列表
    
    参数:
        vox2_dir: VOX2数据集根目录路径
        output_dir: 输出目录
        tiny_size: 小型训练集中的视频数量
    """
    vox2_path = Path(vox2_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"扫描VOX2数据集: {vox2_path}")
    
    # 查找所有mp4文件
    mp4_files = []
    for id_dir in vox2_path.glob("id*"):
        if not id_dir.is_dir():
            continue
        for video_dir in id_dir.iterdir():
            if not video_dir.is_dir():
                continue
            for mp4_file in video_dir.glob("*.mp4"):
                mp4_files.append(str(mp4_file))
    
    print(f"找到 {len(mp4_files)} 个视频文件")
    
    # 创建完整训练列表
    full_list_path = output_path / "vox2_full_train.txt"
    with open(full_list_path, 'w') as f:
        for mp4_file in mp4_files:
            f.write(f"{mp4_file}\n")
    
    print(f"完整训练列表保存至: {full_list_path}")
    
    # 创建小型训练列表
    if tiny_size > len(mp4_files):
        tiny_size = len(mp4_files)
    
    tiny_files = random.sample(mp4_files, tiny_size)
    tiny_list_path = output_path / "vox2_tiny_train.txt"
    with open(tiny_list_path, 'w') as f:
        for mp4_file in tiny_files:
            f.write(f"{mp4_file}\n")
    
    print(f"小型训练列表 ({tiny_size} 个视频) 保存至: {tiny_list_path}")
    
    # 创建单一身份的训练列表
    single_id_files = []
    for mp4_file in mp4_files:
        if "id00012" in mp4_file:  # 使用id00012作为示例
            single_id_files.append(mp4_file)
    
    single_id_list_path = output_path / "vox2_single_id_train.txt"
    with open(single_id_list_path, 'w') as f:
        for mp4_file in single_id_files:
            f.write(f"{mp4_file}\n")
    
    print(f"单一身份训练列表 ({len(single_id_files)} 个视频) 保存至: {single_id_list_path}")
    
    return {
        "full": str(full_list_path),
        "tiny": str(tiny_list_path),
        "single_id": str(single_id_list_path)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="创建VoxCeleb2训练列表")
    parser.add_argument("--vox2_dir", type=str, default="/root/HiFiVFS/data/VOX2/dev",
                       help="VOX2数据集根目录")
    parser.add_argument("--output_dir", type=str, default="/root/HiFiVFS/data",
                       help="输出目录")
    parser.add_argument("--tiny_size", type=int, default=100,
                       help="小型训练集中的视频数量")
    
    args = parser.parse_args()
    lists = create_vox2_lists(args.vox2_dir, args.output_dir, args.tiny_size)
    
    print("\n使用训练列表的示例命令:")
    print(f"python train_fal.py --config ./configs/vox2_tiny_config.yaml")