"""
项目清理工具：删除不必要的文件和目录
"""

import os
import shutil
from pathlib import Path
import sys

def cleanup_project():
    """清理项目文件"""
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent
    
    # 要删除的文件列表
    files_to_delete = [
        "README-0.1.md",
        "README-0.2.md",
        "population_size_performance.png",
        "generation_performance.png",
        "integration_debug.log"
    ]
    
    # 要删除的目录列表
    dirs_to_delete = [
        "AI-Evolution-0.x/AI-Evolution-0.x"  # 重复的目录结构
    ]
    
    # 删除文件
    for file_path in files_to_delete:
        full_path = root_dir / file_path
        if full_path.exists():
            print(f"删除文件: {full_path}")
            full_path.unlink()
    
    # 删除目录
    for dir_path in dirs_to_delete:
        full_path = root_dir / dir_path
        if full_path.exists():
            print(f"删除目录: {full_path}")
            shutil.rmtree(full_path)
    
    # 清理__pycache__目录
    for pycache_dir in root_dir.glob("**/__pycache__"):
        print(f"删除缓存目录: {pycache_dir}")
        shutil.rmtree(pycache_dir)
    
    # 清理.pyc文件
    for pyc_file in root_dir.glob("**/*.pyc"):
        print(f"删除编译文件: {pyc_file}")
        pyc_file.unlink()
    
    print("项目清理完成")

if __name__ == "__main__":
    cleanup_project() 