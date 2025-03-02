"""
性能测试：测试进化系统在不同配置下的性能
"""

import sys
import os
import logging
import tempfile
from pathlib import Path
import yaml
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from engine.evolution_engine import EvolutionEngine

# 设置日志
logging.basicConfig(
    level=logging.WARNING,  # 使用WARNING级别减少日志输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_config(population_size, max_population):
    """创建测试配置"""
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "test_config.yaml"
    
    config = {
        'evolution': {
            'generation_limit': 20,
            'initial_population_size': population_size,
            'max_population_size': max_population,
            'base_mutation_rate': 0.3
        },
        'monitor': {
            'log_interval': 1,
            'report_interval': 10,  # 减少报告生成频率
            'visualization_interval': 10,  # 减少可视化生成频率
            'metrics': [
                'mutation_rate',
                'inheritance_rate',
                'attribute_count',
                'concept_diversity'
            ],
            'storage': {
                'base_path': temp_dir,
                'reports': "reports/",
                'visualizations': "visualizations/",
                'metrics': "metrics/"
            },
            'visualization': {
                'enabled': False,  # 禁用可视化以提高性能
                'format': "png",
                'dpi': 100
            }
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 创建必要的目录
    Path(temp_dir, "reports").mkdir(exist_ok=True)
    Path(temp_dir, "visualizations").mkdir(exist_ok=True)
    Path(temp_dir, "metrics").mkdir(exist_ok=True)
    
    return config_path, temp_dir

def test_population_size_performance():
    """测试不同种群大小的性能"""
    print("\n=== 测试不同种群大小的性能 ===")
    
    population_sizes = [5, 10, 20, 30]
    generations = 10
    times = []
    
    for size in population_sizes:
        config_path, temp_dir = create_test_config(size, size * 2)
        
        try:
            # 创建进化引擎
            engine = EvolutionEngine(config_path)
            
            # 计时运行
            start_time = time.time()
            engine.run(generations=generations)
            elapsed_time = time.time() - start_time
            
            times.append(elapsed_time)
            print(f"种群大小: {size}, 运行时间: {elapsed_time:.2f}秒")
            
        finally:
            # 清理
            shutil.rmtree(temp_dir)
    
    # 绘制性能图表
    plt.figure(figsize=(10, 6))
    plt.bar(population_sizes, times, color='blue')
    plt.xlabel('种群大小')
    plt.ylabel('运行时间 (秒)')
    plt.title(f'不同种群大小的性能 ({generations}代)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, v in enumerate(times):
        plt.text(population_sizes[i], v + 0.1, f'{v:.2f}s', ha='center')
    
    # 保存图表
    plt.savefig('population_size_performance.png')
    print(f"性能图表已保存到 population_size_performance.png")

def test_generation_performance():
    """测试不同代数的性能"""
    print("\n=== 测试不同代数的性能 ===")
    
    generation_counts = [5, 10, 15, 20]
    population_size = 10
    times = []
    
    for gens in generation_counts:
        config_path, temp_dir = create_test_config(population_size, population_size * 2)
        
        try:
            # 创建进化引擎
            engine = EvolutionEngine(config_path)
            
            # 计时运行
            start_time = time.time()
            engine.run(generations=gens)
            elapsed_time = time.time() - start_time
            
            times.append(elapsed_time)
            print(f"代数: {gens}, 运行时间: {elapsed_time:.2f}秒")
            
        finally:
            # 清理
            shutil.rmtree(temp_dir)
    
    # 绘制性能图表
    plt.figure(figsize=(10, 6))
    plt.bar(generation_counts, times, color='green')
    plt.xlabel('代数')
    plt.ylabel('运行时间 (秒)')
    plt.title(f'不同代数的性能 (种群大小: {population_size})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, v in enumerate(times):
        plt.text(generation_counts[i], v + 0.1, f'{v:.2f}s', ha='center')
    
    # 保存图表
    plt.savefig('generation_performance.png')
    print(f"性能图表已保存到 generation_performance.png")

if __name__ == "__main__":
    test_population_size_performance()
    test_generation_performance() 