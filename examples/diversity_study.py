"""
概念多样性研究

这个示例专注于研究概念多样性的形成和维持机制，通过不同的多样性促进策略，
观察种群多样性的变化和影响。
"""

import os
import sys
import yaml
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import traceback
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter
from collections import Counter, defaultdict

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.evolution_engine import EvolutionEngine

# 首先定义单个实验的运行函数
def run_diversity_experiment(experiment_id, strategy, config_path, generations=50):
    """运行单个多样性实验
    
    Args:
        experiment_id: 实验ID
        strategy: 多样性策略名称
        config_path: 配置文件路径
        generations: 要运行的代数
        
    Returns:
        实验结果字典
    """
    print(f"开始多样性实验 {experiment_id}: {strategy}...")
    
    # 初始化进化引擎
    engine = EvolutionEngine(config_path)
    
    # 运行进化
    start_time = time.time()
    
    try:
        final_report = engine.run(generations=generations)
    except Exception as e:
        print(f"运行过程中出错: {e}")
        traceback.print_exc()
        return None
    
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"实验 {experiment_id} ({strategy}) 完成! 总运行时间: {runtime:.2f}秒")
    
    # 保存结果
    results_dir = f'./results/examples/diversity/exp_{experiment_id}'
    results_file = os.path.join(results_dir, 'results.pkl')
    engine.save_results(results_file)
    
    # 保存历史数据
    history_file = os.path.join(results_dir, 'metrics', 'history.csv')
    engine.analytics.save_history(history_file)
    
    # 生成报告
    report_path = os.path.join(results_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write(final_report)
    
    # 返回关键结果
    history = engine.analytics.history
    
    # 计算关键指标
    final_diversity = history['concept_diversity'].iloc[-1] if 'concept_diversity' in history.columns else 0
    avg_diversity = history['concept_diversity'].mean() if 'concept_diversity' in history.columns else 0
    final_population = history['population_size'].iloc[-1] if 'population_size' in history.columns else 0
    avg_attributes = history['avg_attributes'].mean() if 'avg_attributes' in history.columns else 0
    
    # 手动计算属性熵
    attribute_entropy = 0
    if 'unique_attributes' in history.columns and final_population > 0:
        # 获取最后一代的所有个体
        population = engine.population
        
        if population and len(population) > 0:
            # 收集所有属性及其出现次数
            all_attributes = []
            for agent in population:
                all_attributes.extend(agent.attributes.keys())
            
            # 计算属性频率
            attribute_counts = Counter(all_attributes)
            total_attributes = sum(attribute_counts.values())
            
            # 计算熵
            if total_attributes > 0:
                attribute_entropy = 0
                for count in attribute_counts.values():
                    p = count / total_attributes
                    attribute_entropy -= p * np.log2(p)
    
    # 计算多样性指数 (综合多个指标)
    diversity_index = (final_diversity * 0.4 + attribute_entropy * 0.3 + (avg_attributes/10) * 0.3)
    
    return {
        'experiment_id': experiment_id,
        'experiment': strategy,
        'strategy': strategy,
        'final_diversity': final_diversity,
        'avg_diversity': avg_diversity,
        'final_population': final_population,
        'avg_attributes': avg_attributes,
        'attribute_entropy': attribute_entropy,
        'diversity_index': diversity_index,
        'runtime': runtime
    }

# 然后定义配置创建函数
def create_diversity_config(strategy, experiment_id):
    """创建多样性实验配置
    
    Args:
        strategy: 多样性策略名称
        experiment_id: 实验ID
        
    Returns:
        tuple: (配置文件路径, 策略名称)
    """
    # 根据策略设置参数
    if strategy == "基础进化":
        mutation_rate = 0.3
        selection_pressure = 0.7
        inheritance_factor = 0.5
        niche_factor = 0.0
    elif strategy == "高变异率":
        mutation_rate = 0.6
        selection_pressure = 0.7
        inheritance_factor = 0.5
        niche_factor = 0.0
    elif strategy == "低选择压力":
        mutation_rate = 0.3
        selection_pressure = 0.4
        inheritance_factor = 0.5
        niche_factor = 0.0
    elif strategy == "生态位分化":
        mutation_rate = 0.3
        selection_pressure = 0.7
        inheritance_factor = 0.5
        niche_factor = 0.3
    elif strategy == "平衡策略":
        mutation_rate = 0.4
        selection_pressure = 0.6
        inheritance_factor = 0.6
        niche_factor = 0.2
    else:
        # 默认设置
        mutation_rate = 0.3
        selection_pressure = 0.7
        inheritance_factor = 0.5
        niche_factor = 0.0
    
    config = {
        'evolution': {
            'generation_limit': 100,
            'initial_population_size': 30,
            'max_population_size': 100,
            'base_mutation_rate': mutation_rate,
            'selection_pressure': selection_pressure,
            'inheritance_factor': inheritance_factor,
            'niche_factor': niche_factor,  # 生态位因子，用于促进多样性
            'diversity_strategy': strategy
        },
        'agent': {
            'enable_memory': False,  # 禁用记忆功能
        },
        'monitor': {
            'log_interval': 1,
            'report_interval': 5,
            'visualization_interval': 5,
            'metrics': [
                'mutation_rate',
                'inheritance_rate',
                'avg_attributes',
                'unique_attributes',
                'concept_diversity',
                'population_size',
                'total_concepts',
                'natural_mutations',
                'safe_triggers',
                'attribute_entropy'   # 记录属性熵
            ],
            # 添加存储配置
            'storage': {
                'base_path': f'./results/examples/diversity/exp_{experiment_id}',
                'reports': 'reports',
                'visualizations': 'visualizations',
                'metrics': 'metrics'
            },
            # 添加可视化配置
            'visualization': {
                'enabled': True,
                'format': 'png',
                'dpi': 300
            }
        }
    }
    
    # 创建存储目录
    base_dir = f'./results/examples/diversity/exp_{experiment_id}'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'metrics'), exist_ok=True)
    
    # 保存配置文件
    config_path = os.path.join(base_dir, 'diversity_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"多样性实验配置已保存到: {config_path}")
    
    # 返回配置路径和策略名称
    return config_path, strategy

# 定义图表创建函数
def create_diversity_comparison_charts(experiment_results, results_dir):
    """创建多样性比较图表
    
    Args:
        experiment_results: 实验结果字典
        results_dir: 结果保存目录
    """
    print("创建多样性比较图表...")
    
    # 创建图表目录
    charts_dir = os.path.join(results_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # 转换为DataFrame
    results_df = pd.DataFrame.from_dict(experiment_results, orient='index')
    
    # 确保有experiment列
    if 'experiment' not in results_df.columns and 'strategy' in results_df.columns:
        results_df['experiment'] = results_df['strategy']
    
    # 1. 多样性指数比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['experiment'], results_df['diversity_index'], color='skyblue')
    plt.title('不同策略的多样性指数比较', fontsize=16)
    plt.xlabel('多样性策略', fontsize=14)
    plt.ylabel('多样性指数', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'diversity_index_comparison.png'))
    
    # 2. 概念多样性比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['experiment'], results_df['final_diversity'], color='lightgreen')
    plt.title('不同策略的最终概念多样性比较', fontsize=16)
    plt.xlabel('多样性策略', fontsize=14)
    plt.ylabel('概念多样性', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'concept_diversity_comparison.png'))
    
    # 3. 属性熵比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['experiment'], results_df['attribute_entropy'], color='salmon')
    plt.title('不同策略的属性熵比较', fontsize=16)
    plt.xlabel('多样性策略', fontsize=14)
    plt.ylabel('属性熵', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'attribute_entropy_comparison.png'))
    
    # 4. 平均属性数比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['experiment'], results_df['avg_attributes'], color='mediumpurple')
    plt.title('不同策略的平均属性数比较', fontsize=16)
    plt.xlabel('多样性策略', fontsize=14)
    plt.ylabel('平均属性数', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'avg_attributes_comparison.png'))
    
    # 5. 运行时间比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['experiment'], results_df['runtime'], color='lightcoral')
    plt.title('不同策略的运行时间比较', fontsize=16)
    plt.xlabel('多样性策略', fontsize=14)
    plt.ylabel('运行时间 (秒)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'runtime_comparison.png'))
    
    print(f"多样性比较图表已保存到: {charts_dir}")

# 最后定义主实验函数
def run_diversity_experiments(generations=50):
    """运行一系列多样性实验"""
    print("开始概念多样性研究...")
    
    # 定义不同的多样性策略
    strategies = [
        "基础进化",
        "高变异率",
        "低选择压力",
        "生态位分化",
        "平衡策略"
    ]
    
    experiment_results = {}
    
    # 为每个策略运行实验
    for i, strategy in enumerate(strategies):
        print(f"\n运行策略 {i+1}/{len(strategies)}: {strategy}")
        
        # 创建配置文件
        config_path, strategy_name = create_diversity_config(strategy, i+1)
        
        # 运行实验
        result = run_diversity_experiment(i+1, strategy_name, config_path, generations)
        
        if result:
            experiment_results[i+1] = result
    
    # 保存所有结果
    results_dir = './results/examples/diversity'
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建结果数据框
    results_df = pd.DataFrame.from_dict(experiment_results, orient='index')
    
    # 保存结果CSV
    results_csv = os.path.join(results_dir, 'diversity_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"多样性实验结果已保存到: {results_csv}")
    
    # 创建比较图表
    create_diversity_comparison_charts(experiment_results, results_dir)
    
    print("多样性研究完成!")
    
    return results_df

# 主程序入口
if __name__ == "__main__":
    run_diversity_experiments(generations=50) 