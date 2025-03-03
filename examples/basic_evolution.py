"""
基础进化实验

这个示例展示了AI-Evolution框架的基本用法，运行一个标准的进化实验，
并生成详细的报告和可视化结果。
"""

import os
import sys
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import time
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.evolution_engine import EvolutionEngine

def create_basic_config():
    """创建基础实验配置"""
    config = {
        'evolution': {
            'generation_limit': 100,
            'initial_population_size': 20,
            'max_population_size': 50,
            'base_mutation_rate': 0.4,
            'selection_pressure': 0.7,
            'inheritance_factor': 0.5
        },
        'agent': {
            'enable_memory': False,  # 禁用记忆功能
        },
        'monitor': {
            'log_interval': 1,
            'report_interval': 10,
            'visualization_interval': 10,
            'metrics': [
                'mutation_rate',
                'inheritance_rate',
                'avg_attributes',
                'unique_attributes',
                'concept_diversity',
                'population_size',
                'total_concepts',
                'natural_mutations',
                'safe_triggers'
            ],
            # 添加存储配置
            'storage': {
                'base_path': './results/examples/basic',
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
    
    return config

def run_basic_experiment(generations=50):
    """运行基础进化实验
    
    Args:
        generations: 要运行的代数
        
    Returns:
        实验结果报告
    """
    print("开始基础进化实验...")
    
    # 创建结果目录
    results_dir = './results/examples/basic'
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建配置文件
    config = create_basic_config()
    config_path = os.path.join(results_dir, 'config.yaml')
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"基础实验配置已保存到: {config_path}")
    
    # 初始化进化引擎
    engine = EvolutionEngine(config_path)
    
    # 打印初始配置
    print(f"引擎配置: {engine.config}")
    
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
    
    print(f"基础实验完成! 总运行时间: {runtime:.2f}秒")
    
    # 保存结果
    results_file = os.path.join(results_dir, 'results.pkl')
    engine.save_results(results_file)
    
    metrics_dir = os.path.join(results_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    history_file = os.path.join(metrics_dir, 'history.csv')
    engine.analytics.save_history(history_file)
    
    # 生成报告
    report_path = os.path.join(results_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write(final_report)
    
    print(f"实验报告已保存到: {report_path}")
    
    # 创建图表
    create_summary_charts(engine)
    
    return engine

def create_summary_charts(engine):
    """创建结果摘要图表"""
    charts_dir = os.path.join('./results/examples/basic', 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    history = engine.analytics.history
    
    # 打印可用的数据列，帮助调试
    print("可用的数据列:")
    for column in history.columns:
        print(f"- {column}")
    
    # 1. 种群和多样性变化
    plt.figure(figsize=(12, 6))
    
    ax1 = plt.subplot(111)
    if 'population_size' in history.columns:
        ax1.plot(history['generation'], history['population_size'], 'b-', label='种群大小')
        ax1.set_xlabel('代数', fontsize=12)
        ax1.set_ylabel('种群大小', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title('种群大小与概念多样性变化', fontsize=14)
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        
        ax2 = ax1.twinx()
        if 'concept_diversity' in history.columns:
            ax2.plot(history['generation'], history['concept_diversity'], 'r-', label='概念多样性')
            ax2.set_ylabel('概念多样性', color='r', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='r')
            
            # 合并两个坐标轴的图例
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(charts_dir, 'population_diversity.png'))
    else:
        print("警告: 无法创建种群图表，缺少必要的数据列")
    
    # 2. 变异率和继承率变化
    plt.figure(figsize=(12, 6))
    
    if 'mutation_rate' in history.columns and 'inheritance_rate' in history.columns:
        plt.plot(history['generation'], history['mutation_rate'], 'r-', label='变异率')
        plt.plot(history['generation'], history['inheritance_rate'], 'g-', label='继承率')
        plt.title('变异率与继承率变化', fontsize=14)
        plt.xlabel('代数', fontsize=12)
        plt.ylabel('比率', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'mutation_inheritance.png'))
    else:
        print("警告: 无法创建变异率图表，缺少必要的数据列")
    
    # 3. 属性数量变化
    plt.figure(figsize=(12, 6))
    
    if 'avg_attributes' in history.columns and 'unique_attributes' in history.columns:
        plt.plot(history['generation'], history['avg_attributes'], 'b-', label='平均属性数')
        plt.plot(history['generation'], history['unique_attributes'], 'g-', label='唯一属性数')
        plt.title('属性数量变化', fontsize=14)
        plt.xlabel('代数', fontsize=12)
        plt.ylabel('属性数量', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(charts_dir, 'attributes.png'))
    else:
        print("警告: 无法创建属性图表，缺少必要的数据列")
    
    # 4. 概念总数和变异次数
    plt.figure(figsize=(12, 6))
    
    if 'total_concepts' in history.columns and 'natural_mutations' in history.columns:
        plt.plot(history['generation'], history['total_concepts'], 'b-', label='概念总数')
        plt.plot(history['generation'], history['natural_mutations'], 'r-', label='自然变异次数')
        
        if 'safe_triggers' in history.columns:
            plt.plot(history['generation'], history['safe_triggers'], 'g-', label='保障触发次数')
        
        plt.title('概念与变异统计', fontsize=14)
        plt.xlabel('代数', fontsize=12)
        plt.ylabel('数量', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(charts_dir, 'concepts_mutations.png'))
    else:
        print("警告: 无法创建概念统计图表，缺少必要的数据列")
    
    print(f"结果摘要图表已保存到: {charts_dir}")

if __name__ == "__main__":
    run_basic_experiment(generations=50) 