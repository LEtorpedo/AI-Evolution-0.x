"""
参数敏感性分析

这个示例研究不同参数设置对进化过程的影响，通过运行多组实验并比较结果，
帮助理解参数选择的重要性。
"""

import os
import sys
import yaml
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.evolution_engine import EvolutionEngine

def create_parameter_config(experiment_id, mutation_rate, selection_pressure, inheritance_factor):
    """创建参数敏感性实验配置"""
    config = {
        'evolution': {
            'generation_limit': 100,
            'initial_population_size': 20,
            'max_population_size': 50,
            'base_mutation_rate': mutation_rate,
            'selection_pressure': selection_pressure,
            'inheritance_factor': inheritance_factor
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
                'base_path': f'./results/examples/parameter/exp_{experiment_id}',
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
    base_dir = f'./results/examples/parameter/exp_{experiment_id}'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'metrics'), exist_ok=True)
    
    # 保存配置文件
    config_path = os.path.join(base_dir, 'parameter_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"参数实验配置已保存到: {config_path}")
    
    return config_path

def run_parameter_experiment(experiment_id, mutation_rate, selection_pressure, inheritance_factor, generations=50):
    """运行参数敏感性实验
    
    Args:
        experiment_id: 实验ID
        mutation_rate: 变异率
        selection_pressure: 选择压力
        inheritance_factor: 继承因子
        generations: 要运行的代数
        
    Returns:
        实验结果
    """
    print(f"开始参数敏感性实验 {experiment_id}...")
    print(f"参数设置: 变异率={mutation_rate}, 选择压力={selection_pressure}, 继承因子={inheritance_factor}")
    
    # 创建配置文件
    config_path = create_parameter_config(experiment_id, mutation_rate, selection_pressure, inheritance_factor)
    
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
    
    print(f"实验 {experiment_id} 完成! 总运行时间: {runtime:.2f}秒")
    
    # 保存结果
    results_dir = f'./results/examples/parameter/exp_{experiment_id}'
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
    avg_fitness = (final_population * final_diversity) / 100  # 简化的适应度计算
    
    # 添加实验名称
    experiment_name = experiment_names[experiment_id-1] if experiment_id <= len(experiment_names) else f"实验{experiment_id}"
    
    return {
        'experiment_id': experiment_id,
        'experiment': experiment_name,  # 添加实验名称列
        'mutation_rate': mutation_rate,
        'selection_pressure': selection_pressure,
        'inheritance_factor': inheritance_factor,
        'final_diversity': final_diversity,
        'avg_diversity': avg_diversity,
        'final_population': final_population,
        'avg_attributes': avg_attributes,
        'avg_fitness': avg_fitness,
        'runtime': runtime
    }

def run_parameter_experiments(generations=50):
    """运行一系列参数敏感性实验"""
    print("开始参数敏感性分析...")
    
    # 创建结果目录
    results_dir = Path('./results/examples/parameter')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义要测试的参数集
    parameter_sets = [
        (0.1, 0.7, 0.5),  # 低变异率
        (0.6, 0.7, 0.5),  # 高变异率
        (0.3, 0.4, 0.5),  # 低选择压力
        (0.3, 0.9, 0.5),  # 高选择压力
        (0.3, 0.7, 0.2),  # 低继承因子
        (0.3, 0.7, 0.8)   # 高继承因子
    ]
    
    global experiment_names  # 使其成为全局变量，以便在run_parameter_experiment中访问
    experiment_names = [
        "低变异率",
        "高变异率",
        "低选择压力",
        "高选择压力",
        "低继承因子",
        "高继承因子"
    ]
    
    results = []
    
    # 运行每组参数的实验
    for i, (mutation_rate, selection_pressure, inheritance_factor) in enumerate(parameter_sets):
        print(f"\n运行实验 {i+1}/{len(parameter_sets)}: {experiment_names[i]}")
        print(f"参数: 变异率={mutation_rate}, 选择压力={selection_pressure}, 继承因子={inheritance_factor}")
        
        # 运行实验
        result = run_parameter_experiment(i+1, mutation_rate, selection_pressure, inheritance_factor, generations=generations)
        
        if result:
            results.append(result)
            print(f"实验 {i+1} 完成，运行时间: {result['runtime']:.2f}秒")
    
    # 创建结果数据框
    results_df = pd.DataFrame(results)
    
    # 保存结果
    results_dir = Path('./results/examples/parameter')
    results_df.to_csv(results_dir / 'parameter_results.csv', index=False)
    
    # 创建比较图表
    create_comparison_charts(results_df, results_dir)
    
    print("\n参数敏感性分析完成！")
    print(f"结果已保存到: {results_dir}")
    
    return results_df

def create_comparison_charts(results_df, results_dir):
    """创建参数比较图表"""
    charts_dir = results_dir / 'comparison_charts'
    charts_dir.mkdir(exist_ok=True)
    
    # 检查必要的列是否存在
    required_columns = ['experiment', 'final_population', 'final_diversity', 'avg_fitness', 'runtime']
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    
    if missing_columns:
        print(f"警告: 结果数据框缺少以下列: {missing_columns}")
        print("可用的列: ", results_df.columns.tolist())
        
        # 如果缺少experiment列，但有experiment_id列，则创建experiment列
        if 'experiment' in missing_columns and 'experiment_id' in results_df.columns:
            results_df['experiment'] = results_df['experiment_id'].apply(
                lambda x: experiment_names[x-1] if x <= len(experiment_names) else f"实验{x}"
            )
            missing_columns.remove('experiment')
        
        # 如果仍有缺失列，则返回
        if missing_columns:
            print("无法创建比较图表，缺少必要的数据列")
            return
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. 最终种群大小比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['experiment'], results_df['final_population'], color='skyblue')
    plt.title('不同参数设置下的最终种群大小', fontsize=16)
    plt.xlabel('实验参数', fontsize=14)
    plt.ylabel('种群大小', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'population_comparison.png')
    
    # 2. 最终多样性比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['experiment'], results_df['final_diversity'], color='lightgreen')
    plt.title('不同参数设置下的最终概念多样性', fontsize=16)
    plt.xlabel('实验参数', fontsize=14)
    plt.ylabel('多样性指数', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'diversity_comparison.png')
    
    # 3. 平均适应度比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['experiment'], results_df['avg_fitness'], color='salmon')
    plt.title('不同参数设置下的平均适应度', fontsize=16)
    plt.xlabel('实验参数', fontsize=14)
    plt.ylabel('适应度', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'fitness_comparison.png')
    
    # 4. 运行时间比较
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['experiment'], results_df['runtime'], color='mediumpurple')
    plt.title('不同参数设置下的运行时间', fontsize=16)
    plt.xlabel('实验参数', fontsize=14)
    plt.ylabel('时间 (秒)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'runtime_comparison.png')
    
    # 5. 参数相关性热图
    plt.figure(figsize=(10, 8))
    
    # 计算相关性
    corr_columns = ['mutation_rate', 'selection_pressure', 'inheritance_factor', 
                    'final_population', 'final_diversity', 'avg_fitness', 'runtime']
    corr_matrix = results_df[corr_columns].corr()
    
    # 绘制热图
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
    plt.colorbar(label='相关系数')
    plt.title('参数与结果的相关性', fontsize=16)
    
    # 添加标签
    plt.xticks(np.arange(len(corr_columns)), corr_columns, rotation=45, ha='right')
    plt.yticks(np.arange(len(corr_columns)), corr_columns)
    
    # 添加相关系数值
    for i in range(len(corr_columns)):
        for j in range(len(corr_columns)):
            text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black")
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'parameter_correlation.png')
    
    print(f"比较图表已保存到: {charts_dir}")

def compare_parameter_results():
    """比较不同参数设置的结果"""
    results_dir = Path('./results/examples/parameter')
    charts_dir = results_dir / 'comparison'
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有实验的结果
    experiment_results = {}
    
    for exp_dir in results_dir.glob('exp_*'):
        if not exp_dir.is_dir():
            continue
        
        exp_id = exp_dir.name.split('_')[1]
        history_path = exp_dir / 'metrics' / 'history.csv'
        
        if not history_path.exists():
            print(f"警告: 未找到实验{exp_id}的历史数据")
            continue
        
        try:
            # 加载历史数据
            history = pd.read_csv(history_path)
            
            # 获取最后一代的数据
            last_gen = history['generation'].max()
            last_gen_data = history[history['generation'] == last_gen].iloc[0]
            
            # 计算平均值
            avg_population = history['population_size'].mean()
            avg_diversity = history['concept_diversity'].mean() if 'concept_diversity' in history.columns else 0
            
            # 存储结果
            experiment_results[exp_id] = {
                'last_generation': last_gen,
                'final_population': last_gen_data.get('population_size', 0),
                'final_diversity': last_gen_data.get('concept_diversity', 0),
                'avg_population': avg_population,
                'avg_diversity': avg_diversity,
                'mutation_rate': experiment_params[exp_id]['mutation_rate'],
                'selection_pressure': experiment_params[exp_id]['selection_pressure'],
                'inheritance_factor': experiment_params[exp_id]['inheritance_factor'],
                'runtime': experiment_times.get(exp_id, 0)
            }
            
        except Exception as e:
            print(f"处理实验{exp_id}时出错: {e}")
    
    if not experiment_results:
        print("错误: 没有找到任何实验结果")
        return
    
    # 创建比较图表
    
    # 1. 参数对最终种群大小的影响
    plt.figure(figsize=(15, 10))
    
    # 变异率与种群大小
    plt.subplot(2, 2, 1)
    x = [results['mutation_rate'] for _, results in experiment_results.items()]
    y = [results['final_population'] for _, results in experiment_results.items()]
    plt.scatter(x, y, c='blue', alpha=0.7)
    plt.title('变异率与最终种群大小关系', fontsize=14)
    plt.xlabel('变异率', fontsize=12)
    plt.ylabel('最终种群大小', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 选择压力与种群大小
    plt.subplot(2, 2, 2)
    x = [results['selection_pressure'] for _, results in experiment_results.items()]
    y = [results['final_population'] for _, results in experiment_results.items()]
    plt.scatter(x, y, c='red', alpha=0.7)
    plt.title('选择压力与最终种群大小关系', fontsize=14)
    plt.xlabel('选择压力', fontsize=12)
    plt.ylabel('最终种群大小', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 继承因子与种群大小
    plt.subplot(2, 2, 3)
    x = [results['inheritance_factor'] for _, results in experiment_results.items()]
    y = [results['final_population'] for _, results in experiment_results.items()]
    plt.scatter(x, y, c='green', alpha=0.7)
    plt.title('继承因子与最终种群大小关系', fontsize=14)
    plt.xlabel('继承因子', fontsize=12)
    plt.ylabel('最终种群大小', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 运行时间与种群大小
    plt.subplot(2, 2, 4)
    x = [results['runtime'] for _, results in experiment_results.items()]
    y = [results['final_population'] for _, results in experiment_results.items()]
    plt.scatter(x, y, c='purple', alpha=0.7)
    plt.title('运行时间与最终种群大小关系', fontsize=14)
    plt.xlabel('运行时间(秒)', fontsize=12)
    plt.ylabel('最终种群大小', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'parameter_population.png')
    
    # 2. 参数对多样性的影响
    plt.figure(figsize=(15, 10))
    
    # 变异率与多样性
    plt.subplot(2, 2, 1)
    x = [results['mutation_rate'] for _, results in experiment_results.items()]
    y = [results['final_diversity'] for _, results in experiment_results.items()]
    plt.scatter(x, y, c='blue', alpha=0.7)
    plt.title('变异率与最终多样性关系', fontsize=14)
    plt.xlabel('变异率', fontsize=12)
    plt.ylabel('最终多样性', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 选择压力与多样性
    plt.subplot(2, 2, 2)
    x = [results['selection_pressure'] for _, results in experiment_results.items()]
    y = [results['final_diversity'] for _, results in experiment_results.items()]
    plt.scatter(x, y, c='red', alpha=0.7)
    plt.title('选择压力与最终多样性关系', fontsize=14)
    plt.xlabel('选择压力', fontsize=12)
    plt.ylabel('最终多样性', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 继承因子与多样性
    plt.subplot(2, 2, 3)
    x = [results['inheritance_factor'] for _, results in experiment_results.items()]
    y = [results['final_diversity'] for _, results in experiment_results.items()]
    plt.scatter(x, y, c='green', alpha=0.7)
    plt.title('继承因子与最终多样性关系', fontsize=14)
    plt.xlabel('继承因子', fontsize=12)
    plt.ylabel('最终多样性', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 种群大小与多样性
    plt.subplot(2, 2, 4)
    x = [results['final_population'] for _, results in experiment_results.items()]
    y = [results['final_diversity'] for _, results in experiment_results.items()]
    plt.scatter(x, y, c='purple', alpha=0.7)
    plt.title('种群大小与多样性关系', fontsize=14)
    plt.xlabel('最终种群大小', fontsize=12)
    plt.ylabel('最终多样性', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'parameter_diversity.png')
    
    print(f"参数比较图表已保存到: {charts_dir}")

if __name__ == "__main__":
    run_parameter_experiments(generations=50) 