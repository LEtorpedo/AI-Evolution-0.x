"""
长期进化趋势研究

这个示例研究长期进化过程中的模式和趋势，通过运行更长时间的进化实验，
观察种群的长期适应性变化、复杂性增长和进化停滞现象。
"""

import os
import sys
import yaml
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import random
import seaborn as sns
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.evolution_engine import EvolutionEngine

def create_long_term_config():
    """创建长期进化实验配置"""
    config = {
        'evolution': {
            'generation_limit': 300,
            'initial_population_size': 30,
            'max_population_size': 100,
            'base_mutation_rate': 0.4,
            'selection_pressure': 0.7,
            'inheritance_factor': 0.5,
            'adaptive_mutation': True,  # 启用自适应变异
            'adaptive_selection': True,  # 启用自适应选择
            'long_term_memory': True,   # 启用长期记忆
            'concept_reuse': True       # 启用概念重用
        },
        'agent': {
            'enable_memory': True,      # 启用记忆功能
            'memory_capacity': 10,      # 记忆容量
            'memory_decay': 0.1,        # 记忆衰减率
        },
        'monitor': {  # 确保这个部分存在且格式正确
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
                'base_path': './results/examples/long_term',
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
    base_dir = './results/examples/long_term'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'charts'), exist_ok=True)
    
    # 保存配置文件
    config_path = os.path.join(base_dir, 'long_term_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"长期进化实验配置已保存到: {config_path}")
    
    return config_path

def run_long_term_experiment(generations=300):
    """运行长期进化实验"""
    print("开始长期进化实验...")
    
    # 创建配置文件
    config_path = create_long_term_config()
    
    # 验证配置文件是否存在且格式正确
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'monitor' not in config:
                raise ValueError("配置文件缺少'monitor'部分")
    except Exception as e:
        print(f"配置文件验证失败: {e}")
        return
    
    # 初始化进化引擎
    try:
        engine = EvolutionEngine(config_path)
    except Exception as e:
        print(f"初始化进化引擎失败: {e}")
        traceback.print_exc()
        return
    
    # 运行进化
    start_time = time.time()
    
    try:
        final_report = engine.run(generations=generations)
    except Exception as e:
        print(f"运行过程中出错: {e}")
        traceback.print_exc()
        return
    
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"长期进化实验完成! 总运行时间: {runtime:.2f}秒")
    
    # 保存结果
    results_dir = './results/examples/long_term'
    results_file = os.path.join(results_dir, 'results.pkl')
    engine.save_results(results_file)
    
    # 保存历史数据
    history_file = os.path.join(results_dir, 'metrics', 'evolution_data.csv')
    engine.analytics.save_history(history_file)
    
    # 生成报告
    report_path = os.path.join(results_dir, 'long_term_report.md')
    with open(report_path, 'w') as f:
        f.write(final_report)
    
    # 生成长期趋势图表
    create_long_term_charts(engine.analytics.history, generations)
    
    print(f"长期进化实验报告已保存到: {report_path}")

def create_long_term_charts(history, generations):
    """创建长期进化趋势图表"""
    print("生成长期趋势图表...")
    
    # 创建图表目录
    charts_dir = Path('./results/examples/long_term/charts')
    charts_dir.mkdir(exist_ok=True)
    
    # 1. 种群规模长期趋势
    plt.figure(figsize=(14, 8))
    
    if 'population_size' in history.columns:
        plt.plot(history['generation'], history['population_size'], 'b-', label='种群规模')
        
        # 添加平滑趋势线
        if len(history['generation']) > 20:
            window_length = min(21, len(history['generation']) - (1 if len(history['generation']) % 2 == 0 else 0))
            if window_length > 2:
                y_smooth = savgol_filter(history['population_size'], window_length, 2)
                plt.plot(history['generation'], y_smooth, 'r--', linewidth=2, label='趋势线')
        
        plt.title('长期种群规模趋势', fontsize=16)
        plt.xlabel('代数', fontsize=14)
        plt.ylabel('种群规模', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(charts_dir / 'population_trend.png')
    else:
        print("警告: 无法创建种群规模趋势图表，缺少必要的数据列")
    
    # 2. 多样性长期趋势
    plt.figure(figsize=(14, 8))
    
    if 'concept_diversity' in history.columns:
        plt.plot(history['generation'], history['concept_diversity'], 'g-', label='概念多样性')
        
        # 添加平滑趋势线
        if len(history['generation']) > 20:
            window_length = min(21, len(history['generation']) - (1 if len(history['generation']) % 2 == 0 else 0))
            if window_length > 2:
                y_smooth = savgol_filter(history['concept_diversity'], window_length, 2)
                plt.plot(history['generation'], y_smooth, 'r--', linewidth=2, label='趋势线')
        
        plt.title('长期多样性趋势', fontsize=16)
        plt.xlabel('代数', fontsize=14)
        plt.ylabel('概念多样性', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(charts_dir / 'diversity_trend.png')
    else:
        print("警告: 无法创建多样性趋势图表，缺少必要的数据列")
    
    # 3. 属性复杂度长期趋势
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    if 'avg_attributes' in history.columns:
        plt.plot(history['generation'], history['avg_attributes'], 'r-', label='平均属性数量')
        
        # 添加平滑趋势线
        if len(history['generation']) > 20:
            window_length = min(21, len(history['generation']) - (1 if len(history['generation']) % 2 == 0 else 0))
            if window_length > 2:
                y_smooth = savgol_filter(history['avg_attributes'], window_length, 2)
                plt.plot(history['generation'], y_smooth, 'k--', linewidth=2, label='属性数量趋势')
        
        plt.title('长期属性数量趋势', fontsize=16)
        plt.ylabel('平均属性数量', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 多样性趋势
    plt.subplot(2, 1, 2)
    if 'concept_diversity' in history.columns:
        plt.plot(history['generation'], history['concept_diversity'], 'g-', label='概念多样性')
        
        # 添加平滑趋势线
        if len(history['generation']) > 20:
            window_length = min(21, len(history['generation']) - (1 if len(history['generation']) % 2 == 0 else 0))
            if window_length > 2:
                y_smooth = savgol_filter(history['concept_diversity'], window_length, 2)
                plt.plot(history['generation'], y_smooth, 'k--', linewidth=2, label='多样性趋势')
        
        plt.title('长期多样性趋势', fontsize=16)
        plt.xlabel('代数', fontsize=14)
        plt.ylabel('概念多样性', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'complexity_trends.png')
    
    # 4. 变异率与继承率长期趋势
    plt.figure(figsize=(14, 8))
    
    if 'mutation_rate' in history.columns and 'inheritance_rate' in history.columns:
        plt.plot(history['generation'], history['mutation_rate'], 'r-', label='变异率')
        plt.plot(history['generation'], history['inheritance_rate'], 'b-', label='继承率')
        
        # 添加平滑趋势线
        if len(history['generation']) > 20:
            window_length = min(21, len(history['generation']) - (1 if len(history['generation']) % 2 == 0 else 0))
            if window_length > 2:
                y1_smooth = savgol_filter(history['mutation_rate'], window_length, 2)
                y2_smooth = savgol_filter(history['inheritance_rate'], window_length, 2)
                plt.plot(history['generation'], y1_smooth, 'r--', linewidth=2, label='变异率趋势')
                plt.plot(history['generation'], y2_smooth, 'b--', linewidth=2, label='继承率趋势')
        
        plt.title('长期变异与继承趋势', fontsize=16)
        plt.xlabel('代数', fontsize=14)
        plt.ylabel('比率', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(charts_dir / 'mutation_inheritance_trend.png')
    else:
        print("警告: 无法创建变异率趋势图表，缺少必要的数据列")
    
    # 5. 阶段性分析
    if generations > 30:
        # 将进化过程分为多个阶段
        num_phases = min(5, generations // 10)
        phase_size = generations // num_phases
        
        # 准备阶段数据
        phases = []
        for i in range(num_phases):
            start_gen = i * phase_size
            end_gen = (i + 1) * phase_size if i < num_phases - 1 else generations
            
            phase_data = history[(history['generation'] >= start_gen) & 
                                (history['generation'] < end_gen)]
            
            # 确保阶段数据不为空
            if not phase_data.empty:
                phase_info = {
                    'phase': f"阶段{i+1}",
                    'start_gen': start_gen,
                    'end_gen': end_gen,
                    'avg_population': phase_data['population_size'].mean() if 'population_size' in phase_data.columns else 0,
                    'avg_diversity': phase_data['concept_diversity'].mean() if 'concept_diversity' in phase_data.columns else 0,
                    'avg_attributes': phase_data['avg_attributes'].mean() if 'avg_attributes' in phase_data.columns else 0,
                    'mutation_rate': phase_data['mutation_rate'].mean() if 'mutation_rate' in phase_data.columns else 0
                }
                
                # 添加到阶段列表
                phases.append(phase_info)
        
        # 计算增长率
        for i in range(1, len(phases)):
            prev_phase = phases[i-1]
            curr_phase = phases[i]
            
            # 计算种群增长率 (避免除以零)
            if prev_phase['avg_population'] > 0:
                population_growth = ((curr_phase['avg_population'] - prev_phase['avg_population']) / 
                                    prev_phase['avg_population']) * 100
            else:
                population_growth = 0 if curr_phase['avg_population'] == 0 else 100
            
            # 计算多样性增长率 (避免除以零)
            if prev_phase['avg_diversity'] > 0:
                diversity_growth = ((curr_phase['avg_diversity'] - prev_phase['avg_diversity']) / 
                                   prev_phase['avg_diversity']) * 100
            else:
                diversity_growth = 0 if curr_phase['avg_diversity'] == 0 else 100
            
            # 存储增长率
            curr_phase['population_growth'] = population_growth
            curr_phase['diversity_growth'] = diversity_growth
        
        # 为第一个阶段添加默认增长率（因为没有前一个阶段作比较）
        if phases and 'population_growth' not in phases[0]:
            phases[0]['population_growth'] = 0
            phases[0]['diversity_growth'] = 0
        
        # 创建阶段性增长率图表
        plt.figure(figsize=(14, 12))
        
        # 种群增长率
        plt.subplot(2, 1, 1)
        x = [p['phase'] for p in phases[1:]]  # 跳过第一个阶段，因为它没有有意义的增长率
        y = [p['population_growth'] for p in phases[1:]]
        
        # 打印调试信息
        print("种群增长率数据:")
        for i, (phase, growth) in enumerate(zip(x, y)):
            print(f"{phase}: {growth:.2f}%")
        
        plt.bar(x, y, color='blue', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('阶段性种群增长率', fontsize=16)
        plt.ylabel('增长率(%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 多样性增长率
        plt.subplot(2, 1, 2)
        y = [p['diversity_growth'] for p in phases[1:]]
        
        # 打印调试信息
        print("多样性增长率数据:")
        for i, (phase, growth) in enumerate(zip(x, y)):
            print(f"{phase}: {growth:.2f}%")
        
        plt.bar(x, y, color='green', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('阶段性多样性增长率', fontsize=16)
        plt.xlabel('进化阶段', fontsize=14)
        plt.ylabel('增长率(%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'phase_analysis.png')
    
    # 6. 长期稳定性分析
    if generations > 20:
        # 计算滚动方差
        window_size = min(10, generations // 5)
        
        plt.figure(figsize=(14, 10))
        
        # 种群稳定性
        plt.subplot(2, 1, 1)
        if 'population_size' in history.columns:
            # 计算滚动标准差
            rolling_std = history['population_size'].rolling(window=window_size).std()
            
            # 确保x和y数组长度匹配
            valid_indices = rolling_std.notna()
            x_values = history['generation'][valid_indices]
            y_values = rolling_std[valid_indices]
            
            plt.plot(x_values, y_values, 'b-', label='种群波动')
            plt.title('种群规模稳定性分析', fontsize=16)
            plt.ylabel('滚动标准差', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 多样性稳定性
        plt.subplot(2, 1, 2)
        if 'concept_diversity' in history.columns:
            # 计算滚动标准差
            rolling_std = history['concept_diversity'].rolling(window=window_size).std()
            
            # 确保x和y数组长度匹配
            valid_indices = rolling_std.notna()
            x_values = history['generation'][valid_indices]
            y_values = rolling_std[valid_indices]
            
            plt.plot(x_values, y_values, 'g-', label='多样性波动')
            plt.title('多样性稳定性分析', fontsize=16)
            plt.xlabel('代数', fontsize=14)
            plt.ylabel('滚动标准差', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(charts_dir / 'stability_analysis.png')
    
    # 7. 长期相关性分析
    plt.figure(figsize=(14, 10))
    
    # 准备相关性数据
    correlation_data = history.copy()
    
    # 移除非数值列
    for col in correlation_data.columns:
        if correlation_data[col].dtype == 'object':
            correlation_data = correlation_data.drop(col, axis=1)
    
    # 计算相关性矩阵
    if len(correlation_data.columns) > 1:
        corr_matrix = correlation_data.corr()
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('长期进化指标相关性分析', fontsize=16)
        plt.tight_layout()
        plt.savefig(charts_dir / 'correlation_analysis.png')
    
    # 创建阶段特征雷达图
    plt.figure(figsize=(12, 10))

    # 准备雷达图数据
    categories = ['平均种群', '平均多样性', '平均属性数', '变异率']

    # 计算角度
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图

    # 创建子图
    ax = plt.subplot(111, polar=True)

    # 打印调试信息
    print("\n雷达图数据:")
    for phase in phases:
        print(f"{phase['phase']}:")
        print(f"  平均种群: {phase['avg_population']:.2f}")
        print(f"  平均多样性: {phase['avg_diversity']:.2f}")
        print(f"  平均属性数: {phase['avg_attributes']:.2f}")
        print(f"  变异率: {phase['mutation_rate']:.2f}")

    # 计算最大值用于标准化
    max_values = [
        max([p['avg_population'] for p in phases]) if phases else 1,
        max([p['avg_diversity'] for p in phases]) if phases else 1,
        max([p['avg_attributes'] for p in phases]) if phases else 1,
        max([p['mutation_rate'] for p in phases]) if phases else 1
    ]

    # 确保最大值不为零
    max_values = [max(v, 0.001) for v in max_values]

    # 绘制每个阶段的雷达图
    for i, phase in enumerate(phases):
        values = [
            phase['avg_population'],
            phase['avg_diversity'],
            phase['avg_attributes'],
            phase['mutation_rate']
        ]
        
        # 标准化数据
        normalized_values = [v / max_v for v, max_v in zip(values, max_values)]
        normalized_values += normalized_values[:1]  # 闭合雷达图
        
        # 绘制
        ax.plot(angles, normalized_values, linewidth=2, label=phase['phase'])
        ax.fill(angles, normalized_values, alpha=0.1)

    # 设置雷达图
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('进化阶段特征比较', fontsize=16)
    ax.grid(True)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(charts_dir / 'phase_radar.png')
    
    print(f"长期趋势图表已保存到: {charts_dir}")

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    
    # 运行长期进化实验
    run_long_term_experiment(generations=300)