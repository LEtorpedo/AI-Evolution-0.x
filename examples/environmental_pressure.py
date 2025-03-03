"""
环境压力模拟

这个示例模拟外部环境变化对进化过程的影响，通过在不同代数引入不同类型的环境压力，
观察种群如何适应和响应这些变化。
"""

import os
import sys
import yaml
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.evolution_engine import EvolutionEngine

def create_environment_config():
    """创建环境实验配置"""
    config = {
        'evolution': {
            'generation_limit': 100,
            'initial_population_size': 30,
            'max_population_size': 100,
            'base_mutation_rate': 0.4,
            'selection_pressure': 0.7,
            'inheritance_factor': 0.5,
            'dynamic_environment': True
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
                'attribute_count',
                'concept_diversity',
                'population_size',
                'fitness_average',
                'fitness_max',
                'fitness_min',
                'environment_pressure'  # 新增环境压力指标
            ],
            'storage': {
                'base_path': './results/examples/environment/',
                'reports': 'reports/',
                'visualizations': 'visualizations/',
                'metrics': 'metrics/'
            },
            'visualization': {
                'enabled': True,
                'format': 'png',
                'dpi': 300
            }
        }
    }
    
    # 创建存储目录
    base_dir = Path('./results/examples/environment')
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / 'reports').mkdir(exist_ok=True)
    (base_dir / 'visualizations').mkdir(exist_ok=True)
    (base_dir / 'metrics').mkdir(exist_ok=True)
    
    # 保存配置文件
    config_path = base_dir / 'environment_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"环境实验配置已保存到: {config_path}")
    return config_path

def run_environment_experiment(generations=60):
    """运行环境压力实验"""
    print(f"开始环境压力实验，运行{generations}代...")
    
    # 创建配置
    config_path = create_environment_config()
    
    # 创建进化引擎
    engine = EvolutionEngine(config_path)
    
    # 定义环境压力事件
    environment_events = [
        # (代数, 事件类型, 强度, 持续时间)
        (10, 'disaster', 0.7, 3),      # 第10代：灾难事件，强度0.7，持续3代
        (20, 'resource_boom', 0.8, 5), # 第20代：资源丰富，强度0.8，持续5代
        (30, 'predator', 0.6, 4),      # 第30代：捕食者出现，强度0.6，持续4代
        (40, 'climate_change', 0.9, 8),# 第40代：气候变化，强度0.9，持续8代
        (55, 'competition', 0.5, 5)    # 第55代：竞争加剧，强度0.5，持续5代
    ]
    
    # 创建环境压力日志
    env_log_file = Path('./results/examples/environment/environment_events.txt')
    with open(env_log_file, 'w') as f:
        f.write("代数,事件类型,强度,持续时间,影响\n")
    
    # 环境压力历史记录
    pressure_history = []
    
    # 运行进化，每代检查并应用环境压力
    for gen in range(generations):
        # 检查是否有环境事件发生
        active_events = []
        for event_gen, event_type, intensity, duration in environment_events:
            if event_gen <= gen < event_gen + duration:
                active_events.append((event_type, intensity))
        
        # 应用环境压力
        pressure = 0
        if active_events:
            for event_type, intensity in active_events:
                # 根据事件类型应用不同的环境压力
                if event_type == 'disaster':
                    # 灾难：减少种群，增加变异率
                    if hasattr(engine, 'population'):
                        remove_count = int(len(engine.population) * intensity * 0.3)
                        if remove_count > 0 and len(engine.population) > 10:
                            indices = np.random.choice(len(engine.population), remove_count, replace=False)
                            engine.population = [ind for i, ind in enumerate(engine.population) if i not in indices]
                            print(f"第{gen}代: 灾难事件，移除了{remove_count}个个体")
                            
                            with open(env_log_file, 'a') as f:
                                f.write(f"{gen},disaster,{intensity},{duration},移除{remove_count}个体\n")
                    
                    # 增加变异率
                    if hasattr(engine, 'base_mutation_rate'):
                        engine.base_mutation_rate *= (1 + intensity * 0.2)
                    
                    pressure += intensity
                
                elif event_type == 'resource_boom':
                    # 资源丰富：增加种群，降低选择压力
                    if hasattr(engine, 'population'):
                        add_count = int(len(engine.population) * intensity * 0.2)
                        for _ in range(add_count):
                            if engine.population and len(engine.population) < engine.max_population_size:
                                parent = random.choice(engine.population)
                                if hasattr(parent, 'replicate'):
                                    child = parent.replicate()
                                    engine.population.append(child)
                        
                        print(f"第{gen}代: 资源丰富，增加了{add_count}个个体")
                        
                        with open(env_log_file, 'a') as f:
                            f.write(f"{gen},resource_boom,{intensity},{duration},增加{add_count}个体\n")
                    
                    # 降低选择压力
                    if hasattr(engine, 'selection_pressure'):
                        engine.selection_pressure *= (1 - intensity * 0.1)
                    
                    pressure -= intensity * 0.8  # 资源丰富是正面事件，降低压力
                
                elif event_type == 'predator':
                    # 捕食者：增加选择压力，移除低适应度个体
                    if hasattr(engine, 'selection_pressure'):
                        engine.selection_pressure *= (1 + intensity * 0.3)
                    
                    if hasattr(engine, 'population') and hasattr(engine, 'calculate_fitness'):
                        # 计算每个个体的适应度
                        fitnesses = [engine.calculate_fitness(ind) for ind in engine.population]
                        
                        # 找出适应度最低的个体
                        if fitnesses:
                            threshold = np.percentile(fitnesses, intensity * 20)
                            remove_indices = [i for i, fit in enumerate(fitnesses) if fit < threshold]
                            
                            if remove_indices and len(engine.population) > 10:
                                remove_count = min(len(remove_indices), int(len(engine.population) * 0.2))
                                indices_to_remove = np.random.choice(remove_indices, remove_count, replace=False)
                                engine.population = [ind for i, ind in enumerate(engine.population) if i not in indices_to_remove]
                                
                                print(f"第{gen}代: 捕食者出现，移除了{remove_count}个低适应度个体")
                                
                                with open(env_log_file, 'a') as f:
                                    f.write(f"{gen},predator,{intensity},{duration},移除{remove_count}个低适应度个体\n")
                    
                    pressure += intensity * 0.9
                
                elif event_type == 'climate_change':
                    # 气候变化：大幅增加变异率，改变适应度函数
                    if hasattr(engine, 'base_mutation_rate'):
                        engine.base_mutation_rate *= (1 + intensity * 0.5)
                    
                    # 这里我们假设引擎有一个fitness_weights属性，用于调整适应度计算
                    if hasattr(engine, 'fitness_weights'):
                        # 随机调整适应度权重
                        for key in engine.fitness_weights:
                            engine.fitness_weights[key] *= random.uniform(0.5, 1.5)
                    
                    print(f"第{gen}代: 气候变化，变异率增加，适应度标准改变")
                    
                    with open(env_log_file, 'a') as f:
                        f.write(f"{gen},climate_change,{intensity},{duration},变异率增加,适应度标准改变\n")
                    
                    pressure += intensity
                
                elif event_type == 'competition':
                    # 竞争加剧：增加选择压力，减少资源
                    if hasattr(engine, 'selection_pressure'):
                        engine.selection_pressure *= (1 + intensity * 0.2)
                    
                    # 减少种群增长率
                    if hasattr(engine, 'population'):
                        current_size = len(engine.population)
                        target_size = max(15, int(current_size * (1 - intensity * 0.1)))
                        
                        if current_size > target_size:
                            remove_count = current_size - target_size
                            indices = np.random.choice(current_size, remove_count, replace=False)
                            engine.population = [ind for i, ind in enumerate(engine.population) if i not in indices]
                            
                            print(f"第{gen}代: 竞争加剧，资源减少，移除了{remove_count}个个体")
                            
                            with open(env_log_file, 'a') as f:
                                f.write(f"{gen},competition,{intensity},{duration},移除{remove_count}个体\n")
                    
                    pressure += intensity * 0.7
        
        # 记录环境压力
        pressure_history.append(pressure)
        
        # 确保分析器记录环境压力
        if hasattr(engine, 'analytics') and hasattr(engine.analytics, 'record_metric'):
            engine.analytics.record_metric('environment_pressure', pressure)
        
        # 运行一代进化
        engine.evolve_generation()
        
        print(f"第{gen}代进化完成，当前环境压力: {pressure:.2f}")
    
    # 创建环境压力图表
    create_environment_charts(engine, generations)
    
    print("\n环境压力实验完成！")
    print(f"环境事件日志已保存到: {env_log_file}")
    
    return engine

def create_environment_charts(engine, generations):
    """创建环境压力图表"""
    charts_dir = Path('./results/examples/environment/charts')
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    history = engine.analytics.history
    events = engine.environment_events if hasattr(engine, 'environment_events') else []
    
    # 打印可用的数据列，帮助调试
    print("可用的数据列:")
    for column in history.columns:
        print(f"- {column}")
    
    # 1. 环境事件与种群变化
    plt.figure(figsize=(14, 10))
    
    # 种群大小变化
    ax1 = plt.subplot(3, 1, 1)
    if 'population_size' in history.columns:
        ax1.plot(history['generation'], history['population_size'], 'b-', label='种群大小')
        ax1.set_ylabel('种群大小', fontsize=12)
        ax1.set_title('环境事件与种群变化', fontsize=14)
        
        # 标记环境事件
        for event in events:
            if 'generation' in event and 'type' in event:
                gen = event['generation']
                if gen <= generations:
                    ax1.axvline(x=gen, color='r', linestyle='--', alpha=0.7)
                    ax1.text(gen, ax1.get_ylim()[1]*0.9, event['type'], 
                             rotation=90, verticalalignment='top')
        
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 变异率变化
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    if 'mutation_rate' in history.columns:
        ax2.plot(history['generation'], history['mutation_rate'], 'g-', label='变异率')
        ax2.set_ylabel('变异率', fontsize=12)
        
        # 标记环境事件
        for event in events:
            if 'generation' in event and 'type' in event:
                gen = event['generation']
                if gen <= generations:
                    ax2.axvline(x=gen, color='r', linestyle='--', alpha=0.7)
        
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 多样性变化
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    if 'concept_diversity' in history.columns:
        ax3.plot(history['generation'], history['concept_diversity'], 'r-', label='概念多样性')
        ax3.set_xlabel('代数', fontsize=12)
        ax3.set_ylabel('多样性', fontsize=12)
        
        # 标记环境事件
        for event in events:
            if 'generation' in event and 'type' in event:
                gen = event['generation']
                if gen <= generations:
                    ax3.axvline(x=gen, color='r', linestyle='--', alpha=0.7)
        
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'environment_impact.png')
    
    # 2. 环境压力与属性变化
    plt.figure(figsize=(14, 8))
    
    if 'avg_attributes' in history.columns and 'unique_attributes' in history.columns:
        plt.plot(history['generation'], history['avg_attributes'], 'b-', label='平均属性数')
        plt.plot(history['generation'], history['unique_attributes'], 'g-', label='唯一属性数')
        
        # 标记环境事件
        for event in events:
            if 'generation' in event and 'type' in event:
                gen = event['generation']
                if gen <= generations:
                    plt.axvline(x=gen, color='r', linestyle='--', alpha=0.7)
                    plt.text(gen, plt.ylim()[1]*0.9, event['type'], 
                             rotation=90, verticalalignment='top')
        
        plt.title('环境压力与属性变化', fontsize=14)
        plt.xlabel('代数', fontsize=12)
        plt.ylabel('属性数量', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(charts_dir / 'environment_attributes.png')
    
    # 3. 环境事件前后比较
    if events and 'population_size' in history.columns:
        plt.figure(figsize=(14, 10))
        
        event_analysis = []
        
        for i, event in enumerate(events):
            if 'generation' in event and 'type' in event:
                gen = event['generation']
                if gen <= generations:
                    # 获取事件前后的数据
                    before_gen = max(0, gen - 5)
                    after_gen = min(generations, gen + 5)
                    
                    before_data = history[(history['generation'] >= before_gen) & 
                                         (history['generation'] < gen)]
                    after_data = history[(history['generation'] > gen) & 
                                        (history['generation'] <= after_gen)]
                    
                    if not before_data.empty and not after_data.empty:
                        # 计算变化率
                        before_pop = before_data['population_size'].mean()
                        after_pop = after_data['population_size'].mean()
                        pop_change = (after_pop - before_pop) / before_pop if before_pop > 0 else 0
                        
                        before_div = before_data['concept_diversity'].mean() if 'concept_diversity' in before_data else 0
                        after_div = after_data['concept_diversity'].mean() if 'concept_diversity' in after_data else 0
                        div_change = (after_div - before_div) / before_div if before_div > 0 else 0
                        
                        event_analysis.append({
                            'event_type': event['type'],
                            'generation': gen,
                            'population_change': pop_change * 100,  # 转为百分比
                            'diversity_change': div_change * 100    # 转为百分比
                        })
        
        if event_analysis:
            # 绘制事件影响柱状图
            event_df = pd.DataFrame(event_analysis)
            
            plt.subplot(2, 1, 1)
            plt.bar(range(len(event_df)), event_df['population_change'], color='blue', alpha=0.7)
            plt.xticks(range(len(event_df)), event_df['event_type'], rotation=45)
            plt.title('环境事件对种群大小的影响', fontsize=14)
            plt.ylabel('种群变化率(%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.bar(range(len(event_df)), event_df['diversity_change'], color='green', alpha=0.7)
            plt.xticks(range(len(event_df)), event_df['event_type'], rotation=45)
            plt.title('环境事件对多样性的影响', fontsize=14)
            plt.ylabel('多样性变化率(%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(charts_dir / 'event_impact_analysis.png')
    
    print(f"环境压力图表已保存到: {charts_dir}")

if __name__ == "__main__":
    run_environment_experiment(generations=60) 