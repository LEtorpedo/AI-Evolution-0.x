import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class EvolutionAnalytics:
    """进化分析器：追踪和分析进化过程中的关键指标"""
    
    def __init__(self, config_path="config.yaml"):
        """初始化分析器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['monitor']
            
        # 初始化存储路径
        self.base_path = Path(self.config['storage']['base_path'])
        self.metrics_path = self.base_path / self.config['storage']['metrics']
        self.metrics_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据结构
        self.history = pd.DataFrame(columns=[
            'generation',          # 代数
            'population_size',     # 种群大小
            'total_concepts',      # 概念总数
            'mutation_rate',       # 变异率
            'natural_mutations',   # 自然变异数
            'safe_triggers',       # 保障触发数
            'avg_attributes',      # 平均属性数
            'unique_attributes',   # 唯一属性数
            'inheritance_rate',    # 继承率
            'concept_diversity',   # 概念多样性
            'timestamp'            # 时间戳
        ])
        
        # 属性生命周期追踪
        self.attribute_stats = defaultdict(
            lambda: {'first_seen': None, 'count': 0, 'last_seen': None}
        )
    
    def analyze_generation(self, generation, population, mutation_rate):
        """分析一代的进化数据
        
        Args:
            generation: 当前代数
            population: 当前种群
            mutation_rate: 变异率
        """
        if generation % self.config['log_interval'] != 0:
            return
            
        metrics = self._collect_metrics(generation, population, mutation_rate)
        self._update_history(metrics)
        self._save_metrics()
        
        logger.info(f"Generation {generation} analyzed: "
                   f"Mutation rate: {mutation_rate:.2f}, "
                   f"Inheritance rate: {metrics['inheritance_rate']:.2f}")
    
    def _collect_metrics(self, generation, population, mutation_rate):
        """收集一代的指标数据"""
        all_attributes = set()
        natural_mutations = 0
        safe_triggers = 0
        attribute_counts = []
        
        # 收集属性统计
        for replicator in population:
            attrs = replicator.lattice.concepts[replicator.parent_concept]['attrs']
            attribute_counts.append(len(attrs))
            all_attributes.update(attrs)
            
            # 统计变异类型
            for attr in attrs:
                if attr.startswith(('mut_', 'ext_')):
                    natural_mutations += 1
                elif attr.startswith('safe_'):
                    safe_triggers += 1
                
                # 更新属性生命周期
                if attr not in self.attribute_stats:
                    self.attribute_stats[attr]['first_seen'] = generation
                self.attribute_stats[attr]['count'] += 1
                self.attribute_stats[attr]['last_seen'] = generation
        
        # 计算继承率和概念多样性
        inheritance_rate = sum(1 for attr in all_attributes 
                             if attr in ['origin', 'base']) / len(all_attributes)
        concept_diversity = len(all_attributes) / sum(attribute_counts)
        
        return {
            'generation': generation,
            'population_size': len(population),
            'total_concepts': len(population[0].lattice.concepts),
            'mutation_rate': mutation_rate,
            'natural_mutations': natural_mutations,
            'safe_triggers': safe_triggers,
            'avg_attributes': np.mean(attribute_counts),
            'unique_attributes': len(all_attributes),
            'inheritance_rate': inheritance_rate,
            'concept_diversity': concept_diversity,
            'timestamp': datetime.now()
        }
    
    def _update_history(self, metrics):
        """更新历史记录"""
        self.history.loc[len(self.history)] = metrics
    
    def _save_metrics(self):
        """保存指标数据"""
        metrics_file = self.metrics_path / 'evolution_data.csv'
        self.history.to_csv(metrics_file, index=False)
    
    def get_statistics(self):
        """获取统计数据"""
        if len(self.history) == 0:
            return "No evolution data recorded yet."
            
        stats = {
            'total_generations': len(self.history),
            'avg_mutation_rate': self.history['mutation_rate'].mean(),
            'avg_inheritance_rate': self.history['inheritance_rate'].mean(),
            'max_attributes': self.history['avg_attributes'].max(),
            'total_mutations': self.history['natural_mutations'].sum(),
            'total_safe_triggers': self.history['safe_triggers'].sum(),
            'concept_diversity_trend': self.history['concept_diversity'].tolist()
        }
        
        return stats
