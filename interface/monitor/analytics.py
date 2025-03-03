import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime
from pathlib import Path
import yaml
import time
import math
from collections import Counter

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
        # 检查配置中是否有storage键
        if 'storage' in self.config:
            self.base_path = Path(self.config['storage']['base_path'])
            self.metrics_path = self.base_path / self.config['storage']['metrics']
        else:
            # 使用默认路径
            self.base_path = Path('./results')
            self.metrics_path = self.base_path / 'metrics'
        
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
        """收集当前代的指标
        
        Args:
            generation: 当前代数
            population: 当前种群
            mutation_rate: 当前变异率
            
        Returns:
            包含各种指标的字典
        """
        # 基础指标
        metrics = {
            'generation': generation,
            'timestamp': time.time(),
            'population_size': len(population),
            'mutation_rate': mutation_rate
        }
        
        # 属性统计
        all_attributes = []
        attribute_counts = []
        
        # 概念统计
        all_concepts = []
        concept_counts = []
        
        # 收集数据
        for replicator in population:
            # 属性数据
            attrs = list(replicator.attributes.keys()) if hasattr(replicator, 'attributes') else []
            all_attributes.extend(attrs)
            attribute_counts.append(len(attrs))
            
            # 概念数据
            concepts = list(replicator.concepts.keys()) if hasattr(replicator, 'concepts') else []
            all_concepts.extend(concepts)
            concept_counts.append(len(concepts))
        
        # 计算属性指标
        avg_attribute_count = sum(attribute_counts) / len(attribute_counts) if attribute_counts else 0
        unique_attributes = len(set(all_attributes))
        
        # 计算概念指标
        total_concepts = len(all_concepts)
        unique_concepts = len(set(all_concepts))
        
        # 计算多样性指标 (使用Shannon熵)
        if all_concepts:
            concept_counter = Counter(all_concepts)
            concept_probs = [count / len(all_concepts) for count in concept_counter.values()]
            diversity_index = -sum(p * math.log(p) for p in concept_probs if p > 0)
        else:
            diversity_index = 0
        
        # 计算继承率 (如果可用)
        inheritance_rate = sum(1 for r in population if hasattr(r, 'parent_id') and r.parent_id) / len(population) if population else 0
        
        # 计算变异统计
        natural_mutations = sum(1 for r in population if hasattr(r, 'mutation_count') and r.mutation_count > 0)
        safe_triggers = sum(1 for r in population if hasattr(r, 'safe_trigger_count') and r.safe_trigger_count > 0)
        
        # 更新指标
        metrics.update({
            'avg_attributes': avg_attribute_count,
            'unique_attributes': unique_attributes,
            'concept_diversity': diversity_index,
            'inheritance_rate': inheritance_rate,
            'total_concepts': total_concepts,
            'natural_mutations': natural_mutations,
            'safe_triggers': safe_triggers
        })
        
        return metrics
    
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

    def save_history(self, file_path=None):
        """保存历史数据到CSV文件
        
        Args:
            file_path: 保存路径，如果为None则使用默认路径
        """
        if file_path is None:
            file_path = self.metrics_path / 'history.csv'
        
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        self.history.to_csv(file_path, index=False)
        logger.info(f"历史数据已保存到: {file_path}")
