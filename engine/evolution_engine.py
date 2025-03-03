import logging
from pathlib import Path
import yaml
import time
import pickle
from typing import List, Dict, Any
import random

from reproduction.replicator import Replicator
from interface.monitor import EvolutionAnalytics, EvolutionVisualizer, EvolutionReporter

logger = logging.getLogger(__name__)

class EvolutionEngine:
    """进化引擎：控制整个进化过程"""
    
    def __init__(self, config_path="config.yaml"):
        """初始化进化引擎
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化监控系统
        self.analytics = EvolutionAnalytics(config_path)
        self.visualizer = EvolutionVisualizer(self.analytics, config_path)
        self.reporter = EvolutionReporter(self.analytics, config_path)
        
        # 进化参数
        self.generation_limit = self.config['evolution']['generation_limit']
        self.initial_population_size = self.config['evolution']['initial_population_size']
        self.base_mutation_rate = self.config['evolution']['base_mutation_rate']
        self.max_population_size = self.config['evolution']['max_population_size']
        
        # 运行状态
        self.current_generation = 0
        self.population = []
        self.running = False
        self.start_time = None
        
        logger.info(f"Evolution engine initialized with generation limit: {self.generation_limit}")
    
    def initialize_population(self):
        """初始化种群"""
        initial_size = self.config['evolution']['initial_population_size']
        
        logger.info(f"Initializing population with {initial_size} replicators")
        
        self.population = []
        
        for _ in range(initial_size):
            # 创建初始属性
            initial_attrs = {}
            for i in range(random.randint(1, 3)):
                attr_id = f"attr_{random.randint(1000, 9999)}"
                initial_attrs[attr_id] = {
                    "name": f"属性_{random.randint(1, 100)}",
                    "value": random.random()
                }
            
            # 创建初始概念
            initial_concepts = {}
            if initial_attrs and random.random() < 0.7:  # 70%的概率创建初始概念
                concept_id = f"concept_{random.randint(1000, 9999)}"
                concept_attrs = random.sample(
                    list(initial_attrs.keys()),
                    min(random.randint(1, 2), len(initial_attrs))
                )
                initial_concepts[concept_id] = {
                    "name": f"概念_{random.randint(1, 100)}",
                    "attributes": concept_attrs
                }
            
            # 创建复制器
            replicator = Replicator(initial_attrs, initial_concepts)
            self.population.append(replicator)
        
        logger.info(f"Population initialized with {len(self.population)} replicators")
    
    def select_population(self):
        """选择用于繁殖的个体
        
        基于适应度（或其他标准）选择个体进行繁殖
        
        Returns:
            选中的个体列表
        """
        # 如果配置中有选择压力参数，则使用它
        selection_pressure = self.config['evolution'].get('selection_pressure', 0.7)
        
        # 对种群进行排序（这里简单地使用属性和概念数量作为排序标准）
        sorted_population = sorted(
            self.population,
            key=lambda r: len(r.attributes) + len(r.concepts) * 2,  # 概念权重更高
            reverse=True  # 降序排列，使得"更好"的个体排在前面
        )
        
        # 根据选择压力确定选择数量
        selection_size = max(int(len(self.population) * selection_pressure), 1)
        
        # 选择前N个个体
        selected = sorted_population[:selection_size]
        
        # 确保至少选择一个个体
        if not selected and self.population:
            selected = [random.choice(self.population)]
        
        logger.info(f"Selected {len(selected)} replicators for breeding")
        return selected
    
    def calculate_mutation_rate(self, generation):
        """计算当前代的变异率
        
        可以实现自适应变异率，根据进化阶段调整
        
        Args:
            generation: 当前代数
            
        Returns:
            当前代的变异率
        """
        # 基础变异率
        mutation_rate = self.base_mutation_rate
        
        # 如果配置中启用了自适应变异
        if self.config['evolution'].get('adaptive_mutation', False):
            # 随着代数增加，变异率可能降低
            decay = self.config['evolution'].get('mutation_decay', 0.995)
            floor = self.config['evolution'].get('mutation_floor', 0.05)
            
            # 应用衰减
            mutation_rate = max(mutation_rate * (decay ** generation), floor)
        
        return mutation_rate
    
    def evolve_generation(self):
        """进化一代"""
        if not self.population:
            self.initialize_population()
        
        # 计算当前代的变异率
        mutation_rate = self.calculate_mutation_rate(self.current_generation)
        
        # 分析当前代
        self.analytics.analyze_generation(self.current_generation, self.population, mutation_rate)
        
        # 选择过程
        selected_population = self.select_population()
        
        # 生成下一代
        next_generation = []
        
        # 精英保留 - 直接将最佳个体传递到下一代
        elite_count = max(int(len(self.population) * 0.1), 1)
        next_generation.extend(self.population[:elite_count])
        
        # 剩余位置由选中的个体繁殖填充
        while len(next_generation) < self.max_population_size:
            # 随机选择一个父代
            parent = random.choice(selected_population)
            
            # 生成后代
            offspring = parent.replicate(mutation_rate=mutation_rate)
            next_generation.append(offspring)
        
        # 限制种群大小
        if len(next_generation) > self.max_population_size:
            next_generation = next_generation[:self.max_population_size]
        
        # 更新种群
        self.population = next_generation
        
        # 更新代数
        self.current_generation += 1
        
        logger.info(f"Generation {self.current_generation} evolved with {len(self.population)} replicators")
    
    def run(self, generations=None):
        """运行进化过程
        
        Args:
            generations: 要运行的代数，如果为None则使用配置中的值
        
        Returns:
            最终报告
        """
        if generations is None:
            generations = self.generation_limit
        
        self.running = True
        self.start_time = time.time()
        
        logger.info(f"Starting evolution for {generations} generations")
        
        for gen in range(generations):
            self.evolve_generation()
            
            # 记录日志
            if self.config['monitor'].get('log_interval', 1) > 0 and gen % self.config['monitor']['log_interval'] == 0:
                logger.info(f"Generation {gen}: Population size = {len(self.population)}")
            
            # 生成报告
            if self.config['monitor'].get('report_interval', 10) > 0 and gen % self.config['monitor']['report_interval'] == 0:
                report = self.reporter.generate_report()
                logger.info(f"Generation {gen} report:\n{report}")
            
            # 生成可视化
            # 检查visualization键是否存在
            if 'visualization' in self.config['monitor'] and self.config['monitor']['visualization'].get('enabled', False):
                if gen % self.config['monitor'].get('visualization_interval', 10) == 0:
                    self.visualizer.visualize_generation(gen)
            # 如果没有visualization键，但有visualization_interval
            elif 'visualization_interval' in self.config['monitor'] and gen % self.config['monitor']['visualization_interval'] == 0:
                try:
                    self.visualizer.visualize_generation(gen)
                except Exception as e:
                    logger.warning(f"可视化生成失败: {e}")
        
        self.running = False
        end_time = time.time()
        runtime = end_time - self.start_time
        
        logger.info(f"Evolution completed in {runtime:.2f} seconds")
        
        # 生成最终报告
        final_report = self.reporter.generate_report()
        
        return final_report
    
    def stop(self):
        """停止进化过程"""
        self.running = False
        logger.info("Evolution stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态
        
        Returns:
            包含当前状态信息的字典
        """
        return {
            'running': self.running,
            'current_generation': self.current_generation,
            'population_size': len(self.population),
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'generation_limit': self.generation_limit
        }
    
    def save_results(self, file_path):
        """保存实验结果
        
        Args:
            file_path: 保存路径
        """
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 创建结果字典
        results = {
            'config': self.config,
            'generations': self.current_generation,
            'final_population_size': len(self.population),
            'runtime': time.time() - self.start_time if self.start_time else 0
        }
        
        # 保存结果
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"实验结果已保存到: {file_path}")
    
    def generate_report(self, file_path=None):
        """生成实验报告
        
        Args:
            file_path: 报告保存路径，如果为None则返回报告文本
            
        Returns:
            如果file_path为None，则返回报告文本
        """
        report = self.reporter.generate_report()
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(report)
            logger.info(f"实验报告已保存到: {file_path}")
        
        return report 