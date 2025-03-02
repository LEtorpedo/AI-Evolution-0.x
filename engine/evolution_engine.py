import logging
from pathlib import Path
import yaml
import time
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
        self.population = [Replicator() for _ in range(self.initial_population_size)]
        logger.info(f"Initial population created with {len(self.population)} replicators")
    
    def calculate_mutation_rate(self, generation: int) -> float:
        """计算当前代的变异率
        
        Args:
            generation: 当前代数
        
        Returns:
            当前代的变异率
        """
        # 简单实现：基础变异率加上一些随机波动
        variation = random.uniform(-0.1, 0.1)
        mutation_rate = max(0.1, min(0.9, self.base_mutation_rate + variation))
        return mutation_rate
    
    def evolve_generation(self):
        """进化一代"""
        if not self.population:
            self.initialize_population()
        
        # 计算当前代的变异率
        mutation_rate = self.calculate_mutation_rate(self.current_generation)
        
        # 分析当前代
        self.analytics.analyze_generation(self.current_generation, self.population, mutation_rate)
        
        # 生成下一代
        next_generation = []
        for replicator in self.population:
            # 每个复制器生成一个后代
            offspring = replicator.replicate()
            next_generation.append(offspring)
        
        # 限制种群大小
        if len(next_generation) > self.max_population_size:
            next_generation = random.sample(next_generation, self.max_population_size)
        
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
        try:
            self.running = True
            self.start_time = time.time()
            
            # 确定运行的代数
            if generations is None:
                generations = self.generation_limit
            
            logger.info(f"Starting evolution for {generations} generations")
            
            # 初始化种群
            if not self.population:
                self.initialize_population()
            
            # 进化循环
            for gen in range(self.current_generation, self.current_generation + generations):
                if not self.running:
                    logger.info("Evolution stopped")
                    break
                
                self.evolve_generation()
                
                # 定期生成报告和可视化
                if gen % self.config['monitor']['report_interval'] == 0:
                    self.reporter.generate_report()
                
                if self.config['monitor']['visualization']['enabled'] and gen % self.config['monitor']['visualization_interval'] == 0:
                    self.visualizer.plot_evolution_trends()
            
            # 生成最终报告
            final_report = self.reporter.generate_report(report_type='full')
            logger.info("Evolution completed")
            logger.info(f"Total time: {time.time() - self.start_time:.2f} seconds")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Error during evolution: {e}", exc_info=True)
            self.running = False
            raise
    
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