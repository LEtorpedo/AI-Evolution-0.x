import pytest
import tempfile
from pathlib import Path
import yaml
import shutil
import os
import logging

from engine.evolution_engine import EvolutionEngine
from interface.monitor import EvolutionAnalytics, EvolutionVisualizer, EvolutionReporter
from reproduction.replicator import Replicator

# 禁用测试中的日志输出
logging.basicConfig(level=logging.ERROR)

class TestSystemIntegration:
    """测试系统集成"""
    
    @pytest.fixture
    def temp_config(self):
        """创建临时配置文件"""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "test_config.yaml"
        
        config = {
            'evolution': {
                'generation_limit': 5,
                'initial_population_size': 3,
                'max_population_size': 10,
                'base_mutation_rate': 0.3
            },
            'monitor': {
                'log_interval': 1,
                'report_interval': 2,
                'visualization_interval': 2,
                'metrics': [
                    'mutation_rate',
                    'inheritance_rate',
                    'attribute_count',
                    'concept_diversity'
                ],
                'storage': {
                    'base_path': str(temp_dir),  # 确保路径是字符串
                    'reports': "reports/",
                    'visualizations': "visualizations/",
                    'metrics': "metrics/"
                },
                'visualization': {
                    'enabled': False,  # 禁用可视化以加快测试
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
        
        yield config_path
        
        # 清理
        shutil.rmtree(temp_dir)
    
    def test_full_system_integration(self, temp_config):
        """测试完整系统集成"""
        # 创建进化引擎
        engine = EvolutionEngine(temp_config)
        
        # 运行进化（减少代数以加快测试）
        final_report = engine.run(generations=2)
        
        # 验证结果
        assert engine.current_generation == 2
        assert len(engine.population) > 0
        
        # 验证监控系统
        assert len(engine.analytics.history) > 0
        
        # 验证报告生成
        assert isinstance(final_report, str)
        assert "Generation" in final_report or "代" in final_report
    
    def test_component_interaction(self, temp_config):
        """测试组件交互"""
        # 创建各个组件
        analytics = EvolutionAnalytics(config_path=temp_config)
        
        # 添加一些测试数据
        ancestor = Replicator()
        population = [ancestor]
        
        # 手动添加一些数据
        for gen in range(2):
            analytics.analyze_generation(gen, population, 0.3)
            population = [r.replicate() for r in population]
        
        # 创建其他组件
        visualizer = EvolutionVisualizer(analytics, config_path=temp_config)
        reporter = EvolutionReporter(analytics, config_path=temp_config)
        
        # 验证组件交互
        assert len(analytics.history) > 0
        assert analytics.history['generation'].max() >= 1
        
        # 生成报告
        report = reporter.generate_report()
        assert isinstance(report, str) 