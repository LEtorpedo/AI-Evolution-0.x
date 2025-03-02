import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import os

from interface.monitor.analytics import EvolutionAnalytics
from reproduction.replicator import Replicator

class TestEvolutionAnalytics:
    """测试进化分析器"""
    
    @pytest.fixture
    def temp_config(self):
        """创建临时配置文件"""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "test_config.yaml"
        
        with open(config_path, 'w') as f:
            f.write("""
monitor:
  log_interval: 1
  metrics:
    - mutation_rate
    - inheritance_rate
    - attribute_count
    - concept_diversity
  storage:
    base_path: "{}"
    reports: "reports/"
    visualizations: "visualizations/"
    metrics: "metrics/"
  visualization:
    enabled: true
    format: "png"
    dpi: 300
""".format(temp_dir))
        
        yield config_path
        
        # 清理
        shutil.rmtree(temp_dir)
    
    def test_init(self, temp_config):
        """测试初始化"""
        analytics = EvolutionAnalytics(config_path=temp_config)
        assert isinstance(analytics.history, pd.DataFrame)
        assert len(analytics.history) == 0
        assert isinstance(analytics.attribute_stats, dict)
    
    def test_analyze_generation(self, temp_config):
        """测试代数分析"""
        analytics = EvolutionAnalytics(config_path=temp_config)
        
        # 创建测试种群
        ancestor = Replicator()
        population = [ancestor]
        
        # 分析第0代
        analytics.analyze_generation(0, population, 0.3)
        
        # 验证数据记录
        assert len(analytics.history) == 1
        assert analytics.history.iloc[0]['generation'] == 0
        assert analytics.history.iloc[0]['population_size'] == 1
        assert analytics.history.iloc[0]['mutation_rate'] == 0.3
        
        # 生成下一代并分析
        new_gen = [r.replicate() for r in population]
        analytics.analyze_generation(1, new_gen, 0.3)
        
        # 验证数据更新
        assert len(analytics.history) == 2
        assert analytics.history.iloc[1]['generation'] == 1
        
        # 验证属性统计
        assert len(analytics.attribute_stats) > 0
    
    def test_get_statistics(self, temp_config):
        """测试统计数据获取"""
        analytics = EvolutionAnalytics(config_path=temp_config)
        
        # 空数据情况
        assert analytics.get_statistics() == "No evolution data recorded yet."
        
        # 添加测试数据
        ancestor = Replicator()
        population = [ancestor]
        
        for gen in range(5):
            analytics.analyze_generation(gen, population, 0.3)
            population = [r.replicate() for r in population]
        
        # 验证统计数据
        stats = analytics.get_statistics()
        assert isinstance(stats, dict)
        assert 'total_generations' in stats
        assert stats['total_generations'] == 5
        assert 'avg_mutation_rate' in stats
        assert 'concept_diversity_trend' in stats
    
    def test_save_metrics(self, temp_config):
        """测试指标保存"""
        analytics = EvolutionAnalytics(config_path=temp_config)
        
        # 添加测试数据
        ancestor = Replicator()
        population = [ancestor]
        
        for gen in range(3):
            analytics.analyze_generation(gen, population, 0.3)
            population = [r.replicate() for r in population]
        
        # 验证文件保存
        config_dir = Path(temp_config).parent
        metrics_file = config_dir / "metrics" / "evolution_data.csv"
        assert metrics_file.exists()
        
        # 验证文件内容
        saved_data = pd.read_csv(metrics_file)
        assert len(saved_data) == 3
        assert 'generation' in saved_data.columns
        assert 'mutation_rate' in saved_data.columns
