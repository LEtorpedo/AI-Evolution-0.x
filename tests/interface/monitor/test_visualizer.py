import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，用于测试

from interface.monitor.analytics import EvolutionAnalytics
from interface.monitor.visualizer import EvolutionVisualizer
from reproduction.replicator import Replicator

class TestEvolutionVisualizer:
    """测试进化可视化器"""
    
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
        
        # 创建必要的目录
        (Path(temp_dir) / "visualizations").mkdir(exist_ok=True)
        
        yield config_path
        
        # 清理
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def analytics_with_data(self, temp_config):
        """创建带有测试数据的分析器"""
        analytics = EvolutionAnalytics(config_path=temp_config)
        
        # 添加测试数据
        ancestor = Replicator()
        population = [ancestor]
        
        for gen in range(10):
            analytics.analyze_generation(gen, population, 0.3)
            population = [r.replicate() for r in population]
        
        return analytics
    
    def test_init(self, temp_config, analytics_with_data):
        """测试初始化"""
        visualizer = EvolutionVisualizer(analytics_with_data, config_path=temp_config)
        assert visualizer.analytics is analytics_with_data
    
    def test_plot_evolution_trends(self, temp_config, analytics_with_data):
        """测试绘制演化趋势图"""
        visualizer = EvolutionVisualizer(analytics_with_data, config_path=temp_config)
        visualizer.plot_evolution_trends()
        
        # 验证图表文件生成
        config_dir = Path(temp_config).parent
        vis_dir = config_dir / "visualizations"
        
        # 检查是否有png文件生成
        png_files = list(vis_dir.glob("*.png"))
        assert len(png_files) > 0
    
    def test_empty_data(self, temp_config):
        """测试空数据情况"""
        analytics = EvolutionAnalytics(config_path=temp_config)
        visualizer = EvolutionVisualizer(analytics, config_path=temp_config)
        
        # 不应该生成图表，但也不应该报错
        visualizer.plot_evolution_trends()
