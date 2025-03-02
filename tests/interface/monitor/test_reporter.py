import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import os
import json

from interface.monitor.analytics import EvolutionAnalytics
from interface.monitor.reporter import EvolutionReporter
from reproduction.replicator import Replicator

class TestEvolutionReporter:
    """测试进化报告生成器"""
    
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
        (Path(temp_dir) / "reports").mkdir(exist_ok=True)
        
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
        reporter = EvolutionReporter(analytics_with_data, config_path=temp_config)
        assert reporter.analytics is analytics_with_data
    
    def test_generate_full_report(self, temp_config, analytics_with_data):
        """测试生成完整报告"""
        reporter = EvolutionReporter(analytics_with_data, config_path=temp_config)
        report = reporter.generate_report(report_type='full')
        
        # 验证报告内容
        assert "进化分析完整报告" in report
        assert "基础统计" in report
        assert "变异分析" in report
        assert "属性分析" in report
        assert "概念多样性分析" in report
        
        # 验证报告文件生成
        config_dir = Path(temp_config).parent
        report_dir = config_dir / "reports"
        
        # 检查是否有报告文件生成
        report_files = list(report_dir.glob("*full*.txt"))
        assert len(report_files) > 0
    
    def test_generate_summary_report(self, temp_config, analytics_with_data):
        """测试生成摘要报告"""
        reporter = EvolutionReporter(analytics_with_data, config_path=temp_config)
        report = reporter.generate_report(report_type='summary')
        
        # 验证报告内容
        summary = json.loads(report)
        assert 'total_generations' in summary
        assert 'mutation_efficiency' in summary
        assert 'inheritance_stability' in summary
        
        # 验证报告文件生成
        config_dir = Path(temp_config).parent
        report_dir = config_dir / "reports"
        
        # 检查是否有报告文件生成
        report_files = list(report_dir.glob("*summary*.txt"))
        assert len(report_files) > 0
    
    def test_generate_metrics_report(self, temp_config, analytics_with_data):
        """测试生成指标报告"""
        reporter = EvolutionReporter(analytics_with_data, config_path=temp_config)
        report = reporter.generate_report(report_type='metrics')
        
        # 验证报告内容
        assert "count" in report
        assert "mean" in report
        assert "std" in report
        
        # 验证报告文件生成
        config_dir = Path(temp_config).parent
        report_dir = config_dir / "reports"
        
        # 检查是否有报告文件生成
        report_files = list(report_dir.glob("*metrics*.txt"))
        assert len(report_files) > 0
    
    def test_empty_data(self, temp_config):
        """测试空数据情况"""
        analytics = EvolutionAnalytics(config_path=temp_config)
        reporter = EvolutionReporter(analytics, config_path=temp_config)
        
        # 空数据应该返回提示信息
        report = reporter.generate_report()
        assert report == "No evolution data available for reporting."
