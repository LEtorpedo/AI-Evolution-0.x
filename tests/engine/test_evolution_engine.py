import pytest
import tempfile
from pathlib import Path
import yaml
import shutil

from engine.evolution_engine import EvolutionEngine

class TestEvolutionEngine:
    """测试进化引擎"""
    
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
                    'base_path': temp_dir,
                    'reports': "reports/",
                    'visualizations': "visualizations/",
                    'metrics': "metrics/"
                },
                'visualization': {
                    'enabled': True,
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
    
    def test_init(self, temp_config):
        """测试初始化"""
        engine = EvolutionEngine(temp_config)
        assert engine.generation_limit == 5
        assert engine.initial_population_size == 3
        assert engine.base_mutation_rate == 0.3
        assert engine.current_generation == 0
        assert len(engine.population) == 0
    
    def test_initialize_population(self, temp_config):
        """测试种群初始化"""
        engine = EvolutionEngine(temp_config)
        engine.initialize_population()
        assert len(engine.population) == 3
    
    def test_calculate_mutation_rate(self, temp_config):
        """测试变异率计算"""
        engine = EvolutionEngine(temp_config)
        mutation_rate = engine.calculate_mutation_rate(0)
        assert 0.1 <= mutation_rate <= 0.9
    
    def test_evolve_generation(self, temp_config):
        """测试一代进化"""
        engine = EvolutionEngine(temp_config)
        engine.initialize_population()
        initial_population = len(engine.population)
        engine.evolve_generation()
        assert engine.current_generation == 1
        assert len(engine.population) > 0
    
    def test_run(self, temp_config):
        """测试运行进化"""
        engine = EvolutionEngine(temp_config)
        final_report = engine.run(generations=3)
        assert engine.current_generation == 3
        assert len(engine.population) > 0
        assert isinstance(final_report, str)
    
    def test_get_status(self, temp_config):
        """测试获取状态"""
        engine = EvolutionEngine(temp_config)
        engine.initialize_population()
        engine.evolve_generation()
        status = engine.get_status()
        assert status['current_generation'] == 1
        assert status['population_size'] > 0
        assert not status['running'] 