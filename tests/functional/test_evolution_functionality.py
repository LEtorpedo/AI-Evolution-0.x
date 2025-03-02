"""
功能测试：测试进化系统的主要功能
"""

import sys
import os
import logging
import tempfile
from pathlib import Path
import yaml
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from engine.evolution_engine import EvolutionEngine

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_config():
    """创建测试配置"""
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "test_config.yaml"
    
    config = {
        'evolution': {
            'generation_limit': 10,
            'initial_population_size': 5,
            'max_population_size': 20,
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
    
    return config_path, temp_dir

def test_basic_evolution():
    """测试基本进化功能"""
    print("\n=== 测试基本进化功能 ===")
    
    config_path, temp_dir = create_test_config()
    
    try:
        # 创建进化引擎
        engine = EvolutionEngine(config_path)
        
        # 运行进化
        final_report = engine.run(generations=5)
        
        # 验证结果
        print(f"进化完成，当前代数: {engine.current_generation}")
        print(f"种群大小: {len(engine.population)}")
        print(f"报告长度: {len(final_report)} 字符")
        
        # 检查生成的文件
        report_files = list(Path(temp_dir, "reports").glob("*.txt"))
        print(f"生成的报告文件数量: {len(report_files)}")
        
        viz_files = list(Path(temp_dir, "visualizations").glob("*.png"))
        print(f"生成的可视化文件数量: {len(viz_files)}")
        
        metrics_files = list(Path(temp_dir, "metrics").glob("*.csv"))
        print(f"生成的指标文件数量: {len(metrics_files)}")
        
        # 测试通过条件
        success = (
            engine.current_generation == 5 and
            len(engine.population) > 0 and
            len(final_report) > 0 and
            len(report_files) > 0
        )
        
        if success:
            print("✅ 测试通过")
        else:
            print("❌ 测试失败")
            
    finally:
        # 清理
        shutil.rmtree(temp_dir)

def test_evolution_interruption():
    """测试进化中断功能"""
    print("\n=== 测试进化中断功能 ===")
    
    config_path, temp_dir = create_test_config()
    
    try:
        # 创建进化引擎
        engine = EvolutionEngine(config_path)
        
        # 初始化种群
        engine.initialize_population()
        
        # 运行几代
        for _ in range(3):
            engine.evolve_generation()
        
        # 中断进化
        engine.stop()
        
        # 验证状态
        status = engine.get_status()
        print(f"当前状态: {status}")
        
        # 测试通过条件
        success = (
            status['current_generation'] == 3 and
            not status['running']
        )
        
        if success:
            print("✅ 测试通过")
        else:
            print("❌ 测试失败")
            
    finally:
        # 清理
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_basic_evolution()
    test_evolution_interruption() 