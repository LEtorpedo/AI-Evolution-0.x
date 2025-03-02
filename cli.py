import argparse
import logging
import sys
import yaml
from pathlib import Path

from engine.evolution_engine import EvolutionEngine

def setup_logging(verbose=False):
    """设置日志级别"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_default_config():
    """创建默认配置文件"""
    config = {
        'evolution': {
            'generation_limit': 100,
            'initial_population_size': 10,
            'max_population_size': 50,
            'base_mutation_rate': 0.3
        },
        'monitor': {
            'log_interval': 1,
            'report_interval': 10,
            'visualization_interval': 10,
            'metrics': [
                'mutation_rate',
                'inheritance_rate',
                'attribute_count',
                'concept_diversity'
            ],
            'storage': {
                'base_path': './storage/monitor/',
                'reports': 'reports/',
                'visualizations': 'visualizations/',
                'metrics': 'metrics/'
            },
            'visualization': {
                'enabled': True,
                'format': 'png',
                'dpi': 300
            }
        }
    }
    
    config_path = Path('config.yaml')
    if not config_path.exists():
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created default configuration at {config_path}")
    else:
        print(f"Configuration file already exists at {config_path}")
    
    return config_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI Evolution Experiment')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--generations', type=int, help='Number of generations to run')
    parser.add_argument('--create-config', action='store_true', help='Create default configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 创建默认配置
    if args.create_config:
        create_default_config()
        return
    
    # 确定配置文件路径
    config_path = args.config
    if not config_path:
        config_path = 'config.yaml'
        if not Path(config_path).exists():
            config_path = create_default_config()
    
    # 创建并运行进化引擎
    engine = EvolutionEngine(config_path)
    final_report = engine.run(args.generations)
    
    print("\n=== 进化实验完成 ===")
    print(final_report)

if __name__ == '__main__':
    main() 