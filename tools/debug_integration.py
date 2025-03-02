"""
集成调试工具：用于诊断系统集成问题
"""

import sys
import os
import logging
import tempfile
from pathlib import Path
import yaml
import shutil
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.evolution_engine import EvolutionEngine
from interface.monitor import EvolutionAnalytics, EvolutionVisualizer, EvolutionReporter

# 设置日志 - 只输出到文件
log_file = "integration_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
    ]
)

logger = logging.getLogger(__name__)

def create_test_config():
    """创建测试配置"""
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
                'base_path': str(temp_dir),
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
    
    logger.info(f"创建临时配置文件: {config_path}")
    logger.info(f"临时目录: {temp_dir}")
    
    return config_path, temp_dir

def debug_component_initialization():
    """调试组件初始化"""
    logger.info("=== 调试组件初始化 ===")
    
    config_path, temp_dir = create_test_config()
    
    try:
        # 逐个初始化组件并检查
        logger.info("初始化分析器...")
        analytics = EvolutionAnalytics(config_path=config_path)
        logger.info(f"分析器初始化成功: {analytics}")
        
        logger.info("初始化可视化器...")
        visualizer = EvolutionVisualizer(analytics, config_path=config_path)
        logger.info(f"可视化器初始化成功: {visualizer}")
        
        logger.info("初始化报告生成器...")
        reporter = EvolutionReporter(analytics, config_path=config_path)
        logger.info(f"报告生成器初始化成功: {reporter}")
        
        logger.info("初始化进化引擎...")
        engine = EvolutionEngine(config_path)
        logger.info(f"进化引擎初始化成功: {engine}")
    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        logger.error(traceback.format_exc())
    finally:
        shutil.rmtree(temp_dir)

def debug_evolution_process():
    """调试进化过程"""
    logger.info("=== 调试进化过程 ===")
    
    config_path, temp_dir = create_test_config()
    
    try:
        # 创建进化引擎
        engine = EvolutionEngine(config_path)
        
        # 初始化种群
        logger.info("初始化种群...")
        engine.initialize_population()
        logger.info(f"种群初始化成功，大小: {len(engine.population)}")
        
        # 逐代进化
        for i in range(3):
            logger.info(f"运行第{i+1}代进化...")
            engine.evolve_generation()
            logger.info(f"第{i+1}代进化完成")
            logger.info(f"当前种群大小: {len(engine.population)}")
            logger.info(f"分析历史记录大小: {len(engine.analytics.history)}")
        
        # 生成报告
        logger.info("生成报告...")
        report = engine.reporter.generate_report()
        logger.info(f"报告生成成功，长度: {len(report)}")
        
        # 生成可视化
        if engine.config['monitor']['visualization']['enabled']:
            logger.info("生成可视化...")
            engine.visualizer.plot_evolution_trends()
            logger.info("可视化生成成功")
        
        logger.info("进化过程测试通过")
        
    except Exception as e:
        logger.error(f"进化过程失败: {e}")
        logger.error(traceback.format_exc())
    finally:
        shutil.rmtree(temp_dir)

def debug_file_operations():
    """调试文件操作"""
    logger.info("=== 调试文件操作 ===")
    
    config_path, temp_dir = create_test_config()
    
    try:
        # 检查目录结构
        logger.info(f"临时目录: {temp_dir}")
        logger.info(f"目录存在: {os.path.exists(temp_dir)}")
        
        reports_dir = Path(temp_dir) / "reports"
        viz_dir = Path(temp_dir) / "visualizations"
        metrics_dir = Path(temp_dir) / "metrics"
        
        logger.info(f"报告目录: {reports_dir}")
        logger.info(f"报告目录存在: {reports_dir.exists()}")
        
        logger.info(f"可视化目录: {viz_dir}")
        logger.info(f"可视化目录存在: {viz_dir.exists()}")
        
        logger.info(f"指标目录: {metrics_dir}")
        logger.info(f"指标目录存在: {metrics_dir.exists()}")
        
        # 测试文件写入
        logger.info("测试文件写入...")
        test_file = Path(temp_dir) / "test_file.txt"
        with open(test_file, 'w') as f:
            f.write("测试内容")
        
        logger.info(f"测试文件: {test_file}")
        logger.info(f"测试文件存在: {test_file.exists()}")
        
        # 读取测试文件
        with open(test_file, 'r') as f:
            content = f.read()
        logger.info(f"测试文件内容: {content}")
        
        logger.info("文件操作测试通过")
        
    except Exception as e:
        logger.error(f"文件操作失败: {e}")
        logger.error(traceback.format_exc())
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    print(f"开始集成调试，日志将保存到 {log_file}")
    
    debug_component_initialization()
    debug_evolution_process()
    debug_file_operations()
    
    print(f"集成调试完成，请查看 {log_file} 获取详细信息") 