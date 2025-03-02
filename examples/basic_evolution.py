"""
基础进化示例

演示如何使用进化引擎运行简单的进化实验
"""

import sys
import os
import logging

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.evolution_engine import EvolutionEngine

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_basic_evolution():
    """运行基础进化实验"""
    print("=== 开始基础进化实验 ===")
    
    # 创建进化引擎
    engine = EvolutionEngine()
    
    # 运行50代
    final_report = engine.run(generations=50)
    
    print("\n=== 进化实验完成 ===")
    print(final_report)

if __name__ == '__main__':
    run_basic_evolution() 