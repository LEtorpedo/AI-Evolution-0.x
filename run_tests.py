"""
运行所有测试的脚本
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, title):
    """运行命令并返回结果"""
    print(f"运行 {title}...")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    success = result.returncode == 0
    print(f"  {'✓' if success else '✗'} {elapsed_time:.2f}秒")
    
    # 如果失败，将输出保存到日志文件
    if not success:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        error_log = log_dir / f"{title.lower().replace(' ', '_')}_error.log"
        with open(error_log, 'w') as f:
            f.write(f"命令: {command}\n")
            f.write(f"退出代码: {result.returncode}\n")
            f.write("\n=== 标准输出 ===\n")
            f.write(result.stdout)
            f.write("\n=== 错误输出 ===\n")
            f.write(result.stderr)
        
        print(f"  详细错误信息已保存到: {error_log}")
    
    return success

def main():
    """主函数"""
    # 确保当前目录是项目根目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("开始测试运行")
    
    # 运行单元测试
    unit_tests_passed = run_command("pytest tests/engine tests/interface/monitor -v", "单元测试")
    
    # 运行集成测试
    integration_tests_passed = run_command("pytest tests/integration -v", "集成测试")
    
    # 运行功能测试
    functional_tests_passed = run_command("python tests/functional/test_evolution_functionality.py", "功能测试")
    
    # 运行性能测试
    performance_tests_passed = run_command("python tests/performance/test_evolution_performance.py", "性能测试")
    
    # 打印总结
    print("\n=== 测试总结 ===")
    print(f"单元测试: {'通过' if unit_tests_passed else '失败'}")
    print(f"集成测试: {'通过' if integration_tests_passed else '失败'}")
    print(f"功能测试: {'通过' if functional_tests_passed else '失败'}")
    print(f"性能测试: {'通过' if performance_tests_passed else '失败'}")
    
    all_passed = unit_tests_passed and integration_tests_passed and functional_tests_passed and performance_tests_passed
    print(f"\n总体结果: {'全部通过' if all_passed else '部分失败'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 