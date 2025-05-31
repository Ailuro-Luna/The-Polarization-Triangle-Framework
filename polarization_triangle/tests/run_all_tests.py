"""
Run All Tests Script

This script automatically discovers and runs all test files in the tests directory.
It provides comprehensive test coverage reports and execution status.
"""

import os
import sys
import importlib.util
import traceback
import time
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))


def discover_test_files():
    """发现tests目录中的所有Python测试文件"""
    tests_dir = Path(__file__).parent
    test_files = []
    
    for file_path in tests_dir.glob("test_*.py"):
        # 排除运行脚本本身
        if file_path.name != 'run_all_tests.py':
            test_files.append(file_path)
    
    return sorted(test_files)


def run_test_file(file_path):
    """运行单个测试文件"""
    print(f"🧪 正在运行测试: {file_path.name}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 动态导入并执行模块
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        module = importlib.util.module_from_spec(spec)
        
        # 执行模块
        spec.loader.exec_module(module)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ 测试完成 (耗时: {duration:.2f}秒)")
        return True, duration
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ 测试失败: {str(e)}")
        print("\n错误详情:")
        traceback.print_exc()
        return False, duration


def display_test_info(file_path):
    """显示测试文件的基本信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 提取文档字符串
        lines = content.split('\n')
        docstring_lines = []
        in_docstring = False
        
        for line in lines:
            if '"""' in line and not in_docstring:
                in_docstring = True
                docstring_part = line.split('"""')[1] if '"""' in line.split('"""')[1:] else ""
                if docstring_part:
                    docstring_lines.append(docstring_part)
                if line.count('"""') == 2:  # 单行文档字符串
                    break
            elif '"""' in line and in_docstring:
                docstring_part = line.split('"""')[0]
                if docstring_part:
                    docstring_lines.append(docstring_part)
                break
            elif in_docstring:
                docstring_lines.append(line)
        
        if docstring_lines:
            description = '\n'.join(docstring_lines).strip()
            print(f"📋 测试描述: {description}")
        else:
            print("📋 测试描述: 暂无描述")
        
        # 统计测试函数数量
        test_function_count = content.count('def test_')
        print(f"🔢 测试函数数量: {test_function_count}")
            
    except Exception as e:
        print(f"📋 测试描述: 无法读取文件信息 ({str(e)})")


def run_quick_tests():
    """运行快速测试（仅基本功能验证）"""
    print("⚡ 快速测试模式 - 仅运行基本功能验证")
    print("=" * 70)
    
    # 这里可以定义一些快速测试
    try:
        # 测试基本导入
        from polarization_triangle.analysis.statistics import (
            calculate_mean_opinion,
            get_polarization_index
        )
        print("✅ 模块导入测试通过")
        
        # 测试基本配置
        from polarization_triangle.core.config import high_polarization_config
        from polarization_triangle.core.simulation import Simulation
        import copy
        
        config = copy.deepcopy(high_polarization_config)
        config.num_agents = 20  # 使用较小的agent数量以加快速度
        sim = Simulation(config)
        print("✅ Simulation创建测试通过")
        
        # 运行几步并测试统计功能
        for _ in range(3):
            sim.step()
        
        mean_stats = calculate_mean_opinion(sim)
        polarization = get_polarization_index(sim)
        
        print("✅ 统计功能测试通过")
        print(f"   平均意见: {mean_stats['mean_opinion']:.4f}")
        print(f"   极化指数: {polarization:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 快速测试失败: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """主函数：运行所有测试"""
    print("🔬 Polarization Triangle Framework - Test Runner")
    print("=" * 70)
    
    # 发现所有测试文件
    test_files = discover_test_files()
    
    if not test_files:
        print("⚠️  未发现任何测试文件")
        return False
    
    print(f"发现 {len(test_files)} 个测试文件:")
    for i, file_path in enumerate(test_files, 1):
        print(f"  {i}. {file_path.name}")
    print()
    
    # 运行每个测试
    successful_tests = 0
    total_tests = len(test_files)
    total_duration = 0
    
    for i, file_path in enumerate(test_files, 1):
        print(f"\n🧪 测试 {i}/{total_tests}: {file_path.name}")
        print("-" * 60)
        
        # 显示测试信息
        display_test_info(file_path)
        print()
        
        # 运行测试
        try:
            success, duration = run_test_file(file_path)
            total_duration += duration
            if success:
                successful_tests += 1
        except KeyboardInterrupt:
            print("\n⏹️  用户中断，停止测试")
            break
        except Exception as e:
            print(f"❌ 运行测试时出现未预期的错误: {str(e)}")
        
        print("\n" + "=" * 70)
    
    # 显示总结
    print(f"\n📊 测试总结:")
    print(f"   总测试文件数: {total_tests}")
    print(f"   成功运行: {successful_tests}")
    print(f"   失败运行: {total_tests - successful_tests}")
    print(f"   成功率: {(successful_tests/total_tests)*100:.1f}%")
    print(f"   总耗时: {total_duration:.2f}秒")
    
    if successful_tests == total_tests:
        print("\n🎉 所有测试都通过了！")
        return True
    else:
        print(f"\n⚠️  有 {total_tests - successful_tests} 个测试失败")
        return False


def run_specific_test():
    """运行特定的测试（交互模式）"""
    test_files = discover_test_files()
    
    if not test_files:
        print("⚠️  未发现任何测试文件")
        return
    
    print("可用的测试文件:")
    for i, file_path in enumerate(test_files, 1):
        print(f"  {i}. {file_path.name}")
    
    try:
        choice = int(input("\n请选择要运行的测试 (输入编号): ")) - 1
        if 0 <= choice < len(test_files):
            file_path = test_files[choice]
            print(f"\n运行测试: {file_path.name}")
            print("=" * 60)
            display_test_info(file_path)
            print()
            run_test_file(file_path)
        else:
            print("❌ 无效的选择")
    except (ValueError, KeyboardInterrupt):
        print("\n取消运行")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            run_specific_test()
        elif sys.argv[1] == "--quick" or sys.argv[1] == "-q":
            success = run_quick_tests()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("使用方法:")
            print("  python run_all_tests.py          # 运行所有测试")
            print("  python run_all_tests.py -i       # 交互模式，选择运行特定测试")
            print("  python run_all_tests.py -q       # 快速测试模式")
            print("  python run_all_tests.py -h       # 显示帮助信息")
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("使用 -h 查看帮助信息")
    else:
        # 默认运行所有测试
        success = main()
        sys.exit(0 if success else 1) 