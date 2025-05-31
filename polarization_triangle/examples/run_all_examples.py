"""
Run All Examples Script

This script automatically discovers and runs all example files in the examples directory.
It provides a comprehensive overview of all available examples and their execution status.
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))


def discover_example_files():
    """发现examples目录中的所有Python示例文件"""
    examples_dir = Path(__file__).parent
    example_files = []
    
    for file_path in examples_dir.glob("*.py"):
        # 排除__init__.py和运行脚本本身
        if file_path.name not in ['__init__.py', 'run_all_examples.py']:
            example_files.append(file_path)
    
    return sorted(example_files)


def run_example_file(file_path):
    """运行单个示例文件"""
    print(f"📂 正在运行: {file_path.name}")
    print("=" * 50)
    
    try:
        # 动态导入并执行模块
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        module = importlib.util.module_from_spec(spec)
        
        # 执行模块
        spec.loader.exec_module(module)
        
        print("✅ 示例运行成功")
        return True
        
    except Exception as e:
        print(f"❌ 示例运行失败: {str(e)}")
        print("\n错误详情:")
        traceback.print_exc()
        return False


def display_example_info(file_path):
    """显示示例文件的基本信息"""
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
            print(f"📄 描述: {description}")
        else:
            print("📄 描述: 暂无描述")
            
    except Exception as e:
        print(f"📄 描述: 无法读取文件信息 ({str(e)})")


def main():
    """主函数：运行所有示例"""
    print("🚀 Polarization Triangle Framework - Examples Runner")
    print("=" * 70)
    
    # 发现所有示例文件
    example_files = discover_example_files()
    
    if not example_files:
        print("⚠️  未发现任何示例文件")
        return False
    
    print(f"发现 {len(example_files)} 个示例文件:")
    for i, file_path in enumerate(example_files, 1):
        print(f"  {i}. {file_path.name}")
    print()
    
    # 运行每个示例
    successful_runs = 0
    total_runs = len(example_files)
    
    for i, file_path in enumerate(example_files, 1):
        print(f"\n📋 示例 {i}/{total_runs}: {file_path.name}")
        print("-" * 50)
        
        # 显示示例信息
        display_example_info(file_path)
        print()
        
        # 运行示例
        try:
            if run_example_file(file_path):
                successful_runs += 1
        except KeyboardInterrupt:
            print("\n⏹️  用户中断，停止运行")
            break
        except Exception as e:
            print(f"❌ 运行示例时出现未预期的错误: {str(e)}")
        
        print("\n" + "=" * 70)
    
    # 显示总结
    print(f"\n📊 运行总结:")
    print(f"   总示例数: {total_runs}")
    print(f"   成功运行: {successful_runs}")
    print(f"   失败运行: {total_runs - successful_runs}")
    print(f"   成功率: {(successful_runs/total_runs)*100:.1f}%")
    
    if successful_runs == total_runs:
        print("\n🎉 所有示例都运行成功！")
        return True
    else:
        print(f"\n⚠️  有 {total_runs - successful_runs} 个示例运行失败")
        return False


def run_specific_example():
    """运行特定的示例（交互模式）"""
    example_files = discover_example_files()
    
    if not example_files:
        print("⚠️  未发现任何示例文件")
        return
    
    print("可用的示例文件:")
    for i, file_path in enumerate(example_files, 1):
        print(f"  {i}. {file_path.name}")
    
    try:
        choice = int(input("\n请选择要运行的示例 (输入编号): ")) - 1
        if 0 <= choice < len(example_files):
            file_path = example_files[choice]
            print(f"\n运行示例: {file_path.name}")
            print("=" * 50)
            display_example_info(file_path)
            print()
            run_example_file(file_path)
        else:
            print("❌ 无效的选择")
    except (ValueError, KeyboardInterrupt):
        print("\n取消运行")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            run_specific_example()
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("使用方法:")
            print("  python run_all_examples.py          # 运行所有示例")
            print("  python run_all_examples.py -i       # 交互模式，选择运行特定示例")
            print("  python run_all_examples.py -h       # 显示帮助信息")
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("使用 -h 查看帮助信息")
    else:
        # 默认运行所有示例
        success = main()
        sys.exit(0 if success else 1) 