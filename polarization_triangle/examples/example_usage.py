#!/usr/bin/env python3
"""
使用示例：Zealot和Morality分析实验的分离式运行

这个脚本展示了如何使用新的分离式功能：
1. 运行测试并积累数据
2. 从累积数据生成图表

优势：
- 可以多次运行测试，每次积累更多数据
- 数据和绘图分离，可以单独重新绘图
- 图表文件名包含总运行次数信息
"""

from polarization_triangle.experiments.zealot_morality_analysis import (
    run_and_accumulate_data,
    plot_from_accumulated_data,
    run_zealot_morality_analysis
)

def example_incremental_runs():
    """示例：多次增量运行以积累数据"""
    
    output_dir = "results/example_zealot_analysis"
    
    print("🔬 示例：多次增量运行实验")
    print("=" * 60)
    
    # 第一次运行：少量数据快速测试
    print("\n📊 第一批数据收集（快速测试）")
    run_and_accumulate_data(
        output_dir=output_dir,
        num_runs=3,  # 少量运行用于快速测试
        max_zealots=20,  
        max_morality=20,
        batch_name="quick_test"
    )
    
    # 生成初步图表看看结果
    print("\n📈 生成初步图表...")
    plot_from_accumulated_data(output_dir)
    
    # 第二次运行：增加更多数据
    print("\n📊 第二批数据收集（增加数据）")
    run_and_accumulate_data(
        output_dir=output_dir,
        num_runs=5,  # 更多运行次数
        max_zealots=20,  
        max_morality=20,
        batch_name="detailed_run"
    )
    
    # 第三次运行：进一步增加数据
    print("\n📊 第三批数据收集（更多数据）")
    run_and_accumulate_data(
        output_dir=output_dir,
        num_runs=7,  # 更多运行次数
        max_zealots=20,  
        max_morality=20,
        batch_name="extended_run"
    )
    
    # 最终从所有累积数据生成图表
    print("\n📈 从所有累积数据生成最终图表...")
    plot_from_accumulated_data(output_dir)
    
    print("\n✅ 完成！现在你有了基于 3+5+7=15 批次数据的图表")
    print(f"📁 结果保存在: {output_dir}")
    print("📊 图表文件名包含了总运行次数信息")


def example_single_run():
    """示例：传统的一次性运行（向后兼容）"""
    
    print("\n🔬 示例：传统的一次性运行")
    print("=" * 60)
    
    run_zealot_morality_analysis(
        output_dir="results/traditional_run",
        num_runs=8,
        max_zealots=30,
        max_morality=30
    )


def example_data_only():
    """示例：只收集数据，稍后绘图"""
    
    print("\n🔬 示例：只收集数据")
    print("=" * 60)
    
    # 只运行数据收集
    run_and_accumulate_data(
        output_dir="results/data_only_example",
        num_runs=6,
        max_zealots=25,
        max_morality=25,
        batch_name="data_collection_phase"
    )
    
    print("\n💡 数据已收集完毕。稍后可以运行以下命令生成图表：")
    print("plot_from_accumulated_data('results/data_only_example')")


def example_plot_only():
    """示例：只从现有数据生成图表"""
    
    print("\n🔬 示例：从现有数据生成图表")
    print("=" * 60)
    
    # 假设数据已经存在，只生成图表
    plot_from_accumulated_data("results/data_only_example")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "incremental":
            example_incremental_runs()
        elif mode == "single":
            example_single_run()
        elif mode == "data_only":
            example_data_only()
        elif mode == "plot_only":
            example_plot_only()
        else:
            print("❌ 未知模式。可用模式: incremental, single, data_only, plot_only")
    else:
        print("📖 使用说明:")
        print("python example_usage.py incremental  # 多次增量运行示例")
        print("python example_usage.py single       # 传统一次性运行示例")
        print("python example_usage.py data_only    # 只收集数据示例")
        print("python example_usage.py plot_only    # 只生成图表示例")
        print()
        print("🚀 运行默认示例（增量运行）...")
        example_incremental_runs() 