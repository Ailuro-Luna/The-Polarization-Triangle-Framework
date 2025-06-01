#!/usr/bin/env python3
"""
测试新功能的简短演示脚本

这个脚本演示：
1. 总耗时计算（数据收集 + 绘图 + 总耗时）
2. 图表标题和文件名中显示总run数而不是batch数
3. 累积多次运行的数据
"""

import time
from polarization_triangle.experiments.zealot_morality_analysis import (
    run_and_accumulate_data,
    plot_from_accumulated_data
)

def test_new_features():
    """测试新功能的简短演示"""
    
    print("🧪 测试新功能演示")
    print("=" * 60)
    
    total_start_time = time.time()
    
    output_dir = "results/test_new_features"
    
    # 第一次数据收集
    print("\n📊 第一次数据收集（3次运行）")
    data_start_1 = time.time()
    
    run_and_accumulate_data(
        output_dir=output_dir,
        num_runs=3,  # 少量运行用于快速测试
        max_zealots=10,  # 小范围测试
        max_morality=10,
        batch_name="test_batch_1"
    )
    
    data_end_1 = time.time()
    print(f"⏱️  第一次数据收集耗时: {data_end_1 - data_start_1:.2f}s")
    
    # 第二次数据收集
    print("\n📊 第二次数据收集（5次运行）")
    data_start_2 = time.time()
    
    run_and_accumulate_data(
        output_dir=output_dir,
        num_runs=5,  # 更多运行
        max_zealots=10,
        max_morality=10,
        batch_name="test_batch_2"
    )
    
    data_end_2 = time.time()
    print(f"⏱️  第二次数据收集耗时: {data_end_2 - data_start_2:.2f}s")
    
    # 生成图表
    print("\n📈 生成累积数据图表")
    plot_start = time.time()
    
    plot_from_accumulated_data(output_dir)
    
    plot_end = time.time()
    print(f"⏱️  绘图耗时: {plot_end - plot_start:.2f}s")
    
    # 总耗时
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "🎯" * 60)
    print("⏱️  新功能测试耗时总结")
    print("🎯" * 60)
    print(f"📊 第一次数据收集: {data_end_1 - data_start_1:.2f}s")
    print(f"📊 第二次数据收集: {data_end_2 - data_start_2:.2f}s")
    print(f"📈 图表生成阶段: {plot_end - plot_start:.2f}s")
    print(f"🎯 总耗时: {total_duration:.2f}s")
    print("🎯" * 60)
    
    print(f"\n✅ 测试完成！")
    print(f"📁 结果保存在: {output_dir}")
    print(f"📊 现在图表标题显示 '(3+5=8 total runs)' 而不是 batch 数")
    print(f"📈 文件名包含 '_8runs.png' 表示总运行次数")
    
    return total_duration

if __name__ == "__main__":
    test_duration = test_new_features()
    print(f"\n🚀 测试脚本总耗时: {test_duration:.2f}s") 