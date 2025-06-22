#!/usr/bin/env python3
"""
测试脚本：验证zealot_numbers的error bands功能
"""

import sys
import os

# 添加项目路径
sys.path.append('polarization_triangle')

from experiments.zealot_morality_analysis import plot_from_accumulated_data

def test_zealot_error_bands():
    """测试zealot_numbers的error bands功能"""
    print("🧪 Testing Zealot Numbers Error Bands")
    print("=" * 50)
    
    # 测试不同的平滑设置
    test_cases = [
        {"enable_smoothing": True, "description": "用户启用平滑（zealot_numbers仍会显示error bands）"},
        {"enable_smoothing": False, "description": "用户关闭平滑（zealot_numbers显示error bands，morality_ratios不平滑）"}
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 测试案例 {i}: {case['description']}")
        print("-" * 40)
        
        try:
            plot_from_accumulated_data(
                output_dir="results/zealot_morality_analysis",
                enable_smoothing=case["enable_smoothing"],
                target_step=2,
                smooth_method='savgol'
            )
            print(f"✅ 测试案例 {i} 完成")
        except Exception as e:
            print(f"❌ 测试案例 {i} 失败: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")
    print("📁 请检查 results/zealot_morality_analysis/mean_plots/ 目录")
    print("📋 zealot_numbers图表应该包含 'with_error_bands' 在文件名中")
    print("📋 morality_ratios图表文件名会根据平滑设置变化")

if __name__ == "__main__":
    test_zealot_error_bands() 