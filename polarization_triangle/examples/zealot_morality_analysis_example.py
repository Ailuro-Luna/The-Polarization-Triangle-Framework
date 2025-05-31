"""
Zealot and Morality Analysis Experiment Example

This example demonstrates how to use the zealot_morality_analysis experiment
to generate plots analyzing the effects of zealot numbers and morality ratios.
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from polarization_triangle.experiments.zealot_morality_analysis import run_zealot_morality_analysis


def run_full_analysis():
    """运行完整的zealot和morality分析"""
    print("🔬 Running Full Zealot and Morality Analysis")
    print("This will generate 8 plots (2 types × 4 metrics)")
    print("=" * 60)
    
    run_zealot_morality_analysis(
        output_dir="results/zealot_morality_full_analysis",
        num_runs=5,      # 每个参数点运行5次
        max_zealots=50,  # zealot数量从0到50
        max_morality=30  # morality ratio从0%到30%
    )


def run_quick_analysis():
    """运行快速版本的分析（较少的参数点和运行次数）"""
    print("⚡ Running Quick Zealot and Morality Analysis")
    print("This is a faster version for testing purposes")
    print("=" * 60)
    
    run_zealot_morality_analysis(
        output_dir="results/zealot_morality_quick_analysis",
        num_runs=3,      # 每个参数点运行3次
        max_zealots=20,  # zealot数量从0到20
        max_morality=20  # morality ratio从0%到20%
    )


def run_minimal_analysis():
    """运行最小版本的分析（用于快速测试）"""
    print("🚀 Running Minimal Zealot and Morality Analysis")
    print("This is for quick testing and debugging")
    print("=" * 60)
    
    run_zealot_morality_analysis(
        output_dir="results/zealot_morality_minimal_analysis",
        num_runs=2,      # 每个参数点运行2次
        max_zealots=10,  # zealot数量从0到10
        max_morality=10  # morality ratio从0%到10%
    )


def main():
    """主函数 - 选择运行哪种分析"""
    print("🔬 Zealot and Morality Analysis Example")
    print("=" * 70)
    print()
    print("Choose analysis type:")
    print("1. Full Analysis (comprehensive but time-consuming)")
    print("2. Quick Analysis (balanced)")
    print("3. Minimal Analysis (fast, for testing)")
    print("4. Run all analyses")
    print()
    
    # 自动选择快速分析进行演示
    choice = "2"
    print(f"Auto-selecting option {choice} for demonstration...")
    
    try:
        if choice == "1":
            run_full_analysis()
        elif choice == "2":
            run_quick_analysis()
        elif choice == "3":
            run_minimal_analysis()
        elif choice == "4":
            print("Running all analyses...")
            run_minimal_analysis()
            print("\n" + "="*70 + "\n")
            run_quick_analysis()
        else:
            print("Invalid choice. Running quick analysis by default.")
            run_quick_analysis()
            
        print("\n🎉 Analysis completed successfully!")
        print("\nGenerated plots:")
        print("📊 Type 1 - Zealot Numbers Analysis:")
        print("   - zealot_numbers_mean_opinion.png")
        print("   - zealot_numbers_variance.png") 
        print("   - zealot_numbers_variance_per_identity.png")
        print("   - zealot_numbers_polarization_index.png")
        print()
        print("📊 Type 2 - Morality Ratio Analysis:")
        print("   - morality_ratios_mean_opinion.png")
        print("   - morality_ratios_variance.png")
        print("   - morality_ratios_variance_per_identity.png")
        print("   - morality_ratios_polarization_index.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 