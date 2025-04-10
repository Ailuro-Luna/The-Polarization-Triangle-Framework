"""
可视化Alpha-Beta验证实验结果的脚本
"""

import os
import argparse
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from polarization_triangle.visualization.alphabeta_viz import visualize_from_file


def visualize_alphabeta_results(result_file="results/verification/alphabeta/alphabeta_results.csv",
                               output_dir="results/verification/alphabeta/visualizations",
                               title_suffix="无道德化",
                               num_runs=10):
    """
    可视化Alpha-Beta验证结果
    
    参数:
    result_file -- 结果文件路径
    output_dir -- 输出目录
    title_suffix -- 标题后缀
    num_runs -- 每个参数组合的模拟次数
    """
    # 确保结果文件存在
    if not os.path.exists(result_file):
        print(f"错误: 结果文件 {result_file} 不存在!")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化结果
    output_paths = visualize_from_file(
        result_file=result_file,
        output_dir=output_dir,
        title_suffix=title_suffix,
        num_runs=num_runs
    )
    
    print(f"\n可视化完成! 共生成了 {len(output_paths)} 个图表:")
    for metric, path in output_paths.items():
        print(f"- {metric}: {path}")
    
    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha-Beta验证结果可视化工具")
    parser.add_argument("--result-file", type=str, 
                        default="results/verification/alphabeta/alphabeta_results.csv",
                        help="结果CSV文件路径")
    parser.add_argument("--output-dir", type=str, 
                        default="results/verification/alphabeta/visualizations",
                        help="输出目录")
    parser.add_argument("--title-suffix", type=str, default="无道德化",
                        help="标题后缀")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="每个参数组合的模拟次数")
    
    args = parser.parse_args()
    
    visualize_alphabeta_results(
        result_file=args.result_file,
        output_dir=args.output_dir,
        title_suffix=args.title_suffix,
        num_runs=args.num_runs
    ) 