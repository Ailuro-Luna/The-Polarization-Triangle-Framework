"""
Alpha-Beta验证实验的可视化模块

提供用于可视化Alpha-Beta参数扫描结果的函数
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple


def plot_alphabeta_comparison(results_df: pd.DataFrame, 
                             output_dir: str = 'results/verification/alphabeta',
                             title_suffix: str = "",
                             num_runs: Optional[int] = None) -> Dict[str, str]:
    """
    绘制alpha和beta参数对极化指标的影响比较图
    
    参数:
    results_df -- 包含实验结果的DataFrame
    output_dir -- 输出目录
    title_suffix -- 标题后缀，用于区分不同实验
    num_runs -- 每个参数组合的模拟次数（用于标题显示）
    
    返回:
    包含所有保存图片路径的字典
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有唯一的alpha值
    alpha_values = sorted(results_df['alpha'].unique())
    
    # 要绘制的指标
    metrics = ['variance', 'bimodality', 'extreme_ratio', 'mean_extremity']
    
    # 存储输出路径
    output_paths = {}
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        for alpha in alpha_values:
            # 过滤出当前alpha的数据
            alpha_df = results_df[results_df['alpha'] == alpha]
            
            # 对beta值排序
            alpha_df = alpha_df.sort_values('beta')
            
            # 准备绘图数据
            beta_values = alpha_df['beta'].values
            metric_values = alpha_df[metric].values
            
            # 如果存在标准差列，则绘制误差区域
            std_col = f"{metric}_std"
            if std_col in alpha_df.columns:
                std_values = alpha_df[std_col].values
                plt.fill_between(
                    beta_values,
                    np.maximum(0, metric_values - std_values),  # 防止负值
                    np.minimum(1, metric_values + std_values),  # 防止超过1
                    alpha=0.2
                )
            
            # 绘制线图
            plt.plot(beta_values, metric_values, marker='o', label=f"α={alpha}")
        
        # 设置图表属性
        plt.xlabel("Beta (社会影响系数)", fontsize=14)
        plt.ylabel(f"{metric.replace('_', ' ').title()}", fontsize=14)
        
        # 设置标题
        title = f"Alpha和Beta对{metric.replace('_', ' ').title()}的影响"
        if num_runs:
            title += f" (每组参数{num_runs}次运行平均)"
        if title_suffix:
            title += f" - {title_suffix}"
        plt.title(title, fontsize=16)
        
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, f"comparison_{metric}{title_suffix.replace(' ', '_')}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        output_paths[metric] = output_path
        print(f"已保存图表: {output_path}")
    
    return output_paths


def plot_alpha_beta_heatmaps(results_df: pd.DataFrame, 
                            output_dir: str = 'results/verification/alphabeta',
                            title_suffix: str = "") -> Dict[str, str]:
    """
    绘制alpha-beta参数对极化指标影响的热力图
    
    参数:
    results_df -- 包含实验结果的DataFrame
    output_dir -- 输出目录
    title_suffix -- 标题后缀，用于区分不同实验
    
    返回:
    包含所有保存图片路径的字典
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 要绘制的指标
    metrics = ['variance', 'bimodality', 'extreme_ratio', 'mean_extremity']
    
    # 获取所有唯一的alpha和beta值
    alpha_values = sorted(results_df['alpha'].unique())
    beta_values = sorted(results_df['beta'].unique())
    
    # 创建网格
    alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values)
    
    # 存储输出路径
    output_paths = {}
    
    for metric in metrics:
        # 创建热力图数据
        z_data = np.zeros((len(beta_values), len(alpha_values)))
        
        for i, beta in enumerate(beta_values):
            for j, alpha in enumerate(alpha_values):
                # 查找匹配的行
                row = results_df[(results_df['alpha'] == alpha) & (results_df['beta'] == beta)]
                if not row.empty:
                    z_data[i, j] = row[metric].values[0]
        
        plt.figure(figsize=(12, 10))
        
        # 绘制热力图
        cmap = 'viridis'
        if metric == 'bimodality':
            # 对双峰性使用不同的颜色映射，以突出高值
            cmap = 'plasma'
        
        plt.pcolormesh(alpha_grid, beta_grid, z_data, cmap=cmap, shading='auto')
        plt.colorbar(label=metric.replace('_', ' ').title())
        
        # 添加等高线
        contour = plt.contour(alpha_grid, beta_grid, z_data, colors='white', alpha=0.5)
        plt.clabel(contour, inline=True, fontsize=10, fmt='%.2f')
        
        # 设置图表属性
        plt.xlabel("Alpha (自我激活系数)", fontsize=14)
        plt.ylabel("Beta (社会影响系数)", fontsize=14)
        title = f"Alpha-Beta参数对{metric.replace('_', ' ').title()}的影响"
        if title_suffix:
            title += f" - {title_suffix}"
        plt.title(title, fontsize=16)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, f"heatmap_{metric}{title_suffix.replace(' ', '_')}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        output_paths[metric] = output_path
        print(f"已保存热力图: {output_path}")
    
    return output_paths


def visualize_from_file(result_file: str, 
                       output_dir: str = 'results/verification/alphabeta',
                       title_suffix: str = "",
                       num_runs: Optional[int] = None) -> Dict[str, str]:
    """
    从CSV文件加载结果并生成可视化
    
    参数:
    result_file -- 结果CSV文件路径
    output_dir -- 输出目录
    title_suffix -- 标题后缀
    num_runs -- 每个参数组合的模拟次数
    
    返回:
    包含所有保存图片路径的字典
    """
    # 加载数据
    results_df = pd.read_csv(result_file)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制比较图
    comparison_paths = plot_alphabeta_comparison(
        results_df, 
        output_dir=output_dir,
        title_suffix=title_suffix,
        num_runs=num_runs
    )
    
    # 绘制热力图
    heatmap_paths = plot_alpha_beta_heatmaps(
        results_df, 
        output_dir=output_dir,
        title_suffix=title_suffix
    )
    
    # 合并路径字典
    output_paths = {**comparison_paths, **heatmap_paths}
    
    return output_paths


def main():
    """主函数，用于测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alpha-Beta验证实验可视化工具")
    parser.add_argument('--result-file', type=str, default='verification_results/alphabeta/alphabeta_results.csv',
                       help='结果CSV文件路径')
    parser.add_argument('--output-dir', type=str, default='results/verification/alphabeta',
                       help='输出目录')
    parser.add_argument('--title-suffix', type=str, default='无道德化',
                       help='标题后缀')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='每个参数组合的模拟次数')
    
    args = parser.parse_args()
    
    visualize_from_file(
        result_file=args.result_file,
        output_dir=args.output_dir,
        title_suffix=args.title_suffix,
        num_runs=args.num_runs
    )


if __name__ == "__main__":
    main() 