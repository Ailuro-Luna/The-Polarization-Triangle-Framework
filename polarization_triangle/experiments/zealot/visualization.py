"""
Zealot实验可视化模块

提供各种可视化功能
"""

import os
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from polarization_triangle.common.plotting import BasePlotter
from polarization_triangle.visualization.rule_viz import (
    draw_interaction_type_usage, 
    draw_interaction_type_cumulative_usage
)
from polarization_triangle.visualization.activation_viz import (
    draw_activation_components,
    draw_activation_history,
    draw_activation_heatmap,
    draw_activation_trajectory
)


def generate_rule_usage_plots(sim, title_prefix: str, output_dir: str):
    """
    生成规则使用统计图
    
    参数:
    sim -- simulation实例
    title_prefix -- 图表标题前缀
    output_dir -- 输出目录
    """
    # 绘制规则使用统计图
    rule_usage_path = os.path.join(output_dir, f"{title_prefix}_interaction_types.png")
    draw_interaction_type_usage(
        sim.rule_counts_history,
        f"Interaction Types over Time\n{title_prefix}",
        rule_usage_path
    )
    
    # 绘制规则累积使用统计图
    rule_cumulative_path = os.path.join(output_dir, f"{title_prefix}_interaction_types_cumulative.png")
    draw_interaction_type_cumulative_usage(
        sim.rule_counts_history,
        f"Cumulative Interaction Types\n{title_prefix}",
        rule_cumulative_path
    )
    
    # 保存规则使用统计信息
    _save_interaction_stats(sim, title_prefix, output_dir)


def generate_activation_visualizations(sim, trajectory: np.ndarray, 
                                     title_prefix: str, output_dir: str):
    """
    生成激活组件相关的可视化
    
    参数:
    sim -- simulation实例
    trajectory -- 意见轨迹数据
    title_prefix -- 图表标题前缀
    output_dir -- 输出目录
    """
    # 创建激活组件子文件夹
    activation_folder = os.path.join(output_dir, "activation_components")
    os.makedirs(activation_folder, exist_ok=True)
    
    # 1. 自我激活和社会影响散点图
    components_path = os.path.join(activation_folder, f"{title_prefix}_activation_components.png")
    draw_activation_components(sim, f"Activation Components\n{title_prefix}", components_path)
    
    # 2. 自我激活和社会影响随时间的变化
    history_path = os.path.join(activation_folder, f"{title_prefix}_activation_history.png")
    draw_activation_history(sim, f"Activation History\n{title_prefix}", history_path)
    
    # 3. 自我激活和社会影响的热力图
    heatmap_path = os.path.join(activation_folder, f"{title_prefix}_activation_heatmap.png")
    draw_activation_heatmap(sim, f"Activation Heatmap\n{title_prefix}", heatmap_path)
    
    # 4. 选定agent的激活轨迹
    trajectory_path = os.path.join(activation_folder, f"{title_prefix}_activation_trajectory.png")
    draw_activation_trajectory(sim, trajectory, f"Activation Trajectories\n{title_prefix}", trajectory_path)
    
    # 5. 保存激活组件数据
    _save_activation_data(sim, title_prefix, activation_folder)


def plot_comparative_statistics(all_stats: Dict[str, Dict], mode_names: List[str], 
                              results_dir: str):
    """
    绘制比较性统计图表
    
    参数:
    all_stats -- 包含不同模式统计数据的字典
    mode_names -- 不同模式的名称列表
    results_dir -- 结果输出目录
    """
    stats_dir = os.path.join(results_dir, "statistics")
    os.makedirs(stats_dir, exist_ok=True)
    
    plotter = BasePlotter(figsize=(12, 7))
    
    # 准备数据
    num_steps = len(all_stats[mode_names[0]]["mean_opinions"])
    steps = range(num_steps)
    
    # 颜色和线型
    colors = ['blue', 'red', 'green', 'purple']
    
    # 1. 平均意见值对比
    fig1, ax1 = plotter.create_figure()
    for i, mode in enumerate(mode_names):
        ax1.plot(steps, all_stats[mode]["mean_opinions"], 
                label=f'{mode} - Mean Opinion', 
                color=colors[i % len(colors)])
    plotter.setup_axes(ax1, 
                      title='Comparison of Mean Opinions across Different Simulations',
                      xlabel='Step', ylabel='Mean Opinion')
    plotter.add_legend(ax1)
    plotter.save_figure(fig1, os.path.join(stats_dir, "comparison_mean_opinions.png"))
    
    # 2. 平均绝对意见值对比
    fig2, ax2 = plotter.create_figure()
    for i, mode in enumerate(mode_names):
        ax2.plot(steps, all_stats[mode]["mean_abs_opinions"], 
                label=f'{mode} - Mean |Opinion|', 
                color=colors[i % len(colors)])
    plotter.setup_axes(ax2,
                      title='Comparison of Mean Absolute Opinions across Different Simulations',
                      xlabel='Step', ylabel='Mean |Opinion|')
    plotter.add_legend(ax2)
    plotter.save_figure(fig2, os.path.join(stats_dir, "comparison_mean_abs_opinions.png"))
    
    # 3. 非zealot方差对比
    fig3, ax3 = plotter.create_figure()
    for i, mode in enumerate(mode_names):
        ax3.plot(steps, all_stats[mode]["non_zealot_variance"], 
                label=f'{mode}', 
                color=colors[i % len(colors)])
    plotter.setup_axes(ax3,
                      title='Comparison of Opinion Variance (Excluding Zealots)',
                      xlabel='Step', ylabel='Variance')
    plotter.add_legend(ax3)
    plotter.save_figure(fig3, os.path.join(stats_dir, "comparison_non_zealot_variance.png"))
    
    # 4. 社区内部方差对比
    fig4, ax4 = plotter.create_figure()
    for i, mode in enumerate(mode_names):
        ax4.plot(steps, all_stats[mode]["cluster_variance"], 
                label=f'{mode}', 
                color=colors[i % len(colors)])
    plotter.setup_axes(ax4,
                      title='Comparison of Mean Opinion Variance within Clusters',
                      xlabel='Step', ylabel='Mean Intra-Cluster Variance')
    plotter.add_legend(ax4)
    plotter.save_figure(fig4, os.path.join(stats_dir, "comparison_cluster_variance.png"))
    
    # 5. 身份差异对比
    if "identity_opinion_differences" in all_stats[mode_names[0]]:
        fig5, ax5 = plotter.create_figure()
        for i, mode in enumerate(mode_names):
            ax5.plot(steps, all_stats[mode]["identity_opinion_differences"], 
                    label=f'{mode}', 
                    color=colors[i % len(colors)])
        plotter.setup_axes(ax5,
                          title='Identity-based Opinion Differences',
                          xlabel='Step', ylabel='Opinion Difference')
        plotter.add_legend(ax5)
        plotter.save_figure(fig5, os.path.join(stats_dir, "comparison_identity_differences.png"))
    
    # 6. 极化指数对比（如果有）
    if "polarization_index" in all_stats[mode_names[0]] and all_stats[mode_names[0]]["polarization_index"]:
        fig6, ax6 = plotter.create_figure()
        for i, mode in enumerate(mode_names):
            if "polarization_index" in all_stats[mode]:
                ax6.plot(range(len(all_stats[mode]["polarization_index"])), 
                        all_stats[mode]["polarization_index"], 
                        label=f'{mode}', 
                        color=colors[i % len(colors)])
        plotter.setup_axes(ax6,
                          title='Polarization Index Comparison',
                          xlabel='Step', ylabel='Polarization Index')
        plotter.add_legend(ax6)
        plotter.save_figure(fig6, os.path.join(stats_dir, "comparison_polarization_index.png"))


def _save_interaction_stats(sim, title_prefix: str, output_dir: str):
    """保存交互统计信息"""
    interaction_names = [
        "Rule 1: Same dir, Same ID, {0,0}, High Convergence",
        "Rule 2: Same dir, Same ID, {0,1}, Medium Pull",
        "Rule 3: Same dir, Same ID, {1,0}, Medium Pull",
        "Rule 4: Same dir, Same ID, {1,1}, High Polarization",
        "Rule 5: Same dir, Diff ID, {0,0}, Medium Convergence",
        "Rule 6: Same dir, Diff ID, {0,1}, Low Pull",
        "Rule 7: Same dir, Diff ID, {1,0}, Low Pull",
        "Rule 8: Same dir, Diff ID, {1,1}, Medium Polarization",
        "Rule 9: Diff dir, Same ID, {0,0}, Very High Convergence",
        "Rule 10: Diff dir, Same ID, {0,1}, Medium Convergence/Pull",
        "Rule 11: Diff dir, Same ID, {1,0}, Low Resistance",
        "Rule 12: Diff dir, Same ID, {1,1}, Low Polarization",
        "Rule 13: Diff dir, Diff ID, {0,0}, Low Convergence",
        "Rule 14: Diff dir, Diff ID, {0,1}, High Pull",
        "Rule 15: Diff dir, Diff ID, {1,0}, High Resistance",
        "Rule 16: Diff dir, Diff ID, {1,1}, Very High Polarization"
    ]
    
    # 获取交互类型统计
    interaction_stats = sim.get_interaction_counts()
    counts = interaction_stats["counts"]
    total_count = interaction_stats["total_interactions"]
    
    # 保存到文件
    stats_path = os.path.join(output_dir, f"{title_prefix}_interaction_types_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"交互类型统计 - {title_prefix}\n")
        f.write("-" * 50 + "\n")
        for i, interaction_name in enumerate(interaction_names):
            count = counts[i]
            percent = (count / total_count) * 100 if total_count > 0 else 0
            f.write(f"{interaction_name}: {count} 次 ({percent:.1f}%)\n")
        f.write("-" * 50 + "\n")
        f.write(f"总计: {total_count} 次\n")


def _save_activation_data(sim, title_prefix: str, output_dir: str):
    """保存激活组件数据"""
    components = sim.get_activation_components()
    data_path = os.path.join(output_dir, f"{title_prefix}_activation_data.csv")
    
    with open(data_path, "w") as f:
        f.write("agent_id,identity,morality,opinion,self_activation,social_influence,total_activation\n")
        for i in range(sim.num_agents):
            f.write(f"{i},{sim.identities[i]},{sim.morals[i]},{sim.opinions[i]:.4f}")
            f.write(f",{components['self_activation'][i]:.4f},{components['social_influence'][i]:.4f}")
            f.write(f",{components['self_activation'][i] + components['social_influence'][i]:.4f}\n") 