"""
Zealot实验统计模块

提供统计分析功能
"""

import numpy as np
import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt


def generate_opinion_statistics(sim, trajectory: np.ndarray, zealot_ids: List[int], 
                              mode_name: str, results_dir: str) -> Dict[str, Any]:
    """
    计算各种意见统计数据
    
    参数:
    sim -- simulation实例
    trajectory -- 意见轨迹数据
    zealot_ids -- zealot的ID列表
    mode_name -- 模式名称
    results_dir -- 结果输出目录
    
    返回:
    包含各种统计数据的字典
    """
    num_steps = len(trajectory)
    
    # 1. 计算平均opinion和平均abs(opinion)
    mean_opinions = []
    mean_abs_opinions = []
    
    for step_opinions in trajectory:
        mean_opinions.append(np.mean(step_opinions))
        mean_abs_opinions.append(np.mean(np.abs(step_opinions)))
    
    # 2. 计算除zealot外的所有agent的opinion的方差
    non_zealot_var = []
    
    for step_opinions in trajectory:
        # 创建除zealot外的所有agent的意见数组
        non_zealot_opinions = np.delete(step_opinions, zealot_ids) if zealot_ids else step_opinions
        non_zealot_var.append(np.var(non_zealot_opinions))
    
    # 3. 计算所有cluster内部的agent（除了zealot）的opinion的方差
    communities = _get_communities(sim)
    cluster_variances = []
    community_variance_history = {}
    
    for step_opinions in trajectory:
        # 计算每个社区内部的方差
        step_cluster_vars = []
        
        for community_id, members in communities.items():
            # 过滤掉zealot
            community_non_zealots = [m for m in members if m not in zealot_ids]
            if community_non_zealots:
                community_opinions = step_opinions[community_non_zealots]
                community_var = np.var(community_opinions)
                step_cluster_vars.append(community_var)
                
                if community_id not in community_variance_history:
                    community_variance_history[community_id] = []
                community_variance_history[community_id].append(community_var)
        
        if step_cluster_vars:
            cluster_variances.append(np.mean(step_cluster_vars))
        else:
            cluster_variances.append(0)
    
    # 4. 统计正负意见
    negative_counts, negative_means, positive_counts, positive_means = _calculate_opinion_polarity(
        trajectory, zealot_ids
    )
    
    # 5. 计算身份相关统计
    identity_stats = _calculate_identity_statistics(sim, trajectory, zealot_ids)
    
    # 6. 获取极化指数历史
    polarization_history = sim.get_polarization_history() if hasattr(sim, 'get_polarization_history') else []
    
    # 整合所有统计数据
    stats = {
        "mean_opinions": mean_opinions,
        "mean_abs_opinions": mean_abs_opinions,
        "non_zealot_variance": non_zealot_var,
        "cluster_variance": cluster_variances,
        "negative_counts": negative_counts,
        "negative_means": negative_means,
        "positive_counts": positive_counts,
        "positive_means": positive_means,
        "community_variance_history": community_variance_history,
        "communities": communities,
        "polarization_index": polarization_history,
        **identity_stats
    }
    
    # 保存统计数据
    _save_statistics_to_csv(stats, mode_name, results_dir, num_steps)
    
    return stats


def _get_communities(sim) -> Dict[int, List[int]]:
    """获取社区信息"""
    communities = {}
    for node in sim.graph.nodes():
        community = sim.graph.nodes[node].get("community")
        if isinstance(community, (set, frozenset)):
            community = min(community)
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    return communities


def _calculate_opinion_polarity(trajectory: np.ndarray, 
                               zealot_ids: List[int]) -> tuple:
    """计算意见极性统计"""
    negative_counts = []
    negative_means = []
    positive_counts = []
    positive_means = []
    
    for step_opinions in trajectory:
        # 获取非zealot的意见
        non_zealot_opinions = np.delete(step_opinions, zealot_ids) if zealot_ids else step_opinions
        
        # 统计负面意见
        negative_mask = non_zealot_opinions < 0
        negative_opinions = non_zealot_opinions[negative_mask]
        negative_count = len(negative_opinions)
        negative_counts.append(negative_count)
        negative_means.append(np.mean(negative_opinions) if negative_count > 0 else 0)
        
        # 统计正面意见
        positive_mask = non_zealot_opinions > 0
        positive_opinions = non_zealot_opinions[positive_mask]
        positive_count = len(positive_opinions)
        positive_counts.append(positive_count)
        positive_means.append(np.mean(positive_opinions) if positive_count > 0 else 0)
    
    return negative_counts, negative_means, positive_counts, positive_means


def _calculate_identity_statistics(sim, trajectory: np.ndarray, 
                                  zealot_ids: List[int]) -> Dict[str, List[float]]:
    """计算身份相关的统计数据"""
    identity_1_mean_opinions = []
    identity_neg1_mean_opinions = []
    identity_opinion_differences = []
    
    # 找到identity为1和-1的agents（排除zealots）
    identity_1_agents = []
    identity_neg1_agents = []
    
    for i in range(sim.num_agents):
        if zealot_ids and i in zealot_ids:
            continue
        if sim.identities[i] == 1:
            identity_1_agents.append(i)
        elif sim.identities[i] == -1:
            identity_neg1_agents.append(i)
    
    for step_opinions in trajectory:
        # 计算identity为1的agents的平均opinion
        if identity_1_agents:
            identity_1_opinions = step_opinions[identity_1_agents]
            identity_1_mean = np.mean(identity_1_opinions)
        else:
            identity_1_mean = 0.0
        identity_1_mean_opinions.append(identity_1_mean)
        
        # 计算identity为-1的agents的平均opinion
        if identity_neg1_agents:
            identity_neg1_opinions = step_opinions[identity_neg1_agents]
            identity_neg1_mean = np.mean(identity_neg1_opinions)
        else:
            identity_neg1_mean = 0.0
        identity_neg1_mean_opinions.append(identity_neg1_mean)
        
        # 计算差值
        difference = identity_1_mean - identity_neg1_mean
        identity_opinion_differences.append(difference)
    
    return {
        "identity_1_mean_opinions": identity_1_mean_opinions,
        "identity_neg1_mean_opinions": identity_neg1_mean_opinions,
        "identity_opinion_differences": identity_opinion_differences
    }


def _save_statistics_to_csv(stats: Dict[str, Any], mode_name: str, 
                           results_dir: str, num_steps: int):
    """保存统计数据到CSV文件"""
    stats_dir = os.path.join(results_dir, "statistics")
    os.makedirs(stats_dir, exist_ok=True)
    
    file_prefix = mode_name.lower().replace(' ', '_')
    stats_csv = os.path.join(stats_dir, f"{file_prefix}_opinion_stats.csv")
    
    with open(stats_csv, "w") as f:
        # 写入标题
        f.write("step,mean_opinion,mean_abs_opinion,non_zealot_variance,cluster_variance,")
        f.write("negative_count,negative_mean,positive_count,positive_mean")
        if stats.get("polarization_index"):
            f.write(",polarization_index")
        f.write(",identity_1_mean_opinion,identity_neg1_mean_opinion,identity_opinion_difference")
        f.write("\n")
        
        # 写入数据
        for step in range(num_steps):
            f.write(f"{step},{stats['mean_opinions'][step]:.4f},")
            f.write(f"{stats['mean_abs_opinions'][step]:.4f},")
            f.write(f"{stats['non_zealot_variance'][step]:.4f},")
            f.write(f"{stats['cluster_variance'][step]:.4f},")
            f.write(f"{stats['negative_counts'][step]},")
            f.write(f"{stats['negative_means'][step]:.4f},")
            f.write(f"{stats['positive_counts'][step]},")
            f.write(f"{stats['positive_means'][step]:.4f}")
            
            if stats.get("polarization_index") and step < len(stats["polarization_index"]):
                f.write(f",{stats['polarization_index'][step]:.4f}")
            
            f.write(f",{stats['identity_1_mean_opinions'][step]:.4f}")
            f.write(f",{stats['identity_neg1_mean_opinions'][step]:.4f}")
            f.write(f",{stats['identity_opinion_differences'][step]:.4f}")
            f.write("\n")


def plot_community_variances(stats: Dict[str, Any], mode_name: str, results_dir: str):
    """
    绘制每个社区的意见方差变化图
    
    参数:
    stats -- 统计数据字典
    mode_name -- 模式名称
    results_dir -- 结果输出目录
    """
    from polarization_triangle.common.plotting import BasePlotter
    
    stats_dir = os.path.join(results_dir, "statistics")
    os.makedirs(stats_dir, exist_ok=True)
    
    file_prefix = mode_name.lower().replace(' ', '_')
    community_variance_history = stats["community_variance_history"]
    communities = stats["communities"]
    
    # 计算每个社区的大小
    community_sizes = {comm_id: len(members) for comm_id, members in communities.items()}
    
    # 选取前10个最大的社区
    sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
    top_communities = [comm_id for comm_id, size in sorted_communities[:min(10, len(sorted_communities))]]
    
    # 绘制图表
    plotter = BasePlotter(figsize=(12, 8))
    fig, ax = plotter.create_figure()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_communities)))
    linestyles = ['-', '--', '-.', ':'] * 3
    
    for i, community_id in enumerate(top_communities):
        variance_history = community_variance_history[community_id]
        community_size = community_sizes[community_id]
        ax.plot(range(len(variance_history)), variance_history, 
                label=f'Community {community_id} (size: {community_size})', 
                color=colors[i], linestyle=linestyles[i % len(linestyles)])
    
    plotter.setup_axes(ax, 
                      title=f'Opinion Variance within Each Community (Excluding Zealots)\n{mode_name}',
                      xlabel='Step',
                      ylabel='Variance')
    plotter.add_legend(ax, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    filename = os.path.join(stats_dir, f"{file_prefix}_community_variances.png")
    plotter.save_figure(fig, filename) 