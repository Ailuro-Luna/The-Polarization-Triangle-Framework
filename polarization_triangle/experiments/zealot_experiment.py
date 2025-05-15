import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import networkx as nx
from polarization_triangle.core.config import SimulationConfig, lfr_config
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.visualization.network_viz import draw_network
from polarization_triangle.visualization.opinion_viz import draw_opinion_distribution_heatmap
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
from polarization_triangle.analysis.trajectory import run_simulation_with_trajectory


def set_zealot_opinions(sim, zealot_ids):
    """
    将指定的zealot的意见设置为1.0
    
    参数:
    sim -- simulation实例
    zealot_ids -- zealot的ID列表
    """
    for agent_id in zealot_ids:
        sim.opinions[agent_id] = 1.0


def initialize_zealots(sim, num_zealots, mode="random"):
    """
    初始化zealot分布
    
    参数:
    sim -- 原始simulation实例
    num_zealots -- zealot的总数量
    mode -- zealot选择模式: 
            "random" - 随机选择
            "clustered" - 聚类选择(尽量在同一社区)
            "degree" - 选择度数最高的节点
    
    返回:
    list -- zealot的ID列表
    """
    if num_zealots > sim.num_agents:
        num_zealots = sim.num_agents
        print(f"Warning: num_zealots exceeds agent count, setting to {sim.num_agents}")
    
    zealot_ids = []
    
    if mode == "random":
        # 随机选择指定数量的agent作为zealot
        all_nodes = list(range(sim.num_agents))
        zealot_ids = np.random.choice(all_nodes, size=num_zealots, replace=False).tolist()
    
    elif mode == "degree":
        # 选择度数最高的节点作为zealot
        node_degrees = list(sim.graph.degree())
        sorted_nodes_by_degree = sorted(node_degrees, key=lambda x: x[1], reverse=True)
        zealot_ids = [node for node, degree in sorted_nodes_by_degree[:num_zealots]]
    
    elif mode == "clustered":
        # 获取社区信息
        communities = {}
        for node in sim.graph.nodes():
            community = sim.graph.nodes[node].get("community")
            if isinstance(community, (set, frozenset)):
                community = min(community)
            if community not in communities:
                communities[community] = []
            communities[community].append(node)
        
        # 按社区大小排序
        sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
        
        # 尽量在同一社区内选择zealot
        zealots_left = num_zealots
        for community_id, members in sorted_communities:
            if zealots_left <= 0:
                break
            
            # 决定从当前社区选择多少个zealot
            to_select = min(zealots_left, len(members))
            selected = np.random.choice(members, size=to_select, replace=False).tolist()
            zealot_ids.extend(selected)
            zealots_left -= to_select
    
    else:
        raise ValueError(f"Unknown zealot selection mode: {mode}")
    
    return zealot_ids


def generate_rule_usage_plots(sim, title_prefix, output_dir):
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
    
    # 输出规则使用统计信息
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
    
    # 将交互类型统计写入文件
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


def generate_activation_visualizations(sim, trajectory, title_prefix, output_dir):
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
    if not os.path.exists(activation_folder):
        os.makedirs(activation_folder)
    
    # 1. 自我激活和社会影响散点图
    components_path = os.path.join(activation_folder, f"{title_prefix}_activation_components.png")
    draw_activation_components(
        sim,
        f"Activation Components\n{title_prefix}",
        components_path
    )
    
    # 2. 自我激活和社会影响随时间的变化
    history_path = os.path.join(activation_folder, f"{title_prefix}_activation_history.png")
    draw_activation_history(
        sim,
        f"Activation History\n{title_prefix}",
        history_path
    )
    
    # 3. 自我激活和社会影响的热力图
    heatmap_path = os.path.join(activation_folder, f"{title_prefix}_activation_heatmap.png")
    draw_activation_heatmap(
        sim,
        f"Activation Heatmap\n{title_prefix}",
        heatmap_path
    )
    
    # 4. 选定agent的激活轨迹
    trajectory_path = os.path.join(activation_folder, f"{title_prefix}_activation_trajectory.png")
    draw_activation_trajectory(
        sim,
        trajectory,
        f"Activation Trajectories\n{title_prefix}",
        trajectory_path
    )
    
    # 5. 保存激活组件数据到CSV文件
    components = sim.get_activation_components()
    data_path = os.path.join(activation_folder, f"{title_prefix}_activation_data.csv")
    with open(data_path, "w") as f:
        f.write("agent_id,identity,morality,opinion,self_activation,social_influence,total_activation\n")
        for i in range(sim.num_agents):
            f.write(f"{i},{sim.identities[i]},{sim.morals[i]},{sim.opinions[i]:.4f}")
            f.write(f",{components['self_activation'][i]:.4f},{components['social_influence'][i]:.4f}")
            f.write(f",{components['self_activation'][i] + components['social_influence'][i]:.4f}\n")


def generate_opinion_statistics(sim, trajectory, zealot_ids, mode_name, results_dir):
    """
    计算各种意见统计数据但不绘制图表
    
    参数:
    sim -- simulation实例
    trajectory -- 意见轨迹数据
    zealot_ids -- zealot的ID列表
    mode_name -- 模式名称
    results_dir -- 结果输出目录
    
    返回:
    dict -- 包含各种统计数据的字典
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
    # 获取社区信息
    communities = {}
    for node in sim.graph.nodes():
        community = sim.graph.nodes[node].get("community")
        if isinstance(community, (set, frozenset)):
            community = min(community)
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    
    cluster_variances = []
    # 新增：跟踪每个社区的方差历史
    community_variance_history = {}
    
    for step_opinions in trajectory:
        # 计算每个社区内部的方差，然后取平均值
        step_cluster_vars = []
        
        for community_id, members in communities.items():
            # 过滤掉zealot
            community_non_zealots = [m for m in members if m not in zealot_ids]
            if community_non_zealots:  # 确保社区中有非zealot的成员
                community_opinions = step_opinions[community_non_zealots]
                community_var = np.var(community_opinions)
                step_cluster_vars.append(community_var)
                
                # 记录这个社区的方差
                if community_id not in community_variance_history:
                    community_variance_history[community_id] = []
                community_variance_history[community_id].append(community_var)
            else:
                # 如果社区内只有zealot，记录0方差
                if community_id not in community_variance_history:
                    community_variance_history[community_id] = []
                community_variance_history[community_id].append(0)
        
        # 如果有有效的社区方差，计算平均值
        if step_cluster_vars:
            cluster_variances.append(np.mean(step_cluster_vars))
        else:
            cluster_variances.append(0)
    
    # 4. 统计持有negative opinion的个体的个数和negative opinion的均值
    # 5. 统计持有positive opinion的个体的个数和positive opinion的均值
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
        "communities": communities
    }
    
    # 单独保存每个模式的统计数据到CSV文件
    stats_dir = os.path.join(results_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
        
    file_prefix = mode_name.lower().replace(' ', '_')
    stats_csv = os.path.join(stats_dir, f"{file_prefix}_opinion_stats.csv")
    with open(stats_csv, "w") as f:
        f.write("step,mean_opinion,mean_abs_opinion,non_zealot_variance,cluster_variance,")
        f.write("negative_count,negative_mean,positive_count,positive_mean\n")
        
        for step in range(num_steps):
            f.write(f"{step},{mean_opinions[step]:.4f},{mean_abs_opinions[step]:.4f},")
            f.write(f"{non_zealot_var[step]:.4f},{cluster_variances[step]:.4f},")
            f.write(f"{negative_counts[step]},{negative_means[step]:.4f},")
            f.write(f"{positive_counts[step]},{positive_means[step]:.4f}\n")
    
    # 保存每个社区的方差数据到单独的CSV文件
    community_csv = os.path.join(stats_dir, f"{file_prefix}_community_variances.csv")
    with open(community_csv, "w") as f:
        # 写入标题行
        f.write("step")
        for community_id in sorted(community_variance_history.keys()):
            f.write(f",community_{community_id}")
        f.write("\n")
        
        # 写入数据
        for step in range(num_steps):
            f.write(f"{step}")
            for community_id in sorted(community_variance_history.keys()):
                if step < len(community_variance_history[community_id]):
                    f.write(f",{community_variance_history[community_id][step]:.4f}")
                else:
                    f.write(",0.0000")  # 防止索引越界
            f.write("\n")
    
    return stats


def plot_community_variances(stats, mode_name, results_dir):
    """
    绘制每个社区的意见方差变化图
    
    参数:
    stats -- 统计数据字典
    mode_name -- 模式名称
    results_dir -- 结果输出目录
    """
    # 确保统计目录存在
    stats_dir = os.path.join(results_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    file_prefix = mode_name.lower().replace(' ', '_')
    community_variance_history = stats["community_variance_history"]
    communities = stats["communities"]
    
    # 如果社区太多，可能图会很乱，所以限制只显示大的社区
    # 计算每个社区的大小
    community_sizes = {comm_id: len(members) for comm_id, members in communities.items()}
    
    # 按大小排序社区
    sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # 选取前10个最大的社区（或者全部，如果不足10个）
    top_communities = [comm_id for comm_id, size in sorted_communities[:min(10, len(sorted_communities))]]
    
    # 绘制社区方差图
    plt.figure(figsize=(12, 8))
    
    # 使用不同颜色和线型
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_communities)))
    linestyles = ['-', '--', '-.', ':'] * 3  # 重复几次以确保有足够的线型
    
    for i, community_id in enumerate(top_communities):
        variance_history = community_variance_history[community_id]
        community_size = community_sizes[community_id]
        plt.plot(range(len(variance_history)), variance_history, 
                label=f'Community {community_id} (size: {community_size})', 
                color=colors[i], linestyle=linestyles[i % len(linestyles)])
    
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.title(f'Opinion Variance within Each Community (Excluding Zealots)\n{mode_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, f"{file_prefix}_community_variances.png"), dpi=300)
    plt.close()


def plot_comparative_statistics(all_stats, mode_names, results_dir):
    """
    绘制比较性统计图表，将不同模式的统计数据在同一张图上显示
    
    参数:
    all_stats -- 包含不同模式统计数据的字典
    mode_names -- 不同模式的名称列表
    results_dir -- 结果输出目录
    """
    # 确保统计目录存在
    stats_dir = os.path.join(results_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    num_steps = len(all_stats[mode_names[0]]["mean_opinions"])
    steps = range(num_steps)
    
    # 使用不同颜色和线型
    colors = ['blue', 'red', 'green', 'purple']
    linestyles = ['-', '--', '-.', ':']
    
    # 1. 绘制平均意见值对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["mean_opinions"], 
                label=f'{mode} - Mean Opinion', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Opinion')
    plt.title('Comparison of Mean Opinions across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_mean_opinions.png"), dpi=300)
    plt.close()
    
    # 2. 绘制平均绝对意见值对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["mean_abs_opinions"], 
                label=f'{mode} - Mean |Opinion|', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean |Opinion|')
    plt.title('Comparison of Mean Absolute Opinions across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_mean_abs_opinions.png"), dpi=300)
    plt.close()
    
    # 3. 绘制非zealot方差对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["non_zealot_variance"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.title('Comparison of Opinion Variance (Excluding Zealots) across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_non_zealot_variance.png"), dpi=300)
    plt.close()
    
    # 4. 绘制社区内部方差对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["cluster_variance"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Intra-Cluster Variance')
    plt.title('Comparison of Mean Opinion Variance within Clusters across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_cluster_variance.png"), dpi=300)
    plt.close()
    
    # 5. 绘制负面意见数量对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["negative_counts"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.title('Comparison of Negative Opinion Counts across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_negative_counts.png"), dpi=300)
    plt.close()
    
    # 6. 绘制负面意见均值对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["negative_means"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Value')
    plt.title('Comparison of Negative Opinion Means across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_negative_means.png"), dpi=300)
    plt.close()
    
    # 7. 绘制正面意见数量对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["positive_counts"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.title('Comparison of Positive Opinion Counts across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_positive_counts.png"), dpi=300)
    plt.close()
    
    # 8. 绘制正面意见均值对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(steps, all_stats[mode]["positive_means"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Value')
    plt.title('Comparison of Positive Opinion Means across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "comparison_positive_means.png"), dpi=300)
    plt.close()
    
    # 9. 保存组合数据到CSV文件
    stats_csv = os.path.join(stats_dir, "comparison_opinion_stats.csv")
    with open(stats_csv, "w") as f:
        # 写入标题行
        f.write("step")
        for mode in mode_names:
            f.write(f",{mode}_mean_opinion,{mode}_mean_abs_opinion,{mode}_non_zealot_variance,{mode}_cluster_variance")
            f.write(f",{mode}_negative_count,{mode}_negative_mean,{mode}_positive_count,{mode}_positive_mean")
        f.write("\n")
        
        # 写入数据
        for step in range(num_steps):
            f.write(f"{step}")
            for mode in mode_names:
                f.write(f",{all_stats[mode]['mean_opinions'][step]:.4f}")
                f.write(f",{all_stats[mode]['mean_abs_opinions'][step]:.4f}")
                f.write(f",{all_stats[mode]['non_zealot_variance'][step]:.4f}")
                f.write(f",{all_stats[mode]['cluster_variance'][step]:.4f}")
                f.write(f",{all_stats[mode]['negative_counts'][step]}")
                f.write(f",{all_stats[mode]['negative_means'][step]:.4f}")
                f.write(f",{all_stats[mode]['positive_counts'][step]}")
                f.write(f",{all_stats[mode]['positive_means'][step]:.4f}")
            f.write("\n")


def run_simulation_and_generate_results(sim, zealot_ids, mode_name, results_dir, steps):
    """
    运行单个模拟并生成所有可视化结果
    
    参数:
    sim -- simulation实例
    zealot_ids -- zealot的ID列表
    mode_name -- 模式名称
    results_dir -- 结果输出目录
    steps -- 模拟步数
    
    返回:
    dict -- 包含意见历史记录和统计数据的字典
    """
    # 存储意见历史和轨迹
    opinion_history = []
    trajectory = []

    opinion_history.append(sim.opinions.copy())
    trajectory.append(sim.opinions.copy())
    
    # 运行模拟
    for _ in range(steps):
        # 更新zealot的意见
        if zealot_ids:
            set_zealot_opinions(sim, zealot_ids)
        
        # 执行模拟步骤
        sim.step()
        
        # 记录意见历史和轨迹
        opinion_history.append(sim.opinions.copy())
        trajectory.append(sim.opinions.copy())
    
    # 生成热图
    draw_opinion_distribution_heatmap(
        opinion_history, 
        f"Opinion Evolution {mode_name}", 
        f"{results_dir}/{mode_name.lower().replace(' ', '_')}_heatmap.png"
    )
    
    # 绘制网络图 - 意见分布
    draw_network(
        sim, 
        "opinion", 
        f"Opinion Network {mode_name}", 
        f"{results_dir}/{mode_name.lower().replace(' ', '_')}_opinion_network.png"
    )
    
    # 绘制zealot网络图
    draw_zealot_network(
        sim, 
        zealot_ids, 
        f"Network {mode_name}", 
        f"{results_dir}/{mode_name.lower().replace(' ', '_')}_network.png"
    )
    
    # 生成规则使用统计图
    generate_rule_usage_plots(sim, mode_name, results_dir)
    
    # 生成激活组件可视化
    generate_activation_visualizations(sim, trajectory, mode_name, results_dir)
    
    # 计算意见统计数据
    stats = generate_opinion_statistics(sim, trajectory, zealot_ids, mode_name, results_dir)
    
    # 绘制社区方差图
    plot_community_variances(stats, mode_name, results_dir)
    
    return {
        "opinion_history": opinion_history,
        "stats": stats
    }


def run_zealot_experiment(
    steps=500, 
    initial_scale=0.1, 
    num_zealots=50, 
    seed=42, 
    output_dir=None, 
    morality_rate=0.0, 
    zealot_morality=False, 
    identity_clustered=False,
    zealot_mode=None
):
    """
    运行zealot实验，比较无zealot、聚类zealot和随机zealot的影响
    
    参数:
    steps -- 模拟步数
    initial_scale -- 初始意见的缩放因子，模拟对新议题的中立态度
    num_zealots -- zealot的总数量
    seed -- 随机数种子
    output_dir -- 结果输出目录（如果为None，则使用默认目录）
    morality_rate -- moralizing的non-zealot people的比例
    zealot_morality -- zealot是否全部moralizing
    identity_clustered -- 是否按identity进行clustered的初始化
    zealot_mode -- zealot的初始化配置 ("none", "clustered", "random", "high-degree")，若为None则运行所有模式
    
    返回:
    dict -- 包含所有模式结果的字典
    """
    # 设置随机数种子
    np.random.seed(seed)
    
    # 创建基础模拟实例
    base_config = copy.deepcopy(lfr_config)
    base_config.cluster_identity = identity_clustered
    base_config.cluster_morality = False  # 暂时不集群morality
    base_config.cluster_opinion = False
    base_config.opinion_distribution = "uniform"
    base_config.alpha = 0.4
    base_config.beta = 0.12
    print(base_config)
    # 设置道德化率
    base_config.morality_rate = morality_rate
    
    base_sim = Simulation(base_config)
    
    # 缩放所有代理的初始意见
    base_sim.opinions *= initial_scale
    
    # 根据zealot_mode决定运行哪些模式
    run_all_modes = zealot_mode is None
    modes_to_run = []
    
    if run_all_modes or zealot_mode == "none":
        modes_to_run.append("none")
    if run_all_modes or zealot_mode == "clustered":
        modes_to_run.append("clustered")
    if run_all_modes or zealot_mode == "random":
        modes_to_run.append("random")
    if run_all_modes or zealot_mode == "high-degree":
        modes_to_run.append("high-degree")
    
    # 创建副本用于不同的zealot分布
    sims = {}
    zealots = {}
    
    # 对于无zealot模式，使用base_sim
    if "none" in modes_to_run:
        sims["none"] = base_sim
        zealots["none"] = []
    
    # 对于其他模式，创建副本
    if "clustered" in modes_to_run:
        sims["clustered"] = copy.deepcopy(base_sim)
        zealots["clustered"] = initialize_zealots(sims["clustered"], num_zealots, mode="clustered")
    
    if "random" in modes_to_run:
        sims["random"] = copy.deepcopy(base_sim)
        zealots["random"] = initialize_zealots(sims["random"], num_zealots, mode="random")
    
    if "high-degree" in modes_to_run:
        sims["high-degree"] = copy.deepcopy(base_sim)
        zealots["high-degree"] = initialize_zealots(sims["high-degree"], num_zealots, mode="degree")
    
    # 如果zealot_morality为True，将所有zealot的morality设为1
    if zealot_morality:
        for mode, zealot_list in zealots.items():
            if mode == "none":
                continue  # 无zealot模式跳过
            for agent_id in zealot_list:
                sims[mode].morals[agent_id] = 1
    
    # 创建结果目录
    if output_dir is None:
        results_dir = "results/zealot_experiment"
    else:
        results_dir = output_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # 运行各种模式的模拟并生成结果
    results = {}
    
    if "none" in modes_to_run:
        print("Running simulation without zealots...")
        results["without Zealots"] = run_simulation_and_generate_results(
            sims["none"], [], "without Zealots", results_dir, steps
        )
    
    if "clustered" in modes_to_run:
        print("Running simulation with clustered zealots...")
        results["with Clustered Zealots"] = run_simulation_and_generate_results(
            sims["clustered"], zealots["clustered"], "with Clustered Zealots", results_dir, steps
        )
    
    if "random" in modes_to_run:
        print("Running simulation with random zealots...")
        results["with Random Zealots"] = run_simulation_and_generate_results(
            sims["random"], zealots["random"], "with Random Zealots", results_dir, steps
        )
    
    if "high-degree" in modes_to_run:
        print("Running simulation with high-degree zealots...")
        results["with High-Degree Zealots"] = run_simulation_and_generate_results(
            sims["high-degree"], zealots["high-degree"], "with High-Degree Zealots", results_dir, steps
        )
    
    # 收集所有模式的统计数据
    all_stats = {}
    for mode_name, mode_results in results.items():
        all_stats[mode_name] = mode_results["stats"]
    
    # 绘制比较统计图
    mode_names = list(results.keys())
    if len(mode_names) > 1:  # 只有多于一种模式时才绘制比较图
        print("Generating comparative statistics plots...")
        plot_comparative_statistics(all_stats, mode_names, results_dir)
    
    print("All simulations and visualizations completed.")
    
    # 返回所有结果
    return results


def draw_zealot_network(sim, zealot_ids, title, filename):
    """
    绘制网络图，标记zealot节点
    
    参数:
    sim -- simulation实例
    zealot_ids -- zealot的ID列表
    title -- 图表标题
    filename -- 输出文件名
    """
    plt.figure(figsize=(12, 10))
    plt.title(title, fontsize=16)
    
    node_colors = []
    for i in range(sim.num_agents):
        if i in zealot_ids:
            node_colors.append('red')  # zealot颜色为红色
        else:
            opinion = sim.opinions[i]
            # 根据意见值着色其他节点
            if opinion > 0.5:
                node_colors.append('blue')
            elif opinion < -0.5:
                node_colors.append('green')
            else:
                node_colors.append('gray')
    
    # 绘制网络
    nx.draw(
        sim.graph,
        pos=sim.pos,
        node_color=node_colors,
        node_size=50,
        edge_color='lightgray',
        with_labels=False,
        alpha=0.8
    )
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Zealot', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Support (>0.5)', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Oppose (<-0.5)', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Neutral', markerfacecolor='gray', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    # 运行zealot实验
    run_zealot_experiment(
        steps=500,            # 运行500步
        initial_scale=0.1,     # 初始意见缩放到10%
        num_zealots=10,        # 50个zealot
        seed=114514,            # 固定随机种子以便重现结果
        morality_rate=0.0,     # 道德化率
        zealot_morality=False,  # 不全部moralizing
        identity_clustered=False, # 不按identity进行clustered的初始化
        zealot_mode=None        # 运行所有模式
    ) 