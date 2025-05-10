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
    opinion_history -- 意见历史记录
    """
    # 存储意见历史和轨迹
    opinion_history = []
    trajectory = []
    
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
    
    return opinion_history


def run_zealot_experiment(steps=500, initial_scale=0.1, num_zealots=50, seed=42):
    """
    运行zealot实验，比较无zealot、聚类zealot和随机zealot的影响
    
    参数:
    steps -- 模拟步数
    initial_scale -- 初始意见的缩放因子，模拟对新议题的中立态度
    num_zealots -- zealot的总数量
    seed -- 随机数种子
    """
    # 设置随机数种子
    np.random.seed(seed)
    
    # 创建基础模拟实例
    base_config = copy.deepcopy(lfr_config)
    base_config.cluster_identity = False
    base_config.cluster_morality = False
    base_config.cluster_opinion = False
    base_config.opinion_distribution = "uniform"
    base_config.alpha = 0.4
    base_config.beta = 0.12
    print(base_config)
    # 设置道德化率
    base_config.morality_rate = 0
    
    base_sim = Simulation(base_config)
    
    # 缩放所有代理的初始意见
    base_sim.opinions *= initial_scale
    
    # 创建三个副本用于不同的zealot分布
    sim_cluster = copy.deepcopy(base_sim)
    sim_random = copy.deepcopy(base_sim)
    sim_degree = copy.deepcopy(base_sim)
    
    # 分别初始化三种不同模式的zealot
    cluster_zealots = initialize_zealots(sim_cluster, num_zealots, mode="clustered")
    random_zealots = initialize_zealots(sim_random, num_zealots, mode="random")
    degree_zealots = initialize_zealots(sim_degree, num_zealots, mode="degree")
    
    # 创建结果目录
    results_dir = "results/zealot_experiment"
    os.makedirs(results_dir, exist_ok=True)
    
    # 运行各种模式的模拟并生成结果
    print("Running simulation without zealots...")
    run_simulation_and_generate_results(base_sim, [], "without Zealots", results_dir, steps)
    
    print("Running simulation with clustered zealots...")
    run_simulation_and_generate_results(sim_cluster, cluster_zealots, "with Clustered Zealots", results_dir, steps)
    
    print("Running simulation with random zealots...")
    run_simulation_and_generate_results(sim_random, random_zealots, "with Random Zealots", results_dir, steps)
    
    print("Running simulation with high-degree zealots...")
    run_simulation_and_generate_results(sim_degree, degree_zealots, "with High-Degree Zealots", results_dir, steps)


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
        steps=100,            # 运行100步
        initial_scale=0.1,     # 初始意见缩放到10%
        num_zealots=10,        # 50个zealot
        seed=42                # 固定随机种子以便重现结果
    ) 