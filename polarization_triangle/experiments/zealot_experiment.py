import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import networkx as nx
from polarization_triangle.core.config import SimulationConfig, lfr_config
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.visualization.network_viz import draw_network
from polarization_triangle.visualization.opinion_viz import draw_opinion_distribution_heatmap


def set_zealot_opinions(sim, zealot_ids):
    """
    将指定的zealot的意见设置为1.0
    
    参数:
    sim -- simulation实例
    zealot_ids -- zealot的ID列表
    """
    for agent_id in zealot_ids:
        sim.opinions[agent_id] = 1.0


def initialize_zealots(sim, cluster_zealot_ratio=0.3):
    """
    初始化两种不同的zealot分布:
    1. 聚类的zealots (从一个随机社区中选择)
    2. 随机分布的zealots (随机选择相同数量的agents)
    
    参数:
    sim -- 原始simulation实例
    cluster_zealot_ratio -- 选定社区中作为zealot的节点比例
    
    返回:
    tuple -- (cluster_zealots, random_zealots)，分别是聚类和随机的zealot ID列表
    """
    # 获取社区信息
    communities = {}
    for node in sim.graph.nodes():
        community = sim.graph.nodes[node].get("community")
        if isinstance(community, (set, frozenset)):
            community = min(community)
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    
    # 随机选择一个社区
    selected_community = np.random.choice(list(communities.keys()))
    community_nodes = communities[selected_community]
    
    # 计算该社区中要选为zealot的节点数量
    num_zealots = int(len(community_nodes) * cluster_zealot_ratio)
    
    # 随机选择社区内的zealots
    cluster_zealots = np.random.choice(community_nodes, size=num_zealots, replace=False).tolist()
    
    # 为随机分布创建相同数量的zealots
    all_nodes = list(range(sim.num_agents))
    # 计算每个节点成为随机zealot的概率
    random_zealot_prob = num_zealots / sim.num_agents
    # 随机选择节点作为zealots
    random_zealots = []
    for node in all_nodes:
        if np.random.random() < random_zealot_prob:
            random_zealots.append(node)
    
    return cluster_zealots, random_zealots


def run_zealot_experiment(steps=500, initial_scale=0.1, cluster_zealot_ratio=0.3, seed=42):
    """
    运行zealot实验，比较无zealot、聚类zealot和随机zealot的影响
    
    参数:
    steps -- 模拟步数
    initial_scale -- 初始意见的缩放因子，模拟对新议题的中立态度
    cluster_zealot_ratio -- 聚类zealot的比例
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
    base_config.alpha = 0.2
    base_config.beta = 0.16
    print(base_config)
    # 设置道德化率
    base_config.morality_rate = 0.5

    base_sim = Simulation(base_config)
    
    # 缩放所有代理的初始意见
    base_sim.opinions *= initial_scale
    
    # 创建两个副本用于不同的zealot分布
    sim_cluster = copy.deepcopy(base_sim)
    sim_random = copy.deepcopy(base_sim)
    
    # 初始化zealots
    cluster_zealots, random_zealots = initialize_zealots(base_sim, cluster_zealot_ratio)
    
    # 创建结果目录
    results_dir = "results/zealot_experiment"
    os.makedirs(results_dir, exist_ok=True)
    
    # 存储意见历史
    base_opinion_history = []
    cluster_opinion_history = []
    random_opinion_history = []
    
    # 运行模拟
    for step in range(steps):
        # 更新zealot的意见
        set_zealot_opinions(sim_cluster, cluster_zealots)
        set_zealot_opinions(sim_random, random_zealots)
        
        # 执行模拟步骤
        base_sim.step()
        sim_cluster.step()
        sim_random.step()
        
        # 记录意见历史
        base_opinion_history.append(base_sim.opinions.copy())
        cluster_opinion_history.append(sim_cluster.opinions.copy())
        random_opinion_history.append(sim_random.opinions.copy())
    
    # 生成热图
    draw_opinion_distribution_heatmap(
        base_opinion_history, 
        "Opinion Evolution without Zealots", 
        f"{results_dir}/no_zealot_heatmap.png"
    )
    draw_opinion_distribution_heatmap(
        cluster_opinion_history, 
        "Opinion Evolution with Clustered Zealots", 
        f"{results_dir}/cluster_zealot_heatmap.png"
    )
    draw_opinion_distribution_heatmap(
        random_opinion_history, 
        "Opinion Evolution with Random Zealots", 
        f"{results_dir}/random_zealot_heatmap.png"
    )
    
    # 绘制网络图 - 意见分布
    draw_network(base_sim, "opinion", "Opinion Network without Zealots", f"{results_dir}/no_zealot_opinion_network.png")
    draw_network(sim_cluster, "opinion", "Opinion Network with Clustered Zealots", f"{results_dir}/cluster_zealot_opinion_network.png")
    draw_network(sim_random, "opinion", "Opinion Network with Random Zealots", f"{results_dir}/random_zealot_opinion_network.png")
    
    # 绘制zealot网络图 - 在原有网络显示功能基础上创建zealot标记图
    draw_zealot_network(base_sim, [], "Network without Zealots", f"{results_dir}/no_zealot_network.png")
    draw_zealot_network(sim_cluster, cluster_zealots, "Network with Clustered Zealots", f"{results_dir}/cluster_zealot_network.png")
    draw_zealot_network(sim_random, random_zealots, "Network with Random Zealots", f"{results_dir}/random_zealot_network.png")


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
        steps=1000,             # 运行500步
        initial_scale=0.1,     # 初始意见缩放到10%
        cluster_zealot_ratio=1,  # 100%的社区节点将成为zealot
        seed=42                # 固定随机种子以便重现结果
    ) 