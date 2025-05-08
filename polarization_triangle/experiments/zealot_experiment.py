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
    base_config.alpha = 0.5
    base_config.beta = 0.1
    print(base_config)
    # 设置道德化率
    base_config.morality_rate = 0.5
    
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
    
    # 存储意见历史
    base_opinion_history = []
    cluster_opinion_history = []
    random_opinion_history = []
    degree_opinion_history = []
    
    # 运行模拟
    for step in range(steps):
        # 更新zealot的意见
        set_zealot_opinions(sim_cluster, cluster_zealots)
        set_zealot_opinions(sim_random, random_zealots)
        set_zealot_opinions(sim_degree, degree_zealots)
        
        # 执行模拟步骤
        base_sim.step()
        sim_cluster.step()
        sim_random.step()
        sim_degree.step()
        
        # 记录意见历史
        base_opinion_history.append(base_sim.opinions.copy())
        cluster_opinion_history.append(sim_cluster.opinions.copy())
        random_opinion_history.append(sim_random.opinions.copy())
        degree_opinion_history.append(sim_degree.opinions.copy())
    
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
    draw_opinion_distribution_heatmap(
        degree_opinion_history, 
        "Opinion Evolution with High-Degree Zealots", 
        f"{results_dir}/degree_zealot_heatmap.png"
    )
    
    # 绘制网络图 - 意见分布
    draw_network(base_sim, "opinion", "Opinion Network without Zealots", f"{results_dir}/no_zealot_opinion_network.png")
    draw_network(sim_cluster, "opinion", "Opinion Network with Clustered Zealots", f"{results_dir}/cluster_zealot_opinion_network.png")
    draw_network(sim_random, "opinion", "Opinion Network with Random Zealots", f"{results_dir}/random_zealot_opinion_network.png")
    draw_network(sim_degree, "opinion", "Opinion Network with High-Degree Zealots", f"{results_dir}/degree_zealot_opinion_network.png")
    
    # 绘制zealot网络图 - 在原有网络显示功能基础上创建zealot标记图
    draw_zealot_network(base_sim, [], "Network without Zealots", f"{results_dir}/no_zealot_network.png")
    draw_zealot_network(sim_cluster, cluster_zealots, "Network with Clustered Zealots", f"{results_dir}/cluster_zealot_network.png")
    draw_zealot_network(sim_random, random_zealots, "Network with Random Zealots", f"{results_dir}/random_zealot_network.png")
    draw_zealot_network(sim_degree, degree_zealots, "Network with High-Degree Zealots", f"{results_dir}/degree_zealot_network.png")


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
        num_zealots=50,        # 50个zealot
        seed=42                # 固定随机种子以便重现结果
    ) 