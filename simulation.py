import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numba import njit

def sample_morality(mode, leaning_prob=0.7):
    r = np.random.rand()
    if mode == "all_neutral":
        return 0
    elif mode == "evenly_non_neutral":
        return 1 if r < 0.5 else 0
    elif mode == "leaning_conservative":
        return 0 if r < leaning_prob else 1
    elif mode == "leaning_progressive":
        return 1 if r < leaning_prob else 0
    elif mode == "half_neutral_mixed":
        return 1 if r < 0.5 else 0
    else:
        return 1 if r < 0.5 else 0

class Simulation:
    def __init__(self, num_agents=100,
                 network_type='random',
                 network_params=None,
                 opinion_distribution="uniform",
                 coupling="none",
                 extreme_fraction=0.0,
                 moral_correlation="partial",
                 cluster_identity=False,
                 cluster_morality=False,
                 cluster_identity_prob=0.8,
                 cluster_morality_prob=0.8,
                 morality_mode="evenly_non_neutral",
                 morality_leaning_prob=0.7,
                 cluster_opinion=False,
                 cluster_opinion_prob=0.8):

        self.num_agents = num_agents
        self.network_type = network_type

        # 根据 network_type 创建网络
        if network_type == 'random':
            p = network_params.get("p", 0.1) if network_params else 0.1
            self.G = nx.erdos_renyi_graph(n=num_agents, p=p)
        elif network_type == 'lfr':
            # 使用 LFR benchmark 模型
            tau1 = network_params.get("tau1", 3) if network_params else 3
            tau2 = network_params.get("tau2", 1.5) if network_params else 1.5
            mu = network_params.get("mu", 0.1) if network_params else 0.1
            average_degree = network_params.get("average_degree", 5) if network_params else 5
            min_community = network_params.get("min_community", 10) if network_params else 10
            # 注意：LFR_benchmark_graph 可能生成重叠社区，节点属性 'community' 通常为集合
            self.G = nx.LFR_benchmark_graph(
                n=num_agents,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=average_degree,
                min_community=min_community,
                seed=42
            )
        elif network_type == 'community':
            community_sizes = [num_agents // 4] * 4
            intra_p = network_params.get("intra_p", 0.8) if network_params else 0.8
            inter_p = network_params.get("inter_p", 0.1) if network_params else 0.1
            self.G = nx.random_partition_graph(community_sizes, intra_p, inter_p)
        elif network_type == 'ws':  # small-world network
            k = network_params.get("k", 4)
            p = network_params.get("p", 0.1)
            self.G = nx.watts_strogatz_graph(n=num_agents, k=k, p=p)
        elif network_type == 'ba':  # scale-free network
            m = network_params.get("m", 2)
            self.G = nx.barabasi_albert_graph(n=num_agents, m=m)
        else:
            self.G = nx.erdos_renyi_graph(n=num_agents, p=0.1)

        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        # 获取布局（用于后续可视化）
        self.pos = nx.spring_layout(
            self.G,
            k=0.1,  # 调大后节点会更分散
            iterations=50,  # 增加迭代次数
            scale=2.0,  # 整体放大
            seed=42
        )
        # 转换为邻接矩阵（加速 opinion dynamics）
        self.adj_matrix = nx.to_numpy_array(self.G, dtype=np.int32)

        # 针对 community 和 LFR 网络，提前进行 cluster 初始化
        self.cluster_identity_majority = {}
        self.cluster_morality_majority = {}

        self.cluster_opinion = cluster_opinion
        self.cluster_opinion_prob = cluster_opinion_prob
        if network_type in ['community', 'lfr'] and cluster_opinion:
            self.cluster_opinion_majority = {}

        if network_type in ['community', 'lfr']:
            for node in self.G.nodes():
                # 对于 LFR，社区属性存储在 "community" 键中；可能为集合，选择一个代表元素
                if network_type == 'community':
                    block = self.G.nodes[node].get("block")
                elif network_type == 'lfr':
                    block = self.G.nodes[node].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                if cluster_identity and block not in self.cluster_identity_majority:
                    self.cluster_identity_majority[block] = 1 if np.random.rand() < 0.5 else -1
                if cluster_morality and block not in self.cluster_morality_majority:
                    self.cluster_morality_majority[block] = sample_morality(morality_mode, morality_leaning_prob)

        # 保存 cluster 参数与 morality 配置参数
        self.cluster_identity = cluster_identity
        self.cluster_morality = cluster_morality
        self.cluster_identity_prob = cluster_identity_prob
        self.cluster_morality_prob = cluster_morality_prob
        self.morality_mode = morality_mode
        self.morality_leaning_prob = morality_leaning_prob

        # 初始化 agent 属性（用 NumPy 数组）
        self.opinions = np.empty(num_agents, dtype=np.float64)
        self.morals = np.empty(num_agents, dtype=np.int32)    # 1: Progressive, 0: Neutral, -1: Conservative
        self.identities = np.empty(num_agents, dtype=np.int32)  # 1: Left, -1: Right

        # 遍历每个 agent 进行初始化
        for i in range(num_agents):
            # 初始化 identity
            if self.cluster_identity and self.network_type in ['community', 'lfr']:
                if self.network_type == 'community':
                    block = self.G.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.G.nodes[i].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                majority = self.cluster_identity_majority.get(block, 1)
                identity = majority if np.random.rand() < self.cluster_identity_prob else -majority
            else:
                identity = 1 if np.random.rand() < 0.5 else -1
            self.identities[i] = identity

            # 初始化 morality
            if self.cluster_morality and self.network_type in ['community', 'lfr']:
                if self.network_type == 'community':
                    block = self.G.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.G.nodes[i].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                majority = self.cluster_morality_majority.get(block, sample_morality(morality_mode, morality_leaning_prob))
                r = np.random.rand()
                if r < self.cluster_morality_prob:
                    morality = majority
                else:
                    morality = sample_morality(morality_mode, morality_leaning_prob)
            else:
                morality = sample_morality(morality_mode, morality_leaning_prob)
            self.morals[i] = morality

            # 初始化 opinion
            if self.cluster_opinion and self.network_type in ['community', 'lfr']:
                if self.network_type == 'community':
                    block = self.G.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.G.nodes[i].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                if block not in self.cluster_opinion_majority:
                    # 根据 opinion_distribution 选择生成 majority opinion 的方法
                    if opinion_distribution == "twin_peak":
                        majority_opinion = np.random.choice([-1, 1]) * np.abs(np.random.normal(0.7, 0.2))
                    elif opinion_distribution == "uniform":
                        majority_opinion = np.random.uniform(-1, 1)
                    elif opinion_distribution == "single_peak":
                        majority_opinion = np.random.normal(0, 0.3)
                    elif opinion_distribution == "skewed":
                        majority_opinion = np.random.beta(2, 5) * 2 - 1
                    else:
                        majority_opinion = np.random.uniform(-1, 1)
                    self.cluster_opinion_majority[block] = majority_opinion
                majority = self.cluster_opinion_majority.get(block)
                if np.random.rand() < self.cluster_opinion_prob:
                    self.opinions[i] = majority
                else:
                    self.opinions[i] = self.generate_opinion(identity, distribution=opinion_distribution,
                                                             coupling=coupling, extreme_fraction=extreme_fraction)
            else:
                self.opinions[i] = self.generate_opinion(identity, distribution=opinion_distribution,
                                                         coupling=coupling, extreme_fraction=extreme_fraction)

        # 参数设置： opinion 更新动力学
        self.tolerance = 0.6
        self.influence_factor = 0.1

    def generate_opinion(self, identity, distribution="uniform", coupling="none", extreme_fraction=0.0):
        if extreme_fraction > 0 and np.random.rand() < extreme_fraction:
            if coupling != "none":
                if identity == 1:
                    return np.random.uniform(-1, -0.5)
                else:
                    return np.random.uniform(0.5, 1)
            else:
                return np.random.choice([-1, 1]) * np.random.uniform(0.5, 1)
        if distribution == "uniform":
            return np.random.uniform(-1, 1)
        elif distribution == "single_peak":
            return np.random.normal(0, 0.3)
        elif distribution == "twin_peak":
            return np.random.choice([-1, 1]) * np.abs(np.random.normal(0.7, 0.2))
        elif distribution == "skewed":
            return np.random.beta(2, 5) * 2 - 1
        else:
            return np.random.uniform(-1, 1)

    def run(self, steps=100):
        for _ in range(steps):
            self.step()

    def step(self):
        self.opinions = update_opinions(self.opinions, self.adj_matrix,
                                        self.influence_factor,
                                        self.identities, self.morals)

    def generate_histograms(self, output_dir="images"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Degree Distribution Histogram
        degrees = [d for _, d in self.G.degree()]
        plt.figure()
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), align='left',
                 rwidth=0.8, color='skyblue', edgecolor='black')
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "degree_hist.png"))
        plt.close()

        # Opinion Distribution Histogram
        plt.figure()
        plt.hist(self.opinions, bins=np.linspace(-1, 1, 21), rwidth=0.8,
                 color='salmon', edgecolor='black')
        plt.title("Opinion Distribution")
        plt.xlabel("Opinion")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "opinion_hist.png"))
        plt.close()

        # Moral Distribution Histogram
        progressive = np.sum(self.morals == 1)
        neutral = np.sum(self.morals == 0)
        conservative = np.sum(self.morals == -1)
        plt.figure()
        plt.bar(["Progressive", "Neutral", "Conservative"],
                [progressive, neutral, conservative],
                color=['#d73027', '#999999', '#4575b4'], edgecolor='black')
        plt.title("Moral Distribution")
        plt.xlabel("Moral")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "moral_hist.png"))
        plt.close()

@njit
def update_opinions(opinions, adj_matrix, influence_factor, identities, morals):
    n = opinions.shape[0]
    # 定义概率：较高概率与较低概率
    p_radical_high = 0.7
    p_radical_low = 0.3
    p_conv_high = 0.7
    p_conv_low = 0.3

    for i in range(n):
        # 采用 reservoir sampling 随机选择一个邻居
        count = 0
        neighbor = -1
        for j in range(n):
            if adj_matrix[i, j] == 1:
                count += 1
                if np.random.rand() < 1.0 / count:
                    neighbor = j
        if neighbor == -1:
            continue

        o_i = opinions[i]
        o_j = opinions[neighbor]
        m_i = morals[i]
        # 判断是否“同方向”（均为正或均为负）
        same_dir = ((o_i > 0 and o_j > 0) or (o_i < 0 and o_j < 0))

        # 同方向情况
        if same_dir:
            if m_i == 1:
                if np.random.rand() < p_radical_high:
                    # 激进化：向极端方向推进
                    if o_i > 0:
                        o_i = o_i + influence_factor * (1 - o_i)
                    elif o_i < 0:
                        o_i = o_i - influence_factor * (1 + o_i)
            else:  # m_i == 0
                if np.random.rand() < p_radical_low:
                    if o_i > 0:
                        o_i = o_i + influence_factor * (1 - o_i)
                    elif o_i < 0:
                        o_i = o_i - influence_factor * (1 + o_i)
        else:
            # 不同方向情况
            if m_i == 1:
                if identities[i] == identities[neighbor]:
                    if np.random.rand() < p_conv_low:
                        # 略微向邻居靠拢
                        o_i = o_i + influence_factor * (o_j - o_i)
                else:
                    if np.random.rand() < p_radical_low:
                        # 激进化
                        if o_i > 0:
                            o_i = o_i + influence_factor * (1 - o_i)
                        elif o_i < 0:
                            o_i = o_i - influence_factor * (1 + o_i)
            else:  # m_i == 0
                if identities[i] == identities[neighbor]:
                    if np.random.rand() < p_conv_high:
                        o_i = o_i + influence_factor * (o_j - o_i)
                else:
                    if np.random.rand() < p_conv_low:
                        o_i = o_i + influence_factor * (o_j - o_i)
        # 限制 opinion 在 [-1, 1]
        if o_i > 1:
            o_i = 1
        elif o_i < -1:
            o_i = -1
        opinions[i] = o_i
    return opinions

