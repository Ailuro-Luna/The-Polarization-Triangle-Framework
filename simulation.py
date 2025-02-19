import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numba import njit

def sample_morality(mode, leaning_prob=0.7):
    """
    根据 mode 采样 morality 值。
    mode 可选：
      - "all_neutral": 返回 0
      - "evenly_non_neutral": 均等分配 conservative (-1) 与 progressive (1)
      - "leaning_conservative": 返回 -1 的概率为 leaning_prob，否则返回 1
      - "leaning_progressive": 返回 1 的概率为 leaning_prob，否则返回 -1
      - "half_neutral_mixed": 50% 取 neutral (0)，50% 在 conservative 与 progressive 均等采样
    """
    r = np.random.rand()
    if mode == "all_neutral":
        return 0
    elif mode == "evenly_non_neutral":
        return 1 if np.random.rand() < 0.5 else -1
    elif mode == "leaning_conservative":
        return -1 if np.random.rand() < leaning_prob else 1
    elif mode == "leaning_progressive":
        return 1 if np.random.rand() < leaning_prob else -1
    elif mode == "half_neutral_mixed":
        if r < 0.5:
            return 0
        else:
            return 1 if np.random.rand() < 0.5 else -1
    else:
        return 1 if np.random.rand() < 0.5 else -1

class Simulation:
    def __init__(self, num_agents=100,
                 network_type='random',
                 network_params=None,
                 opinion_distribution="uniform",
                 coupling="none",
                 extreme_fraction=0.0,
                 moral_correlation="partial",
                 # 新增 cluster 参数（仅在 community 网络下生效）
                 cluster_identity=False,
                 cluster_morality=False,
                 cluster_identity_prob=0.8,
                 cluster_morality_prob=0.8,
                 # 新增 morality 初始化模式参数：
                 morality_mode="evenly_non_neutral",  # 可选："all_neutral", "evenly_non_neutral", "leaning_conservative", "leaning_progressive", "half_neutral_mixed"
                 morality_leaning_prob=0.7):
        self.num_agents = num_agents
        self.network_type = network_type
        # 创建网络（仅用于结构和布局）
        if network_type == 'random':
            p = network_params.get("p", 0.1) if network_params else 0.1
            self.G = nx.erdos_renyi_graph(n=num_agents, p=p)
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

        # 获取布局（用于后续可视化）
        self.pos = nx.spring_layout(self.G, seed=42)

        # 转换为邻接矩阵（加速 opinion dynamics）
        self.adj_matrix = nx.to_numpy_array(self.G, dtype=np.int32)

        # 如果是 community 网络且需要 cluster 化初始化，
        # 则提前为每个 cluster（节点属性 "block"）随机确定“主导”值
        self.cluster_identity_majority = {}
        self.cluster_morality_majority = {}
        if network_type == 'community':
            for node in self.G.nodes():
                block = self.G.nodes[node].get("block")
                if cluster_identity and block not in self.cluster_identity_majority:
                    self.cluster_identity_majority[block] = 1 if np.random.rand() < 0.5 else -1
                if cluster_morality and block not in self.cluster_morality_majority:
                    # 使用 sample_morality 来确定 cluster 的主导 morality
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

        # 遍历每个 agent
        for i in range(num_agents):
            # 初始化 identity
            if self.cluster_identity and self.network_type == 'community':
                block = self.G.nodes[i].get("block")
                majority = self.cluster_identity_majority.get(block, 1)
                identity = majority if np.random.rand() < self.cluster_identity_prob else -majority
            else:
                identity = 1 if np.random.rand() < 0.5 else -1
            self.identities[i] = identity

            # 初始化 morality
            if self.cluster_morality and self.network_type == 'community':
                block = self.G.nodes[i].get("block")
                majority = self.cluster_morality_majority.get(block, sample_morality(morality_mode, morality_leaning_prob))
                r = np.random.rand()
                if r < self.cluster_morality_prob:
                    morality = majority
                else:
                    morality = sample_morality(morality_mode, morality_leaning_prob)
            else:
                morality = sample_morality(morality_mode, morality_leaning_prob)
            self.morals[i] = morality

            # 生成 opinion（不受 morality cluster 初始化影响，仍采用原方案）
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
        # 传入 identities 与 morals 数组，使 update_opinions 时能考虑它们对门槛的影响
        self.opinions = update_opinions(self.opinions, self.adj_matrix,
                                        self.tolerance, self.influence_factor,
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
def update_opinions(opinions, adj_matrix, tolerance, influence_factor, identities, morals):
    n = opinions.shape[0]
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
        diff = opinions[neighbor] - opinions[i]

        # 调整门槛：身份与道德影响门槛而非更新速度
        if identities[i] == identities[neighbor]:
            id_factor = 1.25
        else:
            id_factor = 0.5

        if morals[i] == 0 or morals[neighbor] == 0:
            mor_factor = 1.0
        else:
            if morals[i] == morals[neighbor]:
                mor_factor = 1.25
            else:
                mor_factor = 0.5

        effective_tolerance = tolerance * id_factor * mor_factor

        if np.abs(diff) <= effective_tolerance:
            opinions[i] += influence_factor * diff
            if opinions[i] > 1:
                opinions[i] = 1
            elif opinions[i] < -1:
                opinions[i] = -1
    return opinions
