# simulation.py
import numpy as np
import networkx as nx
from numba import njit
from config import SimulationConfig

def sample_morality(mode):
    if mode == "all1":
        return 1
    elif mode == "all0":
        return 0
    elif mode == "half":
        return 1 if np.random.rand() < 0.5 else 0
    else:
        # 默认情况也采用 half 模式
        return 1 if np.random.rand() < 0.5 else 0

class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_agents = config.num_agents
        self.network_type = config.network_type

        # 根据网络类型生成网络
        self.G = self._create_network()
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        self.pos = nx.spring_layout(self.G, k=0.1, iterations=50, scale=2.0, seed=42)
        self.adj_matrix = nx.to_numpy_array(self.G, dtype=np.int32)

        # 初始化聚类主导属性
        self.cluster_identity_majority = {}
        self.cluster_morality_majority = {}
        self.cluster_opinion_majority = {} if config.cluster_opinion else None

        if self.network_type in ['community', 'lfr']:
            for node in self.G.nodes():
                block = None
                if self.network_type == 'community':
                    block = self.G.nodes[node].get("block")
                elif self.network_type == 'lfr':
                    block = self.G.nodes[node].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                if config.cluster_identity and block not in self.cluster_identity_majority:
                    self.cluster_identity_majority[block] = 1 if np.random.rand() < 0.5 else -1
                if config.cluster_morality and block not in self.cluster_morality_majority:
                    self.cluster_morality_majority[block] = sample_morality(config.morality_mode)
                if config.cluster_opinion:
                    if block not in self.cluster_opinion_majority:
                        if config.opinion_distribution == "twin_peak":
                            majority_opinion = np.random.choice([-1, 1]) * np.abs(np.random.normal(0.7, 0.2))
                        elif config.opinion_distribution == "uniform":
                            majority_opinion = np.random.uniform(-1, 1)
                        elif config.opinion_distribution == "single_peak":
                            majority_opinion = np.random.normal(0, 0.3)
                        elif config.opinion_distribution == "skewed":
                            majority_opinion = np.random.beta(2, 5) * 2 - 1
                        else:
                            majority_opinion = np.random.uniform(-1, 1)
                        self.cluster_opinion_majority[block] = majority_opinion

        # 初始化代理属性
        self.opinions = np.empty(self.num_agents, dtype=np.float64)
        self.morals = np.empty(self.num_agents, dtype=np.int32)
        self.identities = np.empty(self.num_agents, dtype=np.int32)

        self._init_identities()
        self._init_morality()
        self._init_opinions()

    def _create_network(self):
        params = self.config.network_params
        if self.network_type == 'random':
            p = params.get("p", 0.1) if params else 0.1
            return nx.erdos_renyi_graph(n=self.num_agents, p=p)
        elif self.network_type == 'lfr':
            tau1 = params.get("tau1", 3)
            tau2 = params.get("tau2", 1.5)
            mu = params.get("mu", 0.1)
            average_degree = params.get("average_degree", 5)
            min_community = params.get("min_community", 10)
            return nx.LFR_benchmark_graph(
                n=self.num_agents,
                tau1=tau1,
                tau2=tau2,
                mu=mu,
                average_degree=average_degree,
                min_community=min_community,
                seed=42
            )
        elif self.network_type == 'community':
            community_sizes = [self.num_agents // 4] * 4
            intra_p = params.get("intra_p", 0.8) if params else 0.8
            inter_p = params.get("inter_p", 0.1) if params else 0.1
            return nx.random_partition_graph(community_sizes, intra_p, inter_p)
        elif self.network_type == 'ws':
            k = params.get("k", 4)
            p = params.get("p", 0.1)
            return nx.watts_strogatz_graph(n=self.num_agents, k=k, p=p)
        elif self.network_type == 'ba':
            m = params.get("m", 2)
            return nx.barabasi_albert_graph(n=self.num_agents, m=m)
        else:
            return nx.erdos_renyi_graph(n=self.num_agents, p=0.1)

    def _init_identities(self):
        for i in range(self.num_agents):
            if self.config.cluster_identity and self.network_type in ['community', 'lfr']:
                block = None
                if self.network_type == 'community':
                    block = self.G.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.G.nodes[i].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                majority = self.cluster_identity_majority.get(block, 1)
                prob = self.config.cluster_identity_prob
                self.identities[i] = majority if np.random.rand() < prob else -majority
            else:
                self.identities[i] = 1 if np.random.rand() < 0.5 else -1

    def _init_morality(self):
        for i in range(self.num_agents):
            if self.config.cluster_morality and self.network_type in ['community', 'lfr']:
                block = None
                if self.network_type == 'community':
                    block = self.G.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.G.nodes[i].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                majority = self.cluster_morality_majority.get(block, sample_morality(self.config.morality_mode))
                prob = self.config.cluster_morality_prob
                self.morals[i] = majority if np.random.rand() < prob else sample_morality(self.config.morality_mode)
            else:
                self.morals[i] = sample_morality(self.config.morality_mode)

    def _init_opinions(self):
        for i in range(self.num_agents):
            if self.config.cluster_opinion and self.network_type in ['community', 'lfr']:
                block = None
                if self.network_type == 'community':
                    block = self.G.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.G.nodes[i].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                majority = self.cluster_opinion_majority.get(block)
                if majority is None:
                    if self.config.opinion_distribution == "twin_peak":
                        majority = np.random.choice([-1, 1]) * np.abs(np.random.normal(0.7, 0.2))
                    elif self.config.opinion_distribution == "uniform":
                        majority = np.random.uniform(-1, 1)
                    elif self.config.opinion_distribution == "single_peak":
                        majority = np.random.normal(0, 0.3)
                    elif self.config.opinion_distribution == "skewed":
                        majority = np.random.beta(2, 5) * 2 - 1
                    else:
                        majority = np.random.uniform(-1, 1)
                    self.cluster_opinion_majority[block] = majority
                prob = self.config.cluster_opinion_prob
                self.opinions[i] = majority if np.random.rand() < prob else self.generate_opinion(self.identities[i])
            else:
                self.opinions[i] = self.generate_opinion(self.identities[i])

    def generate_opinion(self, identity):
        if self.config.extreme_fraction > 0 and np.random.rand() < self.config.extreme_fraction:
            if self.config.coupling != "none":
                return np.random.uniform(-1, -0.5) if identity == 1 else np.random.uniform(0.5, 1)
            else:
                return np.random.choice([-1, 1]) * np.random.uniform(0.5, 1)
        if self.config.opinion_distribution == "uniform":
            return np.random.uniform(-1, 1)
        elif self.config.opinion_distribution == "single_peak":
            return np.random.normal(0, 0.3)
        elif self.config.opinion_distribution == "twin_peak":
            return np.random.choice([-1, 1]) * np.abs(np.random.normal(0.7, 0.2))
        elif self.config.opinion_distribution == "skewed":
            return np.random.beta(2, 5) * 2 - 1
        else:
            return np.random.uniform(-1, 1)

    def step(self):
        self.opinions = update_opinions(
            self.opinions,
            self.adj_matrix,
            self.config.influence_factor,
            self.config.p_radical_high,
            self.config.p_radical_low,
            self.config.p_conv_high,
            self.config.p_conv_low,
            self.identities,
            self.morals
        )

@njit
def update_opinions(opinions, adj_matrix, influence_factor, p_radical_high, p_radical_low, p_conv_high, p_conv_low, identities, morals):
    n = opinions.shape[0]
    for i in range(n):
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
        same_dir = ((o_i > 0 and o_j > 0) or (o_i < 0 and o_j < 0))
        if same_dir:
            if m_i == 1:
                if np.random.rand() < p_radical_high:
                    o_i = o_i + influence_factor * (1 - o_i) if o_i > 0 else o_i - influence_factor * (1 + o_i)
            else:
                if np.random.rand() < p_radical_low:
                    o_i = o_i + influence_factor * (1 - o_i) if o_i > 0 else o_i - influence_factor * (1 + o_i)
        else:
            if m_i == 1:
                if identities[i] == identities[neighbor]:
                    if np.random.rand() < p_conv_low:
                        o_i = o_i + influence_factor * (o_j - o_i)
                else:
                    if np.random.rand() < p_radical_low:
                        o_i = o_i + influence_factor * (1 - o_i) if o_i > 0 else o_i - influence_factor * (1 + o_i)
            else:
                if identities[i] == identities[neighbor]:
                    if np.random.rand() < p_conv_high:
                        o_i = o_i + influence_factor * (o_j - o_i)
                else:
                    if np.random.rand() < p_conv_low:
                        o_i = o_i + influence_factor * (o_j - o_i)
        if o_i > 1:
            o_i = 1
        elif o_i < -1:
            o_i = -1
        opinions[i] = o_i
    return opinions
