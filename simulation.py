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

        # 初始化感知群体属性 (添加)
        self.perceived_group_opinion = np.zeros(self.num_agents, dtype=np.float64)
        self.perceived_group_morality = np.zeros(self.num_agents, dtype=np.float64)

        self._init_identities()
        self._init_morality()
        self._init_opinions()
        self._calculate_perceived_group_attributes()  # 计算初始感知属性

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
        # 获取身份-问题关联
        association = self.config.identity_issue_association.get(identity, 0.0)

        # 基于关联调整偏差
        bias = association * 0.3  # 控制偏差的强度

        if self.config.extreme_fraction > 0 and np.random.rand() < self.config.extreme_fraction:
            if self.config.coupling != "none":
                # 根据身份-问题关联调整极端意见
                if association > 0.3:  # 身份倾向于正面意见
                    return np.random.uniform(0.5, 1)
                elif association < -0.3:  # 身份倾向于负面意见
                    return np.random.uniform(-1, -0.5)
                else:  # 无明显关联
                    return np.random.uniform(-1, -0.5) if identity == 1 else np.random.uniform(0.5, 1)
            else:
                return np.random.choice([-1, 1]) * np.random.uniform(0.5, 1)

        if self.config.opinion_distribution == "uniform":
            # 将偏差添加到均匀分布
            unbiased = np.random.uniform(-1, 1)
            return np.clip(unbiased + bias, -1, 1)
        elif self.config.opinion_distribution == "single_peak":
            # 根据偏差调整正态分布的均值
            return np.clip(np.random.normal(bias, 0.3), -1, 1)
        elif self.config.opinion_distribution == "twin_peak":
            # 根据关联选择峰值
            if association > 0.3:  # 强正关联
                peak = 1
            elif association < -0.3:  # 强负关联
                peak = -1
            else:
                peak = np.random.choice([-1, 1])
            return peak * np.abs(np.random.normal(0.7, 0.2))
        elif self.config.opinion_distribution == "skewed":
            # 根据关联调整偏斜
            if association > 0.3:
                return np.random.beta(5, 2) * 2 - 1  # 偏向正面
            elif association < -0.3:
                return np.random.beta(2, 5) * 2 - 1  # 偏向负面
            else:
                return np.random.beta(2, 5) * 2 - 1  # 默认偏斜
        else:
            return np.clip(np.random.uniform(-1, 1) + bias, -1, 1)

    def _calculate_perceived_group_attributes(self):
        """计算每个智能体对其身份群体的平均意见和道德感知"""
        for i in range(self.num_agents):
            identity_i = self.identities[i]
            # 寻找具有相同身份的邻居
            same_identity_neighbors = []
            opinion_sum = 0.0
            moral_sum = 0.0
            count = 0

            for j in range(self.num_agents):
                if self.adj_matrix[i, j] == 1 and self.identities[j] == identity_i:
                    same_identity_neighbors.append(j)
                    opinion_sum += self.opinions[j]
                    moral_sum += self.morals[j]
                    count += 1

            if count > 0:
                # 计算相同身份邻居的平均意见和道德
                self.perceived_group_opinion[i] = opinion_sum / count
                self.perceived_group_morality[i] = moral_sum / count
            else:
                # 如果没有相同身份的邻居，使用身份-问题关联作为默认值
                association = self.config.identity_issue_association.get(identity_i, 0.0)
                self.perceived_group_opinion[i] = association
                self.perceived_group_morality[i] = 0.5  # 默认为中性道德

    def _apply_identity_influence(self):
        """应用身份对意见的影响"""
        for i in range(self.num_agents):
            identity_i = self.identities[i]
            moral_i = self.morals[i]
            opinion_i = self.opinions[i]

            # 获取身份-问题关联
            association = self.config.identity_issue_association.get(identity_i, 0.0)

            # 获取感知的群体意见和道德
            perceived_opinion = self.perceived_group_opinion[i]
            perceived_moral = self.perceived_group_morality[i]

            # 计算身份影响
            # 1. 道德化程度越高，向群体意见靠拢的力量越大
            moralization_factor = perceived_moral * 0.2

            # 2. 身份-问题关联度越高，向关联方向拉力越大
            association_factor = association * 0.1

            # 计算总影响
            identity_influence = 0.0

            # 向群体意见靠拢的影响
            if perceived_moral > 0.5:  # 只有当感知到群体道德化程度较高时
                identity_influence += moralization_factor * (perceived_opinion - opinion_i)

            # 身份-问题关联的直接影响
            if abs(association) > 0.2:  # 只有当关联度足够强时
                # 使用二次抵抗因子使极端值更难达到
                resistance = 1 - (abs(opinion_i) ** 2)
                identity_influence += association_factor * resistance

            # 应用影响，使用config中的身份影响因子控制强度
            self.opinions[i] += identity_influence * self.config.identity_influence_factor

            # 确保意见保持在[-1, 1]范围内
            if self.opinions[i] > 1:
                self.opinions[i] = 1
            elif self.opinions[i] < -1:
                self.opinions[i] = -1

    def step(self):
        # 原有的意见更新
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

        # 应用身份影响
        self._apply_identity_influence()

        # 更新感知群体属性
        self._calculate_perceived_group_attributes()


@njit
def update_opinions(opinions, adj_matrix, influence_factor, p_radical_high, p_radical_low, p_conv_high, p_conv_low,
                    identities, morals):
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
            # 规则1和2：意见同向，moral=0，均以较低概率收敛（使用 p_conv_low）
            if m_i == 0:
                if np.random.rand() < p_conv_low:
                    o_i = o_i + influence_factor * (o_j - o_i)
            # 规则3和4：意见同向，moral=1，均以较低概率使意见走向更极化（使用 p_radical_low）
            else:  # m_i == 1
                if np.random.rand() < p_radical_low:
                    # 修改：添加"极化阻力因子"，使接近极端值更困难
                    resistance = 1 - (abs(o_i) ** 2)  # 二次函数，在极端值处阻力最大
                    if o_i > 0:
                        o_i = o_i + influence_factor * (1 - o_i) * resistance
                    else:
                        o_i = o_i - influence_factor * (1 + o_i) * resistance
        else:
            # 意见不同方向
            if m_i == 0:
                # 规则5：不同方向，moral=0，身份相同，收敛概率较高（p_conv_high）
                if identities[i] == identities[neighbor]:
                    if np.random.rand() < p_conv_high:
                        o_i = o_i + influence_factor * (o_j - o_i)
                # 规则6：不同方向，moral=0，身份不同，收敛概率较低（p_conv_low）
                else:
                    if np.random.rand() < p_conv_low:
                        o_i = o_i + influence_factor * (o_j - o_i)
            else:  # m_i == 1
                # 规则7：不同方向，moral=1，身份相同，收敛概率较低（p_conv_low）
                if identities[i] == identities[neighbor]:
                    if np.random.rand() < p_conv_low:
                        o_i = o_i + influence_factor * (o_j - o_i)
                # 规则8：不同方向，moral=1，身份不同，意见以较高概率极化（p_radical_high）
                else:
                    if np.random.rand() < p_radical_high:
                        # 修改：添加"极化阻力因子"，使接近极端值更困难
                        resistance = 1 - (abs(o_i) ** 2)  # 二次函数，在极端值处阻力最大
                        if o_i > 0:
                            o_i = o_i + influence_factor * (1 - o_i) * resistance
                        else:
                            o_i = o_i - influence_factor * (1 + o_i) * resistance
        if o_i > 1:
            o_i = 1
        elif o_i < -1:
            o_i = -1
        opinions[i] = o_i
    return opinions
