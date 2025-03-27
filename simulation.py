# simulation.py
import numpy as np
import networkx as nx
from numba import njit
from config import SimulationConfig


def sample_morality(morality_rate):
    """
    根据道德化率随机生成一个道德值（0或1）

    参数:
    morality_rate -- 介于0和1之间的浮点数，表示道德化的概率

    返回:
    1（道德化）或0（非道德化）
    """
    return 1 if np.random.rand() < morality_rate else 0

class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_agents = config.num_agents
        self.network_type = config.network_type

        # 根据网络类型生成网络
        self.G = self._create_network()
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        # 检查是否有孤立点
        isolated_count = len([node for node, degree in dict(self.G.degree()).items() if degree == 0])
        if isolated_count > 0:
            print(f"警告: 移除自环后仍有 {isolated_count} 个孤立点，再次处理...")
            self._handle_isolated_nodes(self.G)

        self.pos = nx.spring_layout(self.G, k=0.1, iterations=50, scale=2.0, seed=42)
        self.adj_matrix = nx.to_numpy_array(self.G, dtype=np.int32)

        # 初始化聚类主导属性
        self.cluster_identity_majority = {}
        self.cluster_morality_majority = {}
        self.cluster_opinion_majority = {} if config.cluster_opinion else None

        # 初始化身份与议题关联映射
        self.identity_issue_mapping = config.identity_issue_mapping
        self.identity_influence_factor = config.identity_influence_factor

        # 初始化规则计数器历史记录
        self.rule_counts_history = []

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
                    self.cluster_morality_majority[block] = sample_morality(config.morality_rate)
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

        # 从配置中获取模型参数
        self.delta = config.delta  # 意见衰减率
        self.u = np.ones(self.num_agents) * config.u  # 意见激活系数
        self.alpha = np.ones(self.num_agents) * config.alpha  # 自我激活系数
        self.beta = config.beta  # 社会影响系数
        self.gamma = np.ones(self.num_agents) * config.gamma  # 道德化影响系数
        
        self._init_identities()
        self._init_morality()
        self._init_opinions()
        
        # 存储每个agent的邻居列表
        self.neighbors_list = [[] for _ in range(self.num_agents)]
        # 存储每个agent的同身份邻居列表
        self.same_identity_neighbors_list = [[] for _ in range(self.num_agents)]
        
        # 初始化邻居列表
        self._init_neighbors_lists()
        
        # 初始化用于监控自我激活和社会影响的数组
        self.self_activation = np.zeros(self.num_agents, dtype=np.float64)
        self.social_influence = np.zeros(self.num_agents, dtype=np.float64)
        
        # 存储历史数据
        self.self_activation_history = []
        self.social_influence_history = []

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

    def _handle_isolated_nodes(self, G):
        """处理网络中的孤立点

        参数:
        G -- 网络图

        处理方式:
        1. 找出所有孤立点（度为0的节点）
        2. 为每个孤立点随机连接到网络中的其他节点
        """
        isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]

        if not isolated_nodes:
            return  # 如果没有孤立点，直接返回

        print(f"检测到 {len(isolated_nodes)} 个孤立点，进行处理...")

        # 获取非孤立节点列表
        non_isolated = [node for node in G.nodes() if node not in isolated_nodes]

        if not non_isolated:
            # 如果所有节点都是孤立的（极少情况），创建一个环形连接
            for i in range(len(isolated_nodes)):
                G.add_edge(isolated_nodes[i], isolated_nodes[(i + 1) % len(isolated_nodes)])
            return

        # 为每个孤立点随机连接到1-3个非孤立节点
        for node in isolated_nodes:
            # 随机决定连接数量，最小1个，最大3个或所有非孤立节点数
            num_connections = min(np.random.randint(1, 4), len(non_isolated))
            # 随机选择连接目标
            targets = np.random.choice(non_isolated, num_connections, replace=False)
            # 添加边
            for target in targets:
                G.add_edge(node, target)

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
                majority = self.cluster_morality_majority.get(block, sample_morality(self.config.morality_rate))
                prob = self.config.cluster_morality_prob
                self.morals[i] = majority if np.random.rand() < prob else sample_morality(self.config.morality_rate)
            else:
                self.morals[i] = sample_morality(self.config.morality_rate)

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

        # 应用身份与议题关联的偏移
        for i in range(self.num_agents):
            identity = self.identities[i]
            association = self.identity_issue_mapping.get(identity, 0)
            # 基于身份与议题关联添加偏移
            random_factor = np.random.uniform(0.5, 1.0)
            shift = association * random_factor
            self.opinions[i] = np.clip(self.opinions[i] + shift, -1, 1)

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

    # 基于极化三角框架的感知意见计算
    def calculate_perceived_opinion(self, i, j):
        """
        计算agent i对agent j的意见的感知
        
        参数:
        i -- 观察者agent的索引
        j -- 被观察agent的索引
        
        返回:
        感知意见值，取值为-1, 0, 或1
        """
        z_j = self.opinions[j]
        m_i = self.morals[i]
        m_j = self.morals[j]
        
        if z_j == 0:
            return 0
        elif (m_i == 1 or m_j == 1):
            return np.sign(z_j)  # 返回z_j的符号(1或-1)
        else:
            return z_j  # 返回实际值

    # 计算代理间关系系数
    def calculate_relationship_coefficient(self, i, j):
        """
        计算agent i与agent j之间的关系系数
        
        参数:
        i -- agent i的索引
        j -- agent j的索引
        
        返回:
        关系系数值
        """
        a_ij = self.adj_matrix[i, j]
        if a_ij == 0:  # 如果不是邻居，关系系数为0
            return 0
            
        l_i = self.identities[i]
        l_j = self.identities[j]
        m_i = self.morals[i]
        m_j = self.morals[j]
        
        sigma_ij = self.calculate_perceived_opinion(i, j)
        sigma_ji = self.calculate_perceived_opinion(j, i)
        
        # 根据极化三角框架公式计算关系系数
        if l_i != l_j and m_i == 1 and m_j == 1 and (sigma_ij * sigma_ji) < 0:
            return -a_ij
        elif l_i == l_j and m_i == 1 and m_j == 1 and (sigma_ij * sigma_ji) < 0:
            # 计算tilde{sigma}_{sameIdentity} - agent i的同身份邻居的平均感知意见值
            sigma_same_identity = self.calculate_same_identity_sigma(i)
            return (a_ij / sigma_ji) * sigma_same_identity
        else:
            return a_ij

    def calculate_same_identity_sigma(self, i):
        """
        计算agent i的同身份邻居的平均感知意见值
        
        参数:
        i -- agent i的索引
        
        返回:
        同身份邻居的平均感知意见值，如果没有同身份邻居则返回0
        """
        same_identity_sigmas = []
        
        # 直接使用预先计算的同身份邻居列表
        for j in self.same_identity_neighbors_list[i]:
            sigma_ij = self.calculate_perceived_opinion(i, j)
            same_identity_sigmas.append(sigma_ij)
        
        # 计算平均值
        if same_identity_sigmas:
            return np.mean(same_identity_sigmas)
        else:
            # 如果没有同身份邻居，返回一个默认值
            return 0

    def _init_neighbors_lists(self):
        """初始化每个agent的邻居列表和同身份邻居列表"""
        for i in range(self.num_agents):
            # 获取邻居列表
            for j in range(self.num_agents):
                if i != j and self.adj_matrix[i, j] > 0:
                    self.neighbors_list[i].append(j)
                    
            # 获取同身份邻居列表
            l_i = self.identities[i]
            for j in self.neighbors_list[i]:
                if self.identities[j] == l_i:
                    self.same_identity_neighbors_list[i].append(j)

    # 重构后的step方法，基于极化三角框架
    def step(self):
        """
        执行一步模拟，更新所有代理的意见
        基于极化三角框架的动力学方程实现
        """
        # 初始化规则计数 (为了与原有代码兼容，虽然本方法不再使用规则)
        rule_counts = np.zeros(16, dtype=np.int32)
        
        # 计算每个agent的意见变化
        opinion_changes = np.zeros(self.num_agents)
        
        # 重置自我激活和社会影响的值
        self.self_activation = np.zeros(self.num_agents, dtype=np.float64)
        self.social_influence = np.zeros(self.num_agents, dtype=np.float64)
        
        for i in range(self.num_agents):
            # 计算自我感知
            sigma_ii = np.sign(self.opinions[i]) if self.opinions[i] != 0 else 0
            
            # 计算邻居影响总和
            neighbor_influence = 0
            
            # 使用预先计算的邻居列表
            for j in self.neighbors_list[i]:
                A_ij = self.calculate_relationship_coefficient(i, j)
                sigma_ij = self.calculate_perceived_opinion(i, j)
                neighbor_influence += A_ij * sigma_ij
            
            # 计算并存储自我激活项
            self.self_activation[i] = self.alpha[i] * sigma_ii
            
            # 计算并存储社会影响项
            self.social_influence[i] = (self.beta / (1 + self.gamma[i] * self.morals[i])) * neighbor_influence
            
            # 计算意见变化率
            # 回归中性意见项
            regression_term = -self.delta * self.opinions[i]
            
            # 意见激活项
            activation_term = self.u[i] * np.tanh(
                self.self_activation[i] + self.social_influence[i]
            )
            
            # 总变化
            opinion_changes[i] = regression_term + activation_term
        
        # 应用意见变化，使用小步长避免过大变化
        step_size = self.config.influence_factor  # 使用配置中的影响因子作为步长
        self.opinions += step_size * opinion_changes
        
        # 确保意见值在[-1, 1]范围内
        self.opinions = np.clip(self.opinions, -1, 1)
        
        # 为了与原有代码兼容，存储规则计数
        self.rule_counts_history.append(rule_counts)
        
        # 存储自我激活和社会影响的历史数据
        self.self_activation_history.append(self.self_activation.copy())
        self.social_influence_history.append(self.social_influence.copy())
    
    def get_activation_components(self):
        """
        获取最近一步中的自我激活和社会影响组件
        
        返回:
        字典，包含自我激活和社会影响的数组
        """
        return {
            "self_activation": self.self_activation,
            "social_influence": self.social_influence
        }
    
    def get_activation_history(self):
        """
        获取所有历史步骤的自我激活和社会影响组件
        
        返回:
        字典，包含自我激活和社会影响的历史数据列表
        """
        return {
            "self_activation_history": self.self_activation_history,
            "social_influence_history": self.social_influence_history
        }
        
    def get_agent_activation_details(self, agent_id):
        """
        获取特定代理的自我激活和社会影响详情
        
        参数:
        agent_id -- 代理的ID
        
        返回:
        字典，包含该代理的自我激活和社会影响值
        """
        if 0 <= agent_id < self.num_agents:
            return {
                "self_activation": self.self_activation[agent_id],
                "social_influence": self.social_influence[agent_id],
                "total_activation": self.self_activation[agent_id] + self.social_influence[agent_id],
                "opinion": self.opinions[agent_id],
                "morality": self.morals[agent_id],
                "identity": self.identities[agent_id]
            }
        else:
            return None
