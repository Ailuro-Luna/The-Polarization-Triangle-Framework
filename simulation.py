# final_optimized_simulation.py
import numpy as np
import networkx as nx
from numba import njit, int32, float64, prange, boolean
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


@njit
def calculate_perceived_opinion_func(opinions, morals, i, j):
    """
    计算agent i对agent j的意见的感知
    
    参数:
    opinions -- 所有agent的意见数组
    morals -- 所有agent的道德值数组
    i -- 观察者agent的索引
    j -- 被观察agent的索引
    
    返回:
    感知意见值
    """
    z_j = opinions[j]
    m_i = morals[i]
    m_j = morals[j]
    
    if z_j == 0:
        return 0
    elif (m_i == 1 or m_j == 1):
        return np.sign(z_j)  # 返回z_j的符号(1或-1)
    else:
        return z_j  # 返回实际值


@njit
def calculate_same_identity_sigma_func(opinions, morals, identities, neighbors_indices, neighbors_indptr, i):
    """
    计算agent i的同身份邻居的平均感知意见值（numba加速版本）
    
    参数:
    opinions -- 所有agent的意见数组
    morals -- 所有agent的道德值数组
    identities -- 所有agent的身份数组
    neighbors_indices -- CSR格式的邻居索引
    neighbors_indptr -- CSR格式的邻居指针
    i -- agent i的索引
    
    返回:
    同身份邻居的平均感知意见值，如果没有同身份邻居则返回0
    """
    sigma_sum = 0.0
    count = 0
    l_i = identities[i]
    
    # 遍历i的所有邻居
    for idx in range(neighbors_indptr[i], neighbors_indptr[i+1]):
        j = neighbors_indices[idx]
        # 如果是同身份的
        if identities[j] == l_i:
            sigma_sum += calculate_perceived_opinion_func(opinions, morals, i, j)
            count += 1
    
    # 返回平均值，如果没有同身份邻居，则返回0
    if count > 0:
        return sigma_sum / count
    return 0.0


@njit
def calculate_relationship_coefficient_func(adj_matrix, identities, morals, opinions, i, j, same_identity_sigmas):
    """
    计算agent i与agent j之间的关系系数
    
    参数:
    adj_matrix -- 邻接矩阵
    identities -- 身份数组
    morals -- 道德值数组
    opinions -- 意见数组
    i -- agent i的索引
    j -- agent j的索引
    same_identity_sigmas -- agent i的同身份邻居的感知意见值数组或平均值
    
    返回:
    关系系数值
    """
    a_ij = adj_matrix[i, j]
    if a_ij == 0:  # 如果不是邻居，关系系数为0
        return 0
        
    l_i = identities[i]
    l_j = identities[j]
    m_i = morals[i]
    m_j = morals[j]
    
    # 计算感知意见
    sigma_ij = calculate_perceived_opinion_func(opinions, morals, i, j)
    sigma_ji = calculate_perceived_opinion_func(opinions, morals, j, i)
    
    # 根据极化三角框架公式计算关系系数
    if l_i != l_j and m_i == 1 and m_j == 1 and (sigma_ij * sigma_ji) < 0:
        return -a_ij
    elif l_i == l_j and m_i == 1 and m_j == 1 and (sigma_ij * sigma_ji) < 0:
        # 使用传入的同身份平均感知意见值
        if sigma_ji == 0:  # 避免除零错误
            return a_ij
        return (a_ij / sigma_ji) * same_identity_sigmas
    else:
        return a_ij


@njit
def step_calculation(opinions, morals, identities, adj_matrix, 
                    neighbors_indices, neighbors_indptr,  
                    alpha, beta, gamma, delta, u, influence_factor):
    """
    执行一步模拟计算，使用numba加速
    
    参数:
    opinions -- 代理意见数组
    morals -- 代理道德值数组
    identities -- 代理身份数组
    adj_matrix -- 邻接矩阵
    neighbors_indices -- CSR格式的邻居索引数组
    neighbors_indptr -- CSR格式的邻居指针数组
    alpha -- 自我激活系数
    beta -- 社会影响系数
    gamma -- 道德化影响系数
    delta -- 意见衰减率
    u -- 意见激活系数
    influence_factor -- 影响因子
    
    返回:
    更新后的opinions, self_activation, social_influence
    """
    num_agents = len(opinions)
    opinion_changes = np.zeros(num_agents, dtype=np.float64)
    self_activation = np.zeros(num_agents, dtype=np.float64)
    social_influence = np.zeros(num_agents, dtype=np.float64)
    
    # 预计算所有agent的同身份邻居平均感知意见
    same_identity_sigmas = np.zeros(num_agents, dtype=np.float64)
    for i in range(num_agents):
        same_identity_sigmas[i] = calculate_same_identity_sigma_func(
            opinions, morals, identities, neighbors_indices, neighbors_indptr, i)
    
    for i in range(num_agents):
        # 计算自我感知
        sigma_ii = np.sign(opinions[i]) if opinions[i] != 0 else 0
        
        # 计算邻居影响总和
        neighbor_influence = 0.0
        
        # 遍历i的所有邻居（使用CSR格式）
        for idx in range(neighbors_indptr[i], neighbors_indptr[i+1]):
            j = neighbors_indices[idx]
            A_ij = calculate_relationship_coefficient_func(
                adj_matrix, 
                identities, 
                morals, 
                opinions, 
                i, j, 
                same_identity_sigmas[i]
            )
            sigma_ij = calculate_perceived_opinion_func(opinions, morals, i, j)
            neighbor_influence += A_ij * sigma_ij
        
        # 计算并存储自我激活项
        self_activation[i] = alpha[i] * sigma_ii
        
        # 计算并存储社会影响项
        social_influence[i] = (beta / (1 + gamma[i] * morals[i])) * neighbor_influence
        
        # 计算意见变化率
        # 回归中性意见项
        regression_term = -delta * opinions[i]
        
        # 意见激活项
        activation_term = u[i] * np.tanh(
            self_activation[i] + social_influence[i]
        )
        
        # 总变化
        opinion_changes[i] = regression_term + activation_term
    
    # 应用意见变化，使用小步长避免过大变化
    opinions_new = opinions.copy()
    opinions_new += influence_factor * opinion_changes
    
    # 确保意见值在[-1, 1]范围内
    opinions_new = np.clip(opinions_new, -1, 1)
    
    return opinions_new, self_activation, social_influence


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
        
        # 初始化邻居列表
        self._init_neighbors_lists()
        
        # 初始化用于监控自我激活和社会影响的数组
        self.self_activation = np.zeros(self.num_agents, dtype=np.float64)
        self.social_influence = np.zeros(self.num_agents, dtype=np.float64)
        
        # 存储历史数据
        self.self_activation_history = []
        self.social_influence_history = []
        
        # 优化用的数据结构(CSR格式)
        self._create_csr_neighbors()

    def _create_csr_neighbors(self):
        """
        创建CSR格式的邻居表示
        """
        total_edges = sum(len(neighbors) for neighbors in self.neighbors_list)
        self.neighbors_indices = np.zeros(total_edges, dtype=np.int32)
        self.neighbors_indptr = np.zeros(self.num_agents + 1, dtype=np.int32)
        
        idx = 0
        for i, neighbors in enumerate(self.neighbors_list):
            self.neighbors_indptr[i] = idx
            for j in neighbors:
                self.neighbors_indices[idx] = j
                idx += 1
        self.neighbors_indptr[self.num_agents] = idx

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

    def _init_neighbors_lists(self):
        """初始化每个agent的邻居列表"""
        for i in range(self.num_agents):
            # 获取邻居列表
            for j in range(self.num_agents):
                if i != j and self.adj_matrix[i, j] > 0:
                    self.neighbors_list[i].append(j)

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
        return calculate_perceived_opinion_func(self.opinions, self.morals, i, j)

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
        # 计算同身份邻居的平均感知意见值
        sigma_same_identity = calculate_same_identity_sigma_func(
            self.opinions, self.morals, self.identities, 
            self.neighbors_indices, self.neighbors_indptr, i)
        
        return calculate_relationship_coefficient_func(
            self.adj_matrix, 
            self.identities, 
            self.morals, 
            self.opinions, 
            i, j, 
            sigma_same_identity
        )

    # 优化的step方法，基于极化三角框架
    def step(self):
        """
        执行一步模拟，更新所有代理的意见
        基于极化三角框架的动力学方程实现
        使用numba加速的step_calculation函数
        """
        # 初始化规则计数 (为了与原有代码兼容，虽然本方法不再使用规则)
        rule_counts = np.zeros(16, dtype=np.int32)
        
        # 使用numba加速的函数进行主要计算
        new_opinions, new_self_activation, new_social_influence = step_calculation(
            self.opinions,
            self.morals,
            self.identities,
            self.adj_matrix,
            self.neighbors_indices,
            self.neighbors_indptr,
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            self.u,
            self.config.influence_factor
        )
        
        # 更新状态
        self.opinions = new_opinions
        self.self_activation = new_self_activation
        self.social_influence = new_social_influence
        
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