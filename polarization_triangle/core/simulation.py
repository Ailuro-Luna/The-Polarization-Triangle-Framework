# final_optimized_simulation.py
import numpy as np
import networkx as nx
from polarization_triangle.core.config import SimulationConfig
from polarization_triangle.core.dynamics import *
from polarization_triangle.utils.network import create_network, handle_isolated_nodes
from typing import Dict

class Simulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.num_agents = config.num_agents
        self.network_type = config.network_type

        # 创建网络
        self.graph = create_network(
            num_agents=self.num_agents,
            network_type=config.network_type,
            network_params=config.network_params,
        )
        # 移除自环
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        # 处理孤立节点
        handle_isolated_nodes(self.graph)
        # 获取邻接矩阵
        self.adj_matrix = nx.adjacency_matrix(self.graph).toarray()

        self.pos = nx.spring_layout(self.graph, k=0.1, iterations=50, scale=2.0, seed=42)

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
            for node in self.graph.nodes():
                block = None
                if self.network_type == 'community':
                    block = self.graph.nodes[node].get("block")
                elif self.network_type == 'lfr':
                    block = self.graph.nodes[node].get("community")
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
        
        # 添加轨迹存储
        self.opinion_trajectory = []
        
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

    def _init_identities(self):
        for i in range(self.num_agents):
            if self.config.cluster_identity and self.network_type in ['community', 'lfr']:
                block = None
                if self.network_type == 'community':
                    block = self.graph.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.graph.nodes[i].get("community")
                    if isinstance(block, (set, frozenset)):
                        block = min(block)
                majority = self.cluster_identity_majority.get(block, 1 if np.random.rand() < 0.5 else -1)
                prob = self.config.cluster_identity_prob
                self.identities[i] = majority if np.random.rand() < prob else -majority
            else:
                self.identities[i] = 1 if np.random.rand() < 0.5 else -1

    def _init_morality(self):
        for i in range(self.num_agents):
            if self.config.cluster_morality and self.network_type in ['community', 'lfr']:
                block = None
                if self.network_type == 'community':
                    block = self.graph.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.graph.nodes[i].get("community")
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
                    block = self.graph.nodes[i].get("block")
                elif self.network_type == 'lfr':
                    block = self.graph.nodes[i].get("community")
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

        # # 应用身份与议题关联的偏移
        # for i in range(self.num_agents):
        #     identity = self.identities[i]
        #     association = self.identity_issue_mapping.get(identity, 0)
        #     # 基于身份与议题关联添加偏移
        #     random_factor = np.random.uniform(0.5, 1.0)
        #     shift = association * random_factor
        #     self.opinions[i] = np.clip(self.opinions[i] + shift, -1, 1)

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
        
        # 记录当前意见到轨迹
        self.opinion_trajectory.append(self.opinions.copy())
        
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

    def save_simulation_data(self, output_dir: str, prefix: str = 'sim_data') -> Dict[str, str]:
        """
        保存模拟数据到文件，便于后续进行统计分析
        
        参数:
        output_dir -- 输出目录路径
        prefix -- 文件名前缀
        
        返回:
        包含所有保存文件路径的字典
        """

        from polarization_triangle.utils.data_manager import save_simulation_data
        return save_simulation_data(self, output_dir, prefix) 