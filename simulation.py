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

    # 添加新方法：身份影响步骤
    def identity_influence_step(self):
        for i in range(self.num_agents):
            agent_identity = self.identities[i]

            # 第一部分：处理同身份邻居的影响
            same_identity_neighbors = []
            for j in range(self.num_agents):
                if self.adj_matrix[i, j] == 1 and self.identities[j] == agent_identity:
                    same_identity_neighbors.append(j)

            if same_identity_neighbors:
                # 计算同身份邻居的平均意见和道德值
                avg_opinion = np.mean([self.opinions[j] for j in same_identity_neighbors])
                avg_morality = np.mean([self.morals[j] for j in same_identity_neighbors])

                # 使用感知到的道德值作为被影响的概率
                if np.random.rand() < avg_morality:
                    # 获取身份与议题关联
                    association = self.identity_issue_mapping.get(agent_identity, 0)

                    # 计算影响强度
                    influence = self.identity_influence_factor * abs(association)
                    resistance = 1 - (abs(self.opinions[i]) ** 2)

                    # 应用向群体平均意见的影响
                    self.opinions[i] = self.opinions[i] + influence * (avg_opinion - self.opinions[i])

                    # 确保意见保持在界限内
                    self.opinions[i] = np.clip(self.opinions[i], -1, 1)

            # 第二部分：直接向身份关联立场的拉拢效应
            # 以 identity_influence_factor 作为概率决定是否受到拉拢
            if np.random.rand() < self.identity_influence_factor:
                # 获取该身份关联的议题立场
                target_position = self.identity_issue_mapping.get(agent_identity, 0)

                # 计算拉拢效应强度（与当前立场差距的一小部分）
                pull_strength = 0.05  # 可以调整这个值控制拉拢强度

                # 应用拉拢效应，向目标位置移动
                direction = np.sign(target_position - self.opinions[i])
                if direction != 0:  # 只有当agent不在目标位置时才移动
                    # 移动公式：当前位置 + 方向 * 拉拢强度 * (1 - 到达目标位置后的阻力)
                    resistance = abs(self.opinions[i] - target_position) / 2  # 接近目标位置时阻力增加
                    self.opinions[i] = self.opinions[i] + direction * pull_strength * resistance

                    # 确保不会越过目标位置（防止"过冲"）
                    if (direction > 0 and self.opinions[i] > target_position) or \
                            (direction < 0 and self.opinions[i] < target_position):
                        self.opinions[i] = target_position

                    # 确保意见保持在界限内
                    self.opinions[i] = np.clip(self.opinions[i], -1, 1)

    # 修改step方法
    def step(self):
        # 首先，应用原始意见更新
        self.opinions, rule_counts = update_opinions(
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
        
        # 存储规则计数
        self.rule_counts_history.append(rule_counts)

        # 然后，应用身份影响
        self.identity_influence_step()


@njit
def update_opinions(opinions, adj_matrix, influence_factor, p_radical_high, p_radical_low, p_conv_high, p_conv_low,
                    identities, morals):
    n = opinions.shape[0]
    # 初始化规则计数器 - 从8种规则扩展到16种规则
    rule_counts = np.zeros(16, dtype=np.int32)
    
    # 新的概率参数
    # 收敛概率参数(convergence)
    p_conv_vhigh = 0.9  # 非常高的收敛概率
    p_conv_high = 0.7   # 高收敛概率，保持原值
    p_conv_mid = 0.5    # 中等收敛概率
    p_conv_low = 0.3    # 低收敛概率，保持原值

    # 拉动概率参数(pulling)
    p_pull_high = 0.8   # 高拉动概率
    p_pull_mid = 0.6    # 中等拉动概率
    p_pull_low = 0.4    # 低拉动概率

    # 极化概率参数(polarization)
    p_polar_vhigh = 0.9  # 非常高的极化概率
    p_polar_high = 0.7   # 高极化概率
    p_polar_mid = 0.5    # 中等极化概率
    p_polar_low = 0.3    # 低极化概率
    
    # 抵抗概率参数(resistance)
    p_resist_high = 0.8  # 高抵抗概率
    p_resist_low = 0.4   # 低抵抗概率
    
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
        m_j = morals[neighbor]  # 获取邻居的道德状态
        same_dir = ((o_i > 0 and o_j > 0) or (o_i < 0 and o_j < 0))
        same_id = (identities[i] == identities[neighbor])
        
        # 极化阻力因子，使接近极端值更困难
        resistance = 1 - (abs(o_i) ** 3)  # 在极端值处阻力最大
        
        # 同方向意见
        if same_dir:
            # 规则1-8：意见同向，不同身份关系和道德状态组合
            if same_id:  # 身份相同
                if m_i == 0 and m_j == 0:  # {0,0}
                    # 规则1：意见同向，身份相同，双方不道德化，高度收敛
                    if np.random.rand() < p_conv_high:
                        o_i = o_i + influence_factor * (o_j - o_i)
                        rule_counts[0] += 1
                elif m_i == 0 and m_j == 1:  # {0,1}
                    # 规则2：意见同向，身份相同，自己不道德化对方道德化，中度被拉向极端
                    if np.random.rand() < p_pull_mid:
                        if o_j > 0:
                            o_i = o_i + influence_factor * (o_j + 0.2 * (1 - o_j)) - o_i
                        else:
                            o_i = o_i + influence_factor * (o_j - 0.2 * (1 + o_j)) - o_i
                        rule_counts[1] += 1
                elif m_i == 1 and m_j == 0:  # {1,0}
                    # 规则3：意见同向，身份相同，自己道德化对方不道德化，中度拉他向极端
                    if np.random.rand() < p_pull_mid:
                        if o_i > 0:
                            o_i = o_i + influence_factor * (1 - o_i) * resistance
                        else:
                            o_i = o_i - influence_factor * (1 + o_i) * resistance
                        rule_counts[2] += 1
                else:  # m_i == 1 and m_j == 1, {1,1}
                    # 规则4：意见同向，身份相同，双方道德化，高度极化
                    if np.random.rand() < p_polar_high:
                        if o_i > 0:
                            o_i = o_i + influence_factor * (1 - o_i) * resistance
                        else:
                            o_i = o_i - influence_factor * (1 + o_i) * resistance
                        rule_counts[3] += 1
            else:  # 身份不同
                if m_i == 0 and m_j == 0:  # {0,0}
                    # 规则5：意见同向，身份不同，双方不道德化，中度收敛
                    if np.random.rand() < p_conv_mid:
                        o_i = o_i + influence_factor * (o_j - o_i)
                        rule_counts[4] += 1
                elif m_i == 0 and m_j == 1:  # {0,1}
                    # 规则6：意见同向，身份不同，自己不道德化对方道德化，低度被拉向极端
                    if np.random.rand() < p_pull_low:
                        if o_j > 0:
                            o_i = o_i + influence_factor * (o_j + 0.1 * (1 - o_j)) - o_i
                        else:
                            o_i = o_i + influence_factor * (o_j - 0.1 * (1 + o_j)) - o_i
                        rule_counts[5] += 1
                elif m_i == 1 and m_j == 0:  # {1,0}
                    # 规则7：意见同向，身份不同，自己道德化对方不道德化，低度拉他向极端
                    if np.random.rand() < p_pull_low:
                        if o_i > 0:
                            o_i = o_i + influence_factor * 0.7 * (1 - o_i) * resistance
                        else:
                            o_i = o_i - influence_factor * 0.7 * (1 + o_i) * resistance
                        rule_counts[6] += 1
                else:  # m_i == 1 and m_j == 1, {1,1}
                    # 规则8：意见同向，身份不同，双方道德化，中度极化
                    if np.random.rand() < p_polar_mid:
                        if o_i > 0:
                            o_i = o_i + influence_factor * 0.8 * (1 - o_i) * resistance
                        else:
                            o_i = o_i - influence_factor * 0.8 * (1 + o_i) * resistance
                        rule_counts[7] += 1
        else:  # 不同方向意见
            # 规则9-16：意见不同向，不同身份关系和道德状态组合
            if same_id:  # 身份相同
                if m_i == 0 and m_j == 0:  # {0,0}
                    # 规则9：意见不同向，身份相同，双方不道德化，非常高收敛
                    if np.random.rand() < p_conv_vhigh:
                        o_i = o_i + influence_factor * (o_j - o_i)
                        rule_counts[8] += 1
                elif m_i == 0 and m_j == 1:  # {0,1}
                    # 规则10：意见不同向，身份相同，自己不道德化对方道德化，中度收敛或被拉向他方
                    if np.random.rand() < p_conv_mid:
                        o_i = o_i + influence_factor * 1.2 * (o_j - o_i)
                        rule_counts[9] += 1
                elif m_i == 1 and m_j == 0:  # {1,0}
                    # 规则11：意见不同向，身份相同，自己道德化对方不道德化，低度抵抗并持守立场
                    if np.random.rand() < p_resist_low:
                        o_i = o_i + influence_factor * 0.5 * (abs(o_j)/o_j - o_i)
                        rule_counts[10] += 1
                else:  # m_i == 1 and m_j == 1, {1,1}
                    # 规则12：意见不同向，身份相同，双方道德化，低度双方极化
                    if np.random.rand() < p_polar_low:
                        if o_i > 0:
                            o_i = o_i + influence_factor * 0.6 * (1 - o_i) * resistance
                        else:
                            o_i = o_i - influence_factor * 0.6 * (1 + o_i) * resistance
                        rule_counts[11] += 1
            else:  # 身份不同
                if m_i == 0 and m_j == 0:  # {0,0}
                    # 规则13：意见不同向，身份不同，双方不道德化，低度收敛
                    if np.random.rand() < p_conv_low:
                        o_i = o_i + influence_factor * (o_j - o_i)
                        rule_counts[12] += 1
                elif m_i == 0 and m_j == 1:  # {0,1}
                    # 规则14：意见不同向，身份不同，自己不道德化对方道德化，高度被拉向他方
                    if np.random.rand() < p_pull_high:
                        o_i = o_i + influence_factor * 1.5 * (o_j - o_i)
                        rule_counts[13] += 1
                elif m_i == 1 and m_j == 0:  # {1,0}
                    # 规则15：意见不同向，身份不同，自己道德化对方不道德化，高度抵抗并走向极端
                    if np.random.rand() < p_resist_high:
                        if o_i > 0:
                            o_i = o_i + influence_factor * 1.2 * (1 - o_i) * resistance
                        else:
                            o_i = o_i - influence_factor * 1.2 * (1 + o_i) * resistance
                        rule_counts[14] += 1
                else:  # m_i == 1 and m_j == 1, {1,1}
                    # 规则16：意见不同向，身份不同，双方道德化，极高度双方极化
                    if np.random.rand() < p_polar_vhigh:
                        if o_i > 0:
                            o_i = o_i + influence_factor * 1.5 * (1 - o_i) * resistance
                        else:
                            o_i = o_i - influence_factor * 1.5 * (1 + o_i) * resistance
                        rule_counts[15] += 1
                            
        # 确保意见值在-1到1之间
        if o_i > 1:
            o_i = 1
        elif o_i < -1:
            o_i = -1
        opinions[i] = o_i
    return opinions, rule_counts
