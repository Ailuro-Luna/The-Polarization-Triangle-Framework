# config.py

# Configuration 1: Default community network, twin_peak distribution, partial coupling & moral correlation
model_params_default = {
    "num_agents": 100,
    "network_type": "community",
    "network_params": {"p": 0.1},
    "opinion_distribution": "twin_peak",
    "coupling": "partial",
    "extreme_fraction": 0.1,
    "moral_correlation": "partial"
}

# Configuration 2: High polarization state: more agents, strong coupling & moral correlation, higher extreme fraction
model_params_high = {
    "num_agents": 100,
    "network_type": "community",
    "network_params": {"p": 0.1},
    "opinion_distribution": "twin_peak",
    "coupling": "strong",
    "extreme_fraction": 0.1,
    "moral_correlation": "strong"
}

# Configuration 3: Low polarization state: uniform opinion distribution, no coupling or moral correlation
model_params_low = {
    "num_agents": 100,
    "network_type": "community",
    "network_params": {"p": 0.1},
    "opinion_distribution": "uniform",
    "coupling": "none",
    "extreme_fraction": 0.1,
    "moral_correlation": "none"
}

# Configuration 4: Random network comparison: random network, single_peak distribution, partial coupling & moral correlation
model_params_random = {
    "num_agents": 100,
    "network_type": "random",
    "network_params": {"p": 0.1},
    "opinion_distribution": "single_peak",
    "coupling": "none",
    "extreme_fraction": 0.1,
    "moral_correlation": "partial"
}

# Configuration 5: Community network with sparse parameters
model_params_test = {
    "num_agents": 100,
    "network_type": "community",
    "network_params": {"intra_p": 0.8, "inter_p": 0.1},
    "opinion_distribution": "twin_peak",
    "coupling": "partial",
    "extreme_fraction": 0.1,
    "moral_correlation": "partial"
}

# Small-world network configuration
model_params_ws = {
    "num_agents": 100,
    "network_type": "ws",
    "network_params": {"k": 4, "p": 0.1},
    "opinion_distribution": "twin_peak",
    "coupling": "partial",
    "extreme_fraction": 0.1,
    "moral_correlation": "partial"
}

# Scale-free network configuration
model_params_ba = {
    "num_agents": 100,
    "network_type": "ba",
    "network_params": {"m": 2},
    "opinion_distribution": "single_peak",
    "coupling": "partial",
    "extreme_fraction": 0.1,
    "moral_correlation": "partial"
}

model_params_cluster = {
    "num_agents": 100,
    "network_type": "community",
    "network_params": {"p": 0.1},
    "opinion_distribution": "twin_peak",
    "coupling": "partial",
    "extreme_fraction": 0.1,
    "moral_correlation": "partial",
    "cluster_identity": True,
    "cluster_morality": True,
    "morality_mode": "half_neutral_mixed"
}

# LFR benchmark network configuration
model_params_lfr = {
    "num_agents": 500,
    "network_type": "lfr",
    "network_params": {
        "tau1": 3,             # 度分布的幂律指数（通常大于2）
        "tau2": 1.5,           # 社区规模的幂律指数
        "mu": 0.1,             # 混合参数，代表节点与外部社区连接的比例
        "average_degree": 5,   # 平均度
        "min_community": 10    # 最小社区规模
    },
    "opinion_distribution": "twin_peak",
    "coupling": "partial",
    "extreme_fraction": 0.1,
    "moral_correlation": "partial",
    "cluster_identity": True,
    "cluster_morality": True,
    "morality_mode": "half_neutral_mixed"
}


# Default configuration to use
model_params = model_params_lfr
