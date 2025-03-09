# config.py
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SimulationConfig:
    num_agents: int = 100
    network_type: str = "lfr"  # 可选："random"、"community"、"lfr"、"ws"、"ba"
    network_params: Dict[str, Any] = field(default_factory=lambda: {
        "tau1": 3,
        "tau2": 1.5,
        "mu": 0.1,
        "average_degree": 5,
        "min_community": 10,
    })
    opinion_distribution: str = "twin_peak"  # 可选："uniform"、"single_peak"、"twin_peak"、"skewed"
    coupling: str = "partial"  # 可选："none"、"partial"、"strong"
    extreme_fraction: float = 0.1
    moral_correlation: str = "partial"  # 可选："none"、"partial"、"strong"
    morality_rate: float = 0.5  # 道德化率：0.0 到 1.0 之间的值，表示道德化的比例
    # 聚类参数
    cluster_identity: bool = False
    cluster_morality: bool = False
    cluster_opinion: bool = False
    cluster_identity_prob: float = 1
    cluster_morality_prob: float = 0.8
    cluster_opinion_prob: float = 0.8
    # 仿真更新参数
    influence_factor: float = 0.1
    tolerance: float = 0.6
    # 更新概率参数（意见更新时用到的硬编码概率）
    p_radical_high: float = 0.7
    p_radical_low: float = 0.3
    p_conv_high: float = 0.7
    p_conv_low: float = 0.3

    # 身份与议题关联参数
    identity_issue_mapping: Dict[int, float] = field(default_factory=lambda: {1: 0.3, -1: -0.3})
    identity_influence_factor: float = 0.2

# 各种预设配置：
default_config = SimulationConfig()

high_polarization_config = SimulationConfig(
    network_type="community",
    opinion_distribution="twin_peak",
    coupling="strong",
    moral_correlation="strong",
    morality_rate=0.8,  # 高道德化率
)

low_polarization_config = SimulationConfig(
    network_type="community",
    opinion_distribution="uniform",
    coupling="none",
    moral_correlation="none",
    morality_rate=0.2,  # 低道德化率
)

random_config = SimulationConfig(
    network_type="random",
    opinion_distribution="single_peak",
    coupling="none",
    moral_correlation="partial",
    morality_rate=0.5,  # 中等道德化率
)

test_config = SimulationConfig(
    network_type="community",
    network_params={"intra_p": 0.8, "inter_p": 0.1},
    opinion_distribution="twin_peak",
    coupling="partial",
    moral_correlation="partial",
    morality_rate=0.5,  # 中等道德化率
)

ws_config = SimulationConfig(
    network_type="ws",
    network_params={"k": 4, "p": 0.1},
    opinion_distribution="twin_peak",
    coupling="partial",
    moral_correlation="partial",
    morality_rate=0.5,  # 中等道德化率
)

ba_config = SimulationConfig(
    network_type="ba",
    network_params={"m": 2},
    opinion_distribution="single_peak",
    coupling="partial",
    moral_correlation="partial",
    morality_rate=0.5,  # 中等道德化率
)

cluster_config = SimulationConfig(
    network_type="community",
    opinion_distribution="twin_peak",
    coupling="partial",
    moral_correlation="partial",
    cluster_identity=True,
    cluster_morality=True,
    cluster_opinion=True,
    morality_rate=0.5,  # 中等道德化率
    # 可根据需要调整聚类概率
)

lfr_config = SimulationConfig(
    num_agents=500,
    network_type="lfr",
    network_params={
        "tau1": 3,
        "tau2": 1.5,
        "mu": 0.1,
        "average_degree": 5,
        "min_community": 10
    },
    opinion_distribution="twin_peak",
    coupling="partial",
    extreme_fraction=0.1,
    moral_correlation="partial",
    cluster_identity=True,
    cluster_morality=True,
    cluster_opinion=True,
    cluster_opinion_prob=0.8,
    morality_rate=0.5,  # 中等道德化率
)

# 默认使用的配置
config = lfr_config