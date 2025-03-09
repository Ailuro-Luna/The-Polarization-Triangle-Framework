import os
import copy
import numpy as np
from simulation import Simulation
from config import lfr_config, SimulationConfig
from visualization import draw_network
from trajectory import run_simulation_with_trajectory, draw_opinion_trajectory
from visualization import draw_network, draw_opinion_distribution, draw_opinion_distribution_heatmap

def test_identity_issue_association():
    """测试不同的身份-问题关联配置及其对模拟结果的影响"""
    import os
    from trajectory import run_simulation_with_trajectory, draw_opinion_trajectory
    from visualization import draw_network, draw_opinion_distribution, draw_opinion_distribution_heatmap
    
    # 创建测试结果目录
    test_dir = "identity_test_results"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # 测试不同的配置
    test_configs = {
        "neutral": SimulationConfig(
            num_agents=200,
            network_type="lfr",
            opinion_distribution="uniform",
            morality_mode="half",
            cluster_identity=True,
            identity_issue_association={1: 0.0, -1: 0.0},
            identity_influence_factor=0.0  # 无身份影响（作为基线）
        ),
        "strong_association": SimulationConfig(
            num_agents=200,
            network_type="lfr",
            opinion_distribution="uniform",
            morality_mode="half",
            cluster_identity=True,
            identity_issue_association={1: 0.8, -1: -0.8},
            identity_influence_factor=0.15
        ),
        "high_moral": SimulationConfig(
            num_agents=200,
            network_type="lfr",
            opinion_distribution="uniform",
            morality_mode="all1",  # 全部高道德化
            cluster_identity=True,
            identity_issue_association={1: 0.5, -1: -0.5},
            identity_influence_factor=0.1
        ),
        "low_moral": SimulationConfig(
            num_agents=200,
            network_type="lfr",
            opinion_distribution="uniform",
            morality_mode="all0",  # 全部低道德化
            cluster_identity=True,
            identity_issue_association={1: 0.5, -1: -0.5},
            identity_influence_factor=0.1
        )
    }
    
    for config_name, config in test_configs.items():
        print(f"Running simulation with {config_name} configuration...")
        
        # 创建配置特定的目录
        config_dir = os.path.join(test_dir, config_name)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        # 运行模拟
        sim = Simulation(config)
        
        # 保存初始状态
        draw_network(sim, "opinion", f"Initial Opinion Network - {config_name}", 
                     os.path.join(config_dir, "initial_opinion.png"))
        draw_network(sim, "identity", f"Identity Network - {config_name}", 
                     os.path.join(config_dir, "identity.png"))
        draw_network(sim, "morality", f"Morality Network - {config_name}", 
                     os.path.join(config_dir, "morality.png"))
        draw_opinion_distribution(sim, f"Initial Opinion Distribution - {config_name}", 
                                 os.path.join(config_dir, "initial_distribution.png"))
        
        # 运行模拟并记录轨迹
        trajectory = run_simulation_with_trajectory(sim, steps=300)
        
        # 保存最终状态
        draw_network(sim, "opinion", f"Final Opinion Network - {config_name}", 
                     os.path.join(config_dir, "final_opinion.png"))
        draw_opinion_distribution(sim, f"Final Opinion Distribution - {config_name}", 
                                 os.path.join(config_dir, "final_distribution.png"))
        
        # 绘制意见轨迹
        draw_opinion_trajectory(trajectory, f"Opinion Trajectories - {config_name}", 
                               os.path.join(config_dir, "opinion_trajectory.png"))
        
        # 绘制意见分布热力图
        draw_opinion_distribution_heatmap(
            trajectory, 
            f"Opinion Distribution over Time - {config_name}", 
            os.path.join(config_dir, "opinion_heatmap.png"),
            bins=40,
            log_scale=True
        )
        
        print(f"Completed {config_name} simulation.")

if __name__ == "__main__":
    test_identity_issue_association()
