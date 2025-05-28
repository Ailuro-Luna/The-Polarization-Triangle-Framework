"""
Zealot实验核心模块

包含Zealot实验的主要逻辑
"""

import numpy as np
import copy
from typing import Dict, List, Any, Optional
from polarization_triangle.core.config import SimulationConfig
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.analysis.trajectory import run_simulation_with_trajectory


def run_zealot_experiment(
    steps: int = 500,
    initial_scale: float = 0.1,
    num_zealots: int = 50,
    zealot_opinion: float = 1.0,
    zealot_mode: str = 'random',
    zealot_morality: bool = False,
    zealot_identity_allocation: bool = True,
    morality_rate: float = 0.0,
    identity_clustered: bool = False,
    seed: int = 42,
    network_seed: int = None,
    output_dir: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    运行zealot实验
    
    参数:
    steps -- 模拟步数
    initial_scale -- 初始意见缩放
    num_zealots -- zealot数量
    zealot_opinion -- zealot意见值
    zealot_mode -- zealot模式 ('random', 'clustered', 'high_degree', 'none')
    zealot_morality -- zealot是否道德化
    zealot_identity_allocation -- 是否按身份分配zealot
    morality_rate -- 道德化率
    identity_clustered -- 身份是否聚类
    seed -- 随机种子
    network_seed -- 网络种子
    output_dir -- 输出目录
    
    返回:
    实验结果字典
    """
    # 设置随机种子
    np.random.seed(seed)
    
    if network_seed is None:
        network_seed = seed
    
    # 创建基础配置
    config = _create_base_config(
        morality_rate=morality_rate,
        identity_clustered=identity_clustered,
        network_seed=network_seed
    )
    
    # 创建基础模拟
    base_sim = _create_base_simulation(config, initial_scale)
    
    # 定义要测试的模式
    if zealot_mode == 'none':
        test_modes = ['no_zealot']
    else:
        test_modes = ['no_zealot', zealot_mode]
    
    # 存储结果
    results = {}
    
    # 运行不同模式
    for mode in test_modes:
        print(f"\nRunning {mode} mode...")
        
        # 创建该模式的模拟实例
        sim = _create_mode_simulation(
            base_sim, config, mode, num_zealots, 
            zealot_opinion, zealot_morality, 
            zealot_identity_allocation
        )
        
        # 运行模拟
        trajectory = []
        for step in range(steps):
            sim.step()
            trajectory.append(sim.opinions.copy())
            
            if (step + 1) % 100 == 0:
                print(f"  Step {step + 1}/{steps}")
        
        trajectory = np.array(trajectory)
        
        # 获取zealot IDs
        zealot_ids = sim.get_zealot_ids() if hasattr(sim, 'get_zealot_ids') else []
        
        # 生成统计和可视化
        if output_dir:
            from .statistics import generate_opinion_statistics
            from .visualization import (
                generate_rule_usage_plots,
                generate_activation_visualizations
            )
            
            # 生成统计
            stats = generate_opinion_statistics(
                sim, trajectory, zealot_ids, mode, output_dir
            )
            
            # 生成可视化
            generate_rule_usage_plots(sim, mode, output_dir)
            generate_activation_visualizations(sim, trajectory, mode, output_dir)
            
            results[mode] = {
                'simulation': sim,
                'trajectory': trajectory,
                'zealot_ids': zealot_ids,
                'stats': stats
            }
        else:
            results[mode] = {
                'simulation': sim,
                'trajectory': trajectory,
                'zealot_ids': zealot_ids
            }
    
    # 如果有多个模式，生成比较图
    if len(test_modes) > 1 and output_dir:
        from .visualization import plot_comparative_statistics
        all_stats = {mode: results[mode].get('stats', {}) for mode in test_modes}
        plot_comparative_statistics(all_stats, test_modes, output_dir)
    
    return results


def _create_base_config(morality_rate: float, identity_clustered: bool, 
                       network_seed: int) -> SimulationConfig:
    """创建基础模拟配置"""
    config = SimulationConfig()
    
    # 复制高极化配置的参数
    config.num_agents = 100
    config.network_type = "lfr"
    config.network_params = {
        "tau1": 3,
        "tau2": 1.5,
        "mu": 0.1,
        "average_degree": 5,
        "min_community": 10,
        "seed": network_seed
    }
    
    # 设置其他参数
    config.cluster_identity = identity_clustered
    config.cluster_morality = False
    config.cluster_opinion = False
    config.opinion_distribution = "uniform"
    config.alpha = 0.4
    config.beta = 0.12
    config.morality_rate = morality_rate
    
    return config


def _create_base_simulation(config: SimulationConfig, initial_scale: float) -> Simulation:
    """创建基础模拟实例"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            sim = Simulation(config)
            # 缩放初始意见
            sim.opinions *= initial_scale
            return sim
        except Exception as e:
            if attempt < max_retries - 1:
                # 更新网络种子重试
                config.network_params["seed"] = np.random.randint(10000)
            else:
                raise e


def _create_mode_simulation(base_sim: Simulation, config: SimulationConfig,
                          mode: str, num_zealots: int, zealot_opinion: float,
                          zealot_morality: bool, zealot_identity_allocation: bool) -> Simulation:
    """根据模式创建模拟实例"""
    # 深拷贝基础模拟
    sim = copy.deepcopy(base_sim)
    
    if mode == 'no_zealot':
        # 无zealot模式
        pass
    elif mode in ['clustered', 'random', 'high_degree']:
        # 设置zealot
        config = copy.deepcopy(config)
        config.enable_zealots = True
        config.zealot_count = num_zealots
        config.zealot_opinion = zealot_opinion
        config.zealot_mode = mode if mode != 'high_degree' else 'degree'
        config.zealot_morality = zealot_morality
        config.zealot_identity_allocation = zealot_identity_allocation
        
        # 重新初始化zealots
        sim.config = config
        sim.enable_zealots = True
        sim.zealot_opinion = zealot_opinion
        sim._init_zealots()
    
    return sim 