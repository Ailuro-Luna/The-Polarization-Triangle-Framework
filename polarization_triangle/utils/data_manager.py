#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据管理工具模块
提供数据保存和加载的工具函数
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from polarization_triangle.core.simulation import Simulation


def save_trajectory_to_csv(history: List[np.ndarray], output_path: str) -> str:
    """
    将轨迹数据保存为CSV文件
    
    参数:
    history -- 意见历史数据列表
    output_path -- 输出CSV文件路径
    
    返回:
    保存的文件路径
    """
    # 转换为numpy数组
    history_array = np.array(history)
    steps, num_agents = history_array.shape
    
    # 创建数据框
    data = {
        'step': [],
        'agent_id': [],
        'opinion': []
    }
    
    # 填充数据
    for step in range(steps):
        for agent_id in range(num_agents):
            data['step'].append(step)
            data['agent_id'].append(agent_id)
            data['opinion'].append(history_array[step, agent_id])
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 保存到CSV
    df.to_csv(output_path, index=False)
    
    return output_path


def save_simulation_data(sim: Simulation, output_dir: str, prefix: str = 'sim_data') -> Dict[str, str]:
    """
    保存模拟数据到文件，便于后续进行统计分析
    
    参数:
    sim -- 模拟对象
    output_dir -- 输出目录路径
    prefix -- 文件名前缀
    
    返回:
    包含所有保存文件路径的字典
    """
    # 创建目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存轨迹数据
    trajectory_data = {
        'step': [],
        'agent_id': [],
        'opinion': [],
        'identity': [],
        'morality': [],
        'self_activation': [],
        'social_influence': []
    }
    
    # 获取完整历史
    activation_history = sim.get_activation_history()
    
    # 如果存在历史数据
    if sim.self_activation_history:
        # 为每一步、每个agent添加数据
        for step in range(len(sim.self_activation_history)):
            for agent_id in range(sim.num_agents):
                trajectory_data['step'].append(step)
                trajectory_data['agent_id'].append(agent_id)
                # 对于opinion需要从trajectory中获取，如果没有则用当前值
                if hasattr(sim, 'opinion_trajectory') and step < len(sim.opinion_trajectory):
                    trajectory_data['opinion'].append(sim.opinion_trajectory[step][agent_id])
                else:
                    trajectory_data['opinion'].append(sim.opinions[agent_id])
                
                trajectory_data['identity'].append(sim.identities[agent_id])
                trajectory_data['morality'].append(sim.morals[agent_id])
                trajectory_data['self_activation'].append(activation_history['self_activation_history'][step][agent_id])
                trajectory_data['social_influence'].append(activation_history['social_influence_history'][step][agent_id])
    
    # 将数据转换为DataFrame并保存为CSV
    df = pd.DataFrame(trajectory_data)
    trajectory_csv_path = os.path.join(output_dir, f"{prefix}_trajectory.csv")
    df.to_csv(trajectory_csv_path, index=False)
    
    # 保存最终状态数据
    final_state = {
        'agent_id': list(range(sim.num_agents)),
        'opinion': sim.opinions.tolist(),
        'identity': sim.identities.tolist(),
        'morality': sim.morals.tolist(),
        'self_activation': sim.self_activation.tolist(),
        'social_influence': sim.social_influence.tolist()
    }
    
    df_final = pd.DataFrame(final_state)
    final_csv_path = os.path.join(output_dir, f"{prefix}_final_state.csv")
    df_final.to_csv(final_csv_path, index=False)
    
    # 保存网络结构
    network_data = []
    for i in range(sim.num_agents):
        for j in range(i+1, sim.num_agents):  # 只保存上三角矩阵避免重复
            if sim.adj_matrix[i, j] > 0:
                network_data.append({
                    'source': i,
                    'target': j,
                    'weight': sim.adj_matrix[i, j]
                })
    
    df_network = pd.DataFrame(network_data)
    network_csv_path = os.path.join(output_dir, f"{prefix}_network.csv")
    df_network.to_csv(network_csv_path, index=False)
    
    # 保存模拟配置
    config_dict = vars(sim.config)
    config_data = []
    for key, value in config_dict.items():
        # 跳过无法序列化的复杂对象
        if isinstance(value, (int, float, str, bool)) or value is None:
            config_data.append({'parameter': key, 'value': value})
    
    df_config = pd.DataFrame(config_data)
    config_csv_path = os.path.join(output_dir, f"{prefix}_config.csv")
    df_config.to_csv(config_csv_path, index=False)
    
    return {
        'trajectory': trajectory_csv_path,
        'final_state': final_csv_path,
        'network': network_csv_path,
        'config': config_csv_path
    }
