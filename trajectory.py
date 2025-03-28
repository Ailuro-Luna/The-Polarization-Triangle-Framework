# trajectory.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def run_simulation_with_trajectory(sim, steps=500):
    history = []
    history.append(sim.opinions.copy())
    for _ in range(steps):
        sim.step()
        history.append(sim.opinions.copy())
    return history

def save_trajectory_to_csv(history, output_path):
    """
    将轨迹数据保存为CSV文件
    
    参数:
    history -- 意见历史数据列表
    output_path -- 输出CSV文件路径
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

def draw_opinion_trajectory(history, title, filename):
    history = np.array(history)
    total_steps = history.shape[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(total_steps)
    for i in range(history.shape[1]):
        ax.plot(x, history[:, i], color='gray', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Opinion")
    ax.set_ylim(-1, 1)
    plt.savefig(filename)
    plt.close()
