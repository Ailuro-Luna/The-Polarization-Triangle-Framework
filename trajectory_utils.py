# trajectory_utils.py
import numpy as np
import matplotlib.pyplot as plt

def run_simulation_with_trajectory(sim, steps=500):
    history = []
    # 记录初始状态
    history.append(sim.opinions.copy())
    for _ in range(steps):
        sim.step()
        history.append(sim.opinions.copy())
    return history

def draw_opinion_trajectory(history, title, filename):
    history = np.array(history)  # shape: (steps+1, num_agents)
    total_steps = history.shape[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(total_steps)
    # 为每个 agent 绘制轨迹
    for i in range(history.shape[1]):
        ax.plot(x, history[:, i], color='gray', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Opinion value")
    ax.set_ylim(-1, 1)
    plt.savefig(filename)
    plt.close()
