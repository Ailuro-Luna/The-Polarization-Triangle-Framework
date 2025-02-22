import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
from simulation import Simulation
from config import model_params_lfr


# 帮助函数：绘制并保存 opinion 网络图（包含 color bar）
def draw_opinion_network(sim, title, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = cm.coolwarm
    norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    node_colors = [cmap(norm(op)) for op in sim.opinions]
    nx.draw(
        sim.G,
        pos=sim.pos,
        node_color=node_colors,
        with_labels=False,
        node_size=20,
        alpha=0.8,
        edge_color="#AAAAAA",
        ax=ax
    )
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    # 添加 color bar，注意要防止报错
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Opinion value")
    plt.savefig(filename)
    plt.close()


# 帮助函数：绘制并保存 identity 网络图（使用 legend 说明颜色含义）
def draw_identity_network(sim, title, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    # identity：1 显示为红色('#e41a1c')，-1 显示为蓝色('#377eb8')
    node_colors = ['#e41a1c' if iden == 1 else '#377eb8' for iden in sim.identities]
    nx.draw(
        sim.G,
        pos=sim.pos,
        node_color=node_colors,
        with_labels=False,
        node_size=20,
        alpha=0.8,
        edge_color="#AAAAAA",
        ax=ax
    )
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    from matplotlib.patches import Patch
    patches = [
        Patch(color='#e41a1c', label='Identity: 1'),
        Patch(color='#377eb8', label='Identity: -1')
    ]
    ax.legend(handles=patches, loc='upper right', title="Identity")
    plt.savefig(filename)
    plt.close()


# 帮助函数：绘制并保存 morality 网络图（使用 legend 说明颜色含义）
def draw_morality_network(sim, title, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    # morality：1 显示为绿色('#1a9850')，0 显示为红色('#d73027')
    node_colors = ['#1a9850' if m == 1 else '#d73027' for m in sim.morals]
    nx.draw(
        sim.G,
        pos=sim.pos,
        node_color=node_colors,
        with_labels=False,
        node_size=20,
        alpha=0.8,
        edge_color="#AAAAAA",
        ax=ax
    )
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    from matplotlib.patches import Patch
    patches = [
        Patch(color='#1a9850', label='Morality: 1'),
        Patch(color='#d73027', label='Morality: 0')
    ]
    ax.legend(handles=patches, loc='upper right', title="Morality")
    plt.savefig(filename)
    plt.close()

# 新增：记录 opinion 历史轨迹的函数
def run_simulation_with_trajectory(sim, steps=500):
    history = []
    # 记录初始状态
    history.append(sim.opinions.copy())
    for _ in range(steps):
        sim.step()
        history.append(sim.opinions.copy())
    return history

# 新增：绘制 opinion 随时间变化的轨迹图
def draw_opinion_trajectory(history, title, filename):
    history = np.array(history)  # shape: (steps+1, num_agents)
    steps = history.shape[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(steps)
    # 为每个 agent 绘制一条轨迹
    for i in range(history.shape[1]):
        ax.plot(x, history[:, i], color='gray', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Opinion value")
    ax.set_ylim(-1, 1)
    plt.savefig(filename)
    plt.close()


# 运行仿真，原函数保持不变，但下面将使用新的记录函数替换调用
def run_simulation(sim, steps=500):
    for _ in range(steps):
        sim.step()

# 根据 morality_ratio 参数覆盖仿真对象中 morals 的初始值（保持不变）
def override_morality(sim, ratio):
    n = sim.num_agents
    if ratio == "all1":
        sim.morals[:] = 1
    elif ratio == "all0":
        sim.morals[:] = 0
    elif ratio == "half":
        arr = np.array([1] * (n // 2) + [0] * (n - n // 2))
        np.random.shuffle(arr)
        sim.morals[:] = arr
    else:
        raise ValueError("未知的 morality ratio 参数")

def batch_test():
    base_dir = "batch_results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for id_mode in ["random", "clustered"]:
        for mor_mode in ["random", "clustered"]:
            for op_mode in ["random", "clustered"]:
                for op_dist in ["uniform", "single_peak", "twin_peak"]:
                    for mor_ratio in ["all1", "half", "all0"]:
                        folder_name = f"ID_{id_mode}_M_{mor_mode}_OP_{op_mode}_op_{op_dist}_mor_{mor_ratio}"
                        folder_path = os.path.join(base_dir, folder_name)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)

                        print("Processing configuration:", folder_name)

                        params = copy.deepcopy(model_params_lfr)
                        params["num_agents"] = 500
                        params["cluster_identity"] = (id_mode == "clustered")
                        params["cluster_morality"] = (mor_mode == "clustered")
                        params["cluster_opinion"] = (op_mode == "clustered")
                        params["opinion_distribution"] = op_dist

                        sim = Simulation(**params)
                        override_morality(sim, mor_ratio)

                        # 保存初始状态下的 opinion 网络图
                        start_opinion_path = os.path.join(folder_path, "start_opinion.png")
                        draw_opinion_network(sim, f"Starting Opinion Network\nConfig: {folder_name}", start_opinion_path)

                        # 运行仿真并记录 opinion 历史轨迹
                        trajectory = run_simulation_with_trajectory(sim, steps=500)

                        # 保存 opinion 随时间变化的轨迹图
                        trajectory_path = os.path.join(folder_path, "opinion_trajectory.png")
                        draw_opinion_trajectory(trajectory, f"Opinion Trajectories\nConfig: {folder_name}", trajectory_path)

                        # 保存结束状态下的 opinion 网络图
                        end_opinion_path = os.path.join(folder_path, "end_opinion.png")
                        draw_opinion_network(sim, f"Ending Opinion Network\nConfig: {folder_name}", end_opinion_path)

                        # 保存结束状态下的 identity 网络图
                        end_identity_path = os.path.join(folder_path, "end_identity.png")
                        draw_identity_network(sim, f"Ending Identity Network\nConfig: {folder_name}", end_identity_path)

                        # 保存结束状态下的 morality 网络图
                        end_morality_path = os.path.join(folder_path, "end_morality.png")
                        draw_morality_network(sim, f"Ending Morality Network\nConfig: {folder_name}", end_morality_path)

if __name__ == "__main__":
    batch_test()