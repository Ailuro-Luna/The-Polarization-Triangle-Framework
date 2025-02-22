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


# 运行仿真，更新 opinion 状态（默认 500 步，可根据需要调整）
def run_simulation(sim, steps=500):
    for _ in range(steps):
        sim.step()


# 根据 morality_ratio 参数覆盖仿真对象中 morals 的初始值
def override_morality(sim, ratio):
    # ratio 取值："all1" 表示全部为 1；"half" 表示一半为 1 一半为 0；"all0" 表示全部为 0
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
    # 结果保存的基础目录
    base_dir = "batch_results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 遍历身份（Identity）、道德（Morality）、意见（Opinion）初始化的两种模式，共 2*2*2 = 8 种组合
    for id_mode in ["random", "clustered"]:
        for mor_mode in ["random", "clustered"]:
            for op_mode in ["random", "clustered"]:
                # 遍历意见分布类型：均匀(uniform)、单峰(single_peak)、双峰(twin_peak)
                for op_dist in ["uniform", "single_peak", "twin_peak"]:
                    # 遍历道德比例：全部为 1 (all1)、一半为 1 一半为 0 (half)、全部为 0 (all0)
                    for mor_ratio in ["all1", "half", "all0"]:
                        # 构造文件夹名称，含义如下：
                        # 文件夹名称格式:
                        #   ID_{id_mode}_M_{mor_mode}_OP_{op_mode}_op_{op_dist}_mor_{mor_ratio}
                        # 其中：
                        #   - ID_random：身份随机初始化（每个 agent 以 0.5 概率取 1 或 -1）
                        #   - ID_clustered：基于社区聚类的身份初始化
                        #   - M_random：道德随机初始化
                        #   - M_clustered：基于社区聚类的道德初始化
                        #   - OP_random：意见随机初始化
                        #   - OP_clustered：基于社区聚类的意见初始化
                        #   - op_uniform：意见分布为均匀分布
                        #   - op_single_peak：意见分布为单峰分布
                        #   - op_twin_peak：意见分布为双峰分布
                        #   - mor_all1：道德全部为 1
                        #   - mor_half：道德一半为 1，一半为 0
                        #   - mor_all0：道德全部为 0
                        folder_name = f"ID_{id_mode}_M_{mor_mode}_OP_{op_mode}_op_{op_dist}_mor_{mor_ratio}"
                        folder_path = os.path.join(base_dir, folder_name)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)

                        print("Processing configuration:", folder_name)

                        # 基于 model_params_lfr（LFR benchmark 配置）构造参数字典，并覆盖指定参数
                        params = copy.deepcopy(model_params_lfr)
                        params["num_agents"] = 500  # 保持 500 个 agent 的配置
                        # 设置身份初始化方式：cluster_identity True 表示基于社区聚类
                        params["cluster_identity"] = (id_mode == "clustered")
                        # 设置道德初始化方式：cluster_morality True 表示基于社区聚类
                        params["cluster_morality"] = (mor_mode == "clustered")
                        # 设置意见初始化方式：cluster_opinion True 表示基于社区聚类
                        params["cluster_opinion"] = (op_mode == "clustered")
                        # 设置意见分布类型
                        params["opinion_distribution"] = op_dist

                        # 初始化仿真对象
                        sim = Simulation(**params)
                        # 根据当前配置覆盖初始道德值
                        override_morality(sim, mor_ratio)

                        # 保存初始状态下的 opinion 网络图
                        start_opinion_path = os.path.join(folder_path, "start_opinion.png")
                        draw_opinion_network(sim, f"Starting Opinion Network\nConfig: {folder_name}",
                                             start_opinion_path)

                        # 运行仿真（例如 500 步，可根据需要调整步数）
                        run_simulation(sim, steps=500)

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
