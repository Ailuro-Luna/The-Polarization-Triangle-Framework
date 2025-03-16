# visualization.py
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def draw_network(sim, mode, title, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    if mode == "opinion":
        cmap = cm.coolwarm
        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        node_colors = [cmap(norm(op)) for op in sim.opinions]
        nx.draw(sim.G, pos=sim.pos, node_color=node_colors,
                with_labels=False, node_size=20, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Opinion")
    elif mode == "identity":
        node_colors = ['#e41a1c' if iden == 1 else '#377eb8' for iden in sim.identities]
        nx.draw(sim.G, pos=sim.pos, node_color=node_colors,
                with_labels=False, node_size=20, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        patches = [
            Patch(color='#e41a1c', label='Identity: 1'),
            Patch(color='#377eb8', label='Identity: -1')
        ]
        ax.legend(handles=patches, loc='upper right', title="Identity")
    elif mode == "morality":
        node_colors = ['#1a9850' if m == 1 else '#d73027' for m in sim.morals]
        nx.draw(sim.G, pos=sim.pos, node_color=node_colors,
                with_labels=False, node_size=20, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        patches = [
            Patch(color='#1a9850', label='Morality: 1'),
            Patch(color='#d73027', label='Morality: 0')
        ]
        ax.legend(handles=patches, loc='upper right', title="Morality")
    plt.savefig(filename)
    plt.close()

def draw_opinion_distribution(sim, title, filename, bins=20):
    """
    绘制结束时 agent 的 opinion 分布图。
    横轴为 opinion 区间（默认在 -1 到 1 之间），纵轴为落在该区间内的 agent 数量。
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.hist(sim.opinions, bins=bins, range=(-1, 1), color='blue', alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel("Opinion")
    plt.ylabel("Number of Agents")
    plt.savefig(filename)
    plt.close()


def draw_opinion_distribution_heatmap(history, title, filename, bins=50, log_scale=True):
    """
    绘制三维热力图，展示opinion分布随时间的变化。

    参数:
    history -- 包含每个时间步骤所有agent opinions的数组，形状为(time_steps, n_agents)
    title -- 图表标题
    filename -- 保存文件名
    bins -- opinion值的分箱数量
    log_scale -- 是否使用对数比例表示颜色，对于凸显小峰值很有用
    """
    # 转换为numpy数组确保兼容性
    history = np.array(history)

    # 获取时间步数和智能体数量
    time_steps, n_agents = history.shape

    # 创建opinion的bins
    opinion_bins = np.linspace(-1, 1, bins + 1)

    # 初始化热力图数据矩阵
    heatmap_data = np.zeros((time_steps, bins))

    # 对每个时间步骤计算opinion分布
    for t in range(time_steps):
        # 计算每个bin中的agent数量
        hist, _ = np.histogram(history[t], bins=opinion_bins, range=(-1, 1))
        heatmap_data[t] = hist

    # 创建绘图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 创建坐标
    x = opinion_bins[:-1] + np.diff(opinion_bins) / 2  # opinion值（bin中点）
    y = np.arange(time_steps)  # 时间步骤

    # 绘制热力图
    if log_scale:
        # 使用对数比例，先将0值替换为1（或最小非零值）以避免log(0)错误
        min_nonzero = np.min(heatmap_data[heatmap_data > 0]) if np.any(heatmap_data > 0) else 1
        log_data = np.copy(heatmap_data)
        log_data[log_data == 0] = min_nonzero
        pcm = ax.pcolormesh(x, y, log_data, norm=LogNorm(), cmap='viridis', shading='auto')
    else:
        pcm = ax.pcolormesh(x, y, heatmap_data, cmap='viridis', shading='auto')

    # 添加颜色条
    cbar = fig.colorbar(pcm, ax=ax, label='Agent Count')

    # 设置标签和标题
    ax.set_xlabel('Opinion Value')
    ax.set_ylabel('Time Step')
    ax.set_title(title)

    # 优化Y轴刻度，防止过密
    max_ticks = 10
    step = max(1, time_steps // max_ticks)
    ax.set_yticks(np.arange(0, time_steps, step))

    # 保存图表
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    # 额外创建一个瀑布图版本，提供不同视角
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 选择一些时间步骤来展示（避免过度拥挤）
    step = max(1, time_steps // 20)
    selected_timesteps = np.arange(0, time_steps, step)

    # 为3D图准备数据
    X, Y = np.meshgrid(x, selected_timesteps)
    selected_data = heatmap_data[selected_timesteps]

    # 绘制3D表面
    surf = ax.plot_surface(X, Y, selected_data, cmap='viridis', edgecolor='none', alpha=0.8)

    # 设置标签和标题
    ax.set_xlabel('Opinion Value')
    ax.set_ylabel('Time Step')
    ax.set_zlabel('Agent Count')
    ax.set_title(f"{title} - 3D View")

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Agent Count')

    # 保存3D图
    waterfall_filename = filename.replace('.png', '_3d.png')
    plt.tight_layout()
    plt.savefig(waterfall_filename, dpi=300)
    plt.close()

def draw_rule_usage(rule_counts_history, title, filename, smooth=False, window_size=5):
    """
    绘制规则使用频率随时间的变化图。
    
    参数:
    rule_counts_history -- 包含每个时间步骤规则使用次数的列表，形状为(time_steps, 16)
    title -- 图表标题
    filename -- 保存文件名
    smooth -- 是否平滑曲线
    window_size -- 平滑窗口大小
    """
    # 转换为numpy数组
    rule_counts = np.array(rule_counts_history)
    
    # 检查数组是否为空
    if rule_counts.size == 0:
        print("警告: 规则使用历史记录为空")
        return
    
    # 获取时间步数和规则数量
    time_steps = rule_counts.shape[0]
    num_rules = rule_counts.shape[1] if len(rule_counts.shape) > 1 else 0
    
    # 如果规则数量不符合预期，打印警告
    if num_rules != 16 and num_rules != 8:
        print(f"警告: 规则数量({num_rules})不是预期的8或16")
    
    # 准备时间轴
    time = np.arange(time_steps)
    
    # 规则名称 - 现在是16种规则
    rule_names = [
        "Rule 1: Same dir, Same ID, {0,0}, High Convergence",
        "Rule 2: Same dir, Same ID, {0,1}, Medium Pull",
        "Rule 3: Same dir, Same ID, {1,0}, Medium Pull",
        "Rule 4: Same dir, Same ID, {1,1}, High Polarization",
        "Rule 5: Same dir, Diff ID, {0,0}, Medium Convergence",
        "Rule 6: Same dir, Diff ID, {0,1}, Low Pull",
        "Rule 7: Same dir, Diff ID, {1,0}, Low Pull",
        "Rule 8: Same dir, Diff ID, {1,1}, Medium Polarization",
        "Rule 9: Diff dir, Same ID, {0,0}, Very High Convergence",
        "Rule 10: Diff dir, Same ID, {0,1}, Medium Convergence/Pull",
        "Rule 11: Diff dir, Same ID, {1,0}, Low Resistance",
        "Rule 12: Diff dir, Same ID, {1,1}, Low Polarization",
        "Rule 13: Diff dir, Diff ID, {0,0}, Low Convergence",
        "Rule 14: Diff dir, Diff ID, {0,1}, High Pull",
        "Rule 15: Diff dir, Diff ID, {1,0}, High Resistance",
        "Rule 16: Diff dir, Diff ID, {1,1}, Very High Polarization"
    ]
    
    # 如果是旧模式（8种规则），使用旧的规则名称
    if num_rules == 8:
        rule_names = [
            "Rule 1: Same dir, Same ID, Non-moral, Converge",
            "Rule 2: Same dir, Diff ID, Non-moral, Converge",
            "Rule 3: Same dir, Same ID, Moral, Polarize",
            "Rule 4: Same dir, Diff ID, Moral, Polarize",
            "Rule 5: Diff dir, Same ID, Non-moral, Converge",
            "Rule 6: Diff dir, Diff ID, Non-moral, Converge",
            "Rule 7: Diff dir, Same ID, Moral, Converge",
            "Rule 8: Diff dir, Diff ID, Moral, Polarize"
        ]
    
    # 设置颜色 - 为16种规则扩展颜色列表
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94'
    ]
    
    plt.figure(figsize=(14, 10))
    
    # 平滑数据（如果需要）
    if smooth and time_steps > window_size:
        for i in range(num_rules):
            # 使用移动平均进行平滑
            smoothed = np.convolve(rule_counts[:, i], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            # 调整时间轴以匹配平滑后的数据
            smoothed_time = np.arange(len(smoothed)) + window_size // 2
            plt.plot(smoothed_time, smoothed, label=rule_names[i], color=colors[i], linewidth=2)
    else:
        # 不平滑处理
        for i in range(num_rules):
            plt.plot(time, rule_counts[:, i], label=rule_names[i], color=colors[i], linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Rule Application Count')
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # 额外创建一个堆叠面积图，显示规则使用的比例
    plt.figure(figsize=(14, 10))
    
    # 计算每个时间步的规则使用总数
    total_counts = np.sum(rule_counts, axis=1)
    # 避免除以零
    total_counts = np.where(total_counts == 0, 1, total_counts)
    
    # 计算规则使用的比例
    proportions = rule_counts / total_counts[:, np.newaxis]
    
    # 创建堆叠面积图
    plt.stackplot(time, 
                 [proportions[:, i] for i in range(num_rules)],
                 labels=rule_names,
                 colors=colors,
                 alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Rule Application Proportion')
    plt.title(f"{title} - Proportional Usage")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 保存比例图
    proportion_filename = filename.replace('.png', '_proportions.png')
    plt.tight_layout()
    plt.savefig(proportion_filename)
    plt.close()

def draw_rule_cumulative_usage(rule_counts_history, title, filename, smooth=False, window_size=5):
    """
    绘制规则累积使用次数随时间的变化图。
    
    参数:
    rule_counts_history -- 包含每个时间步骤规则使用次数的列表，形状为(time_steps, 16)
    title -- 图表标题
    filename -- 保存文件名
    smooth -- 是否平滑曲线
    window_size -- 平滑窗口大小
    """
    # 转换为numpy数组
    rule_counts = np.array(rule_counts_history)
    
    # 检查数组是否为空
    if rule_counts.size == 0:
        print("警告: 规则使用历史记录为空")
        return
    
    # 获取时间步数和规则数量
    time_steps = rule_counts.shape[0]
    num_rules = rule_counts.shape[1] if len(rule_counts.shape) > 1 else 0
    
    # 如果规则数量不符合预期，打印警告
    if num_rules != 16 and num_rules != 8:
        print(f"警告: 规则数量({num_rules})不是预期的8或16")
    
    # 计算累积次数
    cumulative_counts = np.cumsum(rule_counts, axis=0)
    
    # 准备时间轴
    time = np.arange(time_steps)
    
    # 规则名称 - 现在是16种规则
    rule_names = [
        "Rule 1: Same dir, Same ID, {0,0}, High Convergence",
        "Rule 2: Same dir, Same ID, {0,1}, Medium Pull",
        "Rule 3: Same dir, Same ID, {1,0}, Medium Pull",
        "Rule 4: Same dir, Same ID, {1,1}, High Polarization",
        "Rule 5: Same dir, Diff ID, {0,0}, Medium Convergence",
        "Rule 6: Same dir, Diff ID, {0,1}, Low Pull",
        "Rule 7: Same dir, Diff ID, {1,0}, Low Pull",
        "Rule 8: Same dir, Diff ID, {1,1}, Medium Polarization",
        "Rule 9: Diff dir, Same ID, {0,0}, Very High Convergence",
        "Rule 10: Diff dir, Same ID, {0,1}, Medium Convergence/Pull",
        "Rule 11: Diff dir, Same ID, {1,0}, Low Resistance",
        "Rule 12: Diff dir, Same ID, {1,1}, Low Polarization",
        "Rule 13: Diff dir, Diff ID, {0,0}, Low Convergence",
        "Rule 14: Diff dir, Diff ID, {0,1}, High Pull",
        "Rule 15: Diff dir, Diff ID, {1,0}, High Resistance",
        "Rule 16: Diff dir, Diff ID, {1,1}, Very High Polarization"
    ]
    
    # 如果是旧模式（8种规则），使用旧的规则名称
    if num_rules == 8:
        rule_names = [
            "Rule 1: Same dir, Same ID, Non-moral, Converge",
            "Rule 2: Same dir, Diff ID, Non-moral, Converge",
            "Rule 3: Same dir, Same ID, Moral, Polarize",
            "Rule 4: Same dir, Diff ID, Moral, Polarize",
            "Rule 5: Diff dir, Same ID, Non-moral, Converge",
            "Rule 6: Diff dir, Diff ID, Non-moral, Converge",
            "Rule 7: Diff dir, Same ID, Moral, Converge",
            "Rule 8: Diff dir, Diff ID, Moral, Polarize"
        ]
    
    # 设置颜色 - 为16种规则扩展颜色列表
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
        '#98df8a', '#ff9896', '#c5b0d5', '#c49c94'
    ]
    
    plt.figure(figsize=(14, 10))
    
    # 平滑数据（如果需要）
    if smooth and time_steps > window_size:
        for i in range(num_rules):
            # 使用移动平均进行平滑
            smoothed = np.convolve(cumulative_counts[:, i], 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            # 调整时间轴以匹配平滑后的数据
            smoothed_time = np.arange(len(smoothed)) + window_size // 2
            plt.plot(smoothed_time, smoothed, label=rule_names[i], color=colors[i], linewidth=2)
    else:
        # 不平滑处理
        for i in range(num_rules):
            plt.plot(time, cumulative_counts[:, i], label=rule_names[i], color=colors[i], linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Rule Application Count')
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # 额外创建一个堆叠面积图，显示规则使用的比例
    plt.figure(figsize=(14, 10))
    
    # 创建堆叠面积图
    plt.stackplot(time, 
                 [cumulative_counts[:, i] for i in range(num_rules)],
                 labels=rule_names,
                 colors=colors,
                 alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Rule Application Count')
    plt.title(f"{title} - Stacked View")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.grid(True, alpha=0.3)
    
    # 保存堆叠图
    stacked_filename = filename.replace('.png', '_stacked.png')
    plt.tight_layout()
    plt.savefig(stacked_filename)
    plt.close()