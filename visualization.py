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

def draw_activation_components(sim, title, filename):
    """
    绘制自我激活和社会影响组件的散点图和直方图
    
    参数:
    sim -- 模拟对象，包含自我激活和社会影响数据
    title -- 图表标题
    filename -- 保存文件名
    """
    # 获取最近一步的激活组件
    components = sim.get_activation_components()
    self_activation = components["self_activation"]
    social_influence = components["social_influence"]
    
    # 设置图形尺寸
    plt.figure(figsize=(16, 12))
    
    # 1. 散点图：自我激活 vs 社会影响
    plt.subplot(2, 2, 1)
    plt.scatter(self_activation, social_influence, c=sim.opinions, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Opinion')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Self Activation')
    plt.ylabel('Social Influence')
    plt.title('Self Activation vs Social Influence')
    plt.grid(True, alpha=0.3)
    
    # 2. 散点图：自我激活 vs 社会影响（按身份着色）
    plt.subplot(2, 2, 2)
    colors = ['#e41a1c' if iden == 1 else '#377eb8' for iden in sim.identities]
    plt.scatter(self_activation, social_influence, c=colors, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Self Activation')
    plt.ylabel('Social Influence')
    plt.title('Self Activation vs Social Influence (by Identity)')
    plt.grid(True, alpha=0.3)
    # 添加图例
    patches = [
        Patch(color='#e41a1c', label='Identity: 1'),
        Patch(color='#377eb8', label='Identity: -1')
    ]
    plt.legend(handles=patches)
    
    # 3. 直方图：自我激活值分布
    plt.subplot(2, 2, 3)
    plt.hist(self_activation, bins=30, alpha=0.7, color='green')
    plt.xlabel('Self Activation Value')
    plt.ylabel('Count')
    plt.title('Distribution of Self Activation')
    plt.grid(True, alpha=0.3)
    
    # 4. 直方图：社会影响值分布
    plt.subplot(2, 2, 4)
    plt.hist(social_influence, bins=30, alpha=0.7, color='purple')
    plt.xlabel('Social Influence Value')
    plt.ylabel('Count')
    plt.title('Distribution of Social Influence')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    plt.savefig(filename)
    plt.close()

def draw_activation_history(sim, title, filename):
    """
    绘制自我激活和社会影响随时间的变化
    
    参数:
    sim -- 模拟对象，包含自我激活和社会影响的历史数据
    title -- 图表标题
    filename -- 保存文件名
    """
    # 获取激活历史数据
    history = sim.get_activation_history()
    self_activation_history = history["self_activation_history"]
    social_influence_history = history["social_influence_history"]
    
    # 如果历史数据为空，则返回
    if not self_activation_history or len(self_activation_history) == 0:
        print("警告: 激活历史数据为空")
        return
    
    # 将列表转换为NumPy数组以便计算
    self_activation_array = np.array(self_activation_history)
    social_influence_array = np.array(social_influence_history)
    
    # 计算每个时间步骤的平均值
    self_activation_mean = np.mean(self_activation_array, axis=1)
    social_influence_mean = np.mean(social_influence_array, axis=1)
    
    # 计算每个时间步骤的标准差
    self_activation_std = np.std(self_activation_array, axis=1)
    social_influence_std = np.std(social_influence_array, axis=1)
    
    # 时间步骤
    time_steps = np.arange(len(self_activation_history))
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 绘制自我激活平均值
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, self_activation_mean, 'g-', label='Mean Self Activation')
    # 添加标准差带
    plt.fill_between(time_steps, 
                     self_activation_mean - self_activation_std, 
                     self_activation_mean + self_activation_std, 
                     color='g', alpha=0.2)
    plt.xlabel('Time Step')
    plt.ylabel('Self Activation')
    plt.title('Mean Self Activation Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制社会影响平均值
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, social_influence_mean, 'b-', label='Mean Social Influence')
    # 添加标准差带
    plt.fill_between(time_steps, 
                     social_influence_mean - social_influence_std, 
                     social_influence_mean + social_influence_std, 
                     color='b', alpha=0.2)
    plt.xlabel('Time Step')
    plt.ylabel('Social Influence')
    plt.title('Mean Social Influence Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    plt.savefig(filename)
    plt.close()
    
    # 额外绘制一个组合图，同时显示自我激活和社会影响的变化
    plt.figure(figsize=(14, 8))
    plt.plot(time_steps, self_activation_mean, 'g-', label='Mean Self Activation')
    plt.plot(time_steps, social_influence_mean, 'b-', label='Mean Social Influence')
    plt.xlabel('Time Step')
    plt.ylabel('Activation Value')
    plt.title(f'{title} - Combined View')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存组合图
    combined_filename = filename.replace('.png', '_combined.png')
    plt.savefig(combined_filename)
    plt.close()

def draw_activation_heatmap(sim, title, filename):
    """
    绘制自我激活和社会影响的热力图，显示这些值随意见和身份的分布
    
    参数:
    sim -- 模拟对象，包含自我激活和社会影响数据
    title -- 图表标题
    filename -- 保存文件名
    """
    # 获取最近一步的激活组件
    components = sim.get_activation_components()
    self_activation = components["self_activation"]
    social_influence = components["social_influence"]
    
    # 创建图形
    plt.figure(figsize=(16, 12))
    
    # 1. 自我激活 vs 意见的热力图
    plt.subplot(2, 2, 1)
    hb = plt.hexbin(sim.opinions, self_activation, gridsize=20, cmap='inferno', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion')
    
    # 2. 社会影响 vs 意见的热力图
    plt.subplot(2, 2, 2)
    hb = plt.hexbin(sim.opinions, social_influence, gridsize=20, cmap='inferno', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion')
    
    # 3. 自我激活 + 社会影响 vs 意见的热力图
    plt.subplot(2, 2, 3)
    total_activation = self_activation + social_influence
    hb = plt.hexbin(sim.opinions, total_activation, gridsize=20, cmap='inferno', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Total Activation')
    plt.title('Total Activation vs Opinion')
    
    # 4. 自我激活 vs 社会影响的热力图
    plt.subplot(2, 2, 4)
    hb = plt.hexbin(self_activation, social_influence, gridsize=20, cmap='inferno', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel('Self Activation')
    plt.ylabel('Social Influence')
    plt.title('Self Activation vs Social Influence')
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    plt.savefig(filename)
    plt.close()
    
    # 按身份分开绘制热力图
    plt.figure(figsize=(16, 12))
    
    # 获取不同身份的agent索引
    identity_1_idx = np.where(sim.identities == 1)[0]
    identity_neg1_idx = np.where(sim.identities == -1)[0]
    
    # 1. 自我激活 vs 意见（身份 = 1）
    plt.subplot(2, 2, 1)
    if len(identity_1_idx) > 0:
        hb = plt.hexbin(sim.opinions[identity_1_idx], self_activation[identity_1_idx], 
                        gridsize=20, cmap='Reds', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion (Identity = 1)')
    
    # 2. 社会影响 vs 意见（身份 = 1）
    plt.subplot(2, 2, 2)
    if len(identity_1_idx) > 0:
        hb = plt.hexbin(sim.opinions[identity_1_idx], social_influence[identity_1_idx], 
                        gridsize=20, cmap='Reds', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion (Identity = 1)')
    
    # 3. 自我激活 vs 意见（身份 = -1）
    plt.subplot(2, 2, 3)
    if len(identity_neg1_idx) > 0:
        hb = plt.hexbin(sim.opinions[identity_neg1_idx], self_activation[identity_neg1_idx], 
                        gridsize=20, cmap='Blues', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion (Identity = -1)')
    
    # 4. 社会影响 vs 意见（身份 = -1）
    plt.subplot(2, 2, 4)
    if len(identity_neg1_idx) > 0:
        hb = plt.hexbin(sim.opinions[identity_neg1_idx], social_influence[identity_neg1_idx], 
                        gridsize=20, cmap='Blues', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion (Identity = -1)')
    
    plt.suptitle(f'{title} - By Identity')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    
    # 保存按身份分类的热力图
    identity_filename = filename.replace('.png', '_by_identity.png')
    plt.savefig(identity_filename)
    plt.close()
    
    # 按道德化值分开绘制热力图
    plt.figure(figsize=(16, 12))
    
    # 获取不同道德化值的agent索引
    moral_1_idx = np.where(sim.morals == 1)[0]
    moral_0_idx = np.where(sim.morals == 0)[0]
    
    # 1. 自我激活 vs 意见（道德化 = 1）
    plt.subplot(2, 2, 1)
    if len(moral_1_idx) > 0:
        hb = plt.hexbin(sim.opinions[moral_1_idx], self_activation[moral_1_idx], 
                        gridsize=20, cmap='Greens', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion (Morality = 1)')
    
    # 2. 社会影响 vs 意见（道德化 = 1）
    plt.subplot(2, 2, 2)
    if len(moral_1_idx) > 0:
        hb = plt.hexbin(sim.opinions[moral_1_idx], social_influence[moral_1_idx], 
                        gridsize=20, cmap='Greens', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion (Morality = 1)')
    
    # 3. 自我激活 vs 意见（道德化 = 0）
    plt.subplot(2, 2, 3)
    if len(moral_0_idx) > 0:
        hb = plt.hexbin(sim.opinions[moral_0_idx], self_activation[moral_0_idx], 
                        gridsize=20, cmap='Oranges', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion (Morality = 0)')
    
    # 4. 社会影响 vs 意见（道德化 = 0）
    plt.subplot(2, 2, 4)
    if len(moral_0_idx) > 0:
        hb = plt.hexbin(sim.opinions[moral_0_idx], social_influence[moral_0_idx], 
                        gridsize=20, cmap='Oranges', mincnt=1)
        plt.colorbar(hb, label='Count')
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion (Morality = 0)')
    
    plt.suptitle(f'{title} - By Morality')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    
    # 保存按道德化值分类的热力图
    morality_filename = filename.replace('.png', '_by_morality.png')
    plt.savefig(morality_filename)
    plt.close()

def draw_activation_trajectory(sim, history, title, filename):
    """
    绘制自我激活和社会影响的轨迹图，显示个别agent的激活组件随时间的变化
    
    参数:
    sim -- 模拟对象
    history -- 意见历史数据，用于选择代表性的agent
    title -- 图表标题
    filename -- 保存文件名
    """
    # 获取激活历史数据
    activation_history = sim.get_activation_history()
    self_activation_history = activation_history["self_activation_history"]
    social_influence_history = activation_history["social_influence_history"]
    
    # 如果历史数据为空，则返回
    if not self_activation_history or len(self_activation_history) == 0:
        print("警告: 激活历史数据为空")
        return
    
    # 将list转换为numpy数组
    self_activation_array = np.array(self_activation_history)
    social_influence_array = np.array(social_influence_history)
    
    # 时间步骤
    time_steps = np.arange(len(self_activation_history))
    
    # 选择一些代表性的agent进行可视化
    # 基于最终意见的极值和中值
    final_opinions = np.array(history[-1]) if history and len(history) > 0 else sim.opinions
    
    # 找到意见最极端和最中庸的agent
    most_positive_idx = np.argmax(final_opinions)
    most_negative_idx = np.argmin(final_opinions)
    moderate_idx = np.argmin(np.abs(final_opinions))
    
    # 随机选择一些其他agent
    num_random = 2
    random_indices = np.random.choice(
        [i for i in range(sim.num_agents) if i not in [most_positive_idx, most_negative_idx, moderate_idx]],
        num_random, replace=False)
    
    # 选择的所有agent
    selected_indices = [most_positive_idx, most_negative_idx, moderate_idx] + list(random_indices)
    selected_names = ["Most Positive", "Most Negative", "Most Neutral"] + [f"Random {i+1}" for i in range(num_random)]
    
    # 颜色列表
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    # 创建图形
    plt.figure(figsize=(14, 10))
    
    # 1. 自我激活轨迹
    plt.subplot(2, 1, 1)
    for i, idx in enumerate(selected_indices):
        plt.plot(time_steps, self_activation_array[:, idx], 
                 label=f"{selected_names[i]} (ID={sim.identities[idx]}, M={sim.morals[idx]})",
                 color=colors[i % len(colors)])
    
    plt.xlabel('Time Step')
    plt.ylabel('Self Activation')
    plt.title('Self Activation Trajectories for Selected Agents')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 社会影响轨迹
    plt.subplot(2, 1, 2)
    for i, idx in enumerate(selected_indices):
        plt.plot(time_steps, social_influence_array[:, idx], 
                 label=f"{selected_names[i]} (ID={sim.identities[idx]}, M={sim.morals[idx]})",
                 color=colors[i % len(colors)])
    
    plt.xlabel('Time Step')
    plt.ylabel('Social Influence')
    plt.title('Social Influence Trajectories for Selected Agents')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    plt.savefig(filename)
    plt.close()
    
    # 额外绘制一个对比图，显示意见、自我激活和社会影响的关系
    plt.figure(figsize=(16, 12))
    
    # 确保history列表和激活历史长度一致
    if history and len(history) > 0:
        # 可能需要裁剪历史数据，确保长度匹配
        min_length = min(len(time_steps), len(history))
        history_array = np.array(history[:min_length])
        time_steps_adjusted = np.arange(min_length)
    else:
        history_array = np.zeros((len(time_steps), sim.num_agents))
        time_steps_adjusted = time_steps
    
    for i, idx in enumerate(selected_indices):
        plt.subplot(len(selected_indices), 1, i+1)
        
        if history and len(history) > 0:
            opinions = history_array[:, idx]
        else:
            opinions = np.zeros_like(time_steps_adjusted)
        
        # 裁剪激活数据以匹配长度
        adj_self_activation = self_activation_array[:min_length, idx] if history and len(history) > 0 else self_activation_array[:, idx]
        adj_social_influence = social_influence_array[:min_length, idx] if history and len(history) > 0 else social_influence_array[:, idx]
        
        # 创建三个y轴
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        # 偏移第三个y轴
        ax3.spines['right'].set_position(('outward', 60))
        
        # 绘制数据
        line1, = ax1.plot(time_steps_adjusted, opinions, 'k-', label='Opinion')
        line2, = ax2.plot(time_steps_adjusted, adj_self_activation, 'g-', label='Self Activation')
        line3, = ax3.plot(time_steps_adjusted, adj_social_influence, 'b-', label='Social Influence')
        
        # 设置标签
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Opinion', color='k')
        ax2.set_ylabel('Self Activation', color='g')
        ax3.set_ylabel('Social Influence', color='b')
        
        # 设置颜色
        ax1.tick_params(axis='y', labelcolor='k')
        ax2.tick_params(axis='y', labelcolor='g')
        ax3.tick_params(axis='y', labelcolor='b')
        
        # 添加图例
        lines = [line1, line2, line3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title(f"{selected_names[i]} (ID={sim.identities[idx]}, M={sim.morals[idx]})")
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - Combined Trajectories')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    
    # 保存组合轨迹图
    combined_filename = filename.replace('.png', '_combined.png')
    plt.savefig(combined_filename)
    plt.close()