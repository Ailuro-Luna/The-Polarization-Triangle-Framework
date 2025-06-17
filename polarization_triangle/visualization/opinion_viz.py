import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
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


def draw_opinion_distribution_heatmap(history, title, filename, bins=50, log_scale=True, 
                                      cmap='viridis', vmin=None, vmax=None, custom_norm=None):
    """
    绘制三维热力图，展示opinion分布随时间的变化。

    参数:
    history -- 包含每个时间步骤所有agent opinions的数组，形状为(time_steps, n_agents)
    title -- 图表标题
    filename -- 保存文件名
    bins -- opinion值的分箱数量
    log_scale -- 是否使用对数比例表示颜色，对于凸显小峰值很有用
    cmap -- 颜色映射方案 ('viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu', 'hot', 'jet', etc.)
    vmin -- 颜色尺度的最小值，如果为None则自动确定
    vmax -- 颜色尺度的最大值，如果为None则自动确定
    custom_norm -- 自定义的颜色标准化对象，如果提供则会覆盖log_scale、vmin、vmax
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

    # 确定颜色标准化
    if custom_norm is not None:
        # 使用自定义标准化
        norm = custom_norm
    elif log_scale:
        # 使用对数比例，先将0值替换为1（或最小非零值）以避免log(0)错误
        min_nonzero = np.min(heatmap_data[heatmap_data > 0]) if np.any(heatmap_data > 0) else -1
        log_data = np.copy(heatmap_data)
        log_data[log_data == 0] = min_nonzero
        
        # 设置对数标准化的范围
        log_vmin = vmin if vmin is not None else min_nonzero
        log_vmax = vmax if vmax is not None else np.max(log_data)
        norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
        heatmap_data = log_data
    else:
        # 使用线性比例
        linear_vmin = vmin if vmin is not None else np.min(heatmap_data)
        linear_vmax = vmax if vmax is not None else np.max(heatmap_data)
        norm = plt.Normalize(vmin=linear_vmin, vmax=linear_vmax)

    # 绘制热力图
    pcm = ax.pcolormesh(x, y, heatmap_data, norm=norm, cmap=cmap, shading='auto')

    # 添加颜色条
    cbar = fig.colorbar(pcm, ax=ax, label='Agent Count')
    
    # 如果设置了具体的数值范围，可以自定义颜色条刻度
    if vmin is not None and vmax is not None:
        if log_scale and not custom_norm:
            # 对数尺度的刻度
            cbar.set_ticks([vmin, vmin*10, vmin*100, vmax])
        else:
            # 线性尺度的刻度
            step = (vmax - vmin) / 5
            cbar.set_ticks([vmin + i*step for i in range(6)])

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
    surf = ax.plot_surface(X, Y, selected_data, cmap=cmap, edgecolor='none', alpha=0.8)

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

def draw_opinion_trajectory(history, title, filename):
    history = np.array(history)
    total_steps = history.shape[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(total_steps)
    for i in range(history.shape[1]):
        ax.plot(history[:, i], y, color='gray', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Opinion")
    ax.set_ylabel("Time step")
    # ax.set_ylim(-1, 1)
    plt.savefig(filename)
    plt.close()