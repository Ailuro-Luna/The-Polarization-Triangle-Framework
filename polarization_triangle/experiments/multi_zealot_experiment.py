import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import random
from polarization_triangle.experiments.zealot_experiment import run_zealot_experiment
from polarization_triangle.visualization.opinion_viz import draw_opinion_distribution_heatmap

def run_multi_zealot_experiment(
    runs=10, 
    steps=500, 
    initial_scale=0.1, 
    morality_rate=0.0,
    zealot_morality=False,
    identity_clustered=False,
    zealot_count=50,
    zealot_mode=None,
    base_seed=42,
    output_dir=None,
    zealot_identity_allocation=True
):
    """
    运行多次zealot实验，并计算平均结果
    
    参数:
    runs -- 运行次数
    steps -- 每次运行的模拟步数
    initial_scale -- 初始意见的缩放因子
    morality_rate -- moralizing的non-zealot people的比例
    zealot_morality -- zealot是否全部moralizing
    identity_clustered -- 是否按identity进行clustered的初始化
    zealot_count -- zealot的总数量
    zealot_mode -- zealot的初始化配置 ("none", "clustered", "random", "high-degree")，若为None则运行所有模式
    base_seed -- 基础随机种子，每次运行会使用不同的种子
    output_dir -- 结果输出目录
    zealot_identity_allocation -- 是否按identity分配zealot，默认启用，启用时zealot只分配给identity为1的agent
    """
    print(f"Running multi-zealot experiment with {runs} runs...")
    print(f"Parameters: morality_rate={morality_rate}, zealot_morality={zealot_morality}, identity_clustered={identity_clustered}")
    print(f"zealot_count={zealot_count}, zealot_mode={zealot_mode}")
    
    # 创建结果目录
    if output_dir is None:
        # 创建包含参数信息的目录名
        dir_name = f"mor_{morality_rate:.1f}_zm_{'T' if zealot_morality else 'F'}_id_{'C' if identity_clustered else 'R'}"
        if zealot_mode:
            dir_name += f"_zn_{zealot_count}_zm_{zealot_mode}"
        results_dir = f"results/multi_zealot_experiment/{dir_name}"
    else:
        results_dir = output_dir
        
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 为每次运行创建单独的子目录
    run_dirs = []
    for i in range(runs):
        run_dir = os.path.join(results_dir, f"run_{i+1}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        run_dirs.append(run_dir)
    
    # 创建平均结果目录
    avg_dir = os.path.join(results_dir, "average_results")
    if not os.path.exists(avg_dir):
        os.makedirs(avg_dir)
        
    # 创建统计子目录
    stats_dir = os.path.join(avg_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    # 运行多次实验
    run_results = []
    
    # 确定要运行的模式
    if zealot_mode is None:
        # 如果未指定模式，运行所有模式
        mode_names = ["without Zealots", "with Clustered Zealots", "with Random Zealots", "with High-Degree Zealots"]
    else:
        # 如果指定了模式，只运行该模式
        if zealot_mode == "none":
            mode_names = ["without Zealots"]
        elif zealot_mode == "clustered":
            mode_names = ["with Clustered Zealots"]
        elif zealot_mode == "random":
            mode_names = ["with Random Zealots"]
        elif zealot_mode == "high-degree":
            mode_names = ["with High-Degree Zealots"]
        else:
            raise ValueError(f"Unknown zealot mode: {zealot_mode}")
    
    # 收集每次运行的统计数据
    all_stats = {mode: [] for mode in mode_names}
    
    # 收集每次运行的意见历史，用于生成平均热图
    all_opinion_histories = {mode: [] for mode in mode_names}
    
    for i in tqdm(range(runs), desc="Running experiments"):
        # 为每次运行使用不同的随机种子
        current_seed = base_seed + i
        
        # 在单独的目录中运行实验，使用新的zealot功能
        print(f"\nRun {i+1}/{runs} with seed {current_seed}")
        result = run_zealot_experiment(
            steps=steps,
            initial_scale=initial_scale,
            num_zealots=zealot_count,
            seed=current_seed,
            output_dir=run_dirs[i],
            morality_rate=morality_rate,
            zealot_morality=zealot_morality,
            identity_clustered=identity_clustered,
            zealot_mode=zealot_mode,
            zealot_identity_allocation=zealot_identity_allocation
        )
        
        # 收集结果
        run_results.append(result)
        
        # 收集统计数据和意见历史
        for mode in mode_names:
            if mode in result:
                all_stats[mode].append(result[mode]["stats"])
                all_opinion_histories[mode].append(result[mode]["opinion_history"])
    
    # 计算平均统计数据
    avg_stats = {}
    for mode in mode_names:
        if all_stats[mode]:  # 确保有数据
            avg_stats[mode] = average_stats(all_stats[mode])
    
    # 绘制平均统计图表
    plot_average_statistics(avg_stats, mode_names, avg_dir, steps)
    
    # 生成平均热图
    generate_average_heatmaps(all_opinion_histories, mode_names, avg_dir)
    
    print(f"\nMulti-zealot experiment completed. Average results saved to {avg_dir}")
    return avg_stats


def average_stats(stats_list):
    """
    计算多次运行的平均统计数据
    
    参数:
    stats_list -- 包含多次运行统计数据的列表
    
    返回:
    dict -- 平均统计数据
    """
    # 初始化结果字典
    avg_stats = {}
    
    # 检查是否有数据
    if not stats_list:
        return avg_stats
    
    # 获取第一个统计数据的键，用于初始化平均值字典
    stat_keys = [
        "mean_opinions", "mean_abs_opinions", "non_zealot_variance", 
        "cluster_variance", "negative_counts", "negative_means", 
        "positive_counts", "positive_means", "polarization_index",
        # 新增identity相关统计
        "identity_1_mean_opinions", "identity_neg1_mean_opinions", "identity_opinion_differences"
    ]
    
    # 初始化每个统计数据的数组
    for key in stat_keys:
        if key in stats_list[0]:
            avg_stats[key] = np.zeros_like(stats_list[0][key])
    
    # 计算所有运行的总和
    for stats in stats_list:
        for key in stat_keys:
            if key in stats and key in avg_stats:
                # 确保数组长度一致
                min_length = min(len(stats[key]), len(avg_stats[key]))
                avg_stats[key][:min_length] += np.array(stats[key][:min_length])
    
    # 计算平均值
    n = len(stats_list)
    for key in stat_keys:
        if key in avg_stats:
            avg_stats[key] = avg_stats[key] / n
    
    return avg_stats


def generate_average_heatmaps(all_opinion_histories, mode_names, output_dir, heatmap_config=None):
    """
    生成平均意见分布热图
    
    参数:
    all_opinion_histories -- 包含所有运行的意见历史的字典
    mode_names -- 模式名称列表
    output_dir -- 输出目录
    heatmap_config -- 热力图配置字典，包含颜色映射、尺度等参数
    """
    print("Generating average heatmaps...")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置默认的热力图配置
    default_config = {
        'bins': 160,
        'log_scale': False,
        'cmap': 'viridis',
        'vmin': None,
        'vmax': None,
        'custom_norm': None
    }
    
    # 合并用户提供的配置
    if heatmap_config:
        default_config.update(heatmap_config)
    
    for mode in mode_names:
        # 获取该模式的所有意见历史
        mode_histories = all_opinion_histories[mode]
        
        if not mode_histories:
            continue
        
        # 计算平均意见分布（而不是平均意见轨迹）
        avg_distribution = calculate_average_opinion_distribution(mode_histories, bins=default_config['bins'])

        # start_step = 900
        start_step = 0
        avg_distribution = avg_distribution[start_step:]
        
        # 绘制平均热图
        heatmap_file = os.path.join(output_dir, f"avg_{mode.lower().replace(' ', '_')}_heatmap.png")
        draw_opinion_distribution_heatmap_from_distribution(
            avg_distribution,
            f"Average Opinion Distribution Evolution {mode} (Multiple Runs)",
            heatmap_file,
            bins=default_config['bins'],
            log_scale=default_config['log_scale'],
            cmap=default_config['cmap'],
            vmin=default_config['vmin'],
            vmax=default_config['vmax'],
            custom_norm=default_config['custom_norm'],
            start_step=start_step
        )


def calculate_average_opinion_distribution(opinion_histories, bins=40):
    """
    计算多次运行的平均意见分布直方图
    
    参数:
    opinion_histories -- 包含多次运行意见历史的列表
    bins -- opinion值的分箱数量
    
    返回:
    numpy.ndarray -- 平均分布直方图数据，形状为(time_steps, bins)
    """
    if not opinion_histories:
        return np.array([])
    
    # 获取第一个历史的时间步长
    num_steps = len(opinion_histories[0])
    
    # 创建opinion的bins
    opinion_bins = np.linspace(-1, 1, bins + 1)
    
    # 初始化所有运行的分布数据存储
    all_distributions = np.zeros((len(opinion_histories), num_steps, bins))
    
    # 对每次运行计算分布直方图
    for run_idx, history in enumerate(opinion_histories):
        for step in range(min(num_steps, len(history))):
            # 计算该时间步的opinion分布
            hist, _ = np.histogram(history[step], bins=opinion_bins, range=(-1, 1))
            all_distributions[run_idx, step] = hist
    
    # 计算平均分布
    avg_distribution = np.mean(all_distributions, axis=0)
    
    return avg_distribution


def draw_opinion_distribution_heatmap_from_distribution(distribution_data, title, filename, bins=40, log_scale=True,
                                                       cmap='viridis', vmin=None, vmax=None, custom_norm=None, start_step=0):
    """
    从预计算的分布数据绘制热力图
    
    参数:
    distribution_data -- 分布数据，形状为(time_steps, bins)
    title -- 图表标题
    filename -- 保存文件名
    bins -- opinion值的分箱数量
    log_scale -- 是否使用对数比例表示颜色
    cmap -- 颜色映射方案 ('viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdBu', 'hot', 'jet', etc.)
    vmin -- 颜色尺度的最小值，如果为None则自动确定
    vmax -- 颜色尺度的最大值，如果为None则自动确定
    custom_norm -- 自定义的颜色标准化对象，如果提供则会覆盖log_scale、vmin、vmax
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    # 获取时间步数
    time_steps = distribution_data.shape[0]
    
    # 创建opinion的bins
    opinion_bins = np.linspace(-1, 1, bins + 1)
    
    # 创建绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建坐标
    x = opinion_bins[:-1] + np.diff(opinion_bins) / 2  # opinion值（bin中点）
    y = np.arange(time_steps)  # 时间步骤索引（用于绘图，从0开始）
    
    # 确定颜色标准化
    if custom_norm is not None:
        # 使用自定义标准化
        norm = custom_norm
        plot_data = distribution_data
    elif log_scale:
        # 使用对数比例，先将0值替换为最小非零值以避免log(0)错误
        min_nonzero = np.min(distribution_data[distribution_data > 0]) if np.any(distribution_data > 0) else 1
        log_data = np.copy(distribution_data)
        log_data[log_data == 0] = min_nonzero
        
        # 设置对数标准化的范围
        log_vmin = vmin if vmin is not None else min_nonzero
        log_vmax = vmax if vmax is not None else np.max(log_data)
        norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
        plot_data = log_data
    else:
        # 使用线性比例
        linear_vmin = vmin if vmin is not None else np.min(distribution_data)
        linear_vmax = vmax if vmax is not None else np.max(distribution_data)
        norm = plt.Normalize(vmin=linear_vmin, vmax=linear_vmax)
        plot_data = distribution_data
    
    # 绘制热力图
    pcm = ax.pcolormesh(x, y, plot_data, norm=norm, cmap=cmap, shading='auto')
    
    # 添加颜色条
    cbar = fig.colorbar(pcm, ax=ax, label='Average Agent Count')
    
    # 如果设置了具体的数值范围，可以自定义颜色条刻度
    if vmin is not None and vmax is not None:
        if log_scale and not custom_norm:
            # 对数尺度的刻度
            ticks = []
            current = vmin
            while current <= vmax:
                ticks.append(current)
                current *= 10
            if ticks[-1] < vmax:
                ticks.append(vmax)
            cbar.set_ticks(ticks)
        else:
            # 线性尺度的刻度
            step = (vmax - vmin) / 5
            cbar.set_ticks([vmin + i*step for i in range(6)])
    
    # 设置标签和标题
    ax.set_xlabel('Opinion Value')
    ax.set_ylabel('Time Step')
    ax.set_title(title)
    
    # 优化Y轴刻度，防止过密，但显示真实的时间步骤
    max_ticks = 10
    tick_step = max(1, time_steps // max_ticks)
    tick_positions = np.arange(0, time_steps, tick_step)  # 在图上的位置（从0开始）
    tick_labels = [str(start_step + pos) for pos in tick_positions]  # 显示的真实时间步骤
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    # 额外创建一个3D视图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 选择一些时间步骤来展示（避免过度拥挤）
    step_interval = max(1, time_steps // 20)
    selected_timesteps = np.arange(0, time_steps, step_interval)
    
    # 为3D图准备数据
    X, Y = np.meshgrid(x, selected_timesteps)
    selected_data = plot_data[selected_timesteps]
    
    # 绘制3D表面
    surf = ax.plot_surface(X, Y, selected_data, cmap=cmap, edgecolor='none', alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel('Opinion Value')
    ax.set_ylabel('Time Step')
    ax.set_zlabel('Average Agent Count')
    ax.set_title(f"{title} - 3D View")
    
    # 修复3D图的Y轴刻度显示真实时间步骤
    y_tick_positions = selected_timesteps[::max(1, len(selected_timesteps)//5)]
    y_tick_labels = [str(start_step + pos) for pos in y_tick_positions]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Average Agent Count')
    
    # 保存3D图
    waterfall_filename = filename.replace('.png', '_3d.png')
    plt.tight_layout()
    plt.savefig(waterfall_filename, dpi=300)
    plt.close()


def plot_average_statistics(avg_stats, mode_names, output_dir, steps):
    """
    绘制平均统计图表
    
    参数:
    avg_stats -- 平均统计数据字典
    mode_names -- 模式名称列表
    output_dir -- 输出目录
    steps -- 模拟步数
    """
    # 确保统计目录存在
    stats_dir = os.path.join(output_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    step_values = range(steps)
    
    # 使用不同颜色和线型
    colors = ['blue', 'red', 'green', 'purple']
    linestyles = ['-', '--', '-.', ':']
    
    # 1. 绘制平均意见值对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["mean_opinions"], 
                label=f'{mode} - Mean Opinion', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Opinion')
    plt.title('Average Mean Opinions across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_mean_opinions.png"), dpi=300)
    plt.close()
    
    # 2. 绘制平均绝对意见值对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["mean_abs_opinions"], 
                label=f'{mode} - Mean |Opinion|', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean |Opinion|')
    plt.title('Average Mean Absolute Opinions across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_mean_abs_opinions.png"), dpi=300)
    plt.close()
    
    # 3. 绘制非zealot方差对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["non_zealot_variance"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Variance')
    plt.title('Average Opinion Variance (Excluding Zealots) across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_non_zealot_variance.png"), dpi=300)
    plt.close()
    
    # 4. 绘制社区内部方差对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["cluster_variance"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Intra-Cluster Variance')
    plt.title('Average Mean Opinion Variance(Excluding Zealots) within Clusters across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_cluster_variance.png"), dpi=300)
    plt.close()
    
    # 5. 绘制负面意见数量对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["negative_counts"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.title('Average Negative Opinion Counts across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_negative_counts.png"), dpi=300)
    plt.close()
    
    # 6. 绘制负面意见均值对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["negative_means"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Value')
    plt.title('Average Negative Opinion Means across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_negative_means.png"), dpi=300)
    plt.close()
    
    # 7. 绘制正面意见数量对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["positive_counts"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.title('Average Positive Opinion Counts across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_positive_counts.png"), dpi=300)
    plt.close()
    
    # 8. 绘制正面意见均值对比图
    plt.figure(figsize=(12, 7))
    for i, mode in enumerate(mode_names):
        plt.plot(step_values, avg_stats[mode]["positive_means"], 
                label=f'{mode}', 
                color=colors[i], linestyle='-')
    plt.xlabel('Step')
    plt.ylabel('Mean Value')
    plt.title('Average Positive Opinion Means across Different Simulations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(stats_dir, "avg_positive_means.png"), dpi=300)
    plt.close()
    
    # 9. 绘制极化指数对比图（如果有数据）
    has_polarization_data = all(
        "polarization_index" in avg_stats[mode] and len(avg_stats[mode]["polarization_index"]) > 0 
        for mode in mode_names
    )
    
    if has_polarization_data:
        plt.figure(figsize=(12, 7))
        for i, mode in enumerate(mode_names):
            plt.plot(range(len(avg_stats[mode]["polarization_index"])), 
                    avg_stats[mode]["polarization_index"], 
                    label=f'{mode}', 
                    color=colors[i], linestyle='-')
        plt.xlabel('Step')
        plt.ylabel('Polarization Index')
        plt.title('Average Polarization Index across Different Simulations')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(stats_dir, "avg_polarization_index.png"), dpi=300)
        plt.close()
    
    # 10. 新增：绘制identity平均意见对比图
    has_identity_data = all(
        "identity_1_mean_opinions" in avg_stats[mode] and "identity_neg1_mean_opinions" in avg_stats[mode]
        for mode in mode_names
    )
    
    if has_identity_data:
        # 10a. 两种identity的平均opinion对比图
        plt.figure(figsize=(15, 7))
        for i, mode in enumerate(mode_names):
            # Identity = 1的平均opinion（实线）
            plt.plot(step_values, avg_stats[mode]["identity_1_mean_opinions"], 
                    label=f'{mode} - Identity +1', 
                    color=colors[i], linestyle='-')
            # Identity = -1的平均opinion（虚线）
            plt.plot(step_values, avg_stats[mode]["identity_neg1_mean_opinions"], 
                    label=f'{mode} - Identity -1', 
                    color=colors[i], linestyle='--')
        plt.xlabel('Step')
        plt.ylabel('Mean Opinion')
        plt.title('Average Mean Opinions by Identity across Different Simulations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, "avg_identity_mean_opinions.png"), dpi=300)
        plt.close()
        
        # 10b. identity意见差值绝对值对比图
        plt.figure(figsize=(12, 7))
        for i, mode in enumerate(mode_names):
            # 计算绝对值
            abs_differences = [abs(diff) for diff in avg_stats[mode]["identity_opinion_differences"]]
            plt.plot(step_values, abs_differences, 
                    label=f'{mode}', 
                    color=colors[i], linestyle='-')
        plt.xlabel('Step')
        plt.ylabel('|Mean Opinion Difference|')
        plt.title('Average Absolute Mean Opinion Differences between Identities')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(stats_dir, "avg_identity_opinion_differences_abs.png"), dpi=300)
        plt.close()
    
    # 11. 保存平均统计数据到CSV文件
    stats_csv = os.path.join(stats_dir, "avg_opinion_stats.csv")
    with open(stats_csv, "w") as f:
        # 写入标题行
        f.write("step")
        for mode in mode_names:
            f.write(f",{mode}_mean_opinion,{mode}_mean_abs_opinion,{mode}_non_zealot_variance,{mode}_cluster_variance")
            f.write(f",{mode}_negative_count,{mode}_negative_mean,{mode}_positive_count,{mode}_positive_mean")
            if "polarization_index" in avg_stats[mode] and len(avg_stats[mode]["polarization_index"]) > 0:
                f.write(f",{mode}_polarization_index")
            # 添加identity相关的列
            if "identity_1_mean_opinions" in avg_stats[mode]:
                f.write(f",{mode}_identity_1_mean_opinion,{mode}_identity_neg1_mean_opinion,{mode}_identity_opinion_difference")
        f.write("\n")
        
        # 写入数据
        for step in range(steps):
            f.write(f"{step}")
            for mode in mode_names:
                f.write(f",{avg_stats[mode]['mean_opinions'][step]:.4f}")
                f.write(f",{avg_stats[mode]['mean_abs_opinions'][step]:.4f}")
                f.write(f",{avg_stats[mode]['non_zealot_variance'][step]:.4f}")
                f.write(f",{avg_stats[mode]['cluster_variance'][step]:.4f}")
                f.write(f",{avg_stats[mode]['negative_counts'][step]:.1f}")
                f.write(f",{avg_stats[mode]['negative_means'][step]:.4f}")
                f.write(f",{avg_stats[mode]['positive_counts'][step]:.1f}")
                f.write(f",{avg_stats[mode]['positive_means'][step]:.4f}")
                # 如果有极化指数数据，添加到CSV
                if "polarization_index" in avg_stats[mode] and step < len(avg_stats[mode]["polarization_index"]):
                    f.write(f",{avg_stats[mode]['polarization_index'][step]:.4f}")
                # 添加identity相关数据
                if "identity_1_mean_opinions" in avg_stats[mode] and step < len(avg_stats[mode]["identity_1_mean_opinions"]):
                    f.write(f",{avg_stats[mode]['identity_1_mean_opinions'][step]:.4f}")
                    f.write(f",{avg_stats[mode]['identity_neg1_mean_opinions'][step]:.4f}")
                    f.write(f",{avg_stats[mode]['identity_opinion_differences'][step]:.4f}")
            f.write("\n")


if __name__ == "__main__":
    # 运行多次zealot实验
    run_multi_zealot_experiment(
        runs=10,                  # 运行10次实验
        steps=100,                # 每次运行100步
        initial_scale=0.1,        # 初始意见缩放到10%
        morality_rate=0.2,        # moralizing的比例为20%
        zealot_morality=True,     # zealot全部moralizing
        identity_clustered=True,  # 按identity进行clustered的初始化
        zealot_count=10,          # 10个zealot
        zealot_mode="clustered",  # 使用clustered zealot模式
        base_seed=42              # 基础随机种子
    ) 