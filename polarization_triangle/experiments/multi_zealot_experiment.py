import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import random
from polarization_triangle.experiments.zealot_experiment import run_zealot_experiment

def run_multi_zealot_experiment(runs=10, steps=500, initial_scale=0.1, num_zealots=50, base_seed=42):
    """
    运行多次zealot实验，并计算平均结果
    
    参数:
    runs -- 运行次数
    steps -- 每次运行的模拟步数
    initial_scale -- 初始意见的缩放因子
    num_zealots -- zealot的总数量
    base_seed -- 基础随机种子，每次运行会使用不同的种子
    """
    print(f"Running multi-zealot experiment with {runs} runs...")
    
    # 创建结果目录
    results_dir = "results/multi_zealot_experiment"
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
    
    # 模式名称
    mode_names = ["without Zealots", "with Clustered Zealots", "with Random Zealots", "with High-Degree Zealots"]
    
    # 收集每次运行的统计数据
    all_stats = {mode: [] for mode in mode_names}
    
    for i in tqdm(range(runs), desc="Running experiments"):
        # 为每次运行使用不同的随机种子
        current_seed = base_seed + i
        
        # 在单独的目录中运行实验
        print(f"\nRun {i+1}/{runs} with seed {current_seed}")
        result = run_zealot_experiment(
            steps=steps,
            initial_scale=initial_scale,
            num_zealots=num_zealots,
            seed=current_seed,
            output_dir=run_dirs[i]
        )
        
        # 收集结果
        run_results.append(result)
        
        # 收集统计数据
        for mode in mode_names:
            all_stats[mode].append(result[mode]["stats"])
    
    # 计算平均统计数据
    avg_stats = {}
    for mode in mode_names:
        avg_stats[mode] = average_stats(all_stats[mode])
    
    # 绘制平均统计图表
    plot_average_statistics(avg_stats, mode_names, avg_dir, steps)
    
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
        "positive_counts", "positive_means"
    ]
    
    # 初始化每个统计数据的数组
    for key in stat_keys:
        if key in stats_list[0]:
            avg_stats[key] = np.zeros_like(stats_list[0][key])
    
    # 计算所有运行的总和
    for stats in stats_list:
        for key in stat_keys:
            if key in stats:
                avg_stats[key] += np.array(stats[key])
    
    # 计算平均值
    n = len(stats_list)
    for key in stat_keys:
        if key in avg_stats:
            avg_stats[key] = avg_stats[key] / n
    
    return avg_stats


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
    plt.title('Average Mean Opinion Variance within Clusters across Different Simulations')
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
    
    # 9. 保存平均统计数据到CSV文件
    stats_csv = os.path.join(stats_dir, "avg_opinion_stats.csv")
    with open(stats_csv, "w") as f:
        # 写入标题行
        f.write("step")
        for mode in mode_names:
            f.write(f",{mode}_mean_opinion,{mode}_mean_abs_opinion,{mode}_non_zealot_variance,{mode}_cluster_variance")
            f.write(f",{mode}_negative_count,{mode}_negative_mean,{mode}_positive_count,{mode}_positive_mean")
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
            f.write("\n")


if __name__ == "__main__":
    # 运行多次zealot实验，默认10次
    run_multi_zealot_experiment(
        runs=10,               # 运行10次实验
        steps=100,            # 每次运行100步
        initial_scale=0.1,    # 初始意见缩放到10%
        num_zealots=10,       # 10个zealot
        base_seed=42          # 基础随机种子
    ) 