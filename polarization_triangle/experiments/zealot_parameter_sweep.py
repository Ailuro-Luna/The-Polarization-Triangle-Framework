import os
import numpy as np
import itertools
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from polarization_triangle.experiments.zealot_experiment import run_zealot_experiment
from polarization_triangle.experiments.multi_zealot_experiment import run_multi_zealot_experiment, average_stats, plot_average_statistics, generate_average_heatmaps

def process_single_parameter_combination(params_and_config):
    """
    处理单个参数组合的函数，用于多进程并行
    
    参数:
    params_and_config -- 包含参数组合和配置信息的元组
    
    返回:
    dict -- 包含参数组合名称、统计数据、执行时间等信息
    """
    params, config = params_and_config
    morality_rate, zealot_morality, id_clustered, zealot_count, zealot_mode = params
    runs_per_config, steps, initial_scale, base_seed, output_base_dir = config
    
    # 跳过无效组合：如果zealot_mode为"none"，但zealot_count不为0
    if zealot_mode == "none" and zealot_count != 0:
        zealot_count = 0  # 如果模式是"none"，强制将zealot数量设为0
    
    # 创建参数组合描述的文件夹名
    folder_name = (
        f"mor_{morality_rate:.1f}_"
        f"zm_{'T' if zealot_morality else 'F'}_"
        f"id_{'C' if id_clustered else 'R'}_"
        f"zn_{zealot_count}_"
        f"zm_{zealot_mode}"
    )
    
    # 创建更易读的配置名称用于图表
    mode_display = {
        "none": "No Zealots",
        "clustered": "Clustered",
        "random": "Random",
        "high-degree": "High-Degree"
    }
    readable_name = (
        f"Morality Rate:{morality_rate:.1f};"
        # f"Zealot Morality:{'T' if zealot_morality else 'F'};"
        f"Identity:{'Clustered' if id_clustered else 'Random'};"
        # f"Zealot Count:{zealot_count};"
        f"Zealot Mode:{mode_display.get(zealot_mode, zealot_mode)}"
    )
    
    # 输出目录
    output_dir = os.path.join(output_base_dir, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
        # 记录开始时间
    start_time = time.time()
    process_id = os.getpid()
    print(f"Process {process_id}: Running combination {folder_name}")
    
    # 运行多次实验并求均值
    try:
        # 运行实验，为每个进程使用不同的种子基础
        adjusted_base_seed = base_seed + (process_id % 10000)  # 基于进程ID调整种子
        
        avg_stats = run_zealot_parameter_experiment(
            runs=runs_per_config,
            steps=steps,
            initial_scale=initial_scale,
            morality_rate=morality_rate,
            zealot_morality=zealot_morality,
            identity_clustered=id_clustered,
            zealot_count=zealot_count,
            zealot_mode=zealot_mode,
            base_seed=adjusted_base_seed,
            output_dir=output_dir
        )
        
        # 记录结束时间和耗时
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Process {process_id}: Completed {folder_name} in {elapsed:.1f} seconds")
        
        # 记录进度到日志文件（每个进程使用独立的日志文件）
        log_file = os.path.join(output_base_dir, f"sweep_progress_{process_id}.log")
        with open(log_file, "a") as f:
            f.write(f"Completed: {folder_name}, Time: {elapsed:.1f}s, Process: {process_id}\n")
        
        return {
            'success': True,
            'readable_name': readable_name,
            'avg_stats': avg_stats,
            'elapsed': elapsed,
            'folder_name': folder_name
        }
        
    except Exception as e:
        print(f"Process {process_id}: Error running {folder_name}: {str(e)}")
        # 记录错误到日志文件（每个进程使用独立的日志文件）
        error_log_file = os.path.join(output_base_dir, f"sweep_errors_{process_id}.log")
        with open(error_log_file, "a") as f:
            f.write(f"Error in {folder_name}: {str(e)}, Process: {process_id}\n")
        
        return {
            'success': False,
            'readable_name': readable_name,
            'error': str(e),
            'folder_name': folder_name
        }


# 生成所有可能的参数组合，并运行实验
def run_parameter_sweep(
    runs_per_config=10,
    steps=100,
    initial_scale=0.1,
    base_seed=42,
    output_base_dir="results/zealot_parameter_sweep",
    num_processes=None
):
    """
    运行参数扫描实验，测试不同参数组合（多进程版本）
    
    参数:
    runs_per_config -- 每种参数配置运行的次数
    steps -- 每次运行的模拟步数
    initial_scale -- 初始意见的缩放因子
    base_seed -- 基础随机种子
    output_base_dir -- 结果输出的基础目录
    num_processes -- 使用的进程数量，默认为None（使用CPU核心数）
    """
    # 记录总体开始时间
    total_start_time = time.time()
    
    # 确定进程数量
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Using {num_processes} processes for parallel execution")
    
    # 定义参数值范围
    morality_rates = [0, 0.3]  # moralizing的non-zealot people的比例
    zealot_moralities = [True]  # zealot是否全部moralizing
    identity_clustered = [True, False]  # 是否按identity进行clustered的初始化
    zealot_counts = [20]  # zealot的数量
    zealot_modes = ["none", "clustered", "random", "high-degree"]  # zealot的初始化配置

    # 创建所有可能的参数组合
    param_combinations = list(itertools.product(
        morality_rates, 
        zealot_moralities, 
        identity_clustered, 
        zealot_counts, 
        zealot_modes
    ))
    
    print(f"Total parameter combinations: {len(param_combinations)}")
    print(f"Each combination will be run {runs_per_config} times")
    print(f"Total experiment runs: {len(param_combinations) * runs_per_config}")
    
    # 确保输出基础目录存在
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # 创建综合结果目录
    combined_dir = os.path.join(output_base_dir, "combined_results")
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    # 准备参数组合和配置信息
    config_tuple = (runs_per_config, steps, initial_scale, base_seed, output_base_dir)
    params_and_configs = [(params, config_tuple) for params in param_combinations]
    
    # 收集所有参数组合的平均统计数据
    all_configs_stats = {}
    config_names = []
    
    # 使用多进程并行处理参数组合
    print("\nStarting parallel processing of parameter combinations...")
    
    if num_processes > 1:
        # 多进程版本
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_parameter_combination, params_and_configs),
                total=len(params_and_configs),
                desc="Processing combinations"
            ))
    else:
        # 单进程版本（用于调试）
        results = []
        for params_and_config in tqdm(params_and_configs, desc="Processing combinations"):
            results.append(process_single_parameter_combination(params_and_config))
    
    # 处理结果
    for result in results:
        if result['success']:
            all_configs_stats[result['readable_name']] = result['avg_stats']
            config_names.append(result['readable_name'])
        else:
            print(f"Failed to process combination: {result['folder_name']}")
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # 绘制所有参数组合的综合对比图
    if len(all_configs_stats) > 1:
        print("\nGenerating combined comparative plots for all parameter combinations...")
        plot_combined_statistics(all_configs_stats, config_names, combined_dir, steps)
    
    # 合并各进程的日志文件
    print("\nMerging log files from all processes...")
    
    # 合并进度日志
    progress_log_file = os.path.join(output_base_dir, "sweep_progress.log")
    with open(progress_log_file, "w") as merged_log:
        for file_name in os.listdir(output_base_dir):
            if file_name.startswith("sweep_progress_") and file_name.endswith(".log"):
                process_log_file = os.path.join(output_base_dir, file_name)
                with open(process_log_file, "r") as f:
                    merged_log.write(f.read())
                # 删除进程特定的日志文件
                os.remove(process_log_file)
    
    # 合并错误日志
    error_log_file = os.path.join(output_base_dir, "sweep_errors.log")
    error_entries = []
    for file_name in os.listdir(output_base_dir):
        if file_name.startswith("sweep_errors_") and file_name.endswith(".log"):
            process_error_file = os.path.join(output_base_dir, file_name)
            with open(process_error_file, "r") as f:
                error_entries.append(f.read())
            # 删除进程特定的日志文件
            os.remove(process_error_file)
    
    # 只有存在错误时才创建错误日志文件
    if error_entries and any(entry.strip() for entry in error_entries):
        with open(error_log_file, "w") as merged_error_log:
            for entry in error_entries:
                merged_error_log.write(entry)
    
    # 计算总用时
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\nParameter sweep completed!")
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Processed {len(all_configs_stats)} successful combinations out of {len(param_combinations)} total combinations")
    
    # 记录总用时到日志文件
    with open(os.path.join(output_base_dir, "sweep_summary.log"), "w") as f:
        f.write(f"Parameter Sweep Summary\n")
        f.write(f"======================\n\n")
        f.write(f"Total parameter combinations: {len(param_combinations)}\n")
        f.write(f"Successful combinations: {len(all_configs_stats)}\n")
        f.write(f"Runs per configuration: {runs_per_config}\n")
        f.write(f"Total experiment runs: {len(param_combinations) * runs_per_config}\n")
        f.write(f"Processes used: {num_processes}\n\n")
        f.write(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
        
    return total_elapsed


def plot_combined_statistics(all_configs_stats, config_names, output_dir, steps):
    """
    绘制所有参数组合的综合对比图
    
    参数:
    all_configs_stats -- 包含所有参数组合平均统计数据的字典
    config_names -- 参数组合名称列表
    output_dir -- 输出目录
    steps -- 模拟步数
    """
    # 确保统计目录存在
    stats_dir = os.path.join(output_dir, "statistics")
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    # 统计数据键列表
    stat_keys = [
        ("mean_opinions", "Mean Opinion", "Mean Opinion Value"),
        # ("mean_abs_opinions", "Mean |Opinion|", "Mean Absolute Opinion Value"),  # 删除：不需要的图表
        ("non_zealot_variance", "Non-Zealot Variance", "Opinion Variance (Excluding Zealots)"),
        ("cluster_variance", "Cluster Variance", "Mean Opinion Variance within Clusters"),
        ("negative_counts", "Negative Counts", "Number of Agents with Negative Opinions"),
        ("negative_means", "Negative Means", "Mean Value of Negative Opinions"),
        ("positive_counts", "Positive Counts", "Number of Agents with Positive Opinions"),
        ("positive_means", "Positive Means", "Mean Value of Positive Opinions"),
        ("polarization_index", "Polarization Index", "Polarization Index"),
        # 新增identity相关统计 - 删除单独的identity统计，只保留特殊处理的综合图表
        # ("identity_1_mean_opinions", "Identity +1 Mean Opinion", "Mean Opinion for Identity +1"),  # 删除：不需要的图表
        # ("identity_neg1_mean_opinions", "Identity -1 Mean Opinion", "Mean Opinion for Identity -1"),  # 删除：不需要的图表
        # ("identity_opinion_differences", "Identity Opinion Difference", "Opinion Difference between Identities")  # 删除：不需要的图表
    ]
    
    # 使用不同颜色和线型
    # 确保有足够多的颜色和线型组合
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(config_names))))
    linestyles = ['-', '--', '-.', ':'] * 5  # 重复以确保足够
    
    step_values = range(steps)
    
    # 绘制每种统计数据的综合图
    for stat_key, stat_label, stat_title in stat_keys:
        # 检查是否有这个统计数据
        has_stat_data = False
        for config_name in config_names:
            if config_name in all_configs_stats:
                # 对于zealot_mode为"none"的情况
                if "without Zealots" in all_configs_stats[config_name]:
                    if stat_key in all_configs_stats[config_name]["without Zealots"]:
                        has_stat_data = True
                        break
                # 对于其他模式
                elif len(all_configs_stats[config_name]) > 0:
                    mode_name = list(all_configs_stats[config_name].keys())[0]
                    if stat_key in all_configs_stats[config_name][mode_name]:
                        has_stat_data = True
                        break
        
        if not has_stat_data:
            continue
            
        plt.figure(figsize=(15, 10))
        
        # 为每个参数组合绘制一条线
        for i, config_name in enumerate(config_names):
            if config_name in all_configs_stats:
                # 对于zealot_mode为"none"的情况，我们只有"without Zealots"模式
                if "without Zealots" in all_configs_stats[config_name]:
                    if stat_key in all_configs_stats[config_name]["without Zealots"]:
                        data = all_configs_stats[config_name]["without Zealots"][stat_key]
                        plt.plot(
                            range(len(data)), 
                            data, 
                            label=config_name,
                            color=colors[i % len(colors)], 
                            linestyle=linestyles[i % len(linestyles)]
                        )
                # 对于其他模式，可以选择使用到的模式
                elif len(all_configs_stats[config_name]) > 0:
                    # 选择第一个可用的模式
                    mode_name = list(all_configs_stats[config_name].keys())[0]
                    if stat_key in all_configs_stats[config_name][mode_name]:
                        data = all_configs_stats[config_name][mode_name][stat_key]
                        plt.plot(
                            range(len(data)), 
                            data, 
                            label=config_name,
                            color=colors[i % len(colors)], 
                            linestyle=linestyles[i % len(linestyles)]
                        )
        
        plt.xlabel('Step')
        plt.ylabel(stat_label)
        plt.title(f'Comparison of {stat_title} across All Parameter Combinations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, f"combined_{stat_key}.png"), dpi=300)
        plt.close()
    
    # 特殊处理：绘制identity综合对比图
    # 检查是否有identity数据
    has_identity_data = False
    for config_name in config_names:
        if config_name in all_configs_stats:
            for mode_key in all_configs_stats[config_name]:
                if "identity_1_mean_opinions" in all_configs_stats[config_name][mode_key]:
                    has_identity_data = True
                    break
            if has_identity_data:
                break
    
    if has_identity_data:
        # 绘制两种identity的平均opinion综合对比图
        plt.figure(figsize=(20, 10))
        for i, config_name in enumerate(config_names):
            if config_name in all_configs_stats:
                mode_key = list(all_configs_stats[config_name].keys())[0]
                if "identity_1_mean_opinions" in all_configs_stats[config_name][mode_key]:
                    # Identity = 1的平均opinion（实线）
                    data_1 = all_configs_stats[config_name][mode_key]["identity_1_mean_opinions"]
                    plt.plot(
                        range(len(data_1)), 
                        data_1, 
                        label=f'{config_name} - Identity +1',
                        color=colors[i % len(colors)], 
                        linestyle='-'
                    )
                    # Identity = -1的平均opinion（虚线）
                    data_neg1 = all_configs_stats[config_name][mode_key]["identity_neg1_mean_opinions"]
                    plt.plot(
                        range(len(data_neg1)), 
                        data_neg1, 
                        label=f'{config_name} - Identity -1',
                        color=colors[i % len(colors)], 
                        linestyle='--'
                    )
        
        plt.xlabel('Step')
        plt.ylabel('Mean Opinion')
        plt.title('Comparison of Mean Opinions by Identity across All Parameter Combinations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, "combined_identity_mean_opinions.png"), dpi=300)
        plt.close()
        
        # 绘制identity意见差值绝对值综合对比图
        plt.figure(figsize=(15, 10))
        for i, config_name in enumerate(config_names):
            if config_name in all_configs_stats:
                mode_key = list(all_configs_stats[config_name].keys())[0]
                if "identity_opinion_differences" in all_configs_stats[config_name][mode_key]:
                    # 计算绝对值
                    differences = all_configs_stats[config_name][mode_key]["identity_opinion_differences"]
                    abs_differences = [abs(diff) for diff in differences]
                    plt.plot(
                        range(len(abs_differences)), 
                        abs_differences, 
                        label=config_name,
                        color=colors[i % len(colors)], 
                        linestyle=linestyles[i % len(linestyles)]
                    )
        
        plt.xlabel('Step')
        plt.ylabel('|Mean Opinion Difference|')
        plt.title('Comparison of Absolute Mean Opinion Differences between Identities')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, "combined_identity_opinion_differences_abs.png"), dpi=300)
        plt.close()
    
    # 保存综合数据到CSV文件
    csv_file = os.path.join(stats_dir, "combined_statistics.csv")
    with open(csv_file, "w") as f:
        # 写入标题行
        f.write("step")
        for config_name in config_names:
            if config_name in all_configs_stats:
                for stat_key, _, _ in stat_keys:
                    # 检查是否有这个统计数据
                    has_stat = False
                    if "without Zealots" in all_configs_stats[config_name]:
                        has_stat = stat_key in all_configs_stats[config_name]["without Zealots"]
                    elif len(all_configs_stats[config_name]) > 0:
                        mode_name = list(all_configs_stats[config_name].keys())[0]
                        has_stat = stat_key in all_configs_stats[config_name][mode_name]
                    
                    if has_stat:
                        f.write(f",{config_name}_{stat_key}")
        f.write("\n")
        
        # 写入数据
        for step in range(steps):
            f.write(f"{step}")
            
            for config_name in config_names:
                if config_name in all_configs_stats:
                    config_stats = all_configs_stats[config_name]
                    
                    # 对于zealot_mode为"none"的情况
                    if "without Zealots" in config_stats:
                        mode_stats = config_stats["without Zealots"]
                        for stat_key, _, _ in stat_keys:
                            if stat_key in mode_stats and step < len(mode_stats[stat_key]):
                                f.write(f",{mode_stats[stat_key][step]:.4f}")
                            elif stat_key in mode_stats:  # 如果存在这个键但步骤超出范围
                                f.write(",0.0000")
                    # 对于其他模式
                    elif len(config_stats) > 0:
                        mode_name = list(config_stats.keys())[0]
                        mode_stats = config_stats[mode_name]
                        for stat_key, _, _ in stat_keys:
                            if stat_key in mode_stats and step < len(mode_stats[stat_key]):
                                f.write(f",{mode_stats[stat_key][step]:.4f}")
                            elif stat_key in mode_stats:  # 如果存在这个键但步骤超出范围
                                f.write(",0.0000")
            
            f.write("\n")
    
    print(f"Combined statistics plots and data saved to {stats_dir}")


def run_zealot_parameter_experiment(
    runs=10,
    steps=100,
    initial_scale=0.1,
    morality_rate=0.0,
    zealot_morality=False,
    identity_clustered=False,
    zealot_count=10,
    zealot_mode="random",
    base_seed=42,
    output_dir=None,
    zealot_identity_allocation=True
):
    """
    运行多次zealot实验，使用指定的参数配置
    
    参数:
    runs -- 运行次数
    steps -- 每次运行的模拟步数
    initial_scale -- 初始意见的缩放因子
    morality_rate -- moralizing的non-zealot people的比例
    zealot_morality -- zealot是否全部moralizing
    identity_clustered -- 是否按identity进行clustered的初始化
    zealot_count -- zealot的数量
    zealot_mode -- zealot的初始化配置
    base_seed -- 基础随机种子
    output_dir -- 结果输出目录
    zealot_identity_allocation -- 是否按identity分配zealot，默认启用，启用时zealot只分配给identity为1的agent
    """
    print(f"Running zealot parameter experiment with parameters:")
    print(f"  - Morality rate: {morality_rate}")
    print(f"  - Zealot morality: {zealot_morality}")
    print(f"  - Identity clustered: {identity_clustered}")
    print(f"  - Zealot count: {zealot_count}")
    print(f"  - Zealot mode: {zealot_mode}")
    print(f"  - Runs: {runs}")
    print(f"  - Steps: {steps}")
    
    # 创建结果目录
    if output_dir is None:
        output_dir = f"results/zealot_parameter_exp_mor{morality_rate}_zm{zealot_morality}_id{identity_clustered}_zn{zealot_count}_zm{zealot_mode}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 为每次运行创建单独的子目录
    run_dirs = []
    for i in range(runs):
        run_dir = os.path.join(output_dir, f"run_{i+1}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        run_dirs.append(run_dir)
    
    # 创建平均结果目录
    avg_dir = os.path.join(output_dir, "average_results")
    if not os.path.exists(avg_dir):
        os.makedirs(avg_dir)
    
    # 运行多次实验
    run_results = []
    
    # 模式名称
    mode_names = ["without Zealots", "with Clustered Zealots", "with Random Zealots", "with High-Degree Zealots"]
    
    # 根据zealot_mode选择要运行的模式
    if zealot_mode == "none":
        # 只运行无zealot模式
        active_mode = "without Zealots"
    elif zealot_mode == "clustered":
        active_mode = "with Clustered Zealots"
    elif zealot_mode == "random":
        active_mode = "with Random Zealots"
    elif zealot_mode == "high-degree":
        active_mode = "with High-Degree Zealots"
    else:
        raise ValueError(f"Unknown zealot mode: {zealot_mode}")
    
    # 收集每次运行的意见历史，用于生成平均热图
    all_opinion_histories = {}
    
    # 收集每次运行的统计数据
    all_stats = {}
    
    for i in tqdm(range(runs), desc="Running experiments"):
        # 为每次运行使用不同的随机种子
        current_seed = base_seed + i
        # 为网络结构使用不同的种子，确保每次运行都有不同的网络
        network_seed = base_seed + i * 1000  # 使用更大的间隔避免种子冲突
        
        # 在单独的目录中运行实验，使用新的内置zealot功能
        print(f"\nRun {i+1}/{runs} with seed {current_seed}, network_seed {network_seed}")
        
        # 添加重试机制，防止LFR网络生成失败
        max_retries = 5
        retry_count = 0
        result = None
        
        while retry_count < max_retries:
            try:
                # 运行指定的模式，使用新的内置zealot功能，并传递网络种子
                result = run_zealot_experiment(
                    steps=steps,
                    initial_scale=initial_scale,
                    morality_rate=morality_rate,
                    zealot_morality=zealot_morality,
                    identity_clustered=identity_clustered,
                    num_zealots=zealot_count,
                    zealot_mode=zealot_mode,
                    seed=current_seed,
                    network_seed=network_seed + retry_count * 100,  # 每次重试使用不同的网络种子
                    output_dir=run_dirs[i],
                    zealot_identity_allocation=zealot_identity_allocation
                )
                break  # 成功则跳出重试循环
            except Exception as e:
                retry_count += 1
                print(f"Attempt {retry_count} failed with error: {str(e)}")
                if retry_count < max_retries:
                    print(f"Retrying with different network seed...")
                else:
                    print(f"All {max_retries} attempts failed. Skipping this run.")
                    break
        
        if result is None:
            print(f"Skipping run {i+1} due to repeated failures.")
            continue
        
        # 收集结果
        run_results.append(result)
        
        # 收集统计数据和意见历史
        for mode_key, mode_data in result.items():
            if mode_key not in all_opinion_histories:
                all_opinion_histories[mode_key] = []
                all_stats[mode_key] = []
            
            all_opinion_histories[mode_key].append(mode_data["opinion_history"])
            all_stats[mode_key].append(mode_data["stats"])
    
    # 计算平均统计数据
    avg_stats = {}
    for mode_key, stats_list in all_stats.items():
        avg_stats[mode_key] = average_stats(stats_list)
    
    # 绘制平均统计图表
    active_mode_names = list(avg_stats.keys())
    plot_average_statistics(avg_stats, active_mode_names, avg_dir, steps)
    
    # 生成平均热图
    generate_average_heatmaps(all_opinion_histories, active_mode_names, avg_dir)
    
    print(f"\nParameter experiment completed. Average results saved to {avg_dir}")
    return avg_stats


if __name__ == "__main__":
    # 运行参数扫描实验（多进程版本）
    # 
    # 性能建议：
    # - num_processes=None: 使用所有CPU核心（推荐）
    # - num_processes=4: 使用4个进程
    # - num_processes=1: 单进程模式（用于调试）
    # 
    # 注意：Windows系统需要确保此代码在 if __name__ == "__main__": 块中运行
    
    run_parameter_sweep(
        runs_per_config=20,  # 每种配置运行10次
        steps=300,           # 每次运行的步数（测试用，生产建议300+）
        initial_scale=0.1,   # 初始意见缩放到10%
        base_seed=42,        # 基础随机种子
        num_processes=None   # 使用所有CPU核心，或者设置为特定数值如4
    ) 