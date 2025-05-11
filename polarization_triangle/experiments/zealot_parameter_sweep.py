import os
import numpy as np
import itertools
from tqdm import tqdm
import time
from polarization_triangle.experiments.zealot_experiment import run_zealot_experiment
from polarization_triangle.experiments.multi_zealot_experiment import run_multi_zealot_experiment

def run_parameter_sweep(
    runs_per_config=10,
    steps=100,
    initial_scale=0.1,
    base_seed=42,
    output_base_dir="results/zealot_parameter_sweep"
):
    """
    运行参数扫描实验，测试不同参数组合
    
    参数:
    runs_per_config -- 每种参数配置运行的次数
    steps -- 每次运行的模拟步数
    initial_scale -- 初始意见的缩放因子
    base_seed -- 基础随机种子
    output_base_dir -- 结果输出的基础目录
    """
    # 定义参数值范围
    morality_rates = [0.0, 0.2, 0.5]  # moralizing的non-zealot people的比例
    zealot_moralities = [True, False]  # zealot是否全部moralizing
    identity_clustered = [True, False]  # 是否按identity进行clustered的初始化
    zealot_counts = [10, 50]  # zealot的数量
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
    
    # 运行所有参数组合
    for i, params in enumerate(tqdm(param_combinations, desc="Parameter combinations")):
        morality_rate, zealot_morality, id_clustered, zealot_count, zealot_mode = params
        
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
        
        # 输出目录
        output_dir = os.path.join(output_base_dir, folder_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 记录开始时间
        start_time = time.time()
        print(f"\nRunning combination {i+1}/{len(param_combinations)}: {folder_name}")
        
        # 运行多次实验并求均值
        try:
            run_zealot_parameter_experiment(
                runs=runs_per_config,
                steps=steps,
                initial_scale=initial_scale,
                morality_rate=morality_rate,
                zealot_morality=zealot_morality,
                identity_clustered=id_clustered,
                zealot_count=zealot_count,
                zealot_mode=zealot_mode,
                base_seed=base_seed,
                output_dir=output_dir
            )
            
            # 记录结束时间和耗时
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Completed in {elapsed:.1f} seconds")
            
            # 记录进度到日志文件
            with open(os.path.join(output_base_dir, "sweep_progress.log"), "a") as f:
                f.write(f"Completed: {folder_name}, Time: {elapsed:.1f}s\n")
                
        except Exception as e:
            print(f"Error running {folder_name}: {str(e)}")
            # 记录错误到日志文件
            with open(os.path.join(output_base_dir, "sweep_errors.log"), "a") as f:
                f.write(f"Error in {folder_name}: {str(e)}\n")
    
    print("\nParameter sweep completed!")
    

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
    output_dir=None
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
        
        # 在单独的目录中运行实验
        print(f"\nRun {i+1}/{runs} with seed {current_seed}")
        
        # 运行指定的模式
        result = run_zealot_experiment(
            steps=steps,
            initial_scale=initial_scale,
            morality_rate=morality_rate,
            zealot_morality=zealot_morality,
            identity_clustered=identity_clustered,
            num_zealots=zealot_count,
            zealot_mode=zealot_mode,
            seed=current_seed,
            output_dir=run_dirs[i]
        )
        
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
    from polarization_triangle.experiments.multi_zealot_experiment import average_stats, plot_average_statistics, generate_average_heatmaps
    
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
    # 运行参数扫描实验
    run_parameter_sweep(
        runs_per_config=6,  # 每种配置运行6次
        steps=100,           # 每次运行100步
        initial_scale=0.1,   # 初始意见缩放到10%
        base_seed=42         # 基础随机种子
    ) 