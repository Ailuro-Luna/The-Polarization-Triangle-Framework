"""
Zealot and Morality Analysis Experiment

This experiment analyzes the effects of zealot numbers and morality ratios on various system metrics.
It generates two types of plots:
1. X-axis: Number of zealots
2. X-axis: Morality ratio

For each plot type, it generates 4 different Y-axis metrics:
- Mean opinion
- Variance 
- Variance per identity
- Polarization index

Total: 8 plots (2 types × 4 metrics)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import itertools

from polarization_triangle.core.config import SimulationConfig, high_polarization_config
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.analysis.statistics import (
    calculate_mean_opinion,
    calculate_variance_metrics,
    calculate_identity_statistics,
    get_polarization_index
)


def create_config_combinations():
    """
    创建参数组合
    
    Returns:
    dict: 包含两类图的参数组合
    """
    # 基础配置
    base_config = copy.deepcopy(high_polarization_config)
    base_config.steps = 300  # 设置运行步数
    
    combinations = {
        'zealot_numbers': [],  # 图1：x轴为zealot numbers
        'morality_ratios': []  # 图2：x轴为morality ratio
    }
    
    # 图1：x轴为zealot numbers的组合
    # 比较 "clustering zealots or not" 和 morality ratio
    zealot_clustering_options = ['random', 'clustered']
    morality_ratios_for_zealot_plot = [0.0, 0.3]  # 两个不同的morality ratio进行比较
    
    for clustering in zealot_clustering_options:
        for morality_ratio in morality_ratios_for_zealot_plot:
            combo = {
                'zealot_mode': clustering,
                'morality_rate': morality_ratio,
                'zealot_identity_allocation': True,  # 固定为True
                'cluster_identity': False,  # 固定为random identity distribution
                'label': f'{clustering.capitalize()} Zealots, Morality={morality_ratio}',
                'steps': base_config.steps
            }
            combinations['zealot_numbers'].append(combo)
    
    # 图2：x轴为morality ratio的组合
    # 比较 "clustering zealots or not", "zealots aligned with identity", "identity distribution"
    zealot_modes = ['random', 'clustered']
    zealot_identity_alignments = [True, False]  # zealots aligned with identity
    identity_distributions = [False, True]  # random vs clustered identity distribution
    
    # 固定zealot数量为20（中等数量）
    fixed_zealot_count = 20
    
    for zealot_mode in zealot_modes:
        for zealot_identity in zealot_identity_alignments:
            for identity_dist in identity_distributions:
                combo = {
                    'zealot_count': fixed_zealot_count,
                    'zealot_mode': zealot_mode,
                    'zealot_identity_allocation': zealot_identity,
                    'cluster_identity': identity_dist,
                    'label': f'{zealot_mode.capitalize()}, ID-align={zealot_identity}, ID-cluster={identity_dist}',
                    'steps': base_config.steps
                }
                combinations['morality_ratios'].append(combo)
    
    return combinations


def run_single_simulation(config: SimulationConfig, steps: int = 500) -> Dict[str, float]:
    """
    运行单次模拟并获取最终状态的统计指标
    
    Args:
    config: 模拟配置
    steps: 运行步数
    
    Returns:
    dict: 包含各项统计指标的字典
    """
    sim = Simulation(config)
    
    # 运行模拟
    for _ in range(steps):
        sim.step()
    
    # 获取统计指标
    mean_stats = calculate_mean_opinion(sim, exclude_zealots=True)
    variance_stats = calculate_variance_metrics(sim, exclude_zealots=True)
    identity_stats = calculate_identity_statistics(sim, exclude_zealots=True)
    polarization = get_polarization_index(sim)
    
    # 计算variance per identity (身份间方差)
    variance_per_identity = 0.0
    if 'identity_difference' in identity_stats:
        variance_per_identity = identity_stats['identity_difference']['abs_mean_opinion_difference']
    else:
        # 如果没有identity_difference，计算所有身份的方差均值
        identity_variances = []
        for key, values in identity_stats.items():
            if key.startswith('identity_') and key != 'identity_difference':
                identity_variances.append(values['variance'])
        if identity_variances:
            variance_per_identity = np.mean(identity_variances)
    
    return {
        'mean_opinion': mean_stats['mean_opinion'],
        'variance': variance_stats['overall_variance'],
        'variance_per_identity': variance_per_identity,
        'polarization_index': polarization
    }


def run_parameter_sweep(plot_type: str, combination: Dict[str, Any], 
                       x_values: List[float], num_runs: int = 5) -> Dict[str, List[List[float]]]:
    """
    对特定参数组合进行参数扫描
    
    Args:
    plot_type: 'zealot_numbers' 或 'morality_ratios'
    combination: 参数组合字典
    x_values: x轴的取值列表
    num_runs: 每个参数点运行的次数
    
    Returns:
    dict: 包含各指标的数据矩阵 {metric: [runs_for_x1, runs_for_x2, ...]}
    """
    results = {
        'mean_opinion': [],
        'variance': [],
        'variance_per_identity': [],
        'polarization_index': []
    }
    
    base_config = copy.deepcopy(high_polarization_config)
    # base_config.steps = 500
    
    # 设置固定参数
    if plot_type == 'zealot_numbers':
        base_config.morality_rate = combination['morality_rate']
        base_config.zealot_identity_allocation = combination['zealot_identity_allocation']
        base_config.cluster_identity = combination['cluster_identity']
        base_config.enable_zealots = True
        base_config.steps = combination['steps']
    else:  # morality_ratios
        base_config.zealot_count = combination['zealot_count']
        base_config.zealot_mode = combination['zealot_mode']
        base_config.zealot_identity_allocation = combination['zealot_identity_allocation']
        base_config.cluster_identity = combination['cluster_identity']
        base_config.enable_zealots = True
        base_config.steps = combination['steps']
    
    # 对每个x值进行多次运行
    for x_val in tqdm(x_values, desc=f"Running {combination['label']}"):
        runs_data = {
            'mean_opinion': [],
            'variance': [],
            'variance_per_identity': [],
            'polarization_index': []
        }
        
        # 设置当前x值对应的参数
        current_config = copy.deepcopy(base_config)
        if plot_type == 'zealot_numbers':
            current_config.zealot_count = int(x_val)
            current_config.zealot_mode = combination['zealot_mode']
            if x_val == 0:
                current_config.enable_zealots = False
        else:  # morality_ratios
            current_config.morality_rate = x_val / 100.0  # 转换为0-1范围
        
        # 运行多次模拟
        for run in range(num_runs):
            try:
                stats = run_single_simulation(current_config)
                for metric in runs_data.keys():
                    runs_data[metric].append(stats[metric])
            except Exception as e:
                print(f"Warning: Simulation failed for x={x_val}, run={run}: {e}")
                # 使用NaN填充失败的运行
                for metric in runs_data.keys():
                    runs_data[metric].append(np.nan)
        
        # 将当前x值的所有运行结果添加到总结果中
        for metric in results.keys():
            results[metric].append(runs_data[metric])
    
    return results


def plot_results(plot_type: str, x_values: List[float], all_results: Dict[str, Dict[str, List[List[float]]]], 
                output_dir: str):
    """
    绘制结果图表
    
    Args:
    plot_type: 'zealot_numbers' 或 'morality_ratios'
    x_values: x轴取值
    all_results: 所有组合的结果数据
    output_dir: 输出目录
    """
    metrics = ['mean_opinion', 'variance', 'variance_per_identity', 'polarization_index']
    metric_labels = {
        'mean_opinion': 'Mean Opinion',
        'variance': 'Opinion Variance',
        'variance_per_identity': 'Variance per Identity',
        'polarization_index': 'Polarization Index'
    }
    
    x_label = 'Number of Zealots' if plot_type == 'zealot_numbers' else 'Morality Ratio (%)'
    
    # 为每个指标创建一个图
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # 为每个参数组合绘制曲线
        for combo_label, results in all_results.items():
            metric_data = results[metric]  # List[List[float]], 每个内层list是一个x值的多次运行结果
            
            means = []
            stds = []
            
            for x_runs in metric_data:
                # 计算均值和标准差，忽略NaN值
                valid_runs = [val for val in x_runs if not np.isnan(val)]
                if valid_runs:
                    means.append(np.mean(valid_runs))
                    stds.append(np.std(valid_runs))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            means = np.array(means)
            stds = np.array(stds)
            
            # 绘制带误差条的曲线
            plt.errorbar(x_values, means, yerr=stds, label=combo_label, 
                        marker='o', linewidth=2, capsize=3, alpha=0.8)
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(metric_labels[metric], fontsize=12)
        plt.title(f'{metric_labels[metric]} vs {x_label}', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        filename = f"{plot_type}_{metric}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filepath}")


def save_raw_data(plot_type: str, x_values: List[float], 
                 all_results: Dict[str, Dict[str, List[List[float]]]], 
                 output_dir: str):
    """
    保存原始数据到CSV文件
    
    Args:
    plot_type: 'zealot_numbers' 或 'morality_ratios'
    x_values: x轴取值
    all_results: 所有组合的结果数据
    output_dir: 输出目录
    """
    # 为每个参数组合保存数据
    for combo_label, results in all_results.items():
        # 创建数据框
        data_rows = []
        
        for i, x_val in enumerate(x_values):
            for metric in ['mean_opinion', 'variance', 'variance_per_identity', 'polarization_index']:
                for run_idx, value in enumerate(results[metric][i]):
                    data_rows.append({
                        'x_value': x_val,
                        'metric': metric,
                        'run': run_idx,
                        'value': value,
                        'combination': combo_label
                    })
        
        df = pd.DataFrame(data_rows)
        
        # 保存到CSV
        safe_label = combo_label.replace('/', '_').replace(' ', '_').replace('=', '_').replace(',', '_')
        filename = f"{plot_type}_{safe_label}_raw_data.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved raw data: {filepath}")


def run_zealot_morality_analysis(output_dir: str = "results/zealot_morality_analysis", 
                                num_runs: int = 5, max_zealots: int = 50, max_morality: int = 30):
    """
    运行zealot和morality分析实验
    
    Args:
    output_dir: 输出目录
    num_runs: 每个参数点的运行次数
    max_zealots: 最大zealot数量
    max_morality: 最大morality ratio (%)
    """
    print("🔬 Starting Zealot and Morality Analysis Experiment")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取参数组合
    combinations = create_config_combinations()
    
    print(f"📊 Experiment Configuration:")
    print(f"   Number of runs per parameter point: {num_runs}")
    print(f"   Max zealots: {max_zealots}")
    print(f"   Max morality ratio: {max_morality}%")
    print(f"   Output directory: {output_dir}")
    print()
    
    # === 图1：x轴为zealot numbers ===
    print("📈 Generating Plot Type 1: Zealot Numbers Analysis")
    print("-" * 50)
    
    plot1_start_time = time.time()
    
    zealot_x_values = list(range(0, max_zealots + 1, 2))  # 0, 2, 4, ..., 50
    zealot_results = {}
    
    for combo in combinations['zealot_numbers']:
        print(f"Running combination: {combo['label']}")
        results = run_parameter_sweep('zealot_numbers', combo, zealot_x_values, num_runs)
        zealot_results[combo['label']] = results
    
    # 绘制zealot numbers的图
    plot_results('zealot_numbers', zealot_x_values, zealot_results, output_dir)
    save_raw_data('zealot_numbers', zealot_x_values, zealot_results, output_dir)
    
    plot1_end_time = time.time()
    plot1_duration = plot1_end_time - plot1_start_time
    hours1, remainder1 = divmod(plot1_duration, 3600)
    minutes1, seconds1 = divmod(remainder1, 60)
    
    print(f"⏱️  Plot Type 1 completed in: {int(hours1)}h {int(minutes1)}m {seconds1:.2f}s")
    print()
    
    # === 图2：x轴为morality ratio ===
    print("📈 Generating Plot Type 2: Morality Ratio Analysis")
    print("-" * 50)
    
    plot2_start_time = time.time()
    
    morality_x_values = list(range(0, max_morality + 1, 2))  # 0, 2, 4, ..., 30
    morality_results = {}
    
    for combo in combinations['morality_ratios']:
        print(f"Running combination: {combo['label']}")
        results = run_parameter_sweep('morality_ratios', combo, morality_x_values, num_runs)
        morality_results[combo['label']] = results
    
    # 绘制morality ratio的图
    plot_results('morality_ratios', morality_x_values, morality_results, output_dir)
    save_raw_data('morality_ratios', morality_x_values, morality_results, output_dir)
    
    plot2_end_time = time.time()
    plot2_duration = plot2_end_time - plot2_start_time
    hours2, remainder2 = divmod(plot2_duration, 3600)
    minutes2, seconds2 = divmod(remainder2, 60)
    
    print(f"⏱️  Plot Type 2 completed in: {int(hours2)}h {int(minutes2)}m {seconds2:.2f}s")
    print()
    
    # 计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("🎉 Experiment Completed Successfully!")
    print(f"📊 Generated 8 plots (2 types × 4 metrics)")
    print()
    print("⏱️  Timing Summary:")
    print(f"   Plot Type 1 (Zealot Numbers): {int(hours1)}h {int(minutes1)}m {seconds1:.2f}s")
    print(f"   Plot Type 2 (Morality Ratios): {int(hours2)}h {int(minutes2)}m {seconds2:.2f}s")
    print(f"   Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"📁 Results saved to: {output_dir}")
    
    # 保存实验信息（包含详细的耗时统计）
    info_file = os.path.join(output_dir, "experiment_info.txt")
    with open(info_file, "w") as f:
        f.write("Zealot and Morality Analysis Experiment\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Timing Summary:\n")
        f.write(f"Plot Type 1 (Zealot Numbers): {int(hours1)}h {int(minutes1)}m {seconds1:.2f}s\n")
        f.write(f"Plot Type 2 (Morality Ratios): {int(hours2)}h {int(minutes2)}m {seconds2:.2f}s\n")
        f.write(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n\n")
        
        f.write("Configuration:\n")
        f.write(f"Number of runs per parameter point: {num_runs}\n")
        f.write(f"Max zealots: {max_zealots}\n")
        f.write(f"Max morality ratio: {max_morality}%\n\n")
        
        f.write("Plot Type 1 - Zealot Numbers Analysis:\n")
        for combo in combinations['zealot_numbers']:
            f.write(f"  - {combo['label']}\n")
        
        f.write("\nPlot Type 2 - Morality Ratio Analysis:\n")
        for combo in combinations['morality_ratios']:
            f.write(f"  - {combo['label']}\n")
        
        f.write(f"\nGenerated plots:\n")
        for plot_type in ['zealot_numbers', 'morality_ratios']:
            for metric in ['mean_opinion', 'variance', 'variance_per_identity', 'polarization_index']:
                f.write(f"  - {plot_type}_{metric}.png\n")
        
        # 添加性能统计
        f.write(f"\nPerformance Statistics:\n")
        f.write(f"Average time per zealot combination: {plot1_duration/len(combinations['zealot_numbers']):.2f}s\n")
        f.write(f"Average time per morality combination: {plot2_duration/len(combinations['morality_ratios']):.2f}s\n")
        f.write(f"Total parameter points processed: {len(zealot_x_values) * len(combinations['zealot_numbers']) + len(morality_x_values) * len(combinations['morality_ratios'])}\n")
        f.write(f"Average time per parameter point: {elapsed_time/(len(zealot_x_values) * len(combinations['zealot_numbers']) + len(morality_x_values) * len(combinations['morality_ratios'])):.2f}s\n")


if __name__ == "__main__":
    # 运行实验
    run_zealot_morality_analysis(
        output_dir="results/zealot_morality_analysis",
        num_runs=20,  # 可以调整运行次数以平衡速度和精度
        max_zealots=30,  # 可以调整最大zealot数量
        max_morality=30  # 可以调整最大morality ratio
    ) 