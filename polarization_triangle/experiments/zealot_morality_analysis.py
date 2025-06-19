"""
Zealot and Morality Analysis Experiment

This experiment analyzes the effects of zealot numbers and morality ratios on various system metrics.
It generates two types of plots:
1. X-axis: Number of zealots
2. X-axis: Morality ratio

For each plot type, it generates 4 different Y-axis metrics:
- Mean opinion
- Variance 
- Identity opinion difference (between identity groups)
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
from glob import glob

from polarization_triangle.core.config import SimulationConfig, high_polarization_config
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.analysis.statistics import (
    calculate_mean_opinion,
    calculate_variance_metrics,
    calculate_identity_statistics,
    get_polarization_index
)
from polarization_triangle.utils.data_manager import ExperimentDataManager


# =====================================
# 工具函数
# =====================================

def format_duration(duration: float) -> str:
    """
    格式化时间显示
    
    Args:
    duration: 持续时间（秒）
    
    Returns:
    str: 格式化的时间字符串
    """
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"


# 注：save_batch_info 函数已被 ExperimentDataManager 的批次元数据功能替代


# =====================================
# 核心实验逻辑函数
# =====================================

def create_config_combinations():
    """
    创建实验参数组合配置
    
    该函数生成两类实验的所有参数组合：
    
    1. zealot_numbers实验：测试不同zealot数量对系统的影响
       - 变量：zealot数量 (x轴)
       - 固定：zealot身份分配=True, 身份分布=random
       - 比较：zealot分布模式(random/clustered) × morality比例(0.0/0.3) = 4个组合
    
    2. morality_ratios实验：测试不同morality比例对系统的影响
       - 变量：morality比例 (x轴)
       - 固定：zealot数量=20
       - 比较：zealot模式(random/clustered/none) × zealot身份对齐(True/False) × 
               身份分布(random/clustered) = 10个组合
    
    Returns:
        dict: 包含两类实验配置的字典
            - 'zealot_numbers': 4个参数组合，用于zealot数量实验
            - 'morality_ratios': 10个参数组合，用于morality比例实验
    """
    # 基础配置：使用高极化配置作为模板
    base_config = copy.deepcopy(high_polarization_config)
    base_config.steps = 300  # 每次模拟运行300步
    
    # 初始化两类实验的参数组合容器
    combinations = {
        'zealot_numbers': [],   # 实验1：x轴为zealot数量的参数组合
        'morality_ratios': []   # 实验2：x轴为morality比例的参数组合
    }
    
    # ===== 实验1：zealot数量扫描实验 =====
    # 比较zealot分布模式和morality比例对系统的影响
    # 固定参数：zealot身份分配=True, 身份分布=random
    zealot_clustering_options = ['random', 'clustered']  # zealot分布模式：随机分布 vs 聚集分布
    morality_ratios_for_zealot_plot = [0.0, 0.3]  # 两个morality水平：无道德约束 vs 中等道德约束
    
    for clustering in zealot_clustering_options:
        for morality_ratio in morality_ratios_for_zealot_plot:
            combo = {
                'zealot_mode': clustering,                    # zealot分布模式
                'morality_rate': morality_ratio,              # morality约束强度
                'zealot_identity_allocation': True,           # zealot按身份分配（固定）
                'cluster_identity': False,                    # 身份随机分布（固定）
                'label': f'{clustering.capitalize()} Zealots, Morality={morality_ratio}',
                'steps': base_config.steps
            }
            combinations['zealot_numbers'].append(combo)
    
    # ===== 实验2：morality比例扫描实验 =====
    # 比较三个关键因素的交互影响：zealot分布、zealot身份对齐、身份分布
    # 固定参数：zealot数量=20（中等水平）
    zealot_modes = ['random', 'clustered', 'none']     # zealot模式：随机/聚集/无zealot
    zealot_identity_alignments = [True, False]         # zealot是否按身份分配
    identity_distributions = [False, True]             # 身份分布：随机 vs 聚集
    
    # 固定zealot数量为20，这是一个中等水平，既不会过度影响系统，也能观察到效果
    fixed_zealot_count = 20
    
    for zealot_mode in zealot_modes:
        if zealot_mode == 'none':
            # 无zealot情况：只需要区分身份分布方式，zealot相关参数无意义
            for identity_dist in identity_distributions:
                combo = {
                    'zealot_count': 0,                           # 无zealot
                    'zealot_mode': zealot_mode,                  # 标记为'none'
                    'zealot_identity_allocation': True,          # 默认值（不影响结果）
                    'cluster_identity': identity_dist,           # 身份分布方式
                    'label': f'{zealot_mode.capitalize()}, ID-cluster={identity_dist}',
                    'steps': base_config.steps
                }
                combinations['morality_ratios'].append(combo)
        else:
            # 有zealot情况：需要考虑zealot身份对齐方式和身份分布方式的组合效应
            for zealot_identity in zealot_identity_alignments:
                for identity_dist in identity_distributions:
                    combo = {
                        'zealot_count': fixed_zealot_count,          # 固定zealot数量
                        'zealot_mode': zealot_mode,                  # zealot分布模式
                        'zealot_identity_allocation': zealot_identity,  # zealot身份对齐方式
                        'cluster_identity': identity_dist,           # 身份分布方式
                        'label': f'{zealot_mode.capitalize()}, ID-align={zealot_identity}, ID-cluster={identity_dist}',
                        'steps': base_config.steps
                    }
                    combinations['morality_ratios'].append(combo)
    
    return combinations


def run_single_simulation(config: SimulationConfig, steps: int = 500) -> Dict[str, float]:
    """
    运行单次模拟并获取最终状态的统计指标
    
    该函数创建一个模拟实例，运行指定步数，然后计算四个关键指标：
    - Mean Opinion: 系统中非zealot agent的平均意见值
    - Variance: 意见分布的方差，衡量意见分化程度
    - Identity Opinion Difference: 不同身份群体间的平均意见差异
    - Polarization Index: 极化指数，衡量系统的极化程度
    
    Args:
        config (SimulationConfig): 模拟配置对象，包含网络、agent、zealot等参数
        steps (int, optional): 模拟运行的步数. Defaults to 500.
    
    Returns:
        Dict[str, float]: 包含四个统计指标的字典
            - 'mean_opinion': 平均意见值
            - 'variance': 意见方差
            - 'identity_opinion_difference': 身份间意见差异
            - 'polarization_index': 极化指数
    
    Raises:
        Exception: 当模拟过程中出现错误时抛出异常
    """
    # 创建模拟实例
    sim = Simulation(config)
    
    # 逐步运行模拟至稳定状态
    for _ in range(steps):
        sim.step()
    
    # 从最终状态计算各项统计指标
    mean_stats = calculate_mean_opinion(sim, exclude_zealots=True)
    variance_stats = calculate_variance_metrics(sim, exclude_zealots=True)
    identity_stats = calculate_identity_statistics(sim, exclude_zealots=True)
    polarization = get_polarization_index(sim)
    
    # 计算identity opinion difference (身份间意见差异)
    identity_opinion_difference = 0.0
    if 'identity_difference' in identity_stats:
        identity_opinion_difference = identity_stats['identity_difference']['abs_mean_opinion_difference']
    else:
        # 理论上在正常情况下不应该到达这里（zealot数量足够小时）
        print("Warning: identity_difference not found, this should not happen under normal conditions")
        identity_opinion_difference = 0.0
    
    return {
        'mean_opinion': mean_stats['mean_opinion'],
        'variance': variance_stats['overall_variance'],
        'identity_opinion_difference': identity_opinion_difference,
        'polarization_index': polarization
    }


def run_parameter_sweep(plot_type: str, combination: Dict[str, Any], 
                       x_values: List[float], num_runs: int = 5) -> Dict[str, List[List[float]]]:
    """
    对特定参数组合进行参数扫描实验
    
    该函数针对给定的参数组合，在x轴的每个取值点运行多次模拟，收集统计数据。
    这是实验的核心执行函数，支持两种类型的扫描：
    - zealot_numbers: 固定morality比例，扫描不同的zealot数量
    - morality_ratios: 固定zealot数量，扫描不同的morality比例
    
    Args:
        plot_type (str): 实验类型
            - 'zealot_numbers': x轴为zealot数量的实验
            - 'morality_ratios': x轴为morality比例的实验
        combination (Dict[str, Any]): 参数组合字典，包含：
            - zealot_mode: zealot分布模式 ('random', 'clustered', 'none')
            - morality_rate: morality比例 (0.0-1.0)
            - zealot_identity_allocation: 是否按身份分配zealot
            - cluster_identity: 是否聚类身份分布
            - label: 组合标签
            - steps: 模拟步数
        x_values (List[float]): x轴扫描的取值列表，如 [0, 1, 2, ...]
        num_runs (int, optional): 每个x值点重复运行次数. Defaults to 5.
    
    Returns:
        Dict[str, List[List[float]]]: 嵌套的结果数据结构
            格式: {metric_name: [x1_runs, x2_runs, ...]}
            其中 x1_runs = [run1_value, run2_value, ...]
            
            包含的指标:
            - 'mean_opinion': 平均意见值的多次运行结果
            - 'variance': 意见方差的多次运行结果  
            - 'identity_opinion_difference': 身份间意见差异的多次运行结果
            - 'polarization_index': 极化指数的多次运行结果
    """
    results = {
        'mean_opinion': [],
        'variance': [],
        'identity_opinion_difference': [],
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
        base_config.enable_zealots = combination['zealot_mode'] != 'none'
        base_config.steps = combination['steps']
    
    # 对每个x值进行多次运行
    for x_val in tqdm(x_values, desc=f"Running {combination['label']}"):
        runs_data = {
            'mean_opinion': [],
            'variance': [],
            'identity_opinion_difference': [],
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


# =====================================
# 数据管理函数 (已重构为使用 ExperimentDataManager)
# =====================================

def save_data_with_manager(data_manager: ExperimentDataManager, 
                          plot_type: str, 
                          x_values: List[float], 
                          all_results: Dict[str, Dict[str, List[List[float]]]], 
                          batch_metadata: Dict[str, Any]) -> None:
    """
    使用新的数据管理器保存实验数据
    
    Args:
        data_manager: 数据管理器实例
        plot_type: 'zealot_numbers' 或 'morality_ratios'
        x_values: x轴取值
        all_results: 所有组合的结果数据
        batch_metadata: 批次元数据
    """
    # 转换数据格式以适配新的数据管理器
    batch_data = {}
    
    for combination_label, results in all_results.items():
        batch_data[combination_label] = {
            'x_values': x_values,
            'results': results
        }
    
    # 使用数据管理器保存数据
    data_manager.save_batch_results(plot_type, batch_data, batch_metadata)


# =====================================
# 绘图相关函数
# =====================================

def get_enhanced_style_config(combo_labels: List[str], plot_type: str) -> Dict[str, Dict[str, Any]]:
    """
    为组合标签生成增强的样式配置，特别针对morality_ratios的10条线进行优化
    
    Args:
    combo_labels: 组合标签列表
    plot_type: 图表类型 ('zealot_numbers' 或 'morality_ratios')
    
    Returns:
    dict: 样式配置字典
    """
    # 定义扩展的颜色调色板
    colors = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色  
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 橄榄色
        '#17becf',  # 青色
        '#aec7e8',  # 浅蓝色
        '#ffbb78'   # 浅橙色
    ]
    
    # 定义多种线型
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 3)), (0, (1, 1))]
    
    # 定义多种标记
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', 'X', '+', 'x']
    
    style_config = {}
    
    if plot_type == 'morality_ratios':
        # 定义颜色映射：按zealot模式和ID-align分组
        zealot_mode_colors = {
            'None': {
                'base': '#505050',      # 深灰色 (ID-cluster=True)
                'light': '#c0c0c0'      # 浅灰色 (ID-cluster=False)
            },
            'Random': {
                'base': '#ff4500',      # 深橙红色 (ID-align=True)
                'light': '#ff8080'      # 浅粉红色 (ID-align=False)  
            },
            'Clustered': {
                'base': '#0066cc',      # 深蓝色 (ID-align=True)
                'light': '#00cc66'      # 亮绿色 (ID-align=False)
            }
        }
        
        # 定义标记映射：按ID-cluster分组
        id_cluster_markers = {
            'True': 'o',      # 圆形表示ID-cluster=True
            'False': '^'      # 三角形表示ID-cluster=False
        }
        
        # 定义标记大小映射：按ID-align分组
        id_align_sizes = {
            'True': 10,        # 大标记表示ID-align=True
            'False': 5         # 小标记表示ID-align=False
        }
        
        for label in combo_labels:
            # 解析标签中的配置信息
            if 'None' in label:
                zealot_mode = 'None'
                if 'ID-cluster=True' in label:
                    id_cluster = 'True'
                    color = zealot_mode_colors[zealot_mode]['base']
                    marker = id_cluster_markers[id_cluster]
                    markersize = 8
                else:
                    id_cluster = 'False'
                    color = zealot_mode_colors[zealot_mode]['light']
                    marker = id_cluster_markers[id_cluster]
                    markersize = 8
                
                style_config[label] = {
                    'color': color,
                    'linestyle': '-',
                    'marker': marker,
                    'markersize': markersize,
                    'group': 'None'
                }
                
            elif 'Random' in label:
                zealot_mode = 'Random'
                id_align = 'True' if 'ID-align=True' in label else 'False'
                id_cluster = 'True' if 'ID-cluster=True' in label else 'False'
                
                color = zealot_mode_colors[zealot_mode]['base'] if id_align == 'True' else zealot_mode_colors[zealot_mode]['light']
                marker = id_cluster_markers[id_cluster]
                markersize = id_align_sizes[id_align]
                
                style_config[label] = {
                    'color': color,
                    'linestyle': '-',
                    'marker': marker,
                    'markersize': markersize,
                    'group': 'Random'
                }
                
            elif 'Clustered' in label:
                zealot_mode = 'Clustered'
                id_align = 'True' if 'ID-align=True' in label else 'False'
                id_cluster = 'True' if 'ID-cluster=True' in label else 'False'
                
                color = zealot_mode_colors[zealot_mode]['base'] if id_align == 'True' else zealot_mode_colors[zealot_mode]['light']
                marker = id_cluster_markers[id_cluster]
                markersize = id_align_sizes[id_align]
                
                style_config[label] = {
                    'color': color,
                    'linestyle': '-',
                    'marker': marker,
                    'markersize': markersize,
                    'group': 'Clustered'
                }
    else:
        # 对于zealot_numbers，使用简单配置
        for i, label in enumerate(combo_labels):
            style_config[label] = {
                'color': colors[i % len(colors)],
                'linestyle': linestyles[i % len(linestyles)],
                'marker': markers[i % len(markers)],
                'markersize': 7,
                'group': 'Default'
            }
    
    return style_config


def simplify_label(combo_label: str) -> str:
    """
    简化组合标签（当前保持原始标签以确保完整含义）
    
    Args:
    combo_label: 原始组合标签
    
    Returns:
    str: 简化后的标签
    """
    return combo_label


def plot_results_with_manager(data_manager: ExperimentDataManager, 
                            plot_type: str) -> None:
    """
    使用数据管理器绘制实验结果图表
    
    Args:
        data_manager: 数据管理器实例  
        plot_type: 'zealot_numbers' 或 'morality_ratios'
    """
    # 从数据管理器获取绘图数据
    all_results, x_values, total_runs_per_combination = data_manager.convert_to_plotting_format(plot_type)
    
    if not all_results:
        print(f"❌ No data found for {plot_type} plotting")
        return
    
    output_dir = str(data_manager.base_dir)
    metrics = ['mean_opinion', 'variance', 'identity_opinion_difference', 'polarization_index']
    metric_labels = {
        'mean_opinion': 'Mean Opinion',
        'variance': 'Opinion Variance',
        'identity_opinion_difference': 'Identity Opinion Difference',
        'polarization_index': 'Polarization Index'
    }
    
    x_label = 'Number of Zealots' if plot_type == 'zealot_numbers' else 'Morality Ratio (%)'
    
    # 计算总运行次数范围（用于文件名）
    min_runs = min(total_runs_per_combination.values()) if total_runs_per_combination else 0
    max_runs = max(total_runs_per_combination.values()) if total_runs_per_combination else 0
    
    if min_runs == max_runs:
        runs_suffix = f"_{min_runs}runs"
    else:
        runs_suffix = f"_{min_runs}-{max_runs}runs"
    
    # 创建 mean_plots 文件夹
    plot_folders = {
        'mean': os.path.join(output_dir, 'mean_plots')
    }
    
    os.makedirs(plot_folders['mean'], exist_ok=True)

    # 获取样式配置
    combo_labels = list(all_results.keys())
    style_config = get_enhanced_style_config(combo_labels, plot_type)
    
    print(f"\n📝 Style Configuration for {plot_type}: {len(combo_labels)} combinations")
    print(f"✅ Style configuration completed successfully")
    
    # 为每个指标生成高质量的 mean plots
    for metric in metrics:
        print(f"  Generating high-quality mean plot for {metric_labels[metric]}...")
        
        # 预处理数据：计算均值
        processed_data = {}
        
        for combo_label, results in all_results.items():
            metric_data = results[metric]
            means = []
            
            for i, x_runs in enumerate(metric_data):
                valid_runs = [val for val in x_runs if not np.isnan(val)]
                if valid_runs:
                    means.append(np.mean(valid_runs))
                else:
                    means.append(np.nan)
            
            processed_data[combo_label] = {
                'means': np.array(means)
            }
        
        # 添加运行次数信息到标题（显示总run数）
        title_suffix = f" ({min_runs}-{max_runs} total runs)" if min_runs != max_runs else f" ({min_runs} total runs)"
        
        # 高质量均值曲线图
        plt.figure(figsize=(20, 12) if plot_type == 'morality_ratios' else (18, 10))
        for combo_label, data in processed_data.items():
            runs_info = total_runs_per_combination.get(combo_label, 0)
            short_label = simplify_label(combo_label)
            label_with_runs = f"{short_label} (n={runs_info})"
            
            style = style_config.get(combo_label, {})
            plt.plot(x_values, data['means'], label=label_with_runs, 
                    color=style.get('color', 'blue'),
                    linestyle=style.get('linestyle', '-'),
                    marker=style.get('marker', 'o'), 
                    linewidth=3.5, markersize=style.get('markersize', 10), alpha=0.85,
                    markeredgewidth=2, markeredgecolor='white')
        
        plt.xlabel(x_label, fontsize=16)
        plt.ylabel(metric_labels[metric], fontsize=16)
        plt.title(f'{metric_labels[metric]} vs {x_label}{title_suffix}', fontsize=18, fontweight='bold')
        
        if plot_type == 'morality_ratios':
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, 
                      fontsize=12, frameon=True, fancybox=True, shadow=True)
        else:
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=12)
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filename = f"{plot_type}_{metric}_mean{runs_suffix}.png"
        filepath = os.path.join(plot_folders['mean'], filename)
        
        # 高质量PNG保存 (DPI 300)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='white', 
                   format='png', transparent=False, 
                   pad_inches=0.1, metadata={'Creator': 'Zealot Morality Analysis'})
        
        plt.close()
    
    print(f"  ✅ Generated high-quality mean plots for {plot_type}:")
    print(f"     - Mean line plots: {plot_folders['mean']}")


# =====================================
# 高级接口函数
# =====================================

def run_and_accumulate_data(output_dir: str = "results/zealot_morality_analysis", 
                           num_runs: int = 5, max_zealots: int = 50, max_morality: int = 30,
                           batch_name: str = ""):
    """
    运行测试并使用新的数据管理器保存数据（第一部分）
    
    Args:
    output_dir: 输出目录
    num_runs: 本次运行的次数
    max_zealots: 最大zealot数量
    max_morality: 最大morality ratio (%)
    batch_name: 批次名称，用于标识本次运行
    """
    print("🔬 Running Tests and Accumulating Data with New Data Manager")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建数据管理器
    data_manager = ExperimentDataManager(output_dir)
    
    # 获取参数组合
    combinations = create_config_combinations()
    
    if not batch_name:
        batch_name = time.strftime("%Y%m%d_%H%M%S")
    
    print(f"📊 Batch Configuration:")
    print(f"   Batch name: {batch_name}")
    print(f"   Number of runs this batch: {num_runs}")
    print(f"   Max zealots: {max_zealots}")
    print(f"   Max morality ratio: {max_morality}%")
    print(f"   Output directory: {output_dir}")
    print(f"   Storage format: Parquet (optimized for space and speed)")
    print()
    
    # === 处理图1：x轴为zealot numbers ===
    print("📈 Running Test Type 1: Zealot Numbers Analysis")
    print("-" * 50)
    
    plot1_start_time = time.time()
    
    zealot_x_values = list(range(0, max_zealots + 1, 1))  # 0, 1, 2, ..., n
    zealot_results = {}
    
    for combo in combinations['zealot_numbers']:
        print(f"Running combination: {combo['label']}")
        results = run_parameter_sweep('zealot_numbers', combo, zealot_x_values, num_runs)
        zealot_results[combo['label']] = results
    
    # 使用新的数据管理器保存zealot numbers的数据
    zealot_batch_metadata = {
        'batch_id': batch_name,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_type': 'zealot_numbers',
        'num_runs': num_runs,
        'max_zealots': max_zealots,
        'x_range': [0, max_zealots],
        'combinations_count': len(combinations['zealot_numbers'])
    }
    
    save_data_with_manager(data_manager, 'zealot_numbers', zealot_x_values, zealot_results, zealot_batch_metadata)
    
    plot1_end_time = time.time()
    plot1_duration = plot1_end_time - plot1_start_time
    
    print(f"⏱️  Test Type 1 completed in: {format_duration(plot1_duration)}")
    print()
    
    # === 处理图2：x轴为morality ratio ===
    print("📈 Running Test Type 2: Morality Ratio Analysis")
    print("-" * 50)
    
    plot2_start_time = time.time()
    
    morality_x_values = list(range(0, max_morality + 1, 1))  # 0, 1, 2, ..., n
    morality_results = {}
    
    for combo in combinations['morality_ratios']:
        print(f"Running combination: {combo['label']}")
        results = run_parameter_sweep('morality_ratios', combo, morality_x_values, num_runs)
        morality_results[combo['label']] = results
    
    # 使用新的数据管理器保存morality ratio的数据
    morality_batch_metadata = {
        'batch_id': batch_name,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_type': 'morality_ratios', 
        'num_runs': num_runs,
        'max_morality': max_morality,
        'x_range': [0, max_morality],
        'combinations_count': len(combinations['morality_ratios'])
    }
    
    save_data_with_manager(data_manager, 'morality_ratios', morality_x_values, morality_results, morality_batch_metadata)
    
    plot2_end_time = time.time()
    plot2_duration = plot2_end_time - plot2_start_time
    
    print(f"⏱️  Test Type 2 completed in: {format_duration(plot2_duration)}")
    print()
    
    # 计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("🎉 Data Collection Completed Successfully!")
    print(f"📊 Batch '{batch_name}' with {num_runs} runs per parameter point")
    print()
    print("⏱️  Timing Summary:")
    print(f"   Test Type 1 (Zealot Numbers): {format_duration(plot1_duration)}")
    print(f"   Test Type 2 (Morality Ratios): {format_duration(plot2_duration)}")
    print(f"   Total execution time: {format_duration(elapsed_time)}")
    print(f"📁 Data saved using Parquet format in: {output_dir}/")
    
    # 保存实验配置到数据管理器
    experiment_config = {
        'batch_name': batch_name,
        'num_runs': num_runs,
        'max_zealots': max_zealots,
        'max_morality': max_morality,
        'elapsed_time': elapsed_time,
        'total_combinations': len(combinations['zealot_numbers']) + len(combinations['morality_ratios'])
    }
    data_manager.save_experiment_config(experiment_config)
    
    # 显示数据管理器摘要
    print("\n" + data_manager.export_summary_report())


def plot_from_accumulated_data(output_dir: str = "results/zealot_morality_analysis"):
    """
    从新的数据管理器中读取数据并生成图表（第二部分）
    
    Args:
    output_dir: 输出目录
    """
    print("📊 Generating Plots from Data Manager")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建数据管理器
    data_manager = ExperimentDataManager(output_dir)
    
    # 显示数据摘要
    print("\n" + data_manager.export_summary_report())
    
    # 生成zealot numbers图表
    print("\n📈 Generating Zealot Numbers Plots...")
    zealot_summary = data_manager.get_experiment_summary('zealot_numbers')
    if zealot_summary['total_records'] > 0:
        plot_results_with_manager(data_manager, 'zealot_numbers')
        print(f"✅ Generated {len(zealot_summary['combinations'])} zealot numbers plots")
    else:
        print("❌ No zealot numbers data found")
    
    # 生成morality ratios图表
    print("\n📈 Generating Morality Ratios Plots...")
    morality_summary = data_manager.get_experiment_summary('morality_ratios')
    if morality_summary['total_records'] > 0:
        plot_results_with_manager(data_manager, 'morality_ratios')
        print(f"✅ Generated {len(morality_summary['combinations'])} morality ratios plots")
    else:
        print("❌ No morality ratios data found")
    
    # 计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("🎉 Plot Generation Completed Successfully!")
    print(f"📊 Generated plots from Parquet data files")
    print(f"⏱️  Total plotting time: {format_duration(elapsed_time)}")
    print(f"📁 Plots saved to: {output_dir}/mean_plots/")


def run_zealot_morality_analysis(output_dir: str = "results/zealot_morality_analysis", 
                                num_runs: int = 5, max_zealots: int = 50, max_morality: int = 30):
    """
    运行完整的zealot和morality分析实验（保持向后兼容）
    
    Args:
    output_dir: 输出目录
    num_runs: 每个参数点的运行次数
    max_zealots: 最大zealot数量
    max_morality: 最大morality ratio (%)
    """
    print("🔬 Starting Complete Zealot and Morality Analysis Experiment")
    print("=" * 70)
    
    # 第一步：运行测试并累积数据
    run_and_accumulate_data(output_dir, num_runs, max_zealots, max_morality)
    
    # 第二步：从累积数据生成图表
    plot_from_accumulated_data(output_dir)


def run_no_zealot_morality_data(output_dir: str = "results/zealot_morality_analysis", 
                               num_runs: int = 5, max_morality: int = 30,
                               batch_name: str = ""):
    """
    单独运行 no zealot 的 morality ratio 数据收集（使用新数据管理器）
    
    Args:
    output_dir: 输出目录
    num_runs: 每个参数点的运行次数
    max_morality: 最大 morality ratio (%)
    batch_name: 批次名称
    """
    print("🔬 Running No Zealot Morality Ratio Data Collection with New Data Manager")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建数据管理器
    data_manager = ExperimentDataManager(output_dir)
    
    # 获取所有参数组合
    combinations = create_config_combinations()
    
    # 只选择 zealot_mode 为 'none' 的组合
    no_zealot_combinations = [combo for combo in combinations['morality_ratios'] 
                             if combo['zealot_mode'] == 'none']
    
    if not no_zealot_combinations:
        print("❌ 没有找到 zealot_mode='none' 的组合")
        return
    
    if not batch_name:
        batch_name = f"no_zealot_{time.strftime('%Y%m%d_%H%M%S')}"
    
    print(f"📊 No Zealot Batch Configuration:")
    print(f"   Batch name: {batch_name}")
    print(f"   Number of runs this batch: {num_runs}")
    print(f"   Max morality ratio: {max_morality}%")
    print(f"   Number of no-zealot combinations: {len(no_zealot_combinations)}")
    print(f"   Output directory: {output_dir}")
    print(f"   Storage format: Parquet (optimized for space and speed)")
    print()
    
    # 设置 morality ratio 的 x 轴取值
    morality_x_values = list(range(0, max_morality + 1, 2))  # 0, 2, 4, ..., max_morality
    morality_results = {}
    
    print("📈 Running No Zealot Morality Ratio Analysis")
    print("-" * 50)
    
    for combo in no_zealot_combinations:
        print(f"Running no-zealot combination: {combo['label']}")
        results = run_parameter_sweep('morality_ratios', combo, morality_x_values, num_runs)
        morality_results[combo['label']] = results
    
    # 使用新的数据管理器保存 no zealot morality ratio 数据
    no_zealot_batch_metadata = {
        'batch_id': batch_name,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_type': 'morality_ratios_no_zealot',
        'num_runs': num_runs,
        'max_morality': max_morality,
        'x_range': [0, max_morality],
        'combinations_count': len(no_zealot_combinations),
        'special_conditions': 'no_zealot_only'
    }
    
    save_data_with_manager(data_manager, 'morality_ratios', morality_x_values, morality_results, no_zealot_batch_metadata)
    
    # 计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("🎉 No Zealot Data Collection Completed Successfully!")
    print(f"📊 Batch '{batch_name}' with {num_runs} runs per parameter point")
    print(f"⏱️  Total execution time: {format_duration(elapsed_time)}")
    print(f"📁 Data saved using Parquet format in: {output_dir}/")
    
    # 保存实验配置到数据管理器
    experiment_config = {
        'batch_name': batch_name,
        'num_runs': num_runs,
        'max_morality': max_morality,
        'elapsed_time': elapsed_time,
        'total_combinations': len(no_zealot_combinations),
        'experiment_type': 'no_zealot_only'
    }
    data_manager.save_experiment_config(experiment_config)
    
    # 显示数据管理器摘要
    print("\n" + data_manager.export_summary_report())


if __name__ == "__main__":
    # 新的分离式使用方法：
    
    # 开始计时
    main_start_time = time.time()
    
    # 方法1：分两步运行
    # 第一步：运行测试并积累数据（可以多次运行以积累更多数据）
    print("=" * 50)
    print("🚀 示例：分步骤运行实验")
    print("=" * 50)
    
    # 数据收集阶段
    data_collection_start_time = time.time()
    
    # 可以多次运行以下命令来积累数据：
    run_and_accumulate_data(
        output_dir="results/zealot_morality_analysis",
        num_runs=2,  # 每次运行100轮测试
        max_zealots=2,  
        max_morality=2,
        # batch_name="batch_001"  # 可选：给批次命名
    )
    
    data_collection_end_time = time.time()
    data_collection_duration = data_collection_end_time - data_collection_start_time
    

    # 第二步：绘图阶段

    plotting_start_time = time.time()

    plot_from_accumulated_data("results/zealot_morality_analysis")
    
    plotting_end_time = time.time()
    plotting_duration = plotting_end_time - plotting_start_time
    
    # 计算总耗时
    main_end_time = time.time()
    total_duration = main_end_time - main_start_time
    
    # 显示耗时总结
    print("\n" + "🕒" * 50)
    print("⏱️  完整实验耗时总结")
    print("🕒" * 50)
    # print(f"📊 数据收集阶段耗时: {format_duration(data_collection_duration)}")
    print(f"📈 图表生成阶段耗时: {format_duration(plotting_duration)}")
    print(f"🎯 总耗时: {format_duration(total_duration)}")
    print("🕒" * 50) 