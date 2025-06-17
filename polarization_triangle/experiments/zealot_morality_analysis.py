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
    zealot_modes = ['random', 'clustered', 'none']
    zealot_identity_alignments = [True, False]  # zealots aligned with identity
    identity_distributions = [False, True]  # random vs clustered identity distribution
    
    # 固定zealot数量为20（中等数量）
    fixed_zealot_count = 20
    
    for zealot_mode in zealot_modes:
        if zealot_mode == 'none':
            # 对于 none zealot，ID-align 参数没有意义，只区分 identity distribution
            for identity_dist in identity_distributions:
                combo = {
                    'zealot_count': 0,
                    'zealot_mode': zealot_mode,
                    'zealot_identity_allocation': True,  # 设为默认值，但不影响结果
                    'cluster_identity': identity_dist,
                    'label': f'{zealot_mode.capitalize()}, ID-cluster={identity_dist}',
                    'steps': base_config.steps
                }
                combinations['morality_ratios'].append(combo)
        else:
            # 对于有 zealot 的情况，需要区分 ID-align 和 ID-cluster
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
    
    # 计算identity opinion difference (身份间意见差异)
    # 注意：为保持数据兼容性，变量名仍使用variance_per_identity
    variance_per_identity = 0.0
    if 'identity_difference' in identity_stats:
        variance_per_identity = identity_stats['identity_difference']['abs_mean_opinion_difference']
    else:
        # 理论上在正常情况下不应该到达这里（zealot数量足够小时）
        print("Warning: identity_difference not found, this should not happen under normal conditions")
        variance_per_identity = 0.0
    
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
        base_config.enable_zealots = combination['zealot_mode'] != 'none'
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


def save_data_incrementally(plot_type: str, x_values: List[float], 
                           all_results: Dict[str, Dict[str, List[List[float]]]], 
                           output_dir: str, batch_info: str = ""):
    """
    以追加模式保存数据到CSV文件，支持累积多次运行的结果
    
    Args:
    plot_type: 'zealot_numbers' 或 'morality_ratios'
    x_values: x轴取值
    all_results: 所有组合的结果数据
    output_dir: 输出目录
    batch_info: 批次信息，用于标识本次运行
    """
    data_dir = os.path.join(output_dir, "accumulated_data")
    os.makedirs(data_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not batch_info:
        batch_info = timestamp
    
    # 为每个参数组合保存数据
    for combo_label, results in all_results.items():
        # 准备新的数据行
        new_data_rows = []
        
        for i, x_val in enumerate(x_values):
            for metric in ['mean_opinion', 'variance', 'variance_per_identity', 'polarization_index']:
                for run_idx, value in enumerate(results[metric][i]):
                    new_data_rows.append({
                        'x_value': x_val,
                        'metric': metric,
                        'run': run_idx,
                        'value': value,
                        'combination': combo_label,
                        'batch_id': batch_info,
                        'timestamp': timestamp
                    })
        
        new_df = pd.DataFrame(new_data_rows)
        
        # 生成文件名
        safe_label = combo_label.replace('/', '_').replace(' ', '_').replace('=', '_').replace(',', '_')
        filename = f"{plot_type}_{safe_label}_accumulated.csv"
        filepath = os.path.join(data_dir, filename)
        
        # 追加或创建文件
        if os.path.exists(filepath):
            # 文件存在，追加数据
            new_df.to_csv(filepath, mode='a', header=False, index=False)
            print(f"Appended data to: {filepath}")
        else:
            # 文件不存在，创建新文件
            new_df.to_csv(filepath, index=False)
            print(f"Created new data file: {filepath}")


def load_accumulated_data(output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    读取累积的数据文件
    
    Args:
    output_dir: 输出目录
    
    Returns:
    dict: 文件名对应的DataFrame字典
    """
    data_dir = os.path.join(output_dir, "accumulated_data")
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return {}
    
    # 查找所有累积数据文件
    pattern = os.path.join(data_dir, "*_accumulated.csv")
    files = glob(pattern)
    
    if not files:
        print(f"No accumulated data files found in: {data_dir}")
        return {}
    
    loaded_data = {}
    
    print("📂 Loading accumulated data files:")
    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            df = pd.read_csv(filepath)
            loaded_data[filename] = df
            
            # 计算总运行次数（与process_accumulated_data_for_plotting中的计算保持一致）
            total_data_points = len(df)
            unique_x_values = len(df['x_value'].unique()) if not df.empty else 0
            unique_metrics = len(df['metric'].unique()) if not df.empty else 0
            
            # 计算总运行次数：总数据点 / (x值数量 * 指标数量)
            total_runs = total_data_points // (unique_x_values * unique_metrics) if unique_x_values > 0 and unique_metrics > 0 else 0
            
            # 统计批次数（用于参考）
            total_batches = len(df['batch_id'].unique()) if 'batch_id' in df.columns and not df.empty else 0
            
            print(f"  ✅ {filename}: {len(df)} records, {total_runs} total runs ({total_batches} batches)")
        except Exception as e:
            print(f"  ❌ Failed to load {filename}: {e}")
    
    return loaded_data


def process_accumulated_data_for_plotting(loaded_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, Dict[str, List[List[float]]]], List[float], Dict[str, int]]:
    """
    将累积数据处理成绘图所需的格式
    
    Args:
    loaded_data: 已加载的数据字典
    
    Returns:
    tuple: (all_results, x_values, total_runs_per_combination)
    """
    if not loaded_data:
        return {}, [], {}
    
    # 确定plot_type（从文件名推断）
    first_filename = list(loaded_data.keys())[0]
    if first_filename.startswith('zealot_numbers'):
        plot_type = 'zealot_numbers'
    elif first_filename.startswith('morality_ratios'):
        plot_type = 'morality_ratios'
    else:
        print("Warning: Cannot determine plot type from filename")
        plot_type = 'unknown'
    
    all_results = {}
    x_values_set = set()
    total_runs_per_combination = {}
    
    for filename, df in loaded_data.items():
        if df.empty:
            continue
            
        # 提取组合标签（从文件名）
        if plot_type == 'zealot_numbers':
            combo_label = filename.replace('zealot_numbers_', '').replace('_accumulated.csv', '').replace('_', ' ')
        elif plot_type == 'morality_ratios':
            combo_label = filename.replace('morality_ratios_', '').replace('_accumulated.csv', '').replace('_', ' ')
        else:
            combo_label = filename.replace('_accumulated.csv', '')
        
        # 恢复原始标签格式
        combo_label = combo_label.replace('Clustered', 'Clustered').replace('Random', 'Random')
        
        # 统计总运行次数（计算实际的数据点数量，而不是batch数）
        total_data_points = len(df)
        unique_x_values = len(df['x_value'].unique())
        unique_metrics = len(df['metric'].unique())
        
        # 计算总运行次数：总数据点 / (x值数量 * 指标数量)
        total_runs = total_data_points // (unique_x_values * unique_metrics) if unique_x_values > 0 and unique_metrics > 0 else 0
        
        total_runs_per_combination[combo_label] = total_runs
        
        # 收集所有x值
        x_values_set.update(df['x_value'].unique())
        
        # 按组合处理数据
        combo_results = {
            'mean_opinion': [],
            'variance': [],
            'variance_per_identity': [],
            'polarization_index': []
        }
        
        # 获取所有x值并排序
        combo_x_values = sorted(df['x_value'].unique())
        
        for x_val in combo_x_values:
            x_data = df[df['x_value'] == x_val]
            
            for metric in ['mean_opinion', 'variance', 'variance_per_identity', 'polarization_index']:
                metric_data = x_data[x_data['metric'] == metric]['value'].tolist()
                combo_results[metric].append(metric_data)
        
        all_results[combo_label] = combo_results
    
    x_values = sorted(list(x_values_set))
    
    return all_results, x_values, total_runs_per_combination


def plot_accumulated_results(plot_type: str, x_values: List[float], 
                           all_results: Dict[str, Dict[str, List[List[float]]]], 
                           total_runs_per_combination: Dict[str, int],
                           output_dir: str):
    """
    绘制累积数据的结果图表，文件名中包含总运行次数信息
    
    Args:
    plot_type: 'zealot_numbers' 或 'morality_ratios'
    x_values: x轴取值
    all_results: 所有组合的结果数据
    total_runs_per_combination: 每个组合的总运行次数
    output_dir: 输出目录
    """
    metrics = ['mean_opinion', 'variance', 'variance_per_identity', 'polarization_index']
    metric_labels = {
        'mean_opinion': 'Mean Opinion',
        'variance': 'Opinion Variance',
        'variance_per_identity': 'Identity Opinion Difference',
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

    # 增强版样式配置函数（专门为morality_ratios优化）
    def get_enhanced_style_config(combo_labels):
        """
        为组合标签生成增强的样式配置，特别针对morality_ratios的10条线进行优化
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
                'False': '^'      # 三角形表示ID-cluster=False (改为更明显的形状)
            }
            
            # 定义标记大小映射：按ID-align分组
            id_align_sizes = {
                'True': 10,        # 大标记表示ID-align=True (增大差异)
                'False': 5         # 小标记表示ID-align=False
            }
            
            # 添加调试信息
            print(f"\n📝 Style Configuration for {plot_type}: {len(combo_labels)} combinations")
            
            for label in combo_labels:
                # 解析标签中的配置信息
                if 'None' in label:
                    zealot_mode = 'None'
                    # 从标签中提取ID-cluster值 (格式: "None  ID-cluster True/False")
                    if 'ID-cluster True' in label:
                        id_cluster = 'True'
                        color = zealot_mode_colors[zealot_mode]['base']  # 深灰色
                        marker = id_cluster_markers[id_cluster]  # 圆形
                        markersize = 8  # None组统一标记大小
                    else:  # ID-cluster False
                        id_cluster = 'False'
                        color = zealot_mode_colors[zealot_mode]['light']  # 浅灰色
                        marker = id_cluster_markers[id_cluster]  # 三角形
                        markersize = 8
                    
                    style_config[label] = {
                        'color': color,
                        'linestyle': '-',  # 统一使用实线
                        'marker': marker,
                        'markersize': markersize,
                        'group': 'None'
                    }
                    
                elif 'Random' in label:
                    zealot_mode = 'Random'
                    # 从标签中提取ID-align和ID-cluster值 (格式: "Random  ID-align True/False  ID-cluster True/False")
                    id_align = 'True' if 'ID-align True' in label else 'False'
                    id_cluster = 'True' if 'ID-cluster True' in label else 'False'
                    
                    # 根据ID-align选择颜色
                    if id_align == 'True':
                        color = zealot_mode_colors[zealot_mode]['base']  # 深橙红色
                    else:
                        color = zealot_mode_colors[zealot_mode]['light']  # 浅粉红色
                    
                    # 根据ID-cluster选择标记
                    marker = id_cluster_markers[id_cluster]
                    # 根据ID-align选择大小
                    markersize = id_align_sizes[id_align]
                    
                    style_config[label] = {
                        'color': color,
                        'linestyle': '-',  # 统一使用实线
                        'marker': marker,
                        'markersize': markersize,
                        'group': 'Random'
                    }
                    
                elif 'Clustered' in label:
                    zealot_mode = 'Clustered'
                    # 从标签中提取ID-align和ID-cluster值 (格式: "Clustered  ID-align True/False  ID-cluster True/False")
                    id_align = 'True' if 'ID-align True' in label else 'False'
                    id_cluster = 'True' if 'ID-cluster True' in label else 'False'
                    
                    # 根据ID-align选择颜色
                    if id_align == 'True':
                        color = zealot_mode_colors[zealot_mode]['base']  # 深蓝色
                    else:
                        color = zealot_mode_colors[zealot_mode]['light']  # 亮绿色
                    
                    # 根据ID-cluster选择标记
                    marker = id_cluster_markers[id_cluster]
                    # 根据ID-align选择大小
                    markersize = id_align_sizes[id_align]
                    
                    style_config[label] = {
                        'color': color,
                        'linestyle': '-',  # 统一使用实线
                        'marker': marker,
                        'markersize': markersize,
                        'group': 'Clustered'
                    }
            
            print(f"✅ Style configuration completed successfully") # 简化的成功信息
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

    # 保持完整标签函数
    def simplify_label(combo_label):
        """保持原始标签，不进行简化以确保读者能理解完整含义"""
        return combo_label
    
    # 获取样式配置
    combo_labels = list(all_results.keys())
    style_config = get_enhanced_style_config(combo_labels)
    
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


def run_and_accumulate_data(output_dir: str = "results/zealot_morality_analysis", 
                           num_runs: int = 5, max_zealots: int = 50, max_morality: int = 30,
                           batch_name: str = ""):
    """
    运行测试并以追加模式保存数据（第一部分）
    
    Args:
    output_dir: 输出目录
    num_runs: 本次运行的次数
    max_zealots: 最大zealot数量
    max_morality: 最大morality ratio (%)
    batch_name: 批次名称，用于标识本次运行
    """
    print("🔬 Running Tests and Accumulating Data")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
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
    print()
    
    # === 处理图1：x轴为zealot numbers ===
    print("📈 Running Test Type 1: Zealot Numbers Analysis")
    print("-" * 50)
    
    plot1_start_time = time.time()
    
    zealot_x_values = list(range(0, max_zealots + 1, 2))  # 0, 2, 4, ..., 50
    zealot_results = {}
    
    for combo in combinations['zealot_numbers']:
        print(f"Running combination: {combo['label']}")
        results = run_parameter_sweep('zealot_numbers', combo, zealot_x_values, num_runs)
        zealot_results[combo['label']] = results
    
    # 保存zealot numbers的数据
    save_data_incrementally('zealot_numbers', zealot_x_values, zealot_results, output_dir, batch_name)
    
    plot1_end_time = time.time()
    plot1_duration = plot1_end_time - plot1_start_time
    hours1, remainder1 = divmod(plot1_duration, 3600)
    minutes1, seconds1 = divmod(remainder1, 60)
    
    print(f"⏱️  Test Type 1 completed in: {int(hours1)}h {int(minutes1)}m {seconds1:.2f}s")
    print()
    
    # === 处理图2：x轴为morality ratio ===
    print("📈 Running Test Type 2: Morality Ratio Analysis")
    print("-" * 50)
    
    plot2_start_time = time.time()
    
    morality_x_values = list(range(0, max_morality + 1, 2))  # 0, 2, 4, ..., 30
    morality_results = {}
    
    for combo in combinations['morality_ratios']:
        print(f"Running combination: {combo['label']}")
        results = run_parameter_sweep('morality_ratios', combo, morality_x_values, num_runs)
        morality_results[combo['label']] = results
    
    # 保存morality ratio的数据
    save_data_incrementally('morality_ratios', morality_x_values, morality_results, output_dir, batch_name)
    
    plot2_end_time = time.time()
    plot2_duration = plot2_end_time - plot2_start_time
    hours2, remainder2 = divmod(plot2_duration, 3600)
    minutes2, seconds2 = divmod(remainder2, 60)
    
    print(f"⏱️  Test Type 2 completed in: {int(hours2)}h {int(minutes2)}m {seconds2:.2f}s")
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
    print(f"   Test Type 1 (Zealot Numbers): {int(hours1)}h {int(minutes1)}m {seconds1:.2f}s")
    print(f"   Test Type 2 (Morality Ratios): {int(hours2)}h {int(minutes2)}m {seconds2:.2f}s")
    print(f"   Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"📁 Data accumulated in: {output_dir}/accumulated_data/")
    
    # 保存批次信息
    batch_info_file = os.path.join(output_dir, "accumulated_data", f"batch_info_{batch_name}.txt")
    with open(batch_info_file, "w") as f:
        f.write(f"Batch Information\n")
        f.write(f"================\n\n")
        f.write(f"Batch name: {batch_name}\n")
        f.write(f"Number of runs: {num_runs}\n")
        f.write(f"Max zealots: {max_zealots}\n")
        f.write(f"Max morality ratio: {max_morality}%\n")
        f.write(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def plot_from_accumulated_data(output_dir: str = "results/zealot_morality_analysis"):
    """
    从累积数据文件中读取数据并生成图表（第二部分）
    
    Args:
    output_dir: 输出目录（包含accumulated_data子文件夹）
    """
    print("📊 Generating Plots from Accumulated Data")
    print("=" * 70)
    
    start_time = time.time()
    
    # 加载累积数据
    loaded_data = load_accumulated_data(output_dir)
    if not loaded_data:
        print("❌ No accumulated data found. Please run data collection first.")
        return
    
    # 按图类型分组数据文件
    zealot_files = {k: v for k, v in loaded_data.items() if k.startswith('zealot_numbers')}
    morality_files = {k: v for k, v in loaded_data.items() if k.startswith('morality_ratios')}
    
    # 处理zealot numbers数据并绘图
    if zealot_files:
        print("\n📈 Processing Zealot Numbers Data...")
        zealot_results, zealot_x_values, zealot_runs_info = process_accumulated_data_for_plotting(zealot_files)
        if zealot_results:
            plot_accumulated_results('zealot_numbers', zealot_x_values, zealot_results, zealot_runs_info, output_dir)
    
    # 处理morality ratios数据并绘图
    if morality_files:
        print("\n📈 Processing Morality Ratios Data...")
        morality_results, morality_x_values, morality_runs_info = process_accumulated_data_for_plotting(morality_files)
        if morality_results:
            plot_accumulated_results('morality_ratios', morality_x_values, morality_results, morality_runs_info, output_dir)
    
    # 计算总耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("🎉 Plot Generation Completed Successfully!")
    print(f"📊 Generated plots from accumulated data")
    print(f"⏱️  Total plotting time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"📁 Plots saved to: {output_dir}")


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
    单独运行 no zealot 的 morality ratio 数据收集
    
    Args:
    output_dir: 输出目录
    num_runs: 每个参数点的运行次数
    max_morality: 最大 morality ratio (%)
    batch_name: 批次名称
    """
    print("🔬 Running No Zealot Morality Ratio Data Collection")
    print("=" * 70)
    
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # 保存 no zealot morality ratio 数据
    save_data_incrementally('morality_ratios', morality_x_values, morality_results, output_dir, batch_name)
    
    # 计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("🎉 No Zealot Data Collection Completed Successfully!")
    print(f"📊 Batch '{batch_name}' with {num_runs} runs per parameter point")
    print(f"⏱️  Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"📁 Data accumulated in: {output_dir}/accumulated_data/")
    
    # 保存批次信息
    batch_info_file = os.path.join(output_dir, "accumulated_data", f"batch_info_{batch_name}.txt")
    with open(batch_info_file, "w") as f:
        f.write(f"No Zealot Batch Information\n")
        f.write(f"===========================\n\n")
        f.write(f"Batch name: {batch_name}\n")
        f.write(f"Number of runs: {num_runs}\n")
        f.write(f"Max morality ratio: {max_morality}%\n")
        f.write(f"Number of combinations: {len(no_zealot_combinations)}\n")
        f.write(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nCombinations run:\n")
        for combo in no_zealot_combinations:
            f.write(f"  - {combo['label']}\n")


def run_complete_no_zealot_analysis(output_dir: str = "results/zealot_morality_analysis", 
                                  num_runs: int = 5, max_morality: int = 30):
    """
    运行完整的 no zealot 分析：数据收集 + 图表生成
    
    Args:
    output_dir: 输出目录
    num_runs: 每个参数点的运行次数
    max_morality: 最大 morality ratio (%)
    """
    print("🔬 Starting Complete No Zealot Analysis")
    print("=" * 70)
    
    # 第一步：运行 no zealot 数据收集
    run_no_zealot_morality_data(output_dir, num_runs, max_morality)
    
    # 第二步：从累积数据生成图表
    plot_from_accumulated_data(output_dir)


def test_combinations():
    """
    测试函数：显示所有参数组合，验证 none zealot 的组合是否正确
    """
    combinations = create_config_combinations()
    
    print("📊 参数组合测试结果:")
    print("=" * 50)
    
    print(f"图1 (Zealot Numbers) 组合数: {len(combinations['zealot_numbers'])}")
    for i, combo in enumerate(combinations['zealot_numbers']):
        print(f"  {i+1}. {combo['label']}")
    
    print(f"\n图2 (Morality Ratios) 组合数: {len(combinations['morality_ratios'])}")
    
    # 按 zealot_mode 分组显示
    none_combos = [c for c in combinations['morality_ratios'] if c['zealot_mode'] == 'none']
    random_combos = [c for c in combinations['morality_ratios'] if c['zealot_mode'] == 'random']
    clustered_combos = [c for c in combinations['morality_ratios'] if c['zealot_mode'] == 'clustered']
    
    print(f"\n  None Zealot 组合数: {len(none_combos)}")
    for i, combo in enumerate(none_combos):
        print(f"    {i+1}. {combo['label']}")
    
    print(f"\n  Random Zealot 组合数: {len(random_combos)}")
    for i, combo in enumerate(random_combos):
        print(f"    {i+1}. {combo['label']}")
    
    print(f"\n  Clustered Zealot 组合数: {len(clustered_combos)}")
    for i, combo in enumerate(clustered_combos):
        print(f"    {i+1}. {combo['label']}")
    
    print(f"\n总计 Morality Ratios 组合: {len(combinations['morality_ratios'])}")
    print("预期组合数: 2 (none) + 4 (random) + 4 (clustered) = 10")


if __name__ == "__main__":
    # 新的分离式使用方法：
    
    # 开始计时
    main_start_time = time.time()
    
    # # 方法1：分两步运行
    # # 第一步：运行测试并积累数据（可以多次运行以积累更多数据）
    # print("=" * 50)
    # print("🚀 示例：分步骤运行实验")
    # print("=" * 50)
    
    # # 数据收集阶段
    # data_collection_start_time = time.time()
    
    # # 可以多次运行以下命令来积累数据：
    # run_and_accumulate_data(
    #     output_dir="results/zealot_morality_analysis",
    #     num_runs=99,  # 每次运行100轮测试
    #     max_zealots=100,  
    #     max_morality=100,
    #     # batch_name="batch_001"  # 可选：给批次命名
    # )
    
    # data_collection_end_time = time.time()
    # data_collection_duration = data_collection_end_time - data_collection_start_time
    

    # 第二步：绘图阶段

    plotting_start_time = time.time()

    plot_from_accumulated_data("results/zealot_morality_analysis")
    
    plotting_end_time = time.time()
    plotting_duration = plotting_end_time - plotting_start_time
    
    # 计算总耗时
    main_end_time = time.time()
    total_duration = main_end_time - main_start_time
    
    # 格式化耗时显示
    def format_duration(duration):
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    
    # 显示耗时总结
    print("\n" + "🕒" * 50)
    print("⏱️  完整实验耗时总结")
    print("🕒" * 50)
    # print(f"📊 数据收集阶段耗时: {format_duration(data_collection_duration)}")
    print(f"📈 图表生成阶段耗时: {format_duration(plotting_duration)}")
    print(f"🎯 总耗时: {format_duration(total_duration)}")
    print("🕒" * 50) 