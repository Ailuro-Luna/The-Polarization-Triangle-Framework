#!/usr/bin/env python3
"""
Sobol敏感性分析主执行脚本
对极化三角框架的关键参数进行敏感性分析
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from polarization_triangle.analysis.sobol_analysis import SobolAnalyzer, SobolConfig
from polarization_triangle.analysis.sensitivity_visualizer import SensitivityVisualizer
from polarization_triangle.core.config import SimulationConfig


def create_analysis_configs():
    """创建不同的分析配置"""
    configs = {}
    
    configs['quick'] = SobolConfig(
        n_samples=50,
        n_runs=2,
        n_processes=2,
        num_steps=100,
        output_dir="results/sobol_results_quick"
    )
    
    configs['standard'] = SobolConfig(
        n_samples=500,
        n_runs=3,
        n_processes=4,
        num_steps=200,
        output_dir="results/sobol_results_standard"
    )
    
    configs['high_precision'] = SobolConfig(
        n_samples=1000,
        n_runs=5,
        n_processes=6,
        num_steps=300,
        output_dir="results/sobol_results_high_precision"
    )

    # configs['full'] = SobolConfig(
    #     n_samples=2000,
    #     n_runs=10,
    #     n_processes=8,
    #     num_steps=500,
    #     output_dir="sobol_results_full"
    # )
    configs['full'] = SobolConfig(
        n_samples=4096,
        n_runs=50,
        n_processes=8,
        num_steps=500,
        output_dir="results/sobol_results_full"
    )
    
    return configs


def run_sensitivity_analysis(config_name: str = 'standard', 
                           custom_config: SobolConfig = None,
                           load_existing: bool = False):
    """运行敏感性分析"""
    
    # 选择配置
    if custom_config:
        config = custom_config
    else:
        configs = create_analysis_configs()
        if config_name not in configs:
            raise ValueError(f"配置 '{config_name}' 不存在。可用配置: {list(configs.keys())}")
        config = configs[config_name]
    
    print(f"使用配置: {config_name}")
    print(f"样本数: {config.n_samples}")
    print(f"运行次数: {config.n_runs}")
    print(f"进程数: {config.n_processes}")
    print(f"模拟步数: {config.num_steps}")
    print(f"输出目录: {config.output_dir}")
    
    # 创建分析器
    analyzer = SobolAnalyzer(config)
    
    # 加载已有结果或运行新分析
    if load_existing:
        print("尝试加载已有结果...")
        try:
            sensitivity_indices = analyzer.load_results()
            if sensitivity_indices:
                print("成功加载已有结果")
            else:
                print("未找到已有结果，开始新分析...")
                sensitivity_indices = analyzer.run_complete_analysis()
        except Exception as e:
            print(f"加载失败: {e}")
            print("开始新分析...")
            sensitivity_indices = analyzer.run_complete_analysis()
    else:
        # 运行完整分析
        sensitivity_indices = analyzer.run_complete_analysis()
    
    return analyzer, sensitivity_indices


def generate_reports(analyzer: SobolAnalyzer, 
                    sensitivity_indices: dict,
                    create_plots: bool = True):
    """生成分析报告"""
    
    print("\n" + "="*60)
    print("生成分析报告")
    print("="*60)
    
    # 生成摘要表
    try:
        summary_df = analyzer.get_summary_table()
        print("\n敏感性分析摘要 (前10行):")
        print(summary_df.head(10).to_string(index=False))
        
        # 导出Excel报告
        analyzer.export_results()
        
    except Exception as e:
        print(f"生成数据报告时出错: {e}")
    
    # 生成可视化报告
    if create_plots:
        try:
            print("\n生成可视化报告...")
            visualizer = SensitivityVisualizer()
            
            # 创建图表输出目录
            plot_dir = os.path.join(analyzer.config.output_dir, "plots")
            plot_files = visualizer.create_comprehensive_report(
                sensitivity_indices,
                analyzer.param_samples,
                analyzer.simulation_results,
                plot_dir
            )
            
            print(f"可视化报告已保存到: {plot_dir}")
            
        except Exception as e:
            print(f"生成可视化报告时出错: {e}")
    
    # 打印关键发现
    print_key_findings(sensitivity_indices)


def print_key_findings(sensitivity_indices: dict):
    """打印关键发现"""
    print("\n" + "="*60)
    print("关键发现")
    print("="*60)
    
    param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
    param_labels = ['α (自我激活)', 'β (社会影响)', 'γ (道德化影响)', 'cohesion_factor (凝聚力)']
    
    # 计算平均敏感性
    all_st_values = []
    all_s1_values = []
    
    for output_name, indices in sensitivity_indices.items():
        all_st_values.append(indices['ST'])
        all_s1_values.append(indices['S1'])
    
    if all_st_values:
        import numpy as np
        mean_st = np.mean(all_st_values, axis=0)
        mean_s1 = np.mean(all_s1_values, axis=0)
        mean_interaction = mean_st - mean_s1
        
        # 参数重要性排序
        importance_ranking = sorted(
            zip(param_labels, mean_st), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("\n1. 参数重要性排序 (基于平均总敏感性指数):")
        for i, (param, value) in enumerate(importance_ranking, 1):
            print(f"   {i}. {param}: {value:.3f}")
        
        # 交互效应分析
        print("\n2. 平均交互效应强度 (ST - S1):")
        for param, interaction in zip(param_labels, mean_interaction):
            if interaction > 0.1:
                level = "强"
            elif interaction > 0.05:
                level = "中等"
            else:
                level = "弱"
            print(f"   {param}: {interaction:.3f} ({level})")
        
        # 最敏感的输出指标
        print("\n3. 各输出指标的最敏感参数:")
        for output_name, indices in sensitivity_indices.items():
            max_idx = np.argmax(indices['ST'])
            max_param = param_labels[max_idx]
            max_value = indices['ST'][max_idx]
            print(f"   {output_name}: {max_param} ({max_value:.3f})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行Sobol敏感性分析')
    parser.add_argument('--config', type=str, default='standard', 
                       choices=['quick', 'standard', 'high_precision', 'full'],
                       help='分析配置类型')
    parser.add_argument('--load', action='store_true', 
                       help='尝试加载已有结果')
    parser.add_argument('--no-plots', action='store_true',
                       help='不生成可视化图表')
    parser.add_argument('--output-dir', type=str, 
                       help='自定义输出目录')
    parser.add_argument('--n-samples', type=int,
                       help='自定义样本数')
    parser.add_argument('--n-runs', type=int,
                       help='自定义运行次数')
    parser.add_argument('--n-processes', type=int,
                       help='自定义进程数')
    
    args = parser.parse_args()
    
    # 创建自定义配置（如果提供了自定义参数）
    custom_config = None
    if any([args.output_dir, args.n_samples, args.n_runs, args.n_processes]):
        configs = create_analysis_configs()
        base_config = configs[args.config]
        
        custom_config = SobolConfig(
            n_samples=args.n_samples or base_config.n_samples,
            n_runs=args.n_runs or base_config.n_runs,
            n_processes=args.n_processes or base_config.n_processes,
            output_dir=args.output_dir or base_config.output_dir,
            num_steps=base_config.num_steps,
            base_config=base_config.base_config
        )
    
    try:
        start_time = time.time()
        
        # 运行分析
        analyzer, sensitivity_indices = run_sensitivity_analysis(
            config_name=args.config,
            custom_config=custom_config,
            load_existing=args.load
        )
        
        # 生成报告
        generate_reports(
            analyzer, 
            sensitivity_indices, 
            create_plots=not args.no_plots
        )
        
        end_time = time.time()
        print(f"\n总耗时: {end_time - start_time:.2f} 秒")
        print(f"结果保存在: {analyzer.config.output_dir}")
        
    except KeyboardInterrupt:
        print("\n分析被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 