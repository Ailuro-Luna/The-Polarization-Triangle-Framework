"""
Sobol敏感性分析使用示例
演示如何对极化三角框架的关键参数进行敏感性分析
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

try:
    from polarization_triangle.analysis import (
        SobolAnalyzer, 
        SobolConfig, 
        SensitivityVisualizer
    )
    from polarization_triangle.core.config import SimulationConfig
    
    SENSITIVITY_AVAILABLE = True
except ImportError as e:
    print(f"敏感性分析模块不可用: {e}")
    print("请安装依赖: pip install SALib pandas seaborn")
    SENSITIVITY_AVAILABLE = False


def basic_sensitivity_analysis():
    """基础敏感性分析示例"""
    if not SENSITIVITY_AVAILABLE:
        return
    
    print("="*60)
    print("基础Sobol敏感性分析示例")
    print("="*60)
    
    # 创建快速测试配置
    config = SobolConfig(
        n_samples=20,  # 非常小的样本数用于快速演示
        n_runs=2,
        n_processes=2,
        num_steps=50,
        output_dir="example_sobol_quick"
    )
    
    print(f"配置信息:")
    print(f"  样本数: {config.n_samples}")
    print(f"  运行次数: {config.n_runs}")
    print(f"  总模拟次数: {config.n_samples * (2 * 4 + 2) * config.n_runs}")
    print(f"  输出目录: {config.output_dir}")
    
    # 创建分析器并运行
    try:
        analyzer = SobolAnalyzer(config)
        sensitivity_indices = analyzer.run_complete_analysis()
        
        # 显示结果摘要
        print("\n敏感性分析完成！")
        summary_df = analyzer.get_summary_table()
        print("\n结果摘要:")
        print(summary_df.head(8).to_string(index=False))
        
        # 导出结果
        analyzer.export_results()
        
        return analyzer, sensitivity_indices
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        return None, None


def create_custom_configuration_example():
    """自定义配置示例"""
    if not SENSITIVITY_AVAILABLE:
        return
    
    print("\n" + "="*60)
    print("自定义配置示例")
    print("="*60)
    
    # 创建自定义基础配置
    custom_base_config = SimulationConfig(
        num_agents=100,  # 较小的Agent数量以便快速演示
        network_type='lfr',
        morality_rate=0.3,  # 较低的道德化率
        alpha=0.5,
        beta=0.15,
        gamma=1.2,
        cohesion_factor=0.15
    )
    
    # 创建自定义敏感性分析配置
    custom_config = SobolConfig(
        # 自定义参数范围
        parameter_bounds={
            'alpha': [0.2, 0.7],        # 较窄的范围
            'beta': [0.08, 0.25],       # 较窄的范围
            'gamma': [0.5, 1.8],        # 较窄的范围
            'cohesion_factor': [0.05, 0.4]  # 较窄的范围
        },
        n_samples=15,
        n_runs=2,
        base_config=custom_base_config,
        output_dir="example_sobol_custom"
    )
    
    print("自定义参数范围:")
    for param, bounds in custom_config.parameter_bounds.items():
        print(f"  {param}: {bounds}")
    
    return custom_config


def visualization_example(analyzer, sensitivity_indices):
    """可视化示例"""
    if not SENSITIVITY_AVAILABLE or not analyzer or not sensitivity_indices:
        return
    
    print("\n" + "="*60)
    print("可视化示例")
    print("="*60)
    
    try:
        # 创建可视化器
        visualizer = SensitivityVisualizer()
        
        # 创建单个输出指标的对比图
        output_names = list(sensitivity_indices.keys())
        if output_names:
            first_output = output_names[0]
            print(f"为 {first_output} 创建敏感性对比图...")
            
            fig = visualizer.plot_sensitivity_comparison(
                sensitivity_indices, 
                first_output, 
                'example_sensitivity_comparison.png'
            )
            # plt.show()  # 注释掉以避免在某些环境下卡住
            plt.close(fig)
        
        # 创建热力图
        print("创建敏感性热力图...")
        fig = visualizer.plot_sensitivity_heatmap(
            sensitivity_indices, 
            'ST', 
            'example_heatmap.png'
        )
        # plt.show()  # 注释掉以避免在某些环境下卡住
        plt.close(fig)
        
        # 创建参数重要性排序图
        print("创建参数重要性排序图...")
        fig = visualizer.plot_parameter_ranking(
            sensitivity_indices, 
            'ST', 
            'example_ranking.png'
        )
        # plt.show()  # 注释掉以避免在某些环境下卡住
        plt.close(fig)
        
        print("可视化图表已生成并显示")
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")


def interpret_results(sensitivity_indices):
    """结果解释示例"""
    if not sensitivity_indices:
        return
    
    print("\n" + "="*60)
    print("结果解释示例")
    print("="*60)
    
    param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
    param_labels = ['α (自我激活)', 'β (社会影响)', 'γ (道德化)', 'cohesion_factor (凝聚力)']
    
    # 分析每个输出指标
    for output_name, indices in sensitivity_indices.items():
        print(f"\n{output_name} 的敏感性分析:")
        
        # 找出最敏感的参数
        max_st_idx = np.argmax(indices['ST'])
        max_st_param = param_labels[max_st_idx]
        max_st_value = indices['ST'][max_st_idx]
        
        print(f"  最敏感参数: {max_st_param} (ST = {max_st_value:.3f})")
        
        # 分析交互效应
        interactions = np.array(indices['ST']) - np.array(indices['S1'])
        max_interaction_idx = np.argmax(interactions)
        max_interaction_param = param_labels[max_interaction_idx]
        max_interaction_value = interactions[max_interaction_idx]
        
        print(f"  最强交互效应: {max_interaction_param} (ST-S1 = {max_interaction_value:.3f})")
        
        # 敏感性分类
        print("  参数敏感性分类:")
        for i, (param, st_val) in enumerate(zip(param_labels, indices['ST'])):
            if st_val > 0.15:
                category = "高敏感"
            elif st_val > 0.1:
                category = "中等敏感"
            else:
                category = "低敏感"
            print(f"    {param}: {category} (ST = {st_val:.3f})")


def main():
    """主示例函数"""
    print("极化三角框架 - Sobol敏感性分析示例")
    print("="*60)
    
    if not SENSITIVITY_AVAILABLE:
        print("敏感性分析功能不可用，请安装必要的依赖包")
        return
    
    # 1. 基础敏感性分析
    analyzer, sensitivity_indices = basic_sensitivity_analysis()
    
    # 2. 结果解释
    interpret_results(sensitivity_indices)
    
    # 3. 可视化演示
    visualization_example(analyzer, sensitivity_indices)
    
    # 4. 自定义配置示例
    custom_config = create_custom_configuration_example()
    
    print("\n" + "="*60)
    print("示例完成")
    print("="*60)
    print("要运行更详细的分析，请使用:")
    print("  python polarization_triangle/scripts/run_sobol_analysis.py")
    print("可用的配置选项:")
    print("  --config quick          # 快速测试")
    print("  --config standard       # 标准分析") 
    print("  --config high_precision # 高精度分析")
    print("  --config full           # 完整分析")


if __name__ == "__main__":
    main() 