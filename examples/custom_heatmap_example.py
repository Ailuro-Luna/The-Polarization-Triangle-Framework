#!/usr/bin/env python3
"""
自定义热力图示例

此脚本展示如何在zealot实验中使用自定义的热力图颜色映射和尺度设置。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polarization_triangle.experiments.multi_zealot_experiment import run_multi_zealot_experiment
from polarization_triangle.config.heatmap_config import get_heatmap_config, create_custom_power_norm
from polarization_triangle.visualization.opinion_viz import draw_opinion_distribution_heatmap
import numpy as np

def run_custom_heatmap_experiment():
    """
    运行带有自定义热力图配置的实验
    """
    print("运行自定义热力图实验...")
    
    # 方法1: 使用预设配置运行实验
    print("\n=== 使用高对比度配置 ===")
    run_multi_zealot_experiment(
        runs=3,                    # 少量运行用于演示
        steps=50,                  # 较少步数
        morality_rate=0.2,
        zealot_morality=True,
        identity_clustered=True,
        zealot_count=10,
        zealot_mode="clustered",
        output_dir="results/custom_heatmap_demo/high_contrast"
    )
    
    # 方法2: 修改已有热力图的显示效果
    print("\n=== 生成自定义颜色的热力图 ===")
    
    # 创建示例数据 (模拟opinion历史)
    demo_data = create_demo_opinion_history()
    
    # 使用不同配置生成多种版本的热力图
    configs_to_test = [
        ('default', 'viridis'),
        ('high_contrast', 'hot'),
        ('symmetric', 'RdBu'),  
        ('colorblind', 'cividis'),
        ('publication', 'gray')
    ]
    
    os.makedirs("results/custom_heatmap_demo/comparisons", exist_ok=True)
    
    for config_name, expected_cmap in configs_to_test:
        print(f"生成 {config_name} 配置的热力图...")
        
        # 获取预设配置
        config = get_heatmap_config(config_name)
        
        # 绘制热力图
        filename = f"results/custom_heatmap_demo/comparisons/demo_{config_name}.png"
        draw_opinion_distribution_heatmap(
            demo_data,
            f"Demo Heatmap - {config_name.title()} Configuration",
            filename,
            bins=20,  # 减少bins用于演示
            **config
        )
    
    # 方法3: 完全自定义的配置
    print("\n=== 生成完全自定义的热力图 ===")
    
    # 自定义配置1: 固定数值范围的线性尺度
    draw_opinion_distribution_heatmap(
        demo_data,
        "Custom Linear Scale (0-25)",
        "results/custom_heatmap_demo/comparisons/custom_linear.png",
        bins=20,
        cmap='plasma',
        log_scale=False,
        vmin=0,
        vmax=25
    )
    
    # 自定义配置2: 幂律标准化
    custom_norm = create_custom_power_norm(gamma=0.3, vmin=1, vmax=30)
    draw_opinion_distribution_heatmap(
        demo_data,
        "Custom Power Norm (γ=0.3)",
        "results/custom_heatmap_demo/comparisons/custom_power.png",
        bins=20,
        cmap='inferno',
        custom_norm=custom_norm
    )
    
    print("\n实验完成！检查以下文件夹中的热力图：")
    print("- results/custom_heatmap_demo/high_contrast/average_results/")
    print("- results/custom_heatmap_demo/comparisons/")

def create_demo_opinion_history():
    """
    创建演示用的opinion历史数据
    """
    # 模拟50个时间步，100个agent的opinion演化
    time_steps = 50
    num_agents = 100
    
    opinion_history = []
    
    # 初始状态：随机分布
    current_opinions = np.random.uniform(-0.2, 0.2, num_agents)
    
    for t in range(time_steps):
        # 模拟opinion随时间的演化
        # 添加一些极化趋势
        for i in range(num_agents):
            if current_opinions[i] > 0:
                current_opinions[i] += np.random.normal(0.02, 0.01)  # 正向增长
            else:
                current_opinions[i] += np.random.normal(-0.02, 0.01)  # 负向增长
        
        # 限制在[-1, 1]范围内
        current_opinions = np.clip(current_opinions, -1, 1)
        
        # 记录当前状态
        opinion_history.append(current_opinions.copy())
    
    return opinion_history

def demonstrate_colormap_comparison():
    """
    演示不同颜色映射的效果
    """
    print("\n=== 颜色映射对比演示 ===")
    
    # 创建更简单的测试数据
    test_data = []
    for t in range(20):
        # 创建具有明显特征的分布
        opinions = []
        opinions.extend(np.random.normal(-0.7, 0.1, 20))  # 负面cluster
        opinions.extend(np.random.normal(0.0, 0.05, 10))   # 中性cluster  
        opinions.extend(np.random.normal(0.7, 0.1, 20))   # 正面cluster
        test_data.append(np.array(opinions))
    
    # 测试不同的颜色映射
    colormaps = ['viridis', 'hot', 'coolwarm', 'RdBu', 'jet']
    
    os.makedirs("results/custom_heatmap_demo/colormap_comparison", exist_ok=True)
    
    for cmap in colormaps:
        filename = f"results/custom_heatmap_demo/colormap_comparison/colormap_{cmap}.png"
        draw_opinion_distribution_heatmap(
            test_data,
            f"Colormap Comparison - {cmap}",
            filename,
            bins=15,
            cmap=cmap,
            log_scale=False,
            vmin=0,
            vmax=15  # 固定范围便于比较
        )
        print(f"生成 {cmap} 颜色映射示例")

if __name__ == "__main__":
    # 运行自定义热力图实验
    run_custom_heatmap_experiment()
    
    # 运行颜色映射对比
    demonstrate_colormap_comparison()
    
    print("\n🎨 热力图自定义演示完成！")
    print("\n使用说明：")
    print("1. 查看生成的图片，比较不同配置的效果")
    print("2. 在实际实验中，可以通过修改函数参数来使用这些配置")
    print("3. 参考 polarization_triangle/config/heatmap_config.py 了解更多选项") 