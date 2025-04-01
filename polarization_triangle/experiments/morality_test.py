import copy
import os
from core.simulation import Simulation
from core.config import lfr_config
from visualization.rule_viz import draw_rule_usage, draw_rule_cumulative_usage
import matplotlib.pyplot as plt
import numpy as np


def batch_test_morality_rates(output_dir = "morality_rate_test", steps = 200, morality_rates = [0.1, 0.3, 0.5, 0.7, 0.9]):
    """单独测试不同道德化率对规则使用的影响"""
    base_dir = output_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    
    # 创建规则累积使用统计的数组
    rule_cumulative_counts_by_rate = {}
    
    for mor_rate in morality_rates:
        print(f"Testing morality rate: {mor_rate}")
        
        # 创建模拟配置
        params = copy.deepcopy(lfr_config)
        params.morality_rate = mor_rate
        
        # 运行模拟
        sim = Simulation(params)
        
        # 运行模拟
        for _ in range(steps):
            sim.step()
        
        # 保存规则累积使用结果
        rule_cumulative_counts_by_rate[mor_rate] = np.array(sim.rule_counts_history)
        
        # 为每个道德化率绘制规则累积使用图
        folder_path = os.path.join(base_dir, f"mor_rate_{mor_rate:.1f}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 保存模拟数据
        data_folder = os.path.join(folder_path, "data")
        sim.save_simulation_data(data_folder, prefix=f"mor_rate_{mor_rate:.1f}")
        print(f"Saved simulation data to {data_folder}")
        
        # 绘制规则使用图
        rule_usage_path = os.path.join(folder_path, "rule_usage.png")
        draw_rule_usage(
            sim.rule_counts_history,
            f"Rule Usage over Time (Morality Rate: {mor_rate:.1f})",
            rule_usage_path
        )
        
        # 绘制规则累积使用图
        rule_cumulative_path = os.path.join(folder_path, "rule_cumulative_usage.png")
        draw_rule_cumulative_usage(
            sim.rule_counts_history,
            f"Cumulative Rule Usage (Morality Rate: {mor_rate:.1f})",
            rule_cumulative_path
        )
    
    # 创建比较图，展示不同道德化率下各规则的累积使用
    plt.figure(figsize=(16, 12))
    
    # 为比较图创建subplots
    for i in range(16):
        plt.subplot(4, 4, i+1)
        for mor_rate in morality_rates:
            data = rule_cumulative_counts_by_rate[mor_rate]
            plt.plot(np.cumsum(data[:, i]), label=f"Rate={mor_rate:.1f}")
        
        plt.title(f"Rule {i+1}")
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.legend()
    plt.suptitle("Comparison of Cumulative Rule Usage Across Morality Rates", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle腾出空间
    
    # 保存比较图
    comparison_path = os.path.join(base_dir, "rule_usage_comparison.png")
    plt.savefig(comparison_path)
    plt.close()