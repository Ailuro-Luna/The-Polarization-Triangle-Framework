# batch_test.py
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from config import lfr_config
from visualization import draw_network
from trajectory import run_simulation_with_trajectory, draw_opinion_trajectory
from visualization import (
    draw_network, 
    draw_opinion_distribution, 
    draw_opinion_distribution_heatmap,
    draw_rule_usage,
    draw_rule_cumulative_usage
)


def override_morality(sim, rate):
    """
    覆盖模拟中所有代理的道德值

    参数:
    sim -- 模拟实例
    rate -- 道德化率（0到1之间的浮点数）
    """
    n = sim.num_agents
    # 生成具有指定道德化率的随机道德值数组
    morals = np.zeros(n, dtype=np.int32)
    moral_indices = np.random.choice(n, int(n * rate), replace=False)
    morals[moral_indices] = 1
    sim.morals[:] = morals


def run_simulation_with_rule_tracking(config, steps=500):
    """
    运行模拟并跟踪规则使用情况
    
    参数:
    config -- 模拟配置
    steps -- 模拟步数
    
    返回:
    sim -- 模拟对象
    """
    sim = Simulation(config)
    
    # 运行模拟
    for _ in range(steps):
        sim.step()
    
    return sim


def batch_test():
    base_dir = "batch_results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    for id_mode in ["random", "clustered"]:
    # for id_mode in ["clustered"]:
        for mor_mode in ["random", "clustered"]:
            for op_mode in ["random", "clustered"]:
                for op_dist in ["uniform", "single_peak", "twin_peak"]:
                    for mor_rate in [0.25, 0.5, 0.75]:  # 使用三个不同的道德化率
                        folder_name = f"ID_{id_mode}_M_{mor_mode}_OP_{op_mode}_op_{op_dist}_mor_{mor_rate:.1f}"
                        folder_path = os.path.join(base_dir, folder_name)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        print("Processing configuration:", folder_name)
                        params = copy.deepcopy(lfr_config)
                        params.cluster_identity = (id_mode == "clustered")
                        params.cluster_morality = (mor_mode == "clustered")
                        params.cluster_opinion = (op_mode == "clustered")
                        params.opinion_distribution = op_dist
                        # 设置道德化率
                        params.morality_rate = mor_rate

                        sim = Simulation(params)

                        # 打印验证morality模式是否正确应用
                        moral_mean = np.mean(sim.morals)
                        print(f"Morality rate: {mor_rate}, Mean moral value: {moral_mean}")

                        start_opinion_path = os.path.join(folder_path, "start_opinion.png")
                        draw_network(sim, "opinion", f"Starting Opinion Network\nConfig: {folder_name}",
                                     start_opinion_path)

                        # 运行模拟并记录完整轨迹
                        trajectory = run_simulation_with_trajectory(sim, steps=500)

                        # 添加：绘制规则使用统计图
                        rule_usage_path = os.path.join(folder_path, "rule_usage.png")
                        draw_rule_usage(
                            sim.rule_counts_history,
                            f"Rule Usage over Time\nConfig: {folder_name}",
                            rule_usage_path
                        )
                        
                        # 添加：绘制规则累积使用统计图
                        rule_cumulative_path = os.path.join(folder_path, "rule_cumulative_usage.png")
                        draw_rule_cumulative_usage(
                            sim.rule_counts_history,
                            f"Cumulative Rule Usage\nConfig: {folder_name}",
                            rule_cumulative_path
                        )

                        # 绘制轨迹图
                        trajectory_path = os.path.join(folder_path, "opinion_trajectory.png")
                        draw_opinion_trajectory(trajectory, f"Opinion Trajectories\nConfig: {folder_name}",
                                                trajectory_path)

                        # 添加：绘制opinion分布热力图
                        heatmap_path = os.path.join(folder_path, "opinion_heatmap.png")
                        draw_opinion_distribution_heatmap(
                            trajectory,
                            f"Opinion Distribution over Time\nConfig: {folder_name}",
                            heatmap_path,
                            bins=40,
                            log_scale=True
                        )

                        # 其他可视化继续保持不变
                        end_opinion_path = os.path.join(folder_path, "end_opinion.png")
                        draw_network(sim, "opinion", f"Ending Opinion Network\nConfig: {folder_name}", end_opinion_path)

                        end_identity_path = os.path.join(folder_path, "end_identity.png")
                        draw_network(sim, "identity", f"Ending Identity Network\nConfig: {folder_name}",
                                     end_identity_path)

                        end_morality_path = os.path.join(folder_path, "end_morality.png")
                        draw_network(sim, "morality", f"Ending Morality Network\nConfig: {folder_name}",
                                     end_morality_path)

                        # 绘制 opinion 分布图
                        end_distribution_path = os.path.join(folder_path, "opinion_distribution.png")
                        draw_opinion_distribution(sim, f"Ending Opinion Distribution\nConfig: {folder_name}",
                                                  end_distribution_path)
                                                  
                        # 输出规则使用统计信息
                        rule_names = [
                            "Rule 1: Same dir, Same ID, Non-moral, Converge",
                            "Rule 2: Same dir, Diff ID, Non-moral, Converge",
                            "Rule 3: Same dir, Same ID, Moral, Polarize",
                            "Rule 4: Same dir, Diff ID, Moral, Polarize",
                            "Rule 5: Diff dir, Same ID, Non-moral, Converge",
                            "Rule 6: Diff dir, Diff ID, Non-moral, Converge",
                            "Rule 7: Diff dir, Same ID, Moral, Converge",
                            "Rule 8: Diff dir, Diff ID, Moral, Polarize"
                        ]
                        
                        # 计算总规则使用次数
                        rule_counts = np.sum(sim.rule_counts_history, axis=0)
                        total_count = np.sum(rule_counts)
                        
                        # 将规则使用统计写入文件
                        stats_path = os.path.join(folder_path, "rule_usage_stats.txt")
                        with open(stats_path, "w") as f:
                            f.write(f"规则使用统计 - 配置: {folder_name}\n")
                            f.write("-" * 50 + "\n")
                            for i, rule_name in enumerate(rule_names):
                                count = rule_counts[i]
                                percent = (count / total_count) * 100 if total_count > 0 else 0
                                f.write(f"{rule_name}: {count} 次 ({percent:.1f}%)\n")
                            f.write("-" * 50 + "\n")
                            f.write(f"总计: {total_count} 次\n")


def batch_test_morality_rates():
    """单独测试不同道德化率对规则使用的影响"""
    base_dir = "morality_rate_test"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    morality_rates = [0.1, 0.3, 0.5, 0.7, 0.9]  # 测试一系列不同的道德化率
    steps = 200  # 每次模拟的步数
    
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
    plt.figure(figsize=(14, 10))
    
    # 为比较图创建subplots
    for i in range(8):
        plt.subplot(2, 4, i+1)
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


if __name__ == "__main__":
    batch_test()
    batch_test_morality_rates()
