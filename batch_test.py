# batch_test.py
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from config import SimulationConfig
from config import lfr_config
from visualization import draw_network
from trajectory import run_simulation_with_trajectory, draw_opinion_trajectory, save_trajectory_to_csv
from visualization import (
    draw_network, 
    draw_opinion_distribution, 
    draw_opinion_distribution_heatmap,
    draw_rule_usage,
    draw_rule_cumulative_usage,
    draw_activation_components,
    draw_activation_history,
    draw_activation_heatmap,
    draw_activation_trajectory
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
                    for mor_rate in [1, 0.5, 0]:  # 使用三个不同的道德化率
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
                        trajectory = run_simulation_with_trajectory(sim, steps=100)
                        
                        # 保存轨迹数据到CSV
                        data_folder = os.path.join(folder_path, "data")
                        if not os.path.exists(data_folder):
                            os.makedirs(data_folder)
                        trajectory_csv = os.path.join(data_folder, f"{folder_name}_trajectory.csv")
                        save_trajectory_to_csv(trajectory, trajectory_csv)
                        print(f"Saved trajectory data to {trajectory_csv}")
                        
                        # 保存模拟数据到CSV文件，便于后续分析
                        sim.save_simulation_data(data_folder, prefix=folder_name)
                        print(f"Saved simulation data to {data_folder}")

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
                            "Rule 1: Same dir, Same ID, {0,0}, High Convergence",
                            "Rule 2: Same dir, Same ID, {0,1}, Medium Pull",
                            "Rule 3: Same dir, Same ID, {1,0}, Medium Pull",
                            "Rule 4: Same dir, Same ID, {1,1}, High Polarization",
                            "Rule 5: Same dir, Diff ID, {0,0}, Medium Convergence",
                            "Rule 6: Same dir, Diff ID, {0,1}, Low Pull",
                            "Rule 7: Same dir, Diff ID, {1,0}, Low Pull",
                            "Rule 8: Same dir, Diff ID, {1,1}, Medium Polarization",
                            "Rule 9: Diff dir, Same ID, {0,0}, Very High Convergence",
                            "Rule 10: Diff dir, Same ID, {0,1}, Medium Convergence/Pull",
                            "Rule 11: Diff dir, Same ID, {1,0}, Low Resistance",
                            "Rule 12: Diff dir, Same ID, {1,1}, Low Polarization",
                            "Rule 13: Diff dir, Diff ID, {0,0}, Low Convergence",
                            "Rule 14: Diff dir, Diff ID, {0,1}, High Pull",
                            "Rule 15: Diff dir, Diff ID, {1,0}, High Resistance",
                            "Rule 16: Diff dir, Diff ID, {1,1}, Very High Polarization"
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
                        
                        # 创建激活组件子文件夹
                        activation_folder = os.path.join(folder_path, "activation_components")
                        if not os.path.exists(activation_folder):
                            os.makedirs(activation_folder)
                        
                        # 绘制自我激活和社会影响组件可视化
                        # 1. 自我激活和社会影响散点图
                        components_path = os.path.join(activation_folder, "activation_components.png")
                        draw_activation_components(
                            sim,
                            f"Activation Components\nConfig: {folder_name}",
                            components_path
                        )
                        
                        # 2. 自我激活和社会影响随时间的变化
                        history_path = os.path.join(activation_folder, "activation_history.png")
                        draw_activation_history(
                            sim,
                            f"Activation History\nConfig: {folder_name}",
                            history_path
                        )
                        
                        # 3. 自我激活和社会影响的热力图
                        heatmap_path = os.path.join(activation_folder, "activation_heatmap.png")
                        draw_activation_heatmap(
                            sim,
                            f"Activation Heatmap\nConfig: {folder_name}",
                            heatmap_path
                        )
                        
                        # 4. 选定agent的激活轨迹
                        trajectory_path = os.path.join(activation_folder, "activation_trajectory.png")
                        draw_activation_trajectory(
                            sim,
                            trajectory,
                            f"Activation Trajectories\nConfig: {folder_name}",
                            trajectory_path
                        )
                        
                        # 5. 保存激活组件数据到CSV文件
                        components = sim.get_activation_components()
                        data_path = os.path.join(activation_folder, "activation_data.csv")
                        with open(data_path, "w") as f:
                            f.write("agent_id,identity,morality,opinion,self_activation,social_influence,total_activation\n")
                            for i in range(sim.num_agents):
                                f.write(f"{i},{sim.identities[i]},{sim.morals[i]},{sim.opinions[i]:.4f}")
                                f.write(f",{components['self_activation'][i]:.4f},{components['social_influence'][i]:.4f}")
                                f.write(f",{components['self_activation'][i] + components['social_influence'][i]:.4f}\n")


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


def batch_test_model_params():
    """测试不同极化三角框架模型参数对结果的影响"""
    base_dir = "model_params_test"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 测试不同的参数组合
    # 低极化、中极化和高极化设置
    param_settings = [
        {
            "name": "low_polarization",
            "delta": 0.2,  # 高衰减
            "u": 0.05,     # 低激活
            "alpha": 0.15, # 低自激活
            "beta": 0.15,  # 低社会影响
            "gamma": 0.5   # 低道德化影响
        },
        {
            "name": "medium_polarization",
            "delta": 0.1,  # 中等衰减
            "u": 0.1,      # 中等激活
            "alpha": 0.25, # 中等自激活
            "beta": 0.25,  # 中等社会影响
            "gamma": 1   # 中等道德化影响
        },
        {
            "name": "high_polarization",
            "delta": 0.05, # 低衰减
            "u": 0.15,     # 高激活
            "alpha": 0.3,  # 高自激活
            "beta": 0.35,  # 高社会影响
            "gamma": 1.5   # 高道德化影响
        }
    ]
    
    steps = 200  # 每次模拟的步数
    
    for param_set in param_settings:
        print(f"Testing parameter set: {param_set['name']}")
        
        # 创建目录
        folder_path = os.path.join(base_dir, param_set['name'])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 创建模拟配置
        params = copy.deepcopy(lfr_config)
        
        # 设置模型参数
        params.delta = param_set['delta']
        params.u = param_set['u']
        params.alpha = param_set['alpha']
        params.beta = param_set['beta']
        params.gamma = param_set['gamma']
        
        # 运行模拟
        sim = Simulation(params)
        
        # 保存初始状态
        start_opinion_path = os.path.join(folder_path, "start_opinion.png")
        draw_network(sim, "opinion", f"Starting Opinion Network\nParams: {param_set['name']}", start_opinion_path)
        
        # 运行模拟并记录完整轨迹
        trajectory = run_simulation_with_trajectory(sim, steps=steps)
        
        # 保存模拟数据
        data_folder = os.path.join(folder_path, "data")
        sim.save_simulation_data(data_folder, prefix=param_set['name'])
        print(f"Saved simulation data to {data_folder}")
        
        # 绘制轨迹图
        trajectory_path = os.path.join(folder_path, "opinion_trajectory.png")
        draw_opinion_trajectory(trajectory, f"Opinion Trajectories\nParams: {param_set['name']}", trajectory_path)
        
        # 绘制opinion分布热力图
        heatmap_path = os.path.join(folder_path, "opinion_heatmap.png")
        draw_opinion_distribution_heatmap(
            trajectory,
            f"Opinion Distribution over Time\nParams: {param_set['name']}",
            heatmap_path,
            bins=40,
            log_scale=True
        )
        
        # 绘制结束状态
        end_opinion_path = os.path.join(folder_path, "end_opinion.png")
        draw_network(sim, "opinion", f"Ending Opinion Network\nParams: {param_set['name']}", end_opinion_path)
        
        # 绘制 opinion 分布图
        end_distribution_path = os.path.join(folder_path, "opinion_distribution.png")
        draw_opinion_distribution(sim, f"Ending Opinion Distribution\nParams: {param_set['name']}", end_distribution_path)
        
        # 添加：绘制自我激活和社会影响组件
        activation_components_path = os.path.join(folder_path, "activation_components.png")
        draw_activation_components(
            sim, 
            f"Activation Components\nParams: {param_set['name']}", 
            activation_components_path
        )
        
        # 添加：绘制自我激活和社会影响的历史变化
        activation_history_path = os.path.join(folder_path, "activation_history.png")
        draw_activation_history(
            sim, 
            f"Activation Components History\nParams: {param_set['name']}", 
            activation_history_path
        )
        
        # 添加：绘制自我激活和社会影响的热力图
        activation_heatmap_path = os.path.join(folder_path, "activation_heatmap.png")
        draw_activation_heatmap(
            sim, 
            f"Activation Components Heatmap\nParams: {param_set['name']}", 
            activation_heatmap_path
        )
        
        # 添加：绘制自我激活和社会影响的轨迹图
        activation_trajectory_path = os.path.join(folder_path, "activation_trajectory.png")
        draw_activation_trajectory(
            sim,
            trajectory,
            f"Activation Components Trajectories\nParams: {param_set['name']}",
            activation_trajectory_path
        )
        
        # 输出模型参数到文件
        params_path = os.path.join(folder_path, "model_params.txt")
        with open(params_path, "w") as f:
            f.write(f"模型参数设置 - {param_set['name']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"delta (意见衰减率): {param_set['delta']}\n")
            f.write(f"u (意见激活系数): {param_set['u']}\n")
            f.write(f"alpha (自我激活系数): {param_set['alpha']}\n")
            f.write(f"beta (社会影响系数): {param_set['beta']}\n")
            f.write(f"gamma (道德化影响系数): {param_set['gamma']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"其他配置参数:\n")
            f.write(f"网络类型: {params.network_type}\n")
            f.write(f"节点数量: {params.num_agents}\n")
            f.write(f"道德化率: {params.morality_rate}\n")
            f.write(f"意见分布: {params.opinion_distribution}\n")
            
        # 输出自我激活和社会影响数据到CSV文件
        components = sim.get_activation_components()
        data_path = os.path.join(folder_path, "activation_components_data.csv")
        with open(data_path, "w") as f:
            f.write("agent_id,identity,morality,opinion,self_activation,social_influence,total_activation\n")
            for i in range(sim.num_agents):
                f.write(f"{i},{sim.identities[i]},{sim.morals[i]},{sim.opinions[i]:.4f}")
                f.write(f",{components['self_activation'][i]:.4f},{components['social_influence'][i]:.4f}")
                f.write(f",{components['self_activation'][i] + components['social_influence'][i]:.4f}\n")


# 添加新的测试函数，专门用于分析自我激活和社会影响
def analyze_activation_components():
    """分析不同设置下的自我激活和社会影响组件"""
    base_dir = "activation_analysis"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 测试不同的道德化率
    morality_rates = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    # 基本配置
    base_params = copy.deepcopy(lfr_config)
    base_params.num_agents = 500  # 减少代理数量以加快测试
    
    steps = 500  # 模拟步数
    
    # 为每个道德化率创建单独的文件夹
    for mor_rate in morality_rates:
        folder_name = f"morality_rate_{mor_rate:.1f}"
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        print(f"Analyzing morality rate: {mor_rate}")
        
        # 设置道德化率
        params = copy.deepcopy(base_params)
        params.morality_rate = mor_rate
        
        # 运行模拟
        sim = Simulation(params)
        
        # 运行模拟并记录轨迹
        trajectory = run_simulation_with_trajectory(sim, steps=steps)
        
        # 保存模拟数据
        data_folder = os.path.join(folder_path, "data")
        sim.save_simulation_data(data_folder, prefix=f"morality_rate_{mor_rate:.1f}")
        print(f"Saved simulation data to {data_folder}")
        
        # 绘制自我激活和社会影响组件
        components_path = os.path.join(folder_path, "activation_components.png")
        draw_activation_components(
            sim,
            f"Activation Components (Morality Rate: {mor_rate:.1f})",
            components_path
        )
        
        # 绘制历史变化
        history_path = os.path.join(folder_path, "activation_history.png")
        draw_activation_history(
            sim,
            f"Activation History (Morality Rate: {mor_rate:.1f})",
            history_path
        )
        
        # 绘制热力图
        heatmap_path = os.path.join(folder_path, "activation_heatmap.png")
        draw_activation_heatmap(
            sim,
            f"Activation Heatmap (Morality Rate: {mor_rate:.1f})",
            heatmap_path
        )
        
        # 绘制轨迹图
        trajectory_path = os.path.join(folder_path, "activation_trajectory.png")
        draw_activation_trajectory(
            sim,
            trajectory,
            f"Activation Trajectories (Morality Rate: {mor_rate:.1f})",
            trajectory_path
        )
        
        # 保存基本的模拟结果
        opinion_path = os.path.join(folder_path, "final_opinion.png")
        draw_network(
            sim,
            "opinion",
            f"Final Opinion Network (Morality Rate: {mor_rate:.1f})",
            opinion_path
        )
        
        # 保存道德化分布
        morality_path = os.path.join(folder_path, "morality.png")
        draw_network(
            sim,
            "morality",
            f"Morality Distribution (Morality Rate: {mor_rate:.1f})",
            morality_path
        )
        
        # 输出激活组件数据到CSV
        components = sim.get_activation_components()
        data_path = os.path.join(folder_path, "activation_data.csv")
        with open(data_path, "w") as f:
            f.write("agent_id,identity,morality,opinion,self_activation,social_influence,total_activation\n")
            for i in range(sim.num_agents):
                f.write(f"{i},{sim.identities[i]},{sim.morals[i]},{sim.opinions[i]:.4f}")
                f.write(f",{components['self_activation'][i]:.4f},{components['social_influence'][i]:.4f}")
                f.write(f",{components['self_activation'][i] + components['social_influence'][i]:.4f}\n")
    
    # 绘制所有道德化率的比较图
    compare_morality_activation(morality_rates, base_dir)

def compare_morality_activation(morality_rates, base_dir):
    """
    绘制不同道德化率下自我激活和社会影响的比较图
    
    参数:
    morality_rates -- 道德化率列表
    base_dir -- 基础目录
    """
    # 创建保存比较图的目录
    comparison_dir = os.path.join(base_dir, "comparison")
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # 分别存储每个道德化率下的数据
    all_data = {}
    
    for mor_rate in morality_rates:
        folder_path = os.path.join(base_dir, f"morality_rate_{mor_rate:.1f}")
        csv_path = os.path.join(folder_path, "activation_data.csv")
        
        # 读取CSV数据
        data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding=None)
        all_data[mor_rate] = data
    
    # 创建比较图
    plt.figure(figsize=(16, 12))
    
    # 1. 不同道德化率下的自我激活分布
    plt.subplot(2, 2, 1)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.hist(data['self_activation'], bins=20, alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Self Activation')
    plt.ylabel('Count')
    plt.title('Self Activation Distribution Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 不同道德化率下的社会影响分布
    plt.subplot(2, 2, 2)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.hist(data['social_influence'], bins=20, alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Social Influence')
    plt.ylabel('Count')
    plt.title('Social Influence Distribution Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 不同道德化率下的总激活分布
    plt.subplot(2, 2, 3)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.hist(data['total_activation'], bins=20, alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Total Activation')
    plt.ylabel('Count')
    plt.title('Total Activation Distribution Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 不同道德化率下的意见分布
    plt.subplot(2, 2, 4)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.hist(data['opinion'], bins=20, alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Opinion')
    plt.ylabel('Count')
    plt.title('Opinion Distribution Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Activation Components Across Morality Rates')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存比较图
    comparison_path = os.path.join(comparison_dir, "activation_distribution_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    
    # 创建散点图比较
    plt.figure(figsize=(16, 12))
    
    # 1. 自我激活 vs 意见
    plt.subplot(2, 2, 1)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.scatter(data['opinion'], data['self_activation'], alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 社会影响 vs 意见
    plt.subplot(2, 2, 2)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.scatter(data['opinion'], data['social_influence'], alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 总激活 vs 意见
    plt.subplot(2, 2, 3)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.scatter(data['opinion'], data['total_activation'], alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Opinion')
    plt.ylabel('Total Activation')
    plt.title('Total Activation vs Opinion Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 自我激活 vs 社会影响
    plt.subplot(2, 2, 4)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.scatter(data['self_activation'], data['social_influence'], alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Self Activation')
    plt.ylabel('Social Influence')
    plt.title('Self Activation vs Social Influence Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Activation Relationships Across Morality Rates')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存散点图比较
    scatter_path = os.path.join(comparison_dir, "activation_scatter_comparison.png")
    plt.savefig(scatter_path)
    plt.close()

if __name__ == "__main__":
    batch_test()
    # batch_test_morality_rates()
    # batch_test_model_params()
    # analyze_activation_components()  # 运行新添加的分析函数
