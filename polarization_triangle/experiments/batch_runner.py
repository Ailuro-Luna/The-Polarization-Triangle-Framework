import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.core.config import lfr_config
from polarization_triangle.visualization.network_viz import draw_network
from polarization_triangle.visualization.opinion_viz import (
    draw_opinion_distribution, 
    draw_opinion_distribution_heatmap,
    draw_opinion_trajectory
)
from polarization_triangle.visualization.rule_viz import (
    draw_interaction_type_usage, 
    draw_interaction_type_cumulative_usage
)
from polarization_triangle.visualization.activation_viz import (
    draw_activation_components,
    draw_activation_history,
    draw_activation_heatmap,
    draw_activation_trajectory
)
from polarization_triangle.analysis.trajectory import run_simulation_with_trajectory
from polarization_triangle.utils.data_manager import save_trajectory_to_csv


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


def batch_test(output_dir = "results/batch_results", steps=100):
    base_dir = output_dir
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
                        trajectory = run_simulation_with_trajectory(sim, steps=steps)
                        
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
                        rule_usage_path = os.path.join(folder_path, "interaction_types.png")
                        draw_interaction_type_usage(
                            sim.rule_counts_history,
                            f"Interaction Types over Time\nConfig: {folder_name}",
                            rule_usage_path
                        )
                        
                        # 添加：绘制规则累积使用统计图
                        rule_cumulative_path = os.path.join(folder_path, "interaction_types_cumulative.png")
                        draw_interaction_type_cumulative_usage(
                            sim.rule_counts_history,
                            f"Cumulative Interaction Types\nConfig: {folder_name}",
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
                        interaction_names = [
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
                        
                        # 获取交互类型统计
                        interaction_stats = sim.get_interaction_counts()
                        counts = interaction_stats["counts"]
                        total_count = interaction_stats["total_interactions"]
                        
                        # 将交互类型统计写入文件
                        stats_path = os.path.join(folder_path, "interaction_types_stats.txt")
                        with open(stats_path, "w") as f:
                            f.write(f"交互类型统计 - 配置: {folder_name}\n")
                            f.write("-" * 50 + "\n")
                            for i, interaction_name in enumerate(interaction_names):
                                count = counts[i]
                                percent = (count / total_count) * 100 if total_count > 0 else 0
                                f.write(f"{interaction_name}: {count} 次 ({percent:.1f}%)\n")
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
