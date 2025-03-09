# batch_test.py
import os
import copy
import numpy as np
from simulation import Simulation
from config import lfr_config
from visualization import draw_network
from trajectory import run_simulation_with_trajectory, draw_opinion_trajectory
from visualization import draw_network, draw_opinion_distribution, draw_opinion_distribution_heatmap


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


if __name__ == "__main__":
    batch_test()
