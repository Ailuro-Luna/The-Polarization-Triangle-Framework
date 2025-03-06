# batch_test.py
import os
import copy
import numpy as np
from simulation import Simulation
from config import lfr_config
from visualization import draw_network
from trajectory import run_simulation_with_trajectory, draw_opinion_trajectory
from visualization import draw_network, draw_opinion_distribution, draw_opinion_distribution_heatmap

def override_morality(sim, ratio):
    n = sim.num_agents
    if ratio == "all1":
        sim.morals[:] = 1
    elif ratio == "all0":
        sim.morals[:] = 0
    elif ratio == "half":
        arr = np.array([1] * (n // 2) + [0] * (n - n // 2))
        np.random.shuffle(arr)
        sim.morals[:] = arr
    else:
        raise ValueError("Unknown morality ratio")


def batch_test():
    base_dir = "batch_results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for id_mode in ["random", "clustered"]:
        for mor_mode in ["random", "clustered"]:
            for op_mode in ["random", "clustered"]:
                for op_dist in ["uniform", "single_peak", "twin_peak"]:
                    for mor_ratio in ["all1", "half", "all0"]:
                        folder_name = f"ID_{id_mode}_M_{mor_mode}_OP_{op_mode}_op_{op_dist}_mor_{mor_ratio}"
                        folder_path = os.path.join(base_dir, folder_name)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        print("Processing configuration:", folder_name)
                        params = copy.deepcopy(lfr_config)
                        params.cluster_identity = (id_mode == "clustered")
                        params.cluster_morality = (mor_mode == "clustered")
                        params.cluster_opinion = (op_mode == "clustered")
                        params.opinion_distribution = op_dist
                        # 修复：添加这行设置morality_mode
                        params.morality_mode = mor_ratio

                        sim = Simulation(params)

                        # 打印验证morality模式是否正确应用
                        moral_mean = np.mean(sim.morals)
                        print(f"Morality mode: {mor_ratio}, Mean moral value: {moral_mean}")

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
