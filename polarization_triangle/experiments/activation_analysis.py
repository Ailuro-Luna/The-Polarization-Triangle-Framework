import copy
import os
from core.simulation import Simulation
from core.config import lfr_config
from analysis.trajectory import run_simulation_with_trajectory
from visualization.activation_viz import draw_activation_components, draw_activation_history, draw_activation_heatmap, draw_activation_trajectory
from visualization.network_viz import draw_network
from analysis.activation import compare_morality_activation

# 添加新的测试函数，专门用于分析自我激活和社会影响
def analyze_activation_components(output_dir = "activation_analysis", steps = 500):
    """分析不同设置下的自我激活和社会影响组件"""
    base_dir = output_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 测试不同的道德化率
    morality_rates = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    # 基本配置
    base_params = copy.deepcopy(lfr_config)
    base_params.num_agents = 500  # 减少代理数量以加快测试
    
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