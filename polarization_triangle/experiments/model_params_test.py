import copy
import os
from core.simulation import Simulation
from core.config import lfr_config
from analysis.trajectory import run_simulation_with_trajectory
from visualization.opinion_viz import draw_opinion_distribution, draw_opinion_distribution_heatmap, draw_opinion_trajectory
from visualization.network_viz import draw_network
from visualization.activation_viz import draw_activation_components, draw_activation_history, draw_activation_heatmap, draw_activation_trajectory

def batch_test_model_params(output_dir = "model_params_test", steps = 200):
    """测试不同极化三角框架模型参数对结果的影响"""
    base_dir = output_dir
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