# analyze_rule_usage.py
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from simulation import Simulation
from visualization import draw_rule_usage
from config import lfr_config

def run_multiple_simulations(config, num_simulations=10, time_steps=100):
    """
    运行多次模拟并收集规则使用统计信息
    
    参数:
    config -- 模拟配置
    num_simulations -- 模拟次数
    time_steps -- 每次模拟的时间步数
    
    返回:
    average_rule_counts -- 平均规则使用次数，形状为(time_steps, 8)
    """
    # 初始化累计规则使用计数
    cumulative_rule_counts = None
    
    print(f"运行 {num_simulations} 次模拟，每次 {time_steps} 步...")
    
    # 运行多次模拟
    for i in tqdm(range(num_simulations)):
        # 创建新模拟
        sim = Simulation(config)
        
        # 运行模拟时间步
        for _ in range(time_steps):
            sim.step()
        
        # 转换规则计数历史为NumPy数组
        rule_counts = np.array(sim.rule_counts_history)
        
        # 确保长度一致（防止某些模拟运行时间不同）
        if rule_counts.shape[0] > time_steps:
            rule_counts = rule_counts[:time_steps]
        
        # 初始化或更新累计计数
        if cumulative_rule_counts is None:
            cumulative_rule_counts = rule_counts
        else:
            cumulative_rule_counts += rule_counts
    
    # 计算平均值
    average_rule_counts = cumulative_rule_counts / num_simulations
    
    return average_rule_counts

def main():
    # 创建输出目录
    output_dir = "rule_usage_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置参数
    num_simulations = 10  # 模拟次数
    time_steps = 100      # 每次模拟的时间步数
    
    # 运行多次模拟并获取平均规则使用计数
    average_rule_counts = run_multiple_simulations(
        lfr_config,       # 使用LFR网络配置
        num_simulations,
        time_steps
    )
    
    # 绘制平均规则使用图表
    title = f"Average Rule Usage ({num_simulations} Simulations)"
    filename = os.path.join(output_dir, "average_rule_usage.png")
    draw_rule_usage(average_rule_counts, title, filename, smooth=True, window_size=5)
    
    # 额外进行单次模拟以便对比
    print("运行单次模拟以供对比...")
    sim = Simulation(lfr_config)
    for _ in tqdm(range(time_steps)):
        sim.step()
    
    # 绘制单次模拟的规则使用
    single_title = "Single Simulation Rule Usage"
    single_filename = os.path.join(output_dir, "single_simulation_rule_usage.png")
    draw_rule_usage(sim.rule_counts_history, single_title, single_filename)
    
    print(f"分析完成，图表已保存至 {output_dir} 目录")
    
    # 输出各规则的平均使用频率
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
    
    print("\n各规则平均使用频率统计:")
    total_rule_counts = np.sum(average_rule_counts, axis=0)
    for i, rule_name in enumerate(rule_names):
        count = total_rule_counts[i]
        percent = (count / np.sum(total_rule_counts)) * 100
        print(f"{rule_name}: {count:.1f} 次 ({percent:.1f}%)")

if __name__ == "__main__":
    main() 