"""
Alpha-Beta Verification Analysis Module

该模块测试不同alpha（自我激活系数）和beta（社会影响系数）组合对模拟结果的影响。
主要验证：
1. 对于低alpha值，增加beta是否能产生类似高alpha的效果
2. 对于高alpha值，增加beta可能只会增强现有效果
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from polarization_triangle.core.config import SimulationConfig
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.analysis.trajectory import run_simulation_with_trajectory
from polarization_triangle.visualization.opinion_viz import (
    draw_opinion_distribution, 
    draw_opinion_distribution_heatmap,
    draw_opinion_trajectory
)


@dataclass
class AlphaBetaVerificationConfig:
    """Alpha-Beta验证实验的配置类"""
    output_dir: str = "results/verification/alphabeta"
    base_config: SimulationConfig = None
    steps: int = 300
    # alpha值设置（低、中、高）
    low_alpha: float = 0.5
    mid_alpha_values: List[float] = None  # 默认为[0.9, 1.0, 1.1]
    high_alpha: float = 1.5
    # beta值范围
    beta_values: List[float] = None  # 默认为从0.1到2.0的范围
    # 道德化率
    morality_rate: float = 0.0  # 将道德化率设置为0
    # 每个参数组合的模拟次数
    num_runs: int = 10


class AlphaBetaVerification:
    """验证不同alpha和beta参数对极化的影响"""
    
    def __init__(self, config: AlphaBetaVerificationConfig = None):
        """初始化验证类"""
        self.config = config or AlphaBetaVerificationConfig()
        
        # 设置默认值（如果未指定）
        if self.config.base_config is None:
            from polarization_triangle.core.config import lfr_config
            self.config.base_config = lfr_config
        
        if self.config.mid_alpha_values is None:
            self.config.mid_alpha_values = [0.9, 1.0, 1.1]
            
        if self.config.beta_values is None:
            self.config.beta_values = np.linspace(0.1, 2.0, 10).tolist()
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 存储结果
        self.results = {}
    
    def measure_polarization(self, opinions: np.ndarray) -> Dict[str, float]:
        """
        计算极化程度的各种指标
        
        参数:
        opinions -- 所有代理的意见数组
        
        返回:
        包含不同极化指标的字典
        """
        # 方差（总体分散程度）
        variance = np.var(opinions)
        
        # 双峰指数（使用双峰检测）
        hist, bin_edges = np.histogram(opinions, bins=20, range=(-1, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # 找到局部最大值
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((bin_centers[i], hist[i]))
        
        # 计算双峰指数（距离 * 高度）
        bimodality = 0
        if len(peaks) >= 2:
            # 按高度排序并获取最高的两个峰
            peaks.sort(key=lambda x: x[1], reverse=True)
            if len(peaks) >= 2:
                peak1, peak2 = peaks[0], peaks[1]
                distance = abs(peak1[0] - peak2[0])
                height_product = peak1[1] * peak2[1]
                bimodality = distance * np.sqrt(height_product / sum(hist))
        
        # 极端比例（|opinion| > 0.7的比例）
        extreme_ratio = np.mean(np.abs(opinions) > 0.7)
        
        # 平均极端度
        mean_extremity = np.mean(np.abs(opinions))
        
        return {
            "variance": variance,
            "bimodality": bimodality,
            "extreme_ratio": extreme_ratio,
            "mean_extremity": mean_extremity
        }
        
    def run_simulation(self, alpha: float, beta: float) -> Tuple[Dict[str, float], Dict[str, float], List[np.ndarray]]:
        """
        运行多次模拟，并返回平均极化指标、标准差和最后一次模拟的轨迹
        
        参数:
        alpha -- 自我激活系数
        beta -- 社会影响系数
        
        返回:
        (平均极化指标字典, 标准差字典, 最后一次模拟的轨迹)
        """
        # 创建配置副本并设置参数
        config = SimulationConfig(**self.config.base_config.__dict__)
        config.alpha = alpha
        config.beta = beta
        config.morality_rate = self.config.morality_rate
        
        # 存储多次运行的指标结果
        all_metrics = []
        last_trajectory = None
        
        # 运行多次模拟
        for _ in range(self.config.num_runs):
            # 创建并运行模拟
            sim = Simulation(config)
            trajectory = run_simulation_with_trajectory(sim, steps=self.config.steps)
            
            # 计算极化指标
            polarization_metrics = self.measure_polarization(sim.opinions)
            all_metrics.append(polarization_metrics)
            
            # 保存最后一次模拟的轨迹（用于可视化）
            last_trajectory = trajectory
        
        # 计算平均值和标准差
        mean_metrics = {}
        std_metrics = {}
        
        # 将列表转换为指标的字典
        metrics_dict = {key: [run[key] for run in all_metrics] for key in all_metrics[0].keys()}
        
        # 计算每个指标的平均值和标准差
        for key, values in metrics_dict.items():
            mean_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
        
        return mean_metrics, std_metrics, last_trajectory
    
    def run_all_combinations(self):
        """运行所有alpha和beta组合的模拟"""
        # 所有alpha值
        all_alpha_values = [self.config.low_alpha] + self.config.mid_alpha_values + [self.config.high_alpha]
        
        # 运行所有组合的模拟
        for alpha in all_alpha_values:
            self.results[alpha] = {}
            alpha_dir = os.path.join(self.config.output_dir, f"alpha_{alpha}")
            os.makedirs(alpha_dir, exist_ok=True)
            
            for beta in self.config.beta_values:
                print(f"Running simulations with alpha={alpha}, beta={beta} ({self.config.num_runs} runs)")
                mean_metrics, std_metrics, trajectory = self.run_simulation(alpha, beta)
                self.results[alpha][beta] = {
                    "mean_metrics": mean_metrics,
                    "std_metrics": std_metrics,
                    "trajectory": trajectory
                }
                
                # 绘制意见分布图
                final_dist_path = os.path.join(alpha_dir, f"opinion_dist_a{alpha}_b{beta}.png")
                draw_opinion_distribution(
                    Simulation(self.config.base_config), 
                    f"Opinion Distribution (α={alpha}, β={beta})",
                    final_dist_path
                )
                
                # 绘制意见轨迹热力图
                heatmap_path = os.path.join(alpha_dir, f"opinion_heatmap_a{alpha}_b{beta}.png")
                draw_opinion_distribution_heatmap(
                    trajectory,
                    f"Opinion Evolution (α={alpha}, β={beta})",
                    heatmap_path
                )
                
                # 绘制意见轨迹图
                trajectory_path = os.path.join(alpha_dir, f"opinion_trajectory_a{alpha}_b{beta}.png")
                draw_opinion_trajectory(
                    trajectory,
                    f"Opinion Trajectories (α={alpha}, β={beta})",
                    trajectory_path
                )
    
    def plot_comparison_results(self):
        """绘制不同alpha和beta组合的极化指标比较图"""
        metrics_to_plot = ["variance", "bimodality", "extreme_ratio", "mean_extremity"]
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 8))
            
            for alpha in self.results.keys():
                beta_values = list(self.results[alpha].keys())
                metric_means = [self.results[alpha][beta]["mean_metrics"][metric] for beta in beta_values]
                metric_stds = [self.results[alpha][beta]["std_metrics"][metric] for beta in beta_values]
                
                # 绘制曲线及误差区域
                plt.plot(beta_values, metric_means, marker='o', label=f"α={alpha}")
                plt.fill_between(
                    beta_values,
                    [m - s for m, s in zip(metric_means, metric_stds)],
                    [m + s for m, s in zip(metric_means, metric_stds)],
                    alpha=0.2
                )
            
            plt.xlabel("Beta (Social Influence Coefficient)")
            plt.ylabel(f"{metric.replace('_', ' ').title()}")
            plt.title(f"Effect of Alpha and Beta on {metric.replace('_', ' ').title()} (Avg of {self.config.num_runs} runs)")
            plt.legend()
            plt.grid(True)
            
            # 保存图表
            output_path = os.path.join(self.config.output_dir, f"comparison_{metric}.png")
            plt.savefig(output_path)
            plt.close()
    
    def save_results_to_csv(self):
        """将平均结果和标准差保存到CSV文件"""
        # 准备数据
        data = []
        for alpha in self.results.keys():
            for beta in self.results[alpha].keys():
                mean_metrics = self.results[alpha][beta]["mean_metrics"]
                std_metrics = self.results[alpha][beta]["std_metrics"]
                
                row = {
                    "alpha": alpha,
                    "beta": beta,
                }
                
                # 添加平均值
                for key, value in mean_metrics.items():
                    row[key] = value
                    
                # 添加标准差
                for key, value in std_metrics.items():
                    row[f"{key}_std"] = value
                    
                data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        output_path = os.path.join(self.config.output_dir, "alphabeta_results.csv")
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def run(self):
        """运行完整的验证流程"""
        print("Starting Alpha-Beta verification analysis...")
        print(f"Each parameter combination will be simulated {self.config.num_runs} times")
        self.run_all_combinations()
        self.plot_comparison_results()
        results_path = self.save_results_to_csv()
        print(f"Analysis complete. Results saved to {results_path}")
        
        return self.results


def main():
    """主入口函数"""
    # 创建一个用于快速测试的配置
    test_config = AlphaBetaVerificationConfig(
        output_dir="verification_results/alphabeta_test",
        num_runs=3,  # 每个参数组合只运行3次
        beta_values=np.linspace(0.1, 2.0, 3).tolist(),  # 只测试3个beta值
        steps=50  # 只运行50步
    )
    verification = AlphaBetaVerification(test_config)
    verification.run()


if __name__ == "__main__":
    main() 