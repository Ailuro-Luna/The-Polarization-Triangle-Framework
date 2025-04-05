"""
运行Alpha-Beta验证的脚本
"""

import os
import argparse
import numpy as np
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from polarization_triangle.verification.alphabeta_analysis import AlphaBetaVerification, AlphaBetaVerificationConfig
from polarization_triangle.core.config import SimulationConfig, lfr_config

def run_alphabeta_verification(output_dir="verification_results/alphabeta_no_morality", 
                             steps=300, 
                             low_alpha=0.5, 
                             high_alpha=1.5,
                             beta_min=0.1,
                             beta_max=2.0,
                             beta_steps=10,
                             morality_rate=0.0,
                             num_runs=10):
    """
    运行Alpha-Beta验证实验
    
    参数:
    output_dir -- 输出目录
    steps -- 模拟步数
    low_alpha -- 低alpha值
    high_alpha -- 高alpha值
    beta_min -- beta最小值
    beta_max -- beta最大值
    beta_steps -- beta步数（测试多少个beta值）
    morality_rate -- 道德化率，默认为0
    num_runs -- 每个参数组合的模拟次数，默认为10
    """
    # 创建配置
    config = AlphaBetaVerificationConfig(
        output_dir=output_dir,
        base_config=lfr_config,
        steps=steps,
        low_alpha=low_alpha,
        mid_alpha_values=[0.9, 1.0, 1.1],
        high_alpha=high_alpha,
        beta_values=list(np.linspace(beta_min, beta_max, beta_steps)),
        morality_rate=morality_rate,
        num_runs=num_runs
    )
    
    # 创建并运行验证
    verification = AlphaBetaVerification(config)
    results = verification.run()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpha-Beta verification analysis")
    parser.add_argument("--output-dir", type=str, default="verification_results/alphabeta_no_morality",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of simulation steps")
    parser.add_argument("--low-alpha", type=float, default=0.5,
                        help="Low alpha value")
    parser.add_argument("--high-alpha", type=float, default=1.5,
                        help="High alpha value")
    parser.add_argument("--beta-min", type=float, default=0.1,
                        help="Minimum beta value")
    parser.add_argument("--beta-max", type=float, default=2.0,
                        help="Maximum beta value")
    parser.add_argument("--beta-steps", type=int, default=10,
                        help="Number of beta values to test")
    parser.add_argument("--morality-rate", type=float, default=0.0,
                        help="Morality rate (0.0-1.0)")
    parser.add_argument("--num-runs", type=int, default=10,
                        help="Number of simulation runs per parameter combination")
    
    args = parser.parse_args()
    
    run_alphabeta_verification(
        output_dir=args.output_dir,
        steps=args.steps,
        low_alpha=args.low_alpha,
        high_alpha=args.high_alpha,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_steps=args.beta_steps,
        morality_rate=args.morality_rate,
        num_runs=args.num_runs
    ) 