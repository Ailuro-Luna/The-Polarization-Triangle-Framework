#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一的实验运行脚本

运行各种实验的入口
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from polarization_triangle.experiments.batch_runner import batch_test
from polarization_triangle.experiments.morality_test import batch_test_morality_rates
from polarization_triangle.experiments.model_params_test import batch_test_model_params
from polarization_triangle.experiments.activation_analysis import analyze_activation_components
from polarization_triangle.experiments.zealot.core import run_zealot_experiment
from polarization_triangle.verification.alpha_analysis import AlphaVerification
from polarization_triangle.verification.agent_interaction_verification import main as agent_verification_main


def run_basic_simulation(output_dir: str = "results/basic", steps: int = 200):
    """运行基础模拟"""
    print("Running basic simulation...")
    batch_test(output_dir=output_dir, steps=steps)


def run_morality_test(output_dir: str = "results/morality", steps: int = 200, 
                     morality_rates: list = None):
    """运行道德化率测试"""
    if morality_rates is None:
        morality_rates = [0.2, 0.4, 0.6, 0.8]
    
    print(f"Running morality rate test with rates: {morality_rates}")
    batch_test_morality_rates(
        output_dir=output_dir,
        steps=steps,
        morality_rates=morality_rates
    )


def run_model_params_test(output_dir: str = "results/params", steps: int = 200):
    """运行模型参数测试"""
    print("Running model parameters test...")
    batch_test_model_params(output_dir=output_dir, steps=steps)


def run_activation_analysis(output_dir: str = "results/activation", steps: int = 200):
    """运行激活组件分析"""
    print("Running activation component analysis...")
    analyze_activation_components(output_dir=output_dir, steps=steps)


def run_zealot_exp(output_dir: str = "results/zealot", steps: int = 500, **kwargs):
    """运行zealot实验"""
    print("Running zealot experiment...")
    return run_zealot_experiment(
        steps=steps,
        num_zealots=kwargs.get('num_zealots', 50),
        zealot_opinion=kwargs.get('zealot_opinion', 1.0),
        zealot_mode=kwargs.get('zealot_mode', 'random'),
        zealot_morality=kwargs.get('zealot_morality', False),
        morality_rate=kwargs.get('morality_rate', 0.0),
        identity_clustered=kwargs.get('identity_clustered', False),
        output_dir=output_dir
    )


def run_alpha_verification(output_dir: str = "results/verification/alpha",
                         alpha_min: float = -1.0, alpha_max: float = 2.0):
    """运行Alpha验证"""
    print("Running alpha verification...")
    verification = AlphaVerification(
        alpha_range=(alpha_min, alpha_max),
        output_dir=output_dir
    )
    verification.run()


def run_alphabeta_verification(output_dir: str = "results/verification/alphabeta",
                             steps: int = 200, **kwargs):
    """运行Alpha-Beta验证"""
    print("Running alpha-beta verification...")
    from polarization_triangle.scripts.run_alphabeta_verification import run_alphabeta_verification
    run_alphabeta_verification(output_dir=output_dir, steps=steps, **kwargs)


def run_agent_interaction_verification(output_dir: str = "results/verification/agent_interaction",
                                     num_steps: int = 1):
    """运行代理交互验证"""
    print("Running agent interaction verification...")
    agent_verification_main(output_dir=output_dir, num_steps=num_steps)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Run polarization triangle experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic simulation
  python run_experiment.py basic
  
  # Run morality test with custom rates
  python run_experiment.py morality --morality-rates 0.1 0.5 0.9
  
  # Run zealot experiment
  python run_experiment.py zealot --num-zealots 50 --zealot-mode clustered
  
  # Run all verifications
  python run_experiment.py verification --verification-type all
        """
    )
    
    # 位置参数
    parser.add_argument("experiment", 
                       choices=["basic", "morality", "params", "activation", 
                               "zealot", "verification"],
                       help="Type of experiment to run")
    
    # 通用参数
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: results/<experiment>)")
    parser.add_argument("--steps", type=int, default=200,
                       help="Number of simulation steps")
    
    # 特定实验参数
    parser.add_argument("--morality-rates", type=float, nargs="+",
                       help="Morality rates for morality test")
    
    # Zealot实验参数
    parser.add_argument("--num-zealots", type=int, default=50,
                       help="Number of zealots")
    parser.add_argument("--zealot-opinion", type=float, default=1.0,
                       help="Zealot opinion value")
    parser.add_argument("--zealot-mode", 
                       choices=["random", "clustered", "high_degree"],
                       default="random",
                       help="Zealot placement mode")
    parser.add_argument("--zealot-morality", action="store_true",
                       help="Whether zealots are moralizing")
    parser.add_argument("--identity-clustered", action="store_true",
                       help="Whether identity is clustered")
    
    # 验证参数
    parser.add_argument("--verification-type", 
                       choices=["alpha", "alphabeta", "agent_interaction", "all"],
                       default="alpha",
                       help="Type of verification to run")
    parser.add_argument("--alpha-min", type=float, default=-1.0,
                       help="Minimum alpha value")
    parser.add_argument("--alpha-max", type=float, default=2.0,
                       help="Maximum alpha value")
    
    args = parser.parse_args()
    
    # 设置默认输出目录
    if args.output_dir is None:
        args.output_dir = f"results/{args.experiment}"
    
    # 运行相应的实验
    if args.experiment == "basic":
        run_basic_simulation(args.output_dir, args.steps)
    
    elif args.experiment == "morality":
        run_morality_test(args.output_dir, args.steps, args.morality_rates)
    
    elif args.experiment == "params":
        run_model_params_test(args.output_dir, args.steps)
    
    elif args.experiment == "activation":
        run_activation_analysis(args.output_dir, args.steps)
    
    elif args.experiment == "zealot":
        run_zealot_exp(
            args.output_dir, 
            args.steps,
            num_zealots=args.num_zealots,
            zealot_opinion=args.zealot_opinion,
            zealot_mode=args.zealot_mode,
            zealot_morality=args.zealot_morality,
            morality_rate=args.morality_rates[0] if args.morality_rates else 0.0,
            identity_clustered=args.identity_clustered
        )
    
    elif args.experiment == "verification":
        if args.verification_type == "alpha" or args.verification_type == "all":
            run_alpha_verification(
                os.path.join(args.output_dir, "alpha"),
                args.alpha_min, args.alpha_max
            )
        
        if args.verification_type == "alphabeta" or args.verification_type == "all":
            run_alphabeta_verification(
                os.path.join(args.output_dir, "alphabeta"),
                args.steps
            )
        
        if args.verification_type == "agent_interaction" or args.verification_type == "all":
            run_agent_interaction_verification(
                os.path.join(args.output_dir, "agent_interaction")
            )
    
    print("\nExperiment completed!")


if __name__ == "__main__":
    main() 