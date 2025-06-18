#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Polarization Triangle Framework Main Entry File
Provides command line interface to run various simulation tests
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from polarization_triangle.experiments.batch_runner import batch_test
from polarization_triangle.experiments.morality_test import batch_test_morality_rates
from polarization_triangle.experiments.model_params_test import batch_test_model_params
from polarization_triangle.experiments.activation_analysis import analyze_activation_components


def run_single_simulation(output_dir="results/single_run", steps=300, 
                         zealot_count=0, zealot_mode="random", zealot_opinion=1.0,
                         zealot_morality=False, zealot_identity_allocation=True):
    """
    运行单次模拟并生成基本的可视化结果
    
    参数:
    output_dir: 输出目录
    steps: 模拟步数
    zealot_count: zealot数量
    zealot_mode: zealot选择模式 (random, degree, clustered)
    zealot_opinion: zealot固定意见值
    zealot_morality: zealot是否都是道德化的
    zealot_identity_allocation: 是否只从identity=1的agent中选择zealot
    """
    import copy
    from polarization_triangle.core.config import base_config
    from polarization_triangle.core.simulation import Simulation
    from polarization_triangle.visualization.network_viz import draw_network
    from polarization_triangle.visualization.opinion_viz import draw_opinion_distribution_heatmap
    from polarization_triangle.analysis.trajectory import run_simulation_with_trajectory
    from polarization_triangle.analysis.statistics import print_statistics_summary
    
    print(f"🚀 运行单次模拟...")
    print(f"📊 模拟步数: {steps}")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用base_config并配置zealot
    config = copy.deepcopy(base_config)
    
    # 配置zealot参数
    has_zealots = zealot_count > 0
    if has_zealots:
        config.enable_zealots = True
        config.zealot_count = zealot_count
        config.zealot_mode = zealot_mode
        config.zealot_opinion = zealot_opinion
        config.zealot_morality = zealot_morality
        config.zealot_identity_allocation = zealot_identity_allocation
    
    print(f"🔧 使用配置: base_config")
    print(f"   Agent数量: {config.num_agents}")
    print(f"   网络类型: {config.network_type}")
    print(f"   道德化率: {config.morality_rate}")
    if has_zealots:
        print(f"   🎯 Zealot配置:")
        print(f"      数量: {zealot_count}")
        print(f"      模式: {zealot_mode}")
        print(f"      意见值: {zealot_opinion}")
        print(f"      道德化: {zealot_morality}")
        print(f"      身份分配: {zealot_identity_allocation}")
    else:
        print(f"   🎯 无Zealot")
    
    # 创建模拟实例
    print("🏗️  创建模拟...")
    sim = Simulation(config)
    
    # 显示zealot信息
    if has_zealots:
        zealot_ids = sim.get_zealot_ids()
        print(f"   Zealot IDs: {zealot_ids}")
        print(f"   实际Zealot数量: {len(zealot_ids)}")
    
    # 绘制初始网络
    print("📈 绘制初始网络...")
    draw_network(sim, "opinion", "Initial Opinion Network", 
                os.path.join(output_dir, "initial_opinion.png"))
    draw_network(sim, "identity", "Initial Identity Network", 
                os.path.join(output_dir, "initial_identity.png"))
    draw_network(sim, "morality", "Initial Morality Network", 
                os.path.join(output_dir, "initial_morality.png"))
    
    # 添加综合网络图（身份+道德化+zealot状态）
    if has_zealots or any(sim.morals == 1):
        draw_network(sim, "identity_morality", "Initial Identity & Morality Network", 
                    os.path.join(output_dir, "initial_identity_morality.png"))
    
    # 运行模拟并记录轨迹
    print(f"⚡ 运行模拟 {steps} 步...")
    if has_zealots:
        # 对于有zealot的情况，手动记录轨迹以确保zealot意见正确记录
        trajectory = []
        trajectory.append(sim.opinions.copy())
        
        for step in range(steps):
            sim.step()  # step方法中会自动调用set_zealot_opinions()
            trajectory.append(sim.opinions.copy())
            
            # 每50步打印一次进度
            if (step + 1) % 50 == 0:
                print(f"   完成 {step + 1}/{steps} 步")
    else:
        # 对于无zealot的情况，使用现有的轨迹记录函数
        trajectory = run_simulation_with_trajectory(sim, steps=steps)
    
    # 生成可视化
    print("📊 生成可视化...")
    draw_opinion_distribution_heatmap(
        trajectory, 
        "Opinion Evolution Over Time", 
        os.path.join(output_dir, "opinion_evolution.png")
    )
    
    # 绘制最终网络
    draw_network(sim, "opinion", "Final Opinion Network", 
                os.path.join(output_dir, "final_opinion.png"))
    draw_network(sim, "identity", "Final Identity Network", 
                os.path.join(output_dir, "final_identity.png"))
    draw_network(sim, "morality", "Final Morality Network", 
                os.path.join(output_dir, "final_morality.png"))
    
    # 添加最终综合网络图
    if has_zealots or any(sim.morals == 1):
        draw_network(sim, "identity_morality", "Identity & Morality Network", 
                    os.path.join(output_dir, "identity_morality.png"))
    
    # 打印统计摘要
    print("\n📋 统计摘要:")
    print("=" * 50)
    print_statistics_summary(sim, exclude_zealots=True)
    
    if has_zealots:
        print("\n🎯 Zealot最终状态:")
        print("-" * 30)
        zealot_ids = sim.get_zealot_ids()
        print(f"Zealot数量: {len(zealot_ids)}")
        print(f"Zealot IDs: {zealot_ids}")
        print(f"Zealot意见值: {[sim.opinions[i] for i in zealot_ids]}")
        print(f"预期Zealot意见值: {zealot_opinion}")
        
        # 也显示包含zealot的统计
        print("\n📊 包含Zealot的统计:")
        print("-" * 30)
        print_statistics_summary(sim, exclude_zealots=False)
    
    print(f"\n🎉 单次模拟完成！结果已保存到: {output_dir}")
    print("📁 生成的文件:")
    print("   - initial_*.png (初始网络)")
    print("   - opinion_evolution.png (意见演化热图)")
    print("   - final_*.png (最终网络)")
    if has_zealots or any(sim.morals == 1):
        print("   - *_identity_morality.png (综合网络：身份+道德化+Zealot)")
    print("\n🎨 可视化规则:")
    print("   - 形状：所有Agent都是圆形")
    print("   - 边框：金色边框 = Zealot，黑色边框 = 道德化普通Agent，无边框 = 非道德化普通Agent")
    if has_zealots:
        print("   - 颜色：根据图表模式显示意见/身份/道德化状态")
    
    return sim


def main():
    """
    Main entry function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Polarization Triangle Framework Simulation")
    parser.add_argument("--test-type", 
                        choices=["basic", "single", "morality", "model-params", "activation", "verification"],
                        default="basic",
                        help="Type of test to run: 'basic' for batch tests, 'single' for one simulation")
    parser.add_argument("--output-dir", type=str, default="batch_results",
                       help="Output directory name")
    parser.add_argument("--steps", type=int, default=200,
                       help="Number of simulation steps")
    parser.add_argument("--morality-rates", type=float, nargs="+", 
                       default=[0.2, 0.4, 0.6, 0.8],
                       help="List of morality rates for morality test")
    parser.add_argument("--verification-type", 
                        choices=["alpha", "alphabeta", "agent_interaction", "all"],
                        default="alpha",
                        help="Verification type (used only when test-type is verification)")
    parser.add_argument("--alpha-min", type=float, default=-1.0,
                       help="Minimum alpha value for analysis (used only when verification-type is alpha)")
    parser.add_argument("--alpha-max", type=float, default=2.0,
                       help="Maximum alpha value for analysis (used only when verification-type is alpha)")
    parser.add_argument("--alpha-values", type=float, nargs="+", default=[0.5, 1.0, 1.5],
                       help="Alpha values for alphabeta analysis")
    parser.add_argument("--beta-min", type=float, default=0.1,
                       help="Minimum beta value for alphabeta analysis")
    parser.add_argument("--beta-max", type=float, default=2.0,
                       help="Maximum beta value for alphabeta analysis")
    parser.add_argument("--beta-steps", type=int, default=10,
                       help="Number of beta steps for alphabeta analysis")
    parser.add_argument("--morality-rate", type=float, default=0.0,
                       help="Morality rate for alphabeta analysis (0.0-1.0)")
    parser.add_argument("--num-runs", type=int, default=10,
                       help="Number of simulation runs per parameter combination for alphabeta analysis")
    
    # Zealot相关参数
    parser.add_argument("--zealot-count", type=int, default=20,
                       help="Number of zealots (0 to disable zealots)")
    parser.add_argument("--zealot-mode", type=str, choices=["random", "degree", "clustered"], 
                       default="random",
                       help="Zealot selection mode")
    parser.add_argument("--zealot-opinion", type=float, default=1.0,
                       help="Fixed opinion value for zealots")
    parser.add_argument("--zealot-morality", action='store_true',
                       help="Make all zealots moralizing (morality=1)")
    parser.add_argument("--zealot-identity-allocation", action='store_true', default=True,
                       help="Only allocate zealots to agents with identity=1")
    
    args = parser.parse_args()
    
    # Run different tests based on test type
    if args.test_type == "basic":
        print("Running basic simulation...")
        # Use batch_test from experiments module
        batch_test(output_dir=args.output_dir, steps=args.steps)
        
    elif args.test_type == "single":
        print("Running single simulation...")
        run_single_simulation(
            output_dir=args.output_dir, 
            steps=args.steps,
            zealot_count=args.zealot_count,
            zealot_mode=args.zealot_mode,
            zealot_opinion=args.zealot_opinion,
            zealot_morality=args.zealot_morality,
            zealot_identity_allocation=args.zealot_identity_allocation
        )
        
    elif args.test_type == "morality":
        print(f"Running morality rate test, morality rates: {args.morality_rates}...")
        # Use batch_test_morality_rates from experiments module
        batch_test_morality_rates(output_dir=args.output_dir, steps=args.steps,
                         morality_rates=args.morality_rates)
        
    elif args.test_type == "model-params":
        print("Running model parameters test...")
        # Use batch_test_model_params from experiments module
        batch_test_model_params(output_dir=args.output_dir, steps=args.steps)
        
    elif args.test_type == "activation":
        print("Running activation component analysis...")
        # run analyze_activation_components from experiments module
        analyze_activation_components(output_dir=args.output_dir, steps=args.steps)
        
    elif args.test_type == "verification":
        print(f"Running verification analysis, type: {args.verification_type}...")
        
        # Function to run alpha verification
        def run_alpha_verification():
            from polarization_triangle.verification.alpha_analysis import AlphaVerification
            verification = AlphaVerification(
                alpha_range=(args.alpha_min, args.alpha_max),
                output_dir=os.path.join(args.output_dir, "alpha_verification")
            )
            verification.run()
            
        # Function to run alphabeta verification    
        def run_alphabeta_verification():
            from polarization_triangle.scripts.run_alphabeta_verification import run_alphabeta_verification
            output_dir = os.path.join(args.output_dir, "alphabeta_verification")
            run_alphabeta_verification(
                output_dir=output_dir,
                steps=args.steps,
                # low_alpha=args.low_alpha,
                # high_alpha=args.high_alpha,
                beta_min=args.beta_min,
                beta_max=args.beta_max,
                beta_steps=args.beta_steps,
                morality_rate=args.morality_rate,
                num_runs=args.num_runs
            )
            
        # Function to run agent interaction verification
        def run_agent_interaction_verification():
            from polarization_triangle.verification.agent_interaction_verification import main as agent_verification_main
            output_dir = os.path.join(args.output_dir, "agent_interaction_verification")
            # os.makedirs(output_dir, exist_ok=True)
            # agent_verification_main(output_dir=output_dir, num_steps=args.steps)
            agent_verification_main(output_dir=output_dir, num_steps=1)
            
        # Run verification based on type
        if args.verification_type == "alpha":
            run_alpha_verification()
        elif args.verification_type == "alphabeta":
            run_alphabeta_verification()
        elif args.verification_type == "agent_interaction":
            run_agent_interaction_verification()
        elif args.verification_type == "all":
            print("Running all verification types...")
            run_alpha_verification()
            run_alphabeta_verification()
            run_agent_interaction_verification()
    
    print("Completed!")


if __name__ == "__main__":
    main()
