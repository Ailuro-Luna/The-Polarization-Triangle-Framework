#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Single Simulation Script
专门用于运行与zealot_morality_analysis.py配置一致的单次模拟
包含variance分析和详细配置信息输出
"""

import os
import sys
import argparse
import numpy as np
import copy

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from polarization_triangle.core.config import high_polarization_config
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.visualization.network_viz import draw_network
from polarization_triangle.visualization.opinion_viz import draw_opinion_distribution_heatmap
from polarization_triangle.analysis.trajectory import run_simulation_with_trajectory
from polarization_triangle.analysis.statistics import print_statistics_summary


def run_enhanced_single_simulation(output_dir="results/enhanced_single_run", steps=300, 
                         zealot_count=20, zealot_mode="random", zealot_opinion=1.0,
                         zealot_morality=False, zealot_identity_allocation=True,
                         initial_scale=0.1, morality_rate=1.0):
    """
    运行增强的单次模拟并生成详细的分析结果
    与zealot_morality_analysis.py的配置保持一致
    
    参数:
    output_dir: 输出目录
    steps: 模拟步数
    zealot_count: zealot数量（默认20，与zealot_morality_analysis.py一致）
    zealot_mode: zealot选择模式 (random, degree, clustered)
    zealot_opinion: zealot固定意见值
    zealot_morality: zealot是否都是道德化的
    zealot_identity_allocation: 是否只从identity=1的agent中选择zealot
    initial_scale: 初始意见缩放因子，用于模拟对新议题的相对中立态度（默认0.1，即除十）
    morality_rate: 道德化率（默认1.0，即100%）
    """
    
    print(f"🚀 运行增强版单次模拟（与zealot_morality_analysis.py配置一致）...")
    print(f"📊 模拟步数: {steps}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📉 初始意见缩放: {initial_scale} (非zealot agent的初始意见将被缩放到原来的{initial_scale}倍)")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用high_polarization_config（与zealot_morality_analysis.py一致）
    config = copy.deepcopy(high_polarization_config)
    
    # 设置高道德化率
    config.morality_rate = morality_rate
    
    # 配置zealot参数
    has_zealots = zealot_count > 0
    if has_zealots:
        config.enable_zealots = True
        config.zealot_count = zealot_count
        config.zealot_mode = zealot_mode
        config.zealot_opinion = zealot_opinion
        config.zealot_morality = zealot_morality
        config.zealot_identity_allocation = zealot_identity_allocation
    
    print(f"🔧 使用配置: high_polarization_config（与zealot_morality_analysis.py一致）")
    print(f"   Agent数量: {config.num_agents}")
    print(f"   网络类型: {config.network_type}")
    print(f"   道德化率: {config.morality_rate} ({morality_rate*100:.0f}%)")
    print(f"   Alpha参数: {config.alpha} (高极化配置)")
    print(f"   Beta参数: {config.beta}")
    print(f"   身份聚类: {config.cluster_identity}")
    if has_zealots:
        print(f"   🎯 Zealot配置:")
        print(f"      数量: {zealot_count} (与morality_ratios实验一致)")
        print(f"      模式: {zealot_mode}")
        print(f"      意见值: {zealot_opinion}")
        print(f"      道德化: {zealot_morality}")
        print(f"      身份分配: {zealot_identity_allocation}")
    else:
        print(f"   🎯 无Zealot")
    
    # 创建模拟实例
    print("🏗️  创建模拟...")
    sim = Simulation(config)
    
    # 应用初始意见缩放（模拟对新议题的相对中立态度）
    if initial_scale != 1.0:
        print(f"📉 应用初始意见缩放 (scale={initial_scale})...")
        print(f"   缩放前意见范围: [{sim.opinions.min():.3f}, {sim.opinions.max():.3f}]")
        
        # 缩放所有agent的初始意见
        sim.opinions *= initial_scale
        
        # 重新设置zealot的意见，避免被缩放影响
        if has_zealots:
            sim.set_zealot_opinions()
            print(f"   ✅ Zealot意见已重新设置为未缩放的值: {zealot_opinion}")
        
        print(f"   缩放后意见范围: [{sim.opinions.min():.3f}, {sim.opinions.max():.3f}]")
    
    # 显示zealot信息
    if has_zealots:
        zealot_ids = sim.get_zealot_ids()
        print(f"   Zealot IDs: {zealot_ids}")
        print(f"   实际Zealot数量: {len(zealot_ids)}")
        print(f"   Zealot意见值: {[sim.opinions[i] for i in zealot_ids]}")
    
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
    
    # 计算并打印意见方差
    print("\n📊 意见方差分析:")
    print("=" * 50)
    
    # 计算不同范围的意见方差
    all_opinions = sim.opinions
    all_variance = float(np.var(all_opinions))
    
    if has_zealots:
        # 获取zealot和非zealot的意见
        zealot_ids = sim.get_zealot_ids()
        zealot_mask = np.zeros(sim.num_agents, dtype=bool)
        zealot_mask[zealot_ids] = True
        
        non_zealot_opinions = all_opinions[~zealot_mask]
        zealot_opinions = all_opinions[zealot_mask]
        
        non_zealot_variance = float(np.var(non_zealot_opinions)) if len(non_zealot_opinions) > 0 else 0.0
        zealot_variance = float(np.var(zealot_opinions)) if len(zealot_opinions) > 1 else 0.0
        
        print(f"📈 总体意见方差 (包含所有agent): {all_variance:.6f}")
        print(f"📉 非Zealot意见方差: {non_zealot_variance:.6f}")
        print(f"🎯 Zealot意见方差: {zealot_variance:.6f}")
        print(f"📊 非Zealot agent数量: {len(non_zealot_opinions)}")
        print(f"🎯 Zealot agent数量: {len(zealot_opinions)}")
        
        # 按身份分组计算方差
        identities = sim.identities
        non_zealot_identities = identities[~zealot_mask]
        
        for identity_val in [1, -1]:
            identity_mask = non_zealot_identities == identity_val
            if np.sum(identity_mask) > 1:
                identity_opinions = non_zealot_opinions[identity_mask]
                identity_variance = float(np.var(identity_opinions))
                print(f"🏷️  Identity={identity_val} 非Zealot方差: {identity_variance:.6f} (n={len(identity_opinions)})")
            else:
                print(f"🏷️  Identity={identity_val} 非Zealot方差: N/A (n={np.sum(identity_mask)})")
    else:
        print(f"📈 总体意见方差: {all_variance:.6f}")
        
        # 按身份分组计算方差
        identities = sim.identities
        for identity_val in [1, -1]:
            identity_mask = identities == identity_val
            if np.sum(identity_mask) > 1:
                identity_opinions = all_opinions[identity_mask]
                identity_variance = float(np.var(identity_opinions))
                print(f"🏷️  Identity={identity_val} 方差: {identity_variance:.6f} (n={len(identity_opinions)})")
            else:
                print(f"🏷️  Identity={identity_val} 方差: N/A (n={np.sum(identity_mask)})")
    
    # 打印统计摘要
    print("\n📋 详细统计摘要:")
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
    
    print(f"\n🎉 增强版单次模拟完成！结果已保存到: {output_dir}")
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
    print(f"\n⚙️  默认配置详情:")
    print(f"   🔧 基础配置: high_polarization_config")
    print(f"      • Agent数量: {config.num_agents}")
    print(f"      • 网络类型: {config.network_type} (LFR网络)")
    print(f"      • Alpha参数: {config.alpha} (自我激活系数，高极化设置)")
    print(f"      • Beta参数: {config.beta} (社会影响系数)")
    print(f"      • Gamma参数: {config.gamma} (道德化影响系数)")
    print(f"   🏷️  身份配置:")
    print(f"      • 身份聚类: {config.cluster_identity} (身份按社群聚集)")
    print(f"      • 道德化聚类: {config.cluster_morality} (道德化按社群聚集)")
    print(f"      • 意见聚类: {config.cluster_opinion} (意见按社群聚集)")
    print(f"   🎯 Zealot配置:")
    print(f"      • 数量: {zealot_count} (固定数量，与morality_ratios实验一致)")
    print(f"      • 分布模式: {zealot_mode} ({'随机分布' if zealot_mode == 'random' else '聚集分布' if zealot_mode == 'clustered' else '按度数选择'})")
    print(f"      • 身份分配限制: {zealot_identity_allocation} ({'只从identity=1中选择' if zealot_identity_allocation else '从所有agent中选择'})")
    print(f"      • Zealot道德化: {zealot_morality} ({'所有zealot都是道德化的' if zealot_morality else 'zealot不强制道德化'})")
    print(f"   💭 意见初始化:")
    print(f"      • 道德化率: {morality_rate*100:.0f}% (非zealot agent中道德化的比例)")
    print(f"      • 初始意见缩放: {initial_scale} ({'模拟对新议题的中立态度' if initial_scale < 1.0 else '标准初始化，无缩放'})")
    print(f"      • 意见分布: {config.opinion_distribution}")
    
    print(f"\n🎛️  关键默认参数总结:")
    print(f"   • Zealot是否聚集: {'是' if zealot_mode == 'clustered' else '否'} (默认: random分布)")
    print(f"   • Identity是否聚集: {'是' if config.cluster_identity else '否'} (默认: 启用聚类)")
    print(f"   • Zealot是否有特定identity: {'是，仅identity=1' if zealot_identity_allocation else '否，任意identity'} (默认: 仅从identity=1选择)")
    print(f"   • 网络结构: LFR社群网络 (真实社会网络结构)")
    print(f"   • 极化倾向: 高 (alpha=0.6，强自我激活)")
    
    return sim


def main():
    """
    命令行入口函数
    """
    parser = argparse.ArgumentParser(description="增强版单次模拟脚本（与zealot_morality_analysis.py配置一致）")
    
    # 基本参数
    parser.add_argument("--output-dir", type=str, default="results/enhanced_single_run",
                       help="输出目录")
    parser.add_argument("--steps", type=int, default=300,
                       help="模拟步数")
    
    # Zealot相关参数
    parser.add_argument("--zealot-count", type=int, default=20,
                       help="Zealot数量（默认20，与zealot_morality_analysis.py一致）")
    parser.add_argument("--zealot-mode", type=str, choices=["random", "degree", "clustered"], 
                       default="random",
                       help="Zealot选择模式")
    parser.add_argument("--zealot-opinion", type=float, default=1.0,
                       help="Zealot固定意见值")
    parser.add_argument("--zealot-morality", action='store_true',
                       help="使所有zealot都是道德化的")
    parser.add_argument("--zealot-identity-allocation", action='store_true', default=True,
                       help="只将zealot分配给identity=1的agent")
    
    # 增强功能参数
    parser.add_argument("--initial-scale", type=float, default=0.1,
                       help="初始意见缩放因子（默认0.1，模拟对新议题的中立态度）")
    parser.add_argument("--morality-rate", type=float, default=1.0,
                       help="道德化率（0.0-1.0，默认1.0为100%%道德化）")
    
    args = parser.parse_args()
    
    print("🔬 增强版单次模拟脚本")
    print("=" * 50)
    print("功能特点：")
    print("• 与zealot_morality_analysis.py配置完全一致")
    print("• 使用high_polarization_config（alpha=0.6高极化设置）")
    print("• 支持初始意见缩放（模拟中立态度）")
    print("• 详细的意见方差分析")
    print("• 按身份分组的统计数据")
    print("• 综合配置信息输出")
    print("=" * 50)
    
    # 运行增强版单次模拟
    run_enhanced_single_simulation(
        output_dir=args.output_dir,
        steps=args.steps,
        zealot_count=args.zealot_count,
        zealot_mode=args.zealot_mode,
        zealot_opinion=args.zealot_opinion,
        zealot_morality=args.zealot_morality,
        zealot_identity_allocation=args.zealot_identity_allocation,
        initial_scale=args.initial_scale,
        morality_rate=args.morality_rate
    )
    
    print("\n✨ 增强版单次模拟完成！")
    print("\n📌 使用说明:")
    print("该脚本专门用于运行与zealot_morality_analysis.py实验配置一致的单次模拟")
    print("相比原版main.py的single simulation，增加了：")
    print("• 详细的意见方差分析（总体、非zealot、zealot、按身份分组）")
    print("• 完整的配置参数说明")
    print("• 初始意见缩放功能（模拟对新议题的中立态度）")
    print("• 高道德化率设置（默认100%）")
    print("• 使用高极化配置（alpha=0.6而不是0.4）")


if __name__ == "__main__":
    main() 