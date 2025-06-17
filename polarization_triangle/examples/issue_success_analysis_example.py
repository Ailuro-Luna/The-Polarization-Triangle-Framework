"""
议题成功度量分析示例

该示例展示如何使用新的议题成功度量指标（ISMI）来区分：
1. 成功推广的议题（形成新共识）
2. 引起极化的议题（导致分化）

核心研究问题：Under which conditions does a new issue have most potential 
to create social change vs lead to opinion segregation?
"""

import numpy as np
import matplotlib.pyplot as plt
from polarization_triangle.core.config import SimulationConfig
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.analysis.issue_success_metrics import (
    IssueSuccessMetrics, 
    analyze_issue_success_across_conditions
)
from polarization_triangle.utils.network import create_network
import pandas as pd

def create_example_scenarios():
    """创建不同的网络和参数场景来测试议题传播效果"""
    
    scenarios = []
    
    # 场景1: 高度连接的随机网络 + 低道德化率 + 强社会影响
    # 预期：有利于共识形成
    config1 = SimulationConfig(
        num_agents=100,
        network_type="random",
        network_params={"p": 0.3},  # 高连接概率
        morality_rate=0.1,          # 低道德化率
        alpha=0.3,                  # 中等自我激活
        beta=0.2,                   # 强社会影响
        gamma=0.8,                  # 中等道德化抑制
        zealot_count=5,             # 5个道德创新者
        zealot_opinion=1.0,         # 正向意见
        enable_zealots=True
    )
    scenarios.append(("High Connection + Low Morality", config1))
    
    # 场景2: 社区网络 + 高道德化率 + 强自我激活
    # 预期：容易引起极化
    config2 = SimulationConfig(
        num_agents=100,
        network_type="community",
        network_params={"num_communities": 4, "p_in": 0.6, "p_out": 0.1},
        morality_rate=0.7,          # 高道德化率
        alpha=0.6,                  # 强自我激活
        beta=0.1,                   # 弱社会影响
        gamma=1.5,                  # 强道德化抑制
        zealot_count=5,
        zealot_opinion=1.0,
        enable_zealots=True,
        cluster_identity=True,      # 身份按社区聚类
        cluster_morality=True       # 道德化按社区聚类
    )
    scenarios.append(("Community + High Morality", config2))
    
    # 场景3: 小世界网络 + 中等道德化率 + 平衡参数
    # 预期：产生复杂的传播模式
    config3 = SimulationConfig(
        num_agents=100,
        network_type="ws",
        network_params={"k": 6, "p": 0.3},
        morality_rate=0.4,          # 中等道德化率
        alpha=0.4,                  # 平衡自我激活
        beta=0.15,                  # 平衡社会影响
        gamma=1.0,                  # 标准道德化抑制
        zealot_count=8,             # 更多道德创新者
        zealot_opinion=0.8,         # 稍微温和的意见
        enable_zealots=True
    )
    scenarios.append(("Small World + Moderate Morality", config3))
    
    # 场景4: 无标度网络 + 低道德化率 + 集中化Zealot
    # 预期：通过hub节点快速传播
    config4 = SimulationConfig(
        num_agents=100,
        network_type="ba",
        network_params={"m": 3},
        morality_rate=0.2,          # 低道德化率
        alpha=0.3,                  # 低自我激活
        beta=0.18,                  # 强社会影响
        gamma=0.5,                  # 弱道德化抑制
        zealot_count=3,             # 少数关键创新者
        zealot_mode="degree",       # 选择高度节点作为Zealot
        zealot_opinion=1.0,
        enable_zealots=True
    )
    scenarios.append(("Scale-Free + Hub Zealots", config4))
    
    return scenarios

def run_comparative_analysis():
    """运行比较分析"""
    
    print("=== 议题成功度量分析 ===")
    print("正在创建不同的网络场景...")
    
    scenarios = create_example_scenarios()
    simulations = []
    conditions = []
    
    # 运行每个场景的仿真
    for condition_name, config in scenarios:
        print(f"\n运行场景: {condition_name}")
        
        # 创建仿真
        sim = Simulation(config)
        
        # 保存初始状态用于方向性分析
        sim.initial_opinions = sim.opinions.copy()
        
        # 运行仿真
        trajectory = []
        steps = 200
        
        for step in range(steps):
            trajectory.append(sim.opinions.copy())
            sim.step()
            
            # 每50步输出进度
            if (step + 1) % 50 == 0:
                mean_opinion = np.mean(sim.opinions)
                polarization = sim.calculate_polarization_index()
                print(f"  步数 {step + 1}: 平均意见 = {mean_opinion:.3f}, 极化指数 = {polarization:.3f}")
        
        # 保存结果
        sim.trajectory = trajectory  # 添加轨迹数据
        simulations.append(sim)
        conditions.append(condition_name)
    
    # 使用新的指标体系分析结果
    print("\n=== 使用议题成功度量指标分析结果 ===")
    
    metrics = IssueSuccessMetrics()
    
    # 分析每个条件下的结果
    for i, (sim, condition) in enumerate(zip(simulations, conditions)):
        print(f"\n【场景 {i+1}: {condition}】")
        
        # 计算综合指标
        results = metrics.calculate_issue_success_index(
            sim,
            moral_innovator_opinion=1.0,  # Zealot的目标意见
            trajectory=sim.trajectory
        )
        
        # 生成报告
        report = metrics.generate_success_report(results)
        print(report)
        
        # 额外的分析
        print(f"\n【详细分析】")
        print(f"最终平均意见: {np.mean(sim.opinions):.4f}")
        print(f"意见标准差: {np.std(sim.opinions):.4f}")
        print(f"极端意见比例 (|opinion| > 0.7): {np.mean(np.abs(sim.opinions) > 0.7):.3f}")
    
    # 跨条件比较
    print("\n=== 跨条件比较分析 ===")
    
    # 使用批量分析功能
    all_results = analyze_issue_success_across_conditions(
        simulations, conditions, moral_innovator_opinion=1.0
    )
    
    # 创建比较表格
    comparison_data = []
    for condition, results in all_results.items():
        comparison_data.append({
            'Condition': condition,
            'ISMI': results.get('issue_success_index', 0),
            'Polarization': results.get('polarization_index', 0),
            'Consensus_Convergence': results.get('consensus_convergence_index', 0),
            'Identity_Bridging': results.get('identity_bridging_index', 0),
            'Outcome_Type': results.get('issue_outcome_type', 'Unknown')
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n比较表格:")
    print(df.to_string(index=False, float_format='%.4f'))
    
    # 找出最优条件
    print(f"\n=== 最有利于议题成功推广的条件 ===")
    
    # 按ISMI排序
    df_sorted = df.sort_values('ISMI', ascending=False)
    print("\n按议题成功指数排序:")
    for idx, row in df_sorted.iterrows():
        print(f"{row['Condition']}: ISMI = {row['ISMI']:.4f}, 类型 = {row['Outcome_Type']}")
    
    # 分析成功 vs 极化的权衡
    print(f"\n=== 成功推广 vs 极化的权衡分析 ===")
    
    for _, row in df.iterrows():
        ratio = row['ISMI'] / row['Polarization'] if row['Polarization'] > 0 else float('inf')
        print(f"{row['Condition']}: 成功/极化比率 = {ratio:.2f}")
    
    return simulations, all_results

def visualize_results(simulations, results):
    """可视化分析结果"""
    
    conditions = list(results.keys())
    
    # 创建综合可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('议题成功度量分析结果', fontsize=16)
    
    # 1. ISMI vs 极化指数散点图
    ax1 = axes[0, 0]
    ismi_values = [results[cond].get('issue_success_index', 0) for cond in conditions]
    polar_values = [results[cond].get('polarization_index', 0) for cond in conditions]
    
    ax1.scatter(polar_values, ismi_values, s=100, alpha=0.7)
    for i, cond in enumerate(conditions):
        ax1.annotate(cond, (polar_values[i], ismi_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('极化指数')
    ax1.set_ylabel('议题成功指数 (ISMI)')
    ax1.set_title('成功度 vs 极化程度')
    ax1.grid(True, alpha=0.3)
    
    # 添加理想区域标记
    ax1.axhspan(0.7, 1.0, alpha=0.2, color='green', label='高成功区域')
    ax1.axvspan(0, 0.3, alpha=0.2, color='lightblue', label='低极化区域')
    ax1.legend()
    
    # 2. 子指标雷达图
    ax2 = axes[0, 1]
    
    # 选择最成功和最极化的条件进行对比
    best_condition = max(conditions, key=lambda c: results[c].get('issue_success_index', 0))
    most_polarized = max(conditions, key=lambda c: results[c].get('polarization_index', 0))
    
    categories = ['共识收敛', '方向性影响', '身份跨越', '时间稳定']
    
    best_values = [
        results[best_condition].get('consensus_convergence_index', 0),
        results[best_condition].get('directional_influence_index', 0),
        results[best_condition].get('identity_bridging_index', 0),
        results[best_condition].get('temporal_stability_index', 0)
    ]
    
    polar_values = [
        results[most_polarized].get('consensus_convergence_index', 0),
        results[most_polarized].get('directional_influence_index', 0),
        results[most_polarized].get('identity_bridging_index', 0),
        results[most_polarized].get('temporal_stability_index', 0)
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x - width/2, best_values, width, label=f'最成功: {best_condition}', alpha=0.8)
    ax2.bar(x + width/2, polar_values, width, label=f'最极化: {most_polarized}', alpha=0.8)
    
    ax2.set_xlabel('指标类型')
    ax2.set_ylabel('指标值')
    ax2.set_title('子指标对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 最终意见分布对比
    ax3 = axes[1, 0]
    
    # 选择两个极端情况展示
    sim_best = simulations[conditions.index(best_condition)]
    sim_polar = simulations[conditions.index(most_polarized)]
    
    ax3.hist(sim_best.opinions, bins=20, alpha=0.7, label=f'最成功: {best_condition}', density=True)
    ax3.hist(sim_polar.opinions, bins=20, alpha=0.7, label=f'最极化: {most_polarized}', density=True)
    
    ax3.set_xlabel('意见值')
    ax3.set_ylabel('密度')
    ax3.set_title('最终意见分布对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 条件排名
    ax4 = axes[1, 1]
    
    # 按ISMI排序
    sorted_conditions = sorted(conditions, key=lambda c: results[c].get('issue_success_index', 0), reverse=True)
    sorted_values = [results[c].get('issue_success_index', 0) for c in sorted_conditions]
    
    bars = ax4.barh(range(len(sorted_conditions)), sorted_values)
    ax4.set_yticks(range(len(sorted_conditions)))
    ax4.set_yticklabels([c.replace(' + ', '\n') for c in sorted_conditions], fontsize=8)
    ax4.set_xlabel('议题成功指数 (ISMI)')
    ax4.set_title('条件排名 (按成功度)')
    
    # 颜色编码
    for i, bar in enumerate(bars):
        if sorted_values[i] > 0.7:
            bar.set_color('green')
        elif sorted_values[i] > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    
    print("开始议题成功度量分析示例...")
    
    # 运行比较分析
    simulations, results = run_comparative_analysis()
    
    # 可视化结果
    print("\n生成可视化结果...")
    visualize_results(simulations, results)
    
    print("\n=== 总结 ===")
    print("通过议题成功度量指数（ISMI），我们可以：")
    print("1. 区分哪些条件有利于议题成功推广（形成共识）")
    print("2. 识别哪些条件容易引起极化分化")
    print("3. 量化评估不同网络结构和参数对议题传播的影响")
    print("4. 为政策制定和干预策略提供数据支持")

if __name__ == "__main__":
    main() 