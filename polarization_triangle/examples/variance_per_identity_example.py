#!/usr/bin/env python3
"""
Variance Per Identity 指标的 Sobol 敏感性分析示例

本示例演示如何使用新增的三个 variance per identity 指标：
1. variance_per_identity_1: identity=1群体的意见方差
2. variance_per_identity_neg1: identity=-1群体的意见方差  
3. variance_per_identity_mean: 两个群体方差的均值

这些指标帮助分析不同身份群体内部的意见分化程度对模型参数的敏感性。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from polarization_triangle.analysis import (
        SobolAnalyzer, 
        SobolConfig, 
        SensitivityVisualizer
    )
    from polarization_triangle.core.config import SimulationConfig
    
    SENSITIVITY_AVAILABLE = True
except ImportError as e:
    print(f"敏感性分析模块不可用: {e}")
    print("请安装依赖: pip install SALib pandas seaborn openpyxl")
    SENSITIVITY_AVAILABLE = False


def analyze_variance_per_identity_sensitivity():
    """分析 variance per identity 指标的参数敏感性"""
    if not SENSITIVITY_AVAILABLE:
        print("❌ 敏感性分析模块不可用，请安装必要依赖")
        return None
    
    print("🔬 Variance Per Identity 敏感性分析示例")
    print("=" * 60)
    
    # 创建分析配置
    config = SobolConfig(
        n_samples=100,      # 适中的样本数，实际使用可增加到500+
        n_runs=3,           # 每个参数组合运行3次
        n_processes=2,      # 使用2个进程并行
        num_steps=100,      # 模拟步数
        output_dir="variance_per_identity_analysis",
        
        # 自定义参数范围（可选）
        parameter_bounds={
            'alpha': [0.2, 0.7],        # 自我激活系数
            'beta': [0.08, 0.25],       # 社会影响系数
            'gamma': [0.5, 1.8],        # 道德化影响系数
            'cohesion_factor': [0.1, 0.4]  # 身份凝聚力因子
        },
        
        # 自定义基础模拟配置
        base_config=SimulationConfig(
            num_agents=150,
            network_type='erdos_renyi',
            network_params={'p': 0.08},
            opinion_distribution='twin_peak',
            morality_rate=0.4,
            cluster_identity=True,
            enable_zealots=True,
            zealot_count=10,
            zealot_mode='random'
        )
    )
    
    print(f"分析配置:")
    print(f"  样本数: {config.n_samples}")
    print(f"  运行次数: {config.n_runs}")
    print(f"  总模拟次数: {config.n_samples * (2 * 4 + 2) * config.n_runs}")
    print(f"  预计耗时: ~{config.n_samples * (2 * 4 + 2) * config.n_runs / 100:.1f} 分钟")
    print(f"  输出目录: {config.output_dir}")
    
    # 运行敏感性分析
    print("\n🚀 开始运行敏感性分析...")
    try:
        analyzer = SobolAnalyzer(config)
        sensitivity_indices = analyzer.run_complete_analysis()
        
        print(f"\n✅ 分析完成！共分析了 {len(sensitivity_indices)} 个输出指标")
        
        # 专门分析 variance per identity 指标
        variance_metrics = [
            'variance_per_identity_1',
            'variance_per_identity_neg1', 
            'variance_per_identity_mean'
        ]
        
        print("\n📊 Variance Per Identity 指标的敏感性结果:")
        print("-" * 50)
        
        param_names = ['α', 'β', 'γ', 'cohesion_factor']
        
        for metric in variance_metrics:
            if metric in sensitivity_indices:
                indices = sensitivity_indices[metric]
                
                print(f"\n🎯 {metric}:")
                print("  参数 \t\t一阶敏感性(S1)\t总敏感性(ST)\t交互效应(ST-S1)")
                print("  " + "-" * 60)
                
                for i, param in enumerate(param_names):
                    s1 = indices['S1'][i]
                    st = indices['ST'][i]
                    interaction = st - s1
                    
                    # 判断敏感性等级
                    if st > 0.3:
                        level = "🔴 极高"
                    elif st > 0.15:
                        level = "🟡 高"
                    elif st > 0.05:
                        level = "🟢 中等"
                    else:
                        level = "⚪ 低"
                    
                    print(f"  {param:12}\t{s1:8.3f}\t\t{st:8.3f}\t\t{interaction:8.3f}\t{level}")
        
        # 识别对 variance per identity 最敏感的参数
        print("\n🔍 重要发现:")
        print("-" * 30)
        
        # 计算每个参数对所有 variance per identity 指标的平均敏感性
        param_importance = {}
        for i, param in enumerate(param_names):
            total_sensitivity = 0
            count = 0
            for metric in variance_metrics:
                if metric in sensitivity_indices:
                    total_sensitivity += sensitivity_indices[metric]['ST'][i]
                    count += 1
            param_importance[param] = total_sensitivity / count if count > 0 else 0
        
        # 按敏感性排序
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("对 Variance Per Identity 指标影响的参数重要性排序:")
        for i, (param, importance) in enumerate(sorted_params, 1):
            print(f"  {i}. {param}: {importance:.3f}")
        
        # 生成详细报告
        print("\n📄 生成详细报告...")
        
        # 生成摘要表
        summary_df = analyzer.get_summary_table()
        variance_summary = summary_df[summary_df['Output'].isin(variance_metrics)]
        
        print("\nVariance Per Identity 指标摘要表:")
        print(variance_summary.to_string(index=False))
        
        # 导出Excel报告
        analyzer.export_results()
        print(f"\n📁 详细结果已导出到: {config.output_dir}/sobol_results.xlsx")
        
        # 生成可视化图表
        print("\n🎨 生成可视化图表...")
        visualizer = SensitivityVisualizer()
        plot_dir = os.path.join(config.output_dir, "plots")
        plot_files = visualizer.create_comprehensive_report(
            sensitivity_indices,
            analyzer.param_samples,
            analyzer.simulation_results,
            plot_dir
        )
        
        print(f"📊 可视化图表已保存到: {plot_dir}")
        print("   包含针对所有指标（包括新的variance per identity指标）的：")
        print("   - 敏感性对比图")
        print("   - 敏感性热力图") 
        print("   - 交互效应分析图")
        print("   - 参数重要性排序图")
        
        return analyzer, sensitivity_indices
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def interpret_variance_per_identity_results():
    """解释 variance per identity 结果的含义"""
    print("\n" + "=" * 60)
    print("📖 Variance Per Identity 指标解释指南")
    print("=" * 60)
    
    print("""
🎯 指标含义:

1. variance_per_identity_1 (identity=1群体方差):
   - 测量identity=1群体内部的意见分化程度
   - 高值表示该群体内部意见分歧较大
   - 低值表示该群体内部意见较为一致

2. variance_per_identity_neg1 (identity=-1群体方差):
   - 测量identity=-1群体内部的意见分化程度
   - 高值表示该群体内部意见分歧较大
   - 低值表示该群体内部意见较为一致

3. variance_per_identity_mean (群体平均方差):
   - 两个身份群体方差的均值
   - 反映系统整体的群体内部分化水平
   - 有助于理解身份认同对内部一致性的影响

🔍 敏感性分析解读:

• 高敏感性参数 (ST > 0.15):
  - 这些参数对群体内部意见分化有显著影响
  - 在实际应用中需要特别关注这些参数的设置

• 强交互效应 (ST - S1 > 0.1):
  - 参数间存在显著的协同或拮抗作用
  - 需要综合考虑参数组合的影响

• 参数作用机制:
  - α (自我激活): 影响个体坚持观点的程度
  - β (社会影响): 影响邻居观点的传播强度
  - γ (道德化影响): 调节道德化对社会影响的抑制
  - cohesion_factor (凝聚力): 增强身份群体内部连接

💡 实际应用建议:

1. 如果关注群体内部一致性，重点调节敏感性最高的参数
2. 如果发现某个身份群体特别敏感，可能需要针对性的干预策略
3. 平均方差指标有助于评估整体的极化风险
4. 结合其他极化指标综合分析，获得更全面的理解
""")


def main():
    """主函数"""
    print("🚀 Variance Per Identity 敏感性分析完整示例")
    print("=" * 70)
    
    # 运行敏感性分析
    analyzer, sensitivity_indices = analyze_variance_per_identity_sensitivity()
    
    if analyzer and sensitivity_indices:
        # 提供结果解释
        interpret_variance_per_identity_results()
        
        print(f"\n🎉 示例完成！")
        print(f"📁 完整结果保存在: {analyzer.config.output_dir}")
        print("\n💡 后续步骤建议:")
        print("1. 查看Excel报告了解详细数值")
        print("2. 查看可视化图表理解敏感性模式")
        print("3. 根据敏感性结果调整模型参数") 
        print("4. 与其他输出指标的敏感性结果进行对比分析")
        
    else:
        print("❌ 示例运行失败，请检查环境配置")


if __name__ == "__main__":
    main() 