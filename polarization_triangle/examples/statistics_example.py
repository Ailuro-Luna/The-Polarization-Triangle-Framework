"""
统计分析模块使用示例

这个文件展示了如何使用statistics.py中的各种函数来分析simulation的统计指标
"""

from polarization_triangle.core.config import SimulationConfig, high_polarization_config
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.analysis.statistics import (
    calculate_mean_opinion,
    calculate_variance_metrics,
    calculate_identity_statistics,
    get_polarization_index,
    get_comprehensive_statistics,
    print_statistics_summary,
    export_statistics_to_dict
)
import copy
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def example_usage():
    """展示统计分析模块的基本用法"""
    
    # 创建一个简单的simulation配置
    config = copy.deepcopy(high_polarization_config)
    config.num_agents = 100
    config.steps = 50
    
    # 创建simulation实例
    sim = Simulation(config)
    
    # 运行几步模拟
    print("运行模拟...")
    for i in range(50):
        sim.step()
        if i % 10 == 0:
            print(f"  完成步骤 {i}/50")
    
    print("\n" + "="*60)
    print("统计分析示例")
    print("="*60)
    
    # 方法1：使用print_statistics_summary快速查看
    print("方法1：快速统计摘要")
    print("-" * 30)
    print_statistics_summary(sim)
    
    print("\n" + "="*60)
    
    # 方法2：分别使用各个函数
    print("方法2：分别计算各项指标")
    print("-" * 30)
    
    # 计算平均意见
    mean_stats = calculate_mean_opinion(sim, exclude_zealots=True)
    print(f"平均意见: {mean_stats['mean_opinion']:.4f}")
    print(f"平均绝对意见: {mean_stats['mean_abs_opinion']:.4f}")
    
    # 计算方差指标
    variance_metrics = calculate_variance_metrics(sim, exclude_zealots=True)
    print(f"整体方差: {variance_metrics['overall_variance']:.4f}")
    print(f"社区内部平均方差: {variance_metrics['mean_intra_community_variance']:.4f}")
    
    # 计算身份统计
    identity_stats = calculate_identity_statistics(sim, exclude_zealots=True)
    for identity, stats in identity_stats.items():
        if not identity.startswith('identity_'):
            continue
        print(f"{identity}: 平均意见={stats['mean_opinion']:.4f}, 方差={stats['variance']:.4f}, 数量={stats['count']}")
    
    # 获取极化指数
    polarization = get_polarization_index(sim)
    print(f"极化指数: {polarization:.4f}")
    
    print("\n" + "="*60)
    
    # 方法3：获取综合统计
    print("方法3：综合统计信息")
    print("-" * 30)
    
    comprehensive_stats = get_comprehensive_statistics(sim, exclude_zealots=True)
    print("系统信息:")
    system_info = comprehensive_stats['system_info']
    print(f"  总agents数量: {system_info['num_agents']}")
    print(f"  zealots数量: {system_info['num_zealots']}")
    print(f"  排除zealots: {system_info['exclude_zealots_flag']}")
    
    print("\n综合指标:")
    print(f"  平均意见: {comprehensive_stats['mean_opinion_stats']['mean_opinion']:.4f}")
    print(f"  整体方差: {comprehensive_stats['variance_metrics']['overall_variance']:.4f}")
    print(f"  极化指数: {comprehensive_stats['polarization_index']:.4f}")
    
    print("\n" + "="*60)
    
    # 方法4：导出为字典格式
    print("方法4：导出扁平化数据")
    print("-" * 30)
    
    flat_data = export_statistics_to_dict(sim, exclude_zealots=True)
    print("扁平化数据（前10个键值对）:")
    for i, (key, value) in enumerate(flat_data.items()):
        if i < 10:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  ... 以及其他 {len(flat_data) - 10} 个指标")
            break


def zealot_comparison_example():
    """展示包含和排除zealots的统计对比"""
    
    print("\n" + "="*60)
    print("Zealots对比示例")
    print("="*60)
    
    # 创建带zealots的simulation
    config = copy.deepcopy(high_polarization_config)
    config.num_agents = 80
    config.enable_zealots = True
    config.zealot_count = 10
    config.zealot_mode = "random"
    
    sim_with_zealots = Simulation(config)
    
    # 运行模拟
    print("运行带zealots的模拟...")
    for _ in range(30):
        sim_with_zealots.step()
    
    # 比较包含和排除zealots的统计
    print("\n包含zealots的统计:")
    print("-" * 30)
    stats_with = calculate_mean_opinion(sim_with_zealots, exclude_zealots=False)
    print(f"平均意见: {stats_with['mean_opinion']:.4f}")
    print(f"统计的agents数量: {stats_with['total_agents']}")
    
    print("\n排除zealots的统计:")
    print("-" * 30)
    stats_without = calculate_mean_opinion(sim_with_zealots, exclude_zealots=True)
    print(f"平均意见: {stats_without['mean_opinion']:.4f}")
    print(f"统计的agents数量: {stats_without['total_agents']}")
    
    print(f"\n差异:")
    print(f"平均意见差异: {abs(stats_with['mean_opinion'] - stats_without['mean_opinion']):.4f}")
    print(f"agents数量差异: {stats_with['total_agents'] - stats_without['total_agents']}")


def main():
    """主函数"""
    print("🔬 Polarization Triangle Framework - Statistics Analysis Example")
    print("="*70)
    
    try:
        # 运行基本示例
        example_usage()
        
        # 运行zealot对比示例
        zealot_comparison_example()
        
        print("\n🎉 所有示例运行完成！")
        
    except Exception as e:
        print(f"❌ 示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 