"""
Test suite for the statistics analysis module

This file contains tests to validate the functionality of the statistics
analysis functions in polarization_triangle.analysis.statistics
"""

import numpy as np
import copy
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from polarization_triangle.core.config import SimulationConfig, high_polarization_config
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.analysis.statistics import (
    calculate_mean_opinion,
    calculate_variance_metrics,
    calculate_identity_statistics,
    get_polarization_index,
    get_comprehensive_statistics,
    export_statistics_to_dict
)


def test_mean_opinion_calculation():
    """测试平均意见计算功能"""
    print("测试平均意见计算功能...")
    
    try:
        # 创建简单配置
        config = copy.deepcopy(high_polarization_config)
        config.num_agents = 50
        
        # 创建simulation
        sim = Simulation(config)
        
        # 运行几步
        for _ in range(10):
            sim.step()
        
        # 测试平均意见计算
        mean_stats = calculate_mean_opinion(sim, exclude_zealots=True)
        
        # 验证返回的数据结构
        assert 'mean_opinion' in mean_stats
        assert 'mean_abs_opinion' in mean_stats
        assert 'total_agents' in mean_stats
        assert 'excluded_zealots' in mean_stats
        
        # 验证数值合理性
        assert isinstance(mean_stats['mean_opinion'], float)
        assert isinstance(mean_stats['mean_abs_opinion'], float)
        assert mean_stats['mean_abs_opinion'] >= 0
        assert mean_stats['total_agents'] > 0
        
        print("✅ 平均意见计算测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 平均意见计算测试失败: {str(e)}")
        return False


def test_variance_metrics():
    """测试方差指标计算功能"""
    print("测试方差指标计算功能...")
    
    try:
        # 创建simulation
        config = copy.deepcopy(high_polarization_config)
        config.num_agents = 50
        sim = Simulation(config)
        
        # 运行几步
        for _ in range(10):
            sim.step()
        
        # 测试方差计算
        variance_stats = calculate_variance_metrics(sim, exclude_zealots=True)
        
        # 验证返回的数据结构
        assert 'overall_variance' in variance_stats
        assert 'mean_intra_community_variance' in variance_stats
        assert 'num_communities' in variance_stats
        assert 'community_details' in variance_stats
        
        # 验证数值合理性
        assert isinstance(variance_stats['overall_variance'], float)
        assert variance_stats['overall_variance'] >= 0
        assert isinstance(variance_stats['mean_intra_community_variance'], float)
        assert variance_stats['mean_intra_community_variance'] >= 0
        assert variance_stats['num_communities'] > 0
        
        print("✅ 方差指标计算测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 方差指标计算测试失败: {str(e)}")
        return False


def test_identity_statistics():
    """测试身份统计功能"""
    print("测试身份统计功能...")
    
    try:
        # 创建simulation
        config = copy.deepcopy(high_polarization_config)
        config.num_agents = 50
        sim = Simulation(config)
        
        # 运行几步
        for _ in range(10):
            sim.step()
        
        # 测试身份统计
        identity_stats = calculate_identity_statistics(sim, exclude_zealots=True)
        
        # 验证返回的数据结构
        assert isinstance(identity_stats, dict)
        
        # 检查是否有身份数据
        identity_keys = [k for k in identity_stats.keys() if k.startswith('identity_') and k != 'identity_difference']
        assert len(identity_keys) > 0
        
        # 验证每个身份的统计数据
        for key in identity_keys:
            identity_data = identity_stats[key]
            assert 'mean_opinion' in identity_data
            assert 'variance' in identity_data
            assert 'count' in identity_data
            assert 'mean_abs_opinion' in identity_data
            
            assert isinstance(identity_data['mean_opinion'], float)
            assert isinstance(identity_data['variance'], float)
            assert identity_data['variance'] >= 0
            assert identity_data['count'] > 0
        
        print("✅ 身份统计测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 身份统计测试失败: {str(e)}")
        return False


def test_polarization_index():
    """测试极化指数计算功能"""
    print("测试极化指数计算功能...")
    
    try:
        # 创建simulation
        config = copy.deepcopy(high_polarization_config)
        config.num_agents = 50
        sim = Simulation(config)
        
        # 运行几步
        for _ in range(10):
            sim.step()
        
        # 测试极化指数计算
        polarization = get_polarization_index(sim)
        
        # 验证返回值
        assert isinstance(polarization, float)
        assert polarization >= 0  # 极化指数应该是非负数
        
        print("✅ 极化指数计算测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 极化指数计算测试失败: {str(e)}")
        return False


def test_comprehensive_statistics():
    """测试综合统计功能"""
    print("测试综合统计功能...")
    
    try:
        # 创建simulation
        config = copy.deepcopy(high_polarization_config)
        config.num_agents = 50
        sim = Simulation(config)
        
        # 运行几步
        for _ in range(10):
            sim.step()
        
        # 测试综合统计
        comprehensive_stats = get_comprehensive_statistics(sim, exclude_zealots=True)
        
        # 验证返回的数据结构
        expected_keys = [
            'mean_opinion_stats',
            'variance_metrics',
            'identity_statistics', 
            'polarization_index',
            'system_info'
        ]
        
        for key in expected_keys:
            assert key in comprehensive_stats, f"Missing key: {key}"
        
        # 验证system_info
        system_info = comprehensive_stats['system_info']
        assert 'num_agents' in system_info
        assert 'num_zealots' in system_info
        assert 'exclude_zealots_flag' in system_info
        
        print("✅ 综合统计测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 综合统计测试失败: {str(e)}")
        return False


def test_export_to_dict():
    """测试导出为字典功能"""
    print("测试导出为字典功能...")
    
    try:
        # 创建simulation
        config = copy.deepcopy(high_polarization_config)
        config.num_agents = 50
        sim = Simulation(config)
        
        # 运行几步
        for _ in range(10):
            sim.step()
        
        # 测试导出功能
        flat_dict = export_statistics_to_dict(sim, exclude_zealots=True)
        
        # 验证返回的是字典
        assert isinstance(flat_dict, dict)
        
        # 验证必要的键存在
        expected_keys = [
            'num_agents',
            'mean_opinion',
            'overall_variance',
            'polarization_index'
        ]
        
        for key in expected_keys:
            assert key in flat_dict, f"Missing key: {key}"
        
        # 验证所有值都是数值类型
        for key, value in flat_dict.items():
            assert isinstance(value, (int, float)), f"Key {key} has non-numeric value: {type(value)}"
        
        print("✅ 导出为字典测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 导出为字典测试失败: {str(e)}")
        return False


def test_zealot_exclusion():
    """测试zealot排除功能"""
    print("测试zealot排除功能...")
    
    try:
        # 创建带zealots的simulation
        config = copy.deepcopy(high_polarization_config)
        config.num_agents = 50
        config.enable_zealots = True
        config.zealot_count = 5
        config.zealot_mode = "random"
        
        sim = Simulation(config)
        
        # 运行几步
        for _ in range(10):
            sim.step()
        
        # 测试包含zealots的统计
        stats_with_zealots = calculate_mean_opinion(sim, exclude_zealots=False)
        
        # 测试排除zealots的统计
        stats_without_zealots = calculate_mean_opinion(sim, exclude_zealots=True)
        
        # 验证排除zealots后统计的agent数量减少
        assert stats_without_zealots['total_agents'] < stats_with_zealots['total_agents']
        assert stats_without_zealots['total_agents'] == config.num_agents - config.zealot_count
        
        print("✅ zealot排除功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ zealot排除功能测试失败: {str(e)}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始运行statistics模块测试套件")
    print("=" * 60)
    
    test_functions = [
        test_mean_opinion_calculation,
        test_variance_metrics,
        test_identity_statistics,
        test_polarization_index,
        test_comprehensive_statistics,
        test_export_to_dict,
        test_zealot_exclusion
    ]
    
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 出现异常: {str(e)}")
        print()
    
    print("=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！")
        return True
    else:
        print(f"⚠️  有 {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 