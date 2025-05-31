"""
Test suite for the Zealot and Morality Analysis experiment

This file contains tests to validate the functionality of the 
zealot_morality_analysis experiment module.
"""

import numpy as np
import os
import tempfile
import shutil
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from polarization_triangle.experiments.zealot_morality_analysis import (
    create_config_combinations,
    run_single_simulation,
    run_parameter_sweep,
    run_zealot_morality_analysis
)
from polarization_triangle.core.config import high_polarization_config
import copy


def test_create_config_combinations():
    """测试参数组合创建功能"""
    print("测试参数组合创建功能...")
    
    try:
        combinations = create_config_combinations()
        
        # 验证返回的数据结构
        assert isinstance(combinations, dict)
        assert 'zealot_numbers' in combinations
        assert 'morality_ratios' in combinations
        
        # 验证zealot_numbers组合
        zealot_combos = combinations['zealot_numbers']
        assert len(zealot_combos) > 0
        for combo in zealot_combos:
            assert 'zealot_mode' in combo
            assert 'morality_rate' in combo
            assert 'label' in combo
            assert combo['zealot_mode'] in ['random', 'clustered']
        
        # 验证morality_ratios组合
        morality_combos = combinations['morality_ratios']
        assert len(morality_combos) > 0
        for combo in morality_combos:
            assert 'zealot_count' in combo
            assert 'zealot_mode' in combo
            assert 'zealot_identity_allocation' in combo
            assert 'cluster_identity' in combo
            assert 'label' in combo
        
        print("✅ 参数组合创建测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 参数组合创建测试失败: {str(e)}")
        return False


def test_run_single_simulation():
    """测试单次模拟运行功能"""
    print("测试单次模拟运行功能...")
    
    try:
        # 创建简单配置
        config = copy.deepcopy(high_polarization_config)
        config.num_agents = 30
        config.enable_zealots = True
        config.zealot_count = 5
        config.zealot_mode = "random"
        
        # 运行模拟
        stats = run_single_simulation(config, steps=20)
        
        # 验证返回的统计指标
        expected_keys = ['mean_opinion', 'variance', 'variance_per_identity', 'polarization_index']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
            assert not np.isnan(stats[key])
        
        print("✅ 单次模拟运行测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 单次模拟运行测试失败: {str(e)}")
        return False


def test_run_parameter_sweep():
    """测试参数扫描功能"""
    print("测试参数扫描功能...")
    
    try:
        # 创建测试用的参数组合
        combination = {
            'zealot_mode': 'random',
            'morality_rate': 0.2,
            'zealot_identity_allocation': True,
            'cluster_identity': False,
            'label': 'Test Combination'
        }
        
        # 测试zealot numbers的参数扫描
        x_values = [0, 5, 10]
        results = run_parameter_sweep('zealot_numbers', combination, x_values, num_runs=2)
        
        # 验证返回的数据结构
        expected_metrics = ['mean_opinion', 'variance', 'variance_per_identity', 'polarization_index']
        for metric in expected_metrics:
            assert metric in results
            assert len(results[metric]) == len(x_values)
            for x_runs in results[metric]:
                assert len(x_runs) == 2  # num_runs = 2
        
        print("✅ 参数扫描测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 参数扫描测试失败: {str(e)}")
        return False


def test_minimal_analysis():
    """测试最小化分析（快速测试）"""
    print("测试最小化分析...")
    
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 运行最小化分析
            run_zealot_morality_analysis(
                output_dir=temp_dir,
                num_runs=1,     # 最少运行次数
                max_zealots=4,  # 最少zealot数量
                max_morality=4  # 最少morality ratio
            )
            
            # 验证输出文件
            expected_plots = []
            for plot_type in ['zealot_numbers', 'morality_ratios']:
                for metric in ['mean_opinion', 'variance', 'variance_per_identity', 'polarization_index']:
                    expected_plots.append(f"{plot_type}_{metric}.png")
            
            for plot_file in expected_plots:
                plot_path = os.path.join(temp_dir, plot_file)
                assert os.path.exists(plot_path), f"Missing plot file: {plot_file}"
            
            # 验证实验信息文件
            info_file = os.path.join(temp_dir, "experiment_info.txt")
            assert os.path.exists(info_file), "Missing experiment info file"
            
            print("✅ 最小化分析测试通过")
            return True
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ 最小化分析测试失败: {str(e)}")
        return False


def test_config_validation():
    """测试配置验证功能"""
    print("测试配置验证功能...")
    
    try:
        # 测试zealot numbers配置
        combinations = create_config_combinations()
        
        for combo in combinations['zealot_numbers']:
            # 验证必要的配置参数
            assert 'zealot_mode' in combo
            assert 'morality_rate' in combo
            assert isinstance(combo['morality_rate'], (int, float))
            assert combo['morality_rate'] >= 0.0
            assert combo['morality_rate'] <= 1.0
        
        for combo in combinations['morality_ratios']:
            # 验证必要的配置参数
            assert 'zealot_count' in combo
            assert 'zealot_mode' in combo
            assert isinstance(combo['zealot_count'], int)
            assert combo['zealot_count'] >= 0
        
        print("✅ 配置验证测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置验证测试失败: {str(e)}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("开始运行Zealot and Morality Analysis实验测试套件")
    print("=" * 70)
    
    test_functions = [
        test_create_config_combinations,
        test_run_single_simulation,
        test_run_parameter_sweep,
        test_config_validation,
        test_minimal_analysis
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
    
    print("=" * 70)
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