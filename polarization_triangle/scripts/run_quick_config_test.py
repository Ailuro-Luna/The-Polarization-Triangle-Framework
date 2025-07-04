#!/usr/bin/env python3
"""
快速配置测试脚本
测试少数几个关键配置组合对Sobol敏感性分析结果的影响
用于快速验证和预览分析结果
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from polarization_triangle.analysis.sobol_analysis import SobolAnalyzer, SobolConfig
from polarization_triangle.core.config import SimulationConfig


class QuickConfigTester:
    """快速配置测试器"""
    
    def __init__(self, base_output_dir: str = "results/quick_config_test"):
        self.base_output_dir = base_output_dir
        self.results = {}
        
        # 创建输出目录
        os.makedirs(base_output_dir, exist_ok=True)
        
        # 定义要测试的关键配置组合（选择8个代表性组合）
        self.test_configurations = [
            # 基准配置 - 所有参数为False，zealot_mode为random
            {
                'name': 'baseline',
                'cluster_identity': False,
                'cluster_morality': False,
                'cluster_opinion': False,
                'zealot_morality': False,
                'zealot_identity_allocation': False,
                'zealot_mode': 'random'
            },
            # 所有cluster参数为True
            {
                'name': 'all_clustered',
                'cluster_identity': True,
                'cluster_morality': True,
                'cluster_opinion': True,
                'zealot_morality': False,
                'zealot_identity_allocation': False,
                'zealot_mode': 'random'
            },
            # 所有zealot参数为True
            {
                'name': 'zealot_enabled',
                'cluster_identity': False,
                'cluster_morality': False,
                'cluster_opinion': False,
                'zealot_morality': True,
                'zealot_identity_allocation': True,
                'zealot_mode': 'random'
            },
            # zealot_mode为clustered
            {
                'name': 'zealot_clustered',
                'cluster_identity': False,
                'cluster_morality': False,
                'cluster_opinion': False,
                'zealot_morality': False,
                'zealot_identity_allocation': False,
                'zealot_mode': 'clustered'
            },
            # 混合配置1：部分cluster + 部分zealot
            {
                'name': 'mixed_1',
                'cluster_identity': True,
                'cluster_morality': False,
                'cluster_opinion': True,
                'zealot_morality': True,
                'zealot_identity_allocation': False,
                'zealot_mode': 'random'
            },
            # 混合配置2：另一种组合
            {
                'name': 'mixed_2',
                'cluster_identity': False,
                'cluster_morality': True,
                'cluster_opinion': False,
                'zealot_morality': False,
                'zealot_identity_allocation': True,
                'zealot_mode': 'clustered'
            },
            # 极端配置：所有参数都为True
            {
                'name': 'all_enabled',
                'cluster_identity': True,
                'cluster_morality': True,
                'cluster_opinion': True,
                'zealot_morality': True,
                'zealot_identity_allocation': True,
                'zealot_mode': 'clustered'
            },
            # 特殊测试：只有identity clustering
            {
                'name': 'identity_only',
                'cluster_identity': True,
                'cluster_morality': False,
                'cluster_opinion': False,
                'zealot_morality': False,
                'zealot_identity_allocation': False,
                'zealot_mode': 'random'
            }
        ]
        
        print(f"将测试 {len(self.test_configurations)} 种配置组合")
    
    def create_config_for_test(self, config_dict: Dict[str, Any]) -> SobolConfig:
        """为测试配置创建SobolConfig"""
        # 使用较小的参数以加快测试速度
        base_config = SobolConfig(
            n_samples=200,  # 较小的样本数用于快速测试
            n_runs=3,       # 较少的运行次数
            n_processes=4,  # 适中的进程数
            num_steps=150,  # 较少的模拟步数
            output_dir="temp"
        )
        
        # 创建SimulationConfig
        sim_config = SimulationConfig(
            num_agents=200,
            network_type='lfr',
            network_params={
                'tau1': 3, 'tau2': 1.5, 'mu': 0.1,
                'average_degree': 5, 'min_community': 10
            },
            opinion_distribution='uniform',
            morality_rate=0.3,
            # 设置测试参数（排除name字段）
            cluster_identity=config_dict['cluster_identity'],
            cluster_morality=config_dict['cluster_morality'],
            cluster_opinion=config_dict['cluster_opinion'],
            # Zealot配置
            zealot_count=30,
            enable_zealots=True,
            zealot_mode=config_dict['zealot_mode'],
            zealot_morality=config_dict['zealot_morality'],
            zealot_identity_allocation=config_dict['zealot_identity_allocation']
        )
        
        # 设置基础配置
        base_config.base_config = sim_config
        
        # 设置输出目录
        config_name = config_dict['name']
        base_config.output_dir = os.path.join(self.base_output_dir, config_name)
        
        return base_config
    
    def run_single_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个配置的分析"""
        config_name = config_dict['name']
        print(f"\n开始运行配置: {config_name}")
        
        try:
            # 创建配置
            config = self.create_config_for_test(config_dict)
            
            # 创建分析器
            analyzer = SobolAnalyzer(config)
            
            # 运行分析
            sensitivity_indices = analyzer.run_complete_analysis()
            
            # 提取关注的结果
            result = {
                'config_name': config_name,
                'config_params': {k: v for k, v in config_dict.items() if k != 'name'},
                'sensitivity_indices': sensitivity_indices,
                'success': True
            }
            
            # 如果有opinion_variance结果，单独提取
            if 'opinion_variance' in sensitivity_indices:
                result['opinion_variance_sensitivity'] = sensitivity_indices['opinion_variance']
            
            print(f"配置 {config_name} 运行成功")
            return result
            
        except Exception as e:
            print(f"配置 {config_name} 执行失败: {e}")
            return {
                'config_name': config_name,
                'config_params': {k: v for k, v in config_dict.items() if k != 'name'},
                'error': str(e),
                'success': False
            }
    
    def run_all_configs(self):
        """运行所有配置的分析"""
        print("="*60)
        print("开始快速配置测试")
        print("="*60)
        
        for config_dict in self.test_configurations:
            result = self.run_single_config(config_dict)
            self.results[config_dict['name']] = result
    
    def analyze_results(self):
        """分析结果差异"""
        print("\n" + "="*60)
        print("分析结果差异")
        print("="*60)
        
        # 收集成功的结果
        successful_results = {
            name: result for name, result in self.results.items()
            if result['success']
        }
        
        print(f"成功的分析: {len(successful_results)}/{len(self.results)}")
        
        if not successful_results:
            print("没有成功的分析结果")
            return
        
        # 分析opinion_variance敏感性的差异
        self._analyze_opinion_variance_sensitivity(successful_results)
    
    def _analyze_opinion_variance_sensitivity(self, results: Dict[str, Dict]):
        """分析opinion_variance敏感性的差异"""
        print("\n分析opinion_variance敏感性差异...")
        
        # 收集所有有opinion_variance结果的数据
        ov_data = []
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        
        for config_name, result in results.items():
            if 'opinion_variance_sensitivity' in result:
                ov_sens = result['opinion_variance_sensitivity']
                config_params = result['config_params']
                
                # 提取S1和ST值
                row_data = {
                    'config_name': config_name,
                    **config_params  # 展开配置参数
                }
                
                # 添加敏感性指数
                for i, param in enumerate(param_names):
                    row_data[f'{param}_S1'] = ov_sens['S1'][i]
                    row_data[f'{param}_ST'] = ov_sens['ST'][i]
                    row_data[f'{param}_interaction'] = ov_sens['ST'][i] - ov_sens['S1'][i]
                
                ov_data.append(row_data)
        
        if not ov_data:
            print("没有找到opinion_variance敏感性结果")
            return
        
        # 创建DataFrame
        df = pd.DataFrame(ov_data)
        
        # 保存详细结果
        df.to_csv(os.path.join(self.base_output_dir, 'quick_test_results.csv'), 
                  index=False)
        
        # 打印敏感性对比
        print("\n各配置的敏感性指数对比:")
        print("-" * 80)
        
        for param in param_names:
            st_col = f'{param}_ST'
            if st_col in df.columns:
                print(f"\n{param} (总敏感性指数 ST):")
                for _, row in df.iterrows():
                    print(f"  {row['config_name']:15s}: {row[st_col]:.3f}")
                
                # 显示范围
                min_val = df[st_col].min()
                max_val = df[st_col].max()
                print(f"  范围: [{min_val:.3f}, {max_val:.3f}], 差异: {max_val - min_val:.3f}")
        
        # 找出敏感性变化最大的参数
        print("\n敏感性变化最大的参数:")
        ranges = {}
        for param in param_names:
            st_col = f'{param}_ST'
            if st_col in df.columns:
                ranges[param] = df[st_col].max() - df[st_col].min()
        
        sorted_ranges = sorted(ranges.items(), key=lambda x: x[1], reverse=True)
        for param, range_val in sorted_ranges:
            print(f"  {param}: {range_val:.3f}")
        
        # 保存结果
        self.comparison_results = df
        
        return df
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "="*60)
        print("生成总结报告")
        print("="*60)
        
        # 统计信息
        total_configs = len(self.test_configurations)
        successful_runs = sum(1 for r in self.results.values() if r['success'])
        failed_runs = total_configs - successful_runs
        
        summary = {
            'test_info': {
                'total_configurations': total_configs,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'success_rate': successful_runs / total_configs if total_configs > 0 else 0
            },
            'configurations_tested': [
                {
                    'name': config['name'],
                    'params': {k: v for k, v in config.items() if k != 'name'}
                }
                for config in self.test_configurations
            ],
            'results': {
                name: {
                    'success': result['success'],
                    'error': result.get('error', None)
                }
                for name, result in self.results.items()
            }
        }
        
        # 保存摘要
        with open(os.path.join(self.base_output_dir, 'quick_test_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"总结报告已保存到: {os.path.join(self.base_output_dir, 'quick_test_summary.json')}")
        
        # 打印摘要
        print(f"\n快速测试摘要:")
        print(f"  总配置数: {total_configs}")
        print(f"  成功运行: {successful_runs}")
        print(f"  失败运行: {failed_runs}")
        print(f"  成功率: {successful_runs/total_configs*100:.1f}%")
        
        if failed_runs > 0:
            print(f"\n失败的配置:")
            for name, result in self.results.items():
                if not result['success']:
                    print(f"  {name}: {result.get('error', 'Unknown error')}")


def main():
    """主函数"""
    print("="*60)
    print("快速配置测试")
    print("="*60)
    
    # 创建测试器
    tester = QuickConfigTester()
    
    try:
        start_time = time.time()
        
        # 运行所有配置
        tester.run_all_configs()
        
        # 分析结果
        tester.analyze_results()
        
        # 生成报告
        tester.generate_summary_report()
        
        end_time = time.time()
        print(f"\n总耗时: {end_time - start_time:.2f} 秒")
        print(f"结果保存在: {tester.base_output_dir}")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 