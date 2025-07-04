#!/usr/bin/env python3
"""
配置对比测试脚本
测试不同配置参数组合对Sobol敏感性分析结果的影响
重点关注四个参数(α, β, γ, cohesion_factor)对opinion_variance的敏感性
"""

import os
import sys
import time
import itertools
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from polarization_triangle.analysis.sobol_analysis import SobolAnalyzer, SobolConfig
from polarization_triangle.core.config import SimulationConfig


class ConfigComparisonAnalyzer:
    """配置对比分析器"""
    
    def __init__(self, base_output_dir: str = "results/config_comparison"):
        self.base_output_dir = base_output_dir
        self.results = {}
        self.comparison_results = {}
        
        # 创建输出目录
        os.makedirs(base_output_dir, exist_ok=True)
        
        # 定义要测试的参数组合
        self.test_parameters = {
            'cluster_identity': [False, True],
            'cluster_morality': [False, True],
            'cluster_opinion': [False, True],
            'zealot_morality': [False, True],
            'zealot_identity_allocation': [False, True],
            'zealot_mode': ['random', 'clustered']
        }
        
        # 生成所有参数组合
        self.param_combinations = self._generate_param_combinations()
        print(f"总共生成了 {len(self.param_combinations)} 种参数组合")
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        param_names = list(self.test_parameters.keys())
        param_values = list(self.test_parameters.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            combination = dict(zip(param_names, combo))
            combinations.append(combination)
        
        return combinations
    
    def create_config_for_combination(self, combination: Dict[str, Any]) -> SobolConfig:
        """为特定参数组合创建SobolConfig"""
        # 基于test1配置创建基础配置
        base_config = SobolConfig(
            n_samples=500,
            n_runs=6,
            n_processes=8,
            num_steps=200,
            output_dir="temp"  # 将在后面设置
        )
        
        # 创建SimulationConfig并设置测试参数
        sim_config = SimulationConfig(
            num_agents=200,
            network_type='lfr',
            network_params={
                'tau1': 3, 'tau2': 1.5, 'mu': 0.1,
                'average_degree': 5, 'min_community': 10
            },
            opinion_distribution='uniform',
            morality_rate=0.3,
            # 设置测试参数
            cluster_identity=combination['cluster_identity'],
            cluster_morality=combination['cluster_morality'],
            cluster_opinion=combination['cluster_opinion'],
            # Zealot配置
            zealot_count=30,
            enable_zealots=True,
            zealot_mode=combination['zealot_mode'],
            zealot_morality=combination['zealot_morality'],
            zealot_identity_allocation=combination['zealot_identity_allocation']
        )
        
        # 设置基础配置
        base_config.base_config = sim_config
        
        # 设置输出目录
        combo_name = self._get_combination_name(combination)
        base_config.output_dir = os.path.join(self.base_output_dir, combo_name)
        
        return base_config
    
    def _get_combination_name(self, combination: Dict[str, Any]) -> str:
        """获取参数组合的名称"""
        parts = []
        for key, value in combination.items():
            if isinstance(value, bool):
                parts.append(f"{key}_{str(value).lower()}")
            else:
                parts.append(f"{key}_{value}")
        return "_".join(parts)
    
    def run_single_combination(self, combination: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """运行单个参数组合的分析"""
        combo_name = self._get_combination_name(combination)
        
        try:
            # 创建配置
            config = self.create_config_for_combination(combination)
            
            # 创建分析器
            analyzer = SobolAnalyzer(config)
            
            # 运行分析
            sensitivity_indices = analyzer.run_complete_analysis()
            
            # 提取关注的结果
            result = {
                'combination': combination,
                'combo_name': combo_name,
                'sensitivity_indices': sensitivity_indices,
                'success': True
            }
            
            # 如果有opinion_variance结果，单独提取
            if 'opinion_variance' in sensitivity_indices:
                result['opinion_variance_sensitivity'] = sensitivity_indices['opinion_variance']
            
            return combo_name, result
            
        except Exception as e:
            print(f"组合 {combo_name} 执行失败: {e}")
            return combo_name, {
                'combination': combination,
                'combo_name': combo_name,
                'error': str(e),
                'success': False
            }
    
    def run_all_combinations(self, n_processes: int = 4):
        """运行所有参数组合的分析"""
        print(f"开始运行 {len(self.param_combinations)} 种参数组合的分析...")
        print(f"使用 {n_processes} 个进程进行并行计算")
        
        if n_processes > 1:
            # 并行执行
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                # 提交任务
                future_to_combo = {
                    executor.submit(self.run_single_combination, combo): combo
                    for combo in self.param_combinations
                }
                
                # 收集结果
                with tqdm(total=len(self.param_combinations), desc="执行配置分析") as pbar:
                    for future in as_completed(future_to_combo):
                        combo_name, result = future.result()
                        self.results[combo_name] = result
                        pbar.update(1)
        else:
            # 串行执行
            for combo in tqdm(self.param_combinations, desc="执行配置分析"):
                combo_name, result = self.run_single_combination(combo)
                self.results[combo_name] = result
    
    def analyze_results(self):
        """分析结果差异"""
        print("\n开始分析结果差异...")
        
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
        
        # 分析参数影响模式
        self._analyze_parameter_influence_patterns(successful_results)
    
    def _analyze_opinion_variance_sensitivity(self, results: Dict[str, Dict]):
        """分析opinion_variance敏感性的差异"""
        print("\n分析opinion_variance敏感性差异...")
        
        # 收集所有有opinion_variance结果的数据
        ov_data = []
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        
        for combo_name, result in results.items():
            if 'opinion_variance_sensitivity' in result:
                ov_sens = result['opinion_variance_sensitivity']
                combination = result['combination']
                
                # 提取S1和ST值
                row_data = {
                    'combo_name': combo_name,
                    **combination  # 展开参数组合
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
        df.to_csv(os.path.join(self.base_output_dir, 'opinion_variance_sensitivity_comparison.csv'), 
                  index=False)
        
        # 分析各参数的敏感性范围
        print("\n各参数敏感性指数范围:")
        for param in param_names:
            s1_col = f'{param}_S1'
            st_col = f'{param}_ST'
            
            if s1_col in df.columns and st_col in df.columns:
                print(f"{param}:")
                print(f"  S1 范围: [{df[s1_col].min():.3f}, {df[s1_col].max():.3f}]")
                print(f"  ST 范围: [{df[st_col].min():.3f}, {df[st_col].max():.3f}]")
                print(f"  ST 标准差: {df[st_col].std():.3f}")
        
        # 找出最大和最小敏感性的配置
        for param in param_names:
            st_col = f'{param}_ST'
            if st_col in df.columns:
                max_idx = df[st_col].idxmax()
                min_idx = df[st_col].idxmin()
                
                print(f"\n{param} 最高敏感性配置:")
                print(f"  值: {df.loc[max_idx, st_col]:.3f}")
                print(f"  配置: {df.loc[max_idx, 'combo_name']}")
                
                print(f"{param} 最低敏感性配置:")
                print(f"  值: {df.loc[min_idx, st_col]:.3f}")
                print(f"  配置: {df.loc[min_idx, 'combo_name']}")
        
        self.comparison_results['opinion_variance_df'] = df
    
    def _analyze_parameter_influence_patterns(self, results: Dict[str, Dict]):
        """分析参数影响模式"""
        print("\n分析参数影响模式...")
        
        # 检查不同测试参数对敏感性的影响
        if 'opinion_variance_df' not in self.comparison_results:
            return
        
        df = self.comparison_results['opinion_variance_df']
        param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
        test_params = list(self.test_parameters.keys())
        
        influence_analysis = {}
        
        for test_param in test_params:
            if test_param in df.columns:
                influence_analysis[test_param] = {}
                
                for model_param in param_names:
                    st_col = f'{model_param}_ST'
                    if st_col in df.columns:
                        # 按测试参数分组，计算敏感性差异
                        grouped = df.groupby(test_param)[st_col].agg(['mean', 'std', 'count'])
                        influence_analysis[test_param][model_param] = grouped.to_dict()
        
        # 保存影响分析结果
        with open(os.path.join(self.base_output_dir, 'parameter_influence_analysis.json'), 'w') as f:
            json.dump(influence_analysis, f, indent=2, default=str)
        
        # 打印关键发现
        print("\n关键发现:")
        for test_param in test_params:
            if test_param in influence_analysis:
                print(f"\n{test_param} 的影响:")
                for model_param in param_names:
                    if model_param in influence_analysis[test_param]:
                        means = influence_analysis[test_param][model_param]['mean']
                        if len(means) > 1:
                            values = list(means.values())
                            max_diff = max(values) - min(values)
                            print(f"  {model_param} 敏感性差异: {max_diff:.3f}")
    
    def save_summary_report(self):
        """保存总结报告"""
        print("\n保存总结报告...")
        
        # 统计信息
        total_combinations = len(self.param_combinations)
        successful_runs = sum(1 for r in self.results.values() if r['success'])
        failed_runs = total_combinations - successful_runs
        
        summary = {
            'analysis_info': {
                'total_combinations': total_combinations,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'success_rate': successful_runs / total_combinations if total_combinations > 0 else 0
            },
            'test_parameters': self.test_parameters,
            'results_summary': {}
        }
        
        # 添加结果摘要
        if 'opinion_variance_df' in self.comparison_results:
            df = self.comparison_results['opinion_variance_df']
            param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
            
            for param in param_names:
                st_col = f'{param}_ST'
                if st_col in df.columns:
                    summary['results_summary'][param] = {
                        'mean_sensitivity': float(df[st_col].mean()),
                        'std_sensitivity': float(df[st_col].std()),
                        'min_sensitivity': float(df[st_col].min()),
                        'max_sensitivity': float(df[st_col].max()),
                        'sensitivity_range': float(df[st_col].max() - df[st_col].min())
                    }
        
        # 保存摘要
        with open(os.path.join(self.base_output_dir, 'summary_report.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"总结报告已保存到: {os.path.join(self.base_output_dir, 'summary_report.json')}")
        
        # 打印摘要
        print(f"\n分析摘要:")
        print(f"  总配置数: {total_combinations}")
        print(f"  成功运行: {successful_runs}")
        print(f"  失败运行: {failed_runs}")
        print(f"  成功率: {successful_runs/total_combinations*100:.1f}%")


def main():
    """主函数"""
    print("="*60)
    print("配置对比测试分析")
    print("="*60)
    
    # 创建分析器
    analyzer = ConfigComparisonAnalyzer()
    
    try:
        start_time = time.time()
        
        # 运行所有组合
        analyzer.run_all_combinations(n_processes=4)
        
        # 分析结果
        analyzer.analyze_results()
        
        # 保存报告
        analyzer.save_summary_report()
        
        end_time = time.time()
        print(f"\n总耗时: {end_time - start_time:.2f} 秒")
        print(f"结果保存在: {analyzer.base_output_dir}")
        
    except KeyboardInterrupt:
        print("\n分析被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 