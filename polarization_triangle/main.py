#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
极化三角框架主入口文件
提供命令行接口来运行各种模拟测试
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from polarization_triangle.experiments.batch_runner import batch_test
from polarization_triangle.experiments.morality_test import batch_test_morality_rates
from polarization_triangle.experiments.model_params_test import batch_test_model_params
from polarization_triangle.experiments.activation_analysis import analyze_activation_components


def main():
    parser = argparse.ArgumentParser(description="极化三角框架模拟测试")
    parser.add_argument("--test-type", type=str, required=True, 
                        choices=["basic", "morality", "model-params", "activation"],
                        help="要运行的测试类型")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出结果的目录路径 (注：仅适用于morality、model-params和activation测试)")
    parser.add_argument("--steps", type=int, default=200,
                        help="每次模拟的步数")
    
    # 道德化率测试特有参数
    parser.add_argument("--morality-rates", type=float, nargs="+", 
                        default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help="要测试的道德化率列表 (仅用于morality测试)")
    
    args = parser.parse_args()
    
    # 设置默认输出目录
    if args.output_dir is None:
        if args.test_type == "basic":
            # 对于基本测试，输出目录是固定的
            pass
        elif args.test_type == "morality":
            args.output_dir = "morality_rate_test"
        elif args.test_type == "model-params":
            args.output_dir = "model_params_test"
        elif args.test_type == "activation":
            args.output_dir = "activation_analysis"
    
    # 根据测试类型运行相应的测试
    if args.test_type == "basic":
        print(f"开始运行基本批量模拟测试，结果将保存到固定目录: batch_results")
        batch_test()  # 不传递output_dir参数
    elif args.test_type == "morality":
        print(f"开始运行道德化率测试，步数: {args.steps}, 道德化率: {args.morality_rates}")
        print(f"结果将保存到: {args.output_dir}")
        batch_test_morality_rates(output_dir=args.output_dir, 
                                  steps=args.steps,
                                  morality_rates=args.morality_rates)
    elif args.test_type == "model-params":
        print(f"开始运行模型参数测试，步数: {args.steps}")
        print(f"结果将保存到: {args.output_dir}")
        batch_test_model_params(output_dir=args.output_dir, 
                               steps=args.steps)
    elif args.test_type == "activation":
        print(f"开始运行激活组件分析，步数: {args.steps}")
        print(f"结果将保存到: {args.output_dir}")
        analyze_activation_components(output_dir=args.output_dir,
                                     steps=args.steps)
    
    print("测试完成！")


if __name__ == "__main__":
    main()
