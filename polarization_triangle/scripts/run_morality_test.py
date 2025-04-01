#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行道德化率测试脚本，测试不同道德化率对规则使用的影响
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from polarization_triangle.experiments.morality_test import batch_test_morality_rates


def main():
    parser = argparse.ArgumentParser(description="运行极化三角框架道德化率测试")
    parser.add_argument("--output-dir", type=str, default="morality_rate_test",
                        help="输出结果的目录路径")
    parser.add_argument("--steps", type=int, default=200,
                        help="每次模拟的步数")
    parser.add_argument("--rates", type=float, nargs="+", 
                        default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help="要测试的道德化率列表")
    args = parser.parse_args()
    
    print(f"开始运行道德化率测试，步数: {args.steps}, 道德化率: {args.rates}...")
    batch_test_morality_rates(output_dir=args.output_dir, 
                              steps=args.steps,
                              morality_rates=args.rates)
    print("道德化率测试完成！")


if __name__ == "__main__":
    main()




