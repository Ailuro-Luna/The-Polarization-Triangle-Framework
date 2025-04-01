#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行模型参数测试脚本，测试不同极化三角框架模型参数对结果的影响
"""

import os
import sys
import argparse
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from polarization_triangle.experiments.model_params_test import batch_test_model_params


def main():
    parser = argparse.ArgumentParser(description="运行极化三角框架模型参数测试")
    parser.add_argument("--output-dir", type=str, default="model_params_test",
                        help="输出结果的目录路径")
    parser.add_argument("--steps", type=int, default=200,
                        help="每次模拟的步数")
    parser.add_argument("--custom-params", type=str, default=None,
                        help="自定义参数设置的JSON文件路径")
    args = parser.parse_args()
    
    custom_param_settings = None
    if args.custom_params and os.path.exists(args.custom_params):
        try:
            with open(args.custom_params, 'r') as f:
                custom_param_settings = json.load(f)
            print(f"已加载自定义参数设置: {args.custom_params}")
        except Exception as e:
            print(f"加载自定义参数设置失败: {e}")
    
    print(f"开始运行模型参数测试，步数: {args.steps}...")
    batch_test_model_params(output_dir=args.output_dir, 
                           steps=args.steps,
                           param_settings=custom_param_settings)
    print("模型参数测试完成！")


if __name__ == "__main__":
    main()




