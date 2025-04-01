#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行基本的极化三角框架批量测试脚本
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from polarization_triangle.experiments.batch_runner import batch_test


def main():
    parser = argparse.ArgumentParser(description="运行极化三角框架基本批量模拟测试")
    parser.add_argument("--output-dir", type=str, default="batch_results",
                        help="输出结果的目录路径 (注：当前仅作为提示，输出目录固定为batch_results)")
    args = parser.parse_args()
    
    print("开始运行基本批量模拟测试...")
    print(f"注意：输出将保存到 batch_results 目录中")
    batch_test()  # 不传递output_dir参数，因为batch_test函数不接受该参数
    print("模拟测试完成！")


if __name__ == "__main__":
    main()
