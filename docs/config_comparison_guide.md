# 配置对比测试使用指南

## 概述

该指南介绍如何使用新创建的配置对比脚本来测试不同参数组合对 Sobol 敏感性分析结果的影响。

## 脚本介绍

### 1. 完整配置对比脚本 (`run_config_comparison.py`)

**功能**：测试所有 64 种参数组合（2^6）对 Sobol 敏感性分析结果的影响
**位置**：`polarization_triangle/scripts/run_config_comparison.py`

**测试参数**：
- `cluster_identity`：身份聚类 (True/False)
- `cluster_morality`：道德聚类 (True/False)
- `cluster_opinion`：观点聚类 (True/False)
- `zealot_morality`：zealot 道德化 (True/False)
- `zealot_identity_allocation`：zealot 身份分配 (True/False)
- `zealot_mode`：zealot 模式 ("random"/"clustered")

**配置参数**：
- 样本数：500
- 运行次数：6
- 进程数：8
- 模拟步数：200

### 2. 快速配置测试脚本 (`run_quick_config_test.py`)

**功能**：测试 8 种代表性配置组合，用于快速验证和预览
**位置**：`polarization_triangle/scripts/run_quick_config_test.py`

**测试配置**：
1. **baseline**：基准配置（所有参数 False，zealot_mode="random"）
2. **all_clustered**：所有聚类参数为 True
3. **zealot_enabled**：所有 zealot 参数为 True
4. **zealot_clustered**：zealot_mode="clustered"
5. **mixed_1**：混合配置 1
6. **mixed_2**：混合配置 2
7. **all_enabled**：所有参数为 True
8. **identity_only**：仅身份聚类

**配置参数**：
- 样本数：200
- 运行次数：3
- 进程数：4
- 模拟步数：150

## 使用方法

### 快速测试（推荐先运行）

```bash
# 进入项目目录
cd The-Polarization-Triangle-Framework

# 运行快速配置测试
python polarization_triangle/scripts/run_quick_config_test.py
```

### 完整分析

```bash
# 运行完整配置对比分析
python polarization_triangle/scripts/run_config_comparison.py
```

## 输出结果

### 文件结构

```
results/
├── quick_config_test/           # 快速测试结果
│   ├── baseline/               # 各配置的详细结果
│   ├── all_clustered/
│   ├── zealot_enabled/
│   ├── ...
│   ├── quick_test_results.csv  # 敏感性对比结果
│   └── quick_test_summary.json # 测试摘要
└── config_comparison/          # 完整对比结果
    ├── cluster_identity_false_cluster_morality_false_.../ # 各配置详细结果
    ├── ...
    ├── opinion_variance_sensitivity_comparison.csv  # 敏感性对比结果
    ├── parameter_influence_analysis.json           # 参数影响分析
    └── summary_report.json                         # 总结报告
```

### 主要输出文件

1. **敏感性对比结果 (CSV)**
   - 包含所有配置的四个参数（α, β, γ, cohesion_factor）敏感性指数
   - 包含 S1（一阶敏感性）和 ST（总敏感性）值
   - 包含交互效应强度

2. **参数影响分析 (JSON)**
   - 分析各测试参数对敏感性的影响
   - 提供统计摘要（均值、标准差等）

3. **总结报告 (JSON)**
   - 测试统计信息
   - 成功率
   - 敏感性范围摘要

## 结果解读

### 敏感性指数含义

- **S1 (一阶敏感性指数)**：参数单独对输出的影响
- **ST (总敏感性指数)**：参数的总影响（包括交互效应）
- **交互效应 (ST - S1)**：参数与其他参数的交互影响

### 关键指标

重点关注四个参数对 `opinion_variance` 的敏感性：

1. **α (alpha)**：自我激活系数
2. **β (beta)**：社会影响系数
3. **γ (gamma)**：道德化影响系数
4. **cohesion_factor**：身份凝聚力因子

### 分析要点

1. **敏感性范围**：查看不同配置下各参数敏感性的变化范围
2. **配置影响**：识别哪些配置参数对敏感性影响最大
3. **参数排序**：确定在不同配置下参数重要性的变化
4. **交互效应**：分析参数间的交互作用强度

## 性能考虑

### 运行时间估计

- **快速测试**：约 10-30 分钟（8 个配置）
- **完整分析**：约 5-10 小时（64 个配置）

### 系统要求

- **内存**：建议 8GB 以上
- **CPU**：多核处理器（脚本支持并行计算）
- **存储**：预留 2-5GB 空间存储结果

### 优化建议

1. **先运行快速测试**：验证脚本正常运行
2. **调整进程数**：根据 CPU 核心数调整并行进程数
3. **分批运行**：如果资源有限，可以修改脚本分批运行
4. **监控资源**：运行期间监控内存和 CPU 使用情况

## 自定义配置

### 修改测试参数

可以编辑脚本中的配置参数：

```python
# 修改 SobolConfig 参数
config = SobolConfig(
    n_samples=200,      # 样本数
    n_runs=3,          # 运行次数
    n_processes=4,     # 进程数
    num_steps=150,     # 模拟步数
    output_dir="custom_results"
)
```

### 添加新的测试配置

在 `run_quick_config_test.py` 中添加新的配置：

```python
# 添加到 test_configurations 列表
{
    'name': 'custom_config',
    'cluster_identity': True,
    'cluster_morality': False,
    'cluster_opinion': True,
    'zealot_morality': False,
    'zealot_identity_allocation': True,
    'zealot_mode': 'random'
}
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少样本数和运行次数
   - 减少并行进程数
   - 分批运行配置

2. **运行时间过长**
   - 先运行快速测试
   - 减少模拟步数
   - 增加并行进程数

3. **结果文件缺失**
   - 检查输出目录权限
   - 确认脚本正常完成
   - 查看错误日志

### 调试模式

添加调试输出：

```python
# 在脚本开始处添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 结果可视化

脚本生成的 CSV 文件可以用于进一步的可视化分析：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('results/quick_config_test/quick_test_results.csv')

# 绘制敏感性对比图
params = ['alpha', 'beta', 'gamma', 'cohesion_factor']
for param in params:
    plt.figure(figsize=(10, 6))
    plt.bar(df['config_name'], df[f'{param}_ST'])
    plt.title(f'{param} Sensitivity Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

## 注意事项

1. **随机种子**：每次运行可能产生略有不同的结果
2. **计算资源**：完整分析需要大量计算资源
3. **结果解读**：需要结合具体应用场景解读敏感性结果
4. **参数范围**：确保测试的参数范围符合实际应用需求

## 联系支持

如果在使用过程中遇到问题，请检查：
1. 依赖库是否正确安装
2. 文件路径是否正确
3. 系统资源是否充足
4. 错误日志中的详细信息 