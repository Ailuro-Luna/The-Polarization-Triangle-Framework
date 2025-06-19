# Identity和Opinion关系分析功能

## 概述

本文档介绍了新添加的identity和opinion关系分析功能，该功能可以探究不同身份群体在模拟过程中的意见动态变化。

## 功能特性

### 1. 统计计算

在每个时间步骤中，系统会自动计算以下统计数据：

- **Identity +1 群体的平均opinion**：所有identity值为+1的非zealot代理的平均意见值
- **Identity -1 群体的平均opinion**：所有identity值为-1的非zealot代理的平均意见值  
- **两群体意见差值**：Identity +1群体平均意见 - Identity -1群体平均意见

### 2. 可视化图表

#### 单次实验图表
对于包含多种zealot模式的单次实验，会生成以下比较图表：

- `comparison_identity_mean_opinions.png`：不同zealot模式下两种identity群体的平均opinion演化对比
  - 实线表示Identity +1群体
  - 虚线表示Identity -1群体
  - 不同颜色表示不同的zealot模式

- `comparison_identity_opinion_differences_abs.png`：不同zealot模式下两种identity群体意见差值的绝对值对比

#### 多次实验平均图表
对于多次运行的平均结果，会生成相应的平均图表：

- `avg_identity_mean_opinions.png`：多次运行平均后的identity群体意见对比
- `avg_identity_opinion_differences_abs.png`：多次运行平均后的意见差值绝对值对比

#### 参数扫描综合图表
在参数扫描中，会生成跨所有参数组合的综合对比图表：

- `combined_identity_mean_opinions.png`：所有参数组合下两种identity群体的意见演化
  - 颜色区分不同的参数配置
  - 实线表示Identity +1群体，虚线表示Identity -1群体

- `combined_identity_opinion_differences_abs.png`：所有参数组合下两种identity群体意见差值绝对值的对比
  - 颜色区分不同的参数配置

### 3. 数据输出

#### CSV文件
所有统计数据都会保存到CSV文件中，包含以下新增列：

- `identity_1_mean_opinion`：Identity +1群体的平均意见
- `identity_neg1_mean_opinion`：Identity -1群体的平均意见  
- `identity_opinion_difference`：两群体的意见差值

#### 文件位置
- **单次实验**：`{output_dir}/statistics/{mode}_opinion_stats.csv`
- **多次实验平均**：`{output_dir}/statistics/avg_opinion_stats.csv`
- **比较分析**：`{output_dir}/statistics/comparison_opinion_stats.csv`
- **参数扫描综合**：`{output_dir}/combined_results/statistics/combined_statistics.csv`

## 技术实现

### 代码修改位置

1. **统计计算**：`polarization_triangle/experiments/zealot_experiment.py`
   - 在`generate_opinion_statistics`函数中添加identity相关统计计算

2. **单次实验绘图**：`polarization_triangle/experiments/zealot_experiment.py`
   - 在`plot_comparative_statistics`函数中添加identity相关图表绘制

3. **多次实验绘图**：`polarization_triangle/experiments/multi_zealot_experiment.py`
   - 在`average_stats`函数中添加identity统计的平均计算
   - 在`plot_average_statistics`函数中添加identity相关图表绘制

4. **参数扫描绘图**：`polarization_triangle/experiments/zealot_parameter_sweep.py`
   - 在`plot_combined_statistics`函数中添加identity相关的综合对比图表

### 核心算法

```python
# 找到不同identity的agents（排除zealots）
identity_1_agents = []
identity_neg1_agents = []

for i in range(sim.num_agents):
    if zealot_ids and i in zealot_ids:
        continue  # 跳过zealots
    if sim.identities[i] == 1:
        identity_1_agents.append(i)
    elif sim.identities[i] == -1:
        identity_neg1_agents.append(i)

# 计算每个时间步的统计数据
for step_opinions in trajectory:
    # Identity +1群体平均意见
    if identity_1_agents:
        identity_1_mean = np.mean(step_opinions[identity_1_agents])
    else:
        identity_1_mean = 0.0
    
    # Identity -1群体平均意见
    if identity_neg1_agents:
        identity_neg1_mean = np.mean(step_opinions[identity_neg1_agents])
    else:
        identity_neg1_mean = 0.0
    
    # 计算差值
    difference = identity_1_mean - identity_neg1_mean
```

## 使用示例

### 运行单次实验
```python
from polarization_triangle.experiments.zealot_experiment import run_zealot_experiment

result = run_zealot_experiment(
    steps=500,
    identity_clustered=True,  # 按identity聚类初始化
    zealot_mode=None,         # 运行所有zealot模式进行对比
    output_dir="identity_analysis"
)

# 访问identity统计数据
for mode_name, mode_data in result.items():
    stats = mode_data['stats']
    identity_1_opinions = stats['identity_1_mean_opinions']
    identity_neg1_opinions = stats['identity_neg1_mean_opinions']
    opinion_differences = stats['identity_opinion_differences']
```

### 运行参数扫描
```python
from polarization_triangle.experiments.zealot_parameter_sweep import run_parameter_sweep

run_parameter_sweep(
    runs_per_config=10,
    steps=1000,
    output_base_dir="identity_parameter_sweep"
)

# 检查生成的综合图表
# combined_results/statistics/combined_identity_mean_opinions.png
# combined_results/statistics/combined_identity_opinion_differences_abs.png
```

## 分析价值

这个功能可以帮助研究者：

1. **观察群体极化**：通过观察两种identity群体意见差值的变化，可以量化群体间的极化程度
2. **比较干预效果**：不同zealot策略对identity群体意见分化的影响
3. **参数敏感性分析**：不同参数组合下identity群体的意见动态模式
4. **时间序列分析**：意见分化的时间演化特征

## 输出文件结构

```
output_directory/
├── statistics/
│   ├── comparison_identity_mean_opinions.png
│   ├── comparison_identity_opinion_differences_abs.png
│   ├── avg_identity_mean_opinions.png
│   ├── avg_identity_opinion_differences_abs.png
│   └── *_opinion_stats.csv (包含identity列)
└── combined_results/
    └── statistics/
        ├── combined_identity_mean_opinions.png
        ├── combined_identity_opinion_differences_abs.png
        └── combined_statistics.csv (包含所有identity列)
``` 