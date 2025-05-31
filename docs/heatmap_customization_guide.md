# 热力图颜色和尺度自定义指南

## 概述

这份指南详细说明了如何在 Polarization Triangle 框架中自定义热力图的颜色映射、数值尺度和其他视觉参数。

## 热力图设置位置

### 1. 单次实验热力图
**函数**: `draw_opinion_distribution_heatmap`  
**文件**: `polarization_triangle/visualization/opinion_viz.py`  
**用途**: 绘制单次实验的意见分布演化热力图

### 2. 多次实验平均热力图
**函数**: `draw_opinion_distribution_heatmap_from_distribution`  
**文件**: `polarization_triangle/experiments/multi_zealot_experiment.py`  
**用途**: 绘制多次实验的平均意见分布热力图

## 可自定义的参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cmap` | str | 'viridis' | 颜色映射方案 |
| `log_scale` | bool | True | 是否使用对数尺度 |
| `vmin` | float/None | None | 颜色尺度最小值 |
| `vmax` | float/None | None | 颜色尺度最大值 |
| `custom_norm` | object/None | None | 自定义标准化对象 |
| `bins` | int | 40/50 | opinion值的分箱数量 |

## 常用颜色映射方案

### 单色渐变
- `'viridis'`: 绿-蓝-紫，默认选择，科学友好
- `'plasma'`: 紫-粉-黄，高对比度
- `'inferno'`: 黑-红-黄，类似火焰
- `'magma'`: 黑-紫-粉-白，柔和渐变
- `'cividis'`: 蓝-黄，色盲友好

### 双色映射
- `'coolwarm'`: 蓝-白-红，适合有中心值的数据
- `'RdBu'`: 红-白-蓝，强对比
- `'seismic'`: 红-白-蓝，地震学常用

### 经典映射
- `'hot'`: 黑-红-黄-白，高对比度
- `'jet'`: 蓝-青-绿-黄-红，彩虹色
- `'gray'`: 灰度，适合打印

## 使用方法

### 方法1: 直接设置参数

```python
from polarization_triangle.visualization.opinion_viz import draw_opinion_distribution_heatmap

# 自定义颜色和尺度
draw_opinion_distribution_heatmap(
    opinion_history,
    "Custom Heatmap",
    "custom_heatmap.png",
    cmap='hot',           # 使用热力颜色映射
    log_scale=True,       # 对数尺度
    vmin=1,               # 最小值设为1
    vmax=100,             # 最大值设为100
    bins=30               # 30个分箱
)

# 线性尺度，固定范围
draw_opinion_distribution_heatmap(
    opinion_history,
    "Linear Scale Heatmap", 
    "linear_heatmap.png",
    cmap='viridis',
    log_scale=False,      # 线性尺度
    vmin=0,               # 从0开始
    vmax=50               # 到50结束
)
```

### 方法2: 使用预设配置

```python
from polarization_triangle.config.heatmap_config import get_heatmap_config

# 使用预设的高对比度配置
config = get_heatmap_config('high_contrast')
draw_opinion_distribution_heatmap(
    opinion_history,
    "High Contrast Heatmap",
    "high_contrast.png",
    **config
)

# 可用的预设配置：
# - 'default': 默认设置
# - 'high_contrast': 高对比度，突出极值
# - 'symmetric': 对称配置，适合有中心值的数据
# - 'detail': 细节强调，线性尺度
# - 'colorblind': 色盲友好
# - 'publication': 论文发表，灰度友好
```

### 方法3: 在多次实验中使用

```python
from polarization_triangle.experiments.multi_zealot_experiment import run_multi_zealot_experiment

# 自定义热力图配置
heatmap_config = {
    'cmap': 'plasma',
    'log_scale': False,
    'vmin': 0,
    'vmax': 30,
    'bins': 50
}

# 运行实验时传递配置
run_multi_zealot_experiment(
    runs=10,
    steps=100,
    output_dir="results/custom_experiment",
    # 注意：需要修改函数以支持heatmap_config参数
)
```

### 方法4: 使用自定义标准化

```python
from polarization_triangle.config.heatmap_config import create_custom_power_norm
import matplotlib.pyplot as plt

# 创建幂律标准化（强调中等数值）
custom_norm = create_custom_power_norm(
    gamma=0.5,    # 幂指数，<1强调小值，>1强调大值
    vmin=1,
    vmax=100
)

draw_opinion_distribution_heatmap(
    opinion_history,
    "Power Norm Heatmap",
    "power_norm.png",
    custom_norm=custom_norm,
    cmap='inferno'
)

# 或者使用matplotlib的其他标准化
from matplotlib.colors import SymLogNorm

symlog_norm = SymLogNorm(
    linthresh=10,   # 线性阈值
    vmin=0,
    vmax=1000
)

draw_opinion_distribution_heatmap(
    opinion_history,
    "SymLog Heatmap",
    "symlog.png", 
    custom_norm=symlog_norm,
    cmap='viridis'
)
```

## 设置具体数值范围的技巧

### 1. 保持多张图的尺度一致
```python
# 为了比较多个实验，使用相同的vmin和vmax
vmin, vmax = 1, 50

for experiment in experiments:
    draw_opinion_distribution_heatmap(
        experiment.data,
        experiment.title,
        experiment.filename,
        vmin=vmin,
        vmax=vmax,
        cmap='viridis'
    )
```

### 2. 突出特定数值范围
```python
# 突出1-20范围内的变化，忽略极值
draw_opinion_distribution_heatmap(
    opinion_history,
    "Focus on 1-20 Range",
    "focused_range.png",
    vmin=1,
    vmax=20,
    log_scale=False
)
```

### 3. 处理稀疏数据
```python
# 对于大部分值为0的稀疏数据，使用对数尺度
draw_opinion_distribution_heatmap(
    sparse_data,
    "Sparse Data",
    "sparse.png",
    log_scale=True,
    vmin=1,     # 避免log(0)
    vmax=100
)
```

## 颜色条自定义

热力图函数会自动根据设置的`vmin`和`vmax`调整颜色条刻度：

- **对数尺度**: 刻度会按10的倍数分布
- **线性尺度**: 刻度会均匀分布

## 实用建议

### 1. 选择颜色映射的原则
- **数据有明确中心值**: 使用双色映射如`'coolwarm'`、`'RdBu'`
- **强调极值**: 使用`'hot'`、`'plasma'`
- **科学发表**: 使用`'viridis'`、`'cividis'`
- **黑白打印**: 使用`'gray'`

### 2. 对数vs线性尺度
- **数据跨越多个数量级**: 使用对数尺度
- **数据范围较小且连续**: 使用线性尺度

### 3. 固定数值范围的时机
- **比较多个实验**: 必须固定范围
- **关注特定区间**: 设置合适的vmin/vmax
- **排除异常值影响**: 根据数据分布设置范围

## 示例脚本

完整的使用示例可以在以下文件中找到：
- `examples/custom_heatmap_example.py` - 综合示例
- `polarization_triangle/config/heatmap_config.py` - 配置选项

运行示例：
```bash
cd examples
python custom_heatmap_example.py
```

这将生成多种不同配置的热力图供您比较和选择。 