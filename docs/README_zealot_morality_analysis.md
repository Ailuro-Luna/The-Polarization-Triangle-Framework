# Zealot和Morality分析实验 - 分离式运行指南

## 概述

我们将原来的 `run_zealot_morality_analysis` 函数分成了两个独立的部分：

1. **数据收集阶段**: `run_and_accumulate_data()` - 运行测试并以追加模式保存数据
2. **绘图阶段**: `plot_from_accumulated_data()` - 从累积数据生成图表

## 主要优势

✅ **累积数据**: 可以多次运行测试，每次积累更多数据  
✅ **分离处理**: 数据收集和绘图可以分开进行  
✅ **运行次数标识**: 图表文件名自动包含总运行次数信息  
✅ **向后兼容**: 原有的一次性运行功能仍然可用  

## 使用方法

### 方法1: 分步骤运行（推荐）

#### 第一步: 收集数据
```python
from polarization_triangle.experiments.zealot_morality_analysis import run_and_accumulate_data

# 第一次运行 - 快速测试
run_and_accumulate_data(
    output_dir="results/my_experiment",
    num_runs=5,  # 本次运行5轮
    max_zealots=30,
    max_morality=30,
    batch_name="quick_test"  # 可选：批次名称
)

# 第二次运行 - 增加更多数据
run_and_accumulate_data(
    output_dir="results/my_experiment",
    num_runs=10,  # 再运行10轮
    max_zealots=30,
    max_morality=30,
    batch_name="detailed_run"
)

# 可以继续运行更多批次...
```

#### 第二步: 生成图表
```python
from polarization_triangle.experiments.zealot_morality_analysis import plot_from_accumulated_data

# 从所有累积数据生成图表
plot_from_accumulated_data("results/my_experiment")
```

### 方法2: 一次性运行（向后兼容）

```python
from polarization_triangle.experiments.zealot_morality_analysis import run_zealot_morality_analysis

# 传统的一次性运行
run_zealot_morality_analysis(
    output_dir="results/traditional_run",
    num_runs=15,
    max_zealots=30,
    max_morality=30
)
```

## 文件结构

运行后会生成以下文件结构：

```
results/my_experiment/
├── accumulated_data/                    # 累积数据文件夹
│   ├── zealot_numbers_*_accumulated.csv # Zealot数量分析数据
│   ├── morality_ratios_*_accumulated.csv # Morality比例分析数据
│   └── batch_info_*.txt                 # 批次信息文件
├── error_bar_plots/                     # 误差条图表
│   ├── zealot_numbers_*_15runs.png
│   └── morality_ratios_*_15runs.png
├── scatter_plots/                       # 散点图表
├── mean_plots/                          # 均值曲线图表
└── combined_plots/                      # 组合图表
```

## 文件命名规则

图表文件名自动包含运行次数信息：
- `zealot_numbers_mean_opinion_15runs.png` - 使用了15次总运行数据
- `morality_ratios_variance_5-12runs.png` - 不同组合使用了5-12次总运行数据

## 耗时统计

系统会自动计算和显示各阶段耗时：

### 数据收集阶段
- 每个参数组合的运行耗时
- 两种测试类型（Zealot Numbers, Morality Ratios）的分别耗时  
- 数据收集阶段总耗时

### 绘图阶段
- 图表生成阶段总耗时

### 总耗时
- main函数执行的完整耗时
- 包含数据收集和绘图的所有时间

示例输出：
```
🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒
⏱️  完整实验耗时总结
🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒
📊 数据收集阶段耗时: 0h 25m 30.45s
📈 图表生成阶段耗时: 0h 0m 5.20s
🎯 总耗时: 0h 25m 35.65s
🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒
```

## 数据格式

累积数据CSV文件包含以下列：
- `x_value`: X轴取值（zealot数量或morality比例）
- `metric`: 指标名称（mean_opinion, variance, etc.）
- `run`: 运行编号
- `value`: 测量值
- `combination`: 参数组合标识
- `batch_id`: 批次标识
- `timestamp`: 时间戳

## 实际使用示例

查看 `example_usage.py` 文件获取完整的使用示例：

```bash
# 运行多次增量测试示例
python example_usage.py incremental

# 运行传统一次性测试示例  
python example_usage.py single

# 只收集数据（稍后绘图）
python example_usage.py data_only

# 只从现有数据生成图表
python example_usage.py plot_only
```

## 注意事项

1. **数据追加**: 每次运行 `run_and_accumulate_data()` 都会追加新数据，不会覆盖现有数据
2. **批次标识**: 建议为每次运行提供有意义的 `batch_name`，便于管理
3. **参数一致性**: 确保多次运行使用相同的 `max_zealots` 和 `max_morality` 参数
4. **存储空间**: 累积数据会占用一定存储空间，定期清理旧数据

## 性能建议

- **快速测试**: 先用少量 `num_runs` (3-5) 进行快速测试
- **详细分析**: 再用更多 `num_runs` (10-20) 获得精确结果  
- **分批运行**: 可以分多次运行，避免一次运行时间过长 