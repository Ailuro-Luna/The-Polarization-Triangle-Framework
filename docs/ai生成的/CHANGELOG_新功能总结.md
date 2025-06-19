# Zealot和Morality分析实验 - 新功能总结

## 🎉 功能改进概览

我们成功实现了您要求的所有功能改进，将原来的 `run_zealot_morality_analysis` 函数分离成两个独立部分，并增加了多项实用功能。

## ✅ 已实现的功能

### 1. 分离式运行架构

#### 🔄 **数据收集与绘图分离**
- **`run_and_accumulate_data()`**: 专门负责运行测试并以追加模式保存数据
- **`plot_from_accumulated_data()`**: 专门负责从累积数据生成图表
- **`run_zealot_morality_analysis()`**: 保持向后兼容的一次性运行功能

#### 📊 **数据累积功能**
- ✅ 支持多次运行，每次追加新数据而不覆盖
- ✅ 每次运行都有独特的 `batch_id` 和 `timestamp` 标识
- ✅ 数据自动保存到 `accumulated_data/` 子文件夹
- ✅ 批次信息文件记录每次运行的详细信息

### 2. 耗时统计系统

#### ⏱️ **全面的耗时计算**
- ✅ **Main函数总耗时**: 完整实验的总耗时
- ✅ **数据收集阶段耗时**: 包含所有测试运行的时间
- ✅ **绘图阶段耗时**: 图表生成的专门耗时
- ✅ **分阶段耗时**: 两种测试类型（Zealot Numbers, Morality Ratios）的分别耗时
- ✅ **进度显示**: 使用 `tqdm` 显示数据生成进度

#### 📈 **耗时显示格式**
```
🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒
⏱️  完整实验耗时总结
🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒
📊 数据收集阶段耗时: 0h 25m 30.45s
📈 图表生成阶段耗时: 0h 0m 5.20s
🎯 总耗时: 0h 25m 35.65s
🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒🕒
```

### 3. 总运行次数显示

#### 🔢 **智能运行次数计算**
- ✅ **计算总run数**: 基于 `总数据点 / (x值数量 × 指标数量)` 而不是batch数
- ✅ **图表标题显示**: `(15 total runs)` 或 `(5-12 total runs)` 对于不同范围
- ✅ **文件名包含运行次数**: `zealot_numbers_mean_opinion_15runs.png`
- ✅ **图例显示运行次数**: `Combination Label (n=15)`

#### 📊 **数据加载信息改进**
```
📂 Loading accumulated data files:
  ✅ zealot_numbers_Random_Morality_0_0_accumulated.csv: 1680 records, 15 total runs (3 batches)
  ✅ morality_ratios_Random_ID_align_True_ID_cluster_False_accumulated.csv: 1232 records, 11 total runs (2 batches)
```

### 4. 文件命名与组织

#### 📁 **智能文件命名**
- ✅ 文件名自动包含总运行次数: `_15runs.png`
- ✅ 不同组合的运行次数范围: `_5-12runs.png`
- ✅ 累积数据文件: `*_accumulated.csv`
- ✅ 批次信息文件: `batch_info_*.txt`

#### 🗂️ **文件结构**
```
results/
├── accumulated_data/                    # 累积数据
│   ├── zealot_numbers_*_accumulated.csv # 数据文件
│   ├── morality_ratios_*_accumulated.csv
│   └── batch_info_*.txt                 # 批次信息
├── error_bar_plots/                     # 图表（按类型分类）
│   └── *_15runs.png                     # 包含运行次数
├── scatter_plots/
├── mean_plots/
└── combined_plots/
```

## 🚀 使用示例

### 分步骤运行（推荐）
```python
# 第一次：快速测试
run_and_accumulate_data(
    output_dir="results/my_experiment",
    num_runs=5,
    batch_name="quick_test"
)

# 第二次：增加数据
run_and_accumulate_data(
    output_dir="results/my_experiment", 
    num_runs=10,
    batch_name="detailed_run"
)

# 生成图表（基于所有15次运行）
plot_from_accumulated_data("results/my_experiment")
```

### 一次性运行（向后兼容）
```python
run_zealot_morality_analysis(
    output_dir="results/traditional_run",
    num_runs=15
)
```

## 📋 技术实现细节

### 数据处理改进
1. **`process_accumulated_data_for_plotting()`**: 正确计算总运行次数
2. **`load_accumulated_data()`**: 显示总运行次数和批次数
3. **`plot_accumulated_results()`**: 图表标题和文件名包含运行次数信息

### 耗时计算实现
1. **Main函数级别**: 整个实验的总耗时
2. **阶段级别**: 数据收集vs绘图的分别耗时
3. **组合级别**: 每个参数组合的运行耗时（在原有函数中）

### 向后兼容性
- ✅ 原有的 `run_zealot_morality_analysis()` 函数保持完全兼容
- ✅ 现有脚本无需修改即可继续使用
- ✅ 新功能作为可选增强功能提供

## 🎯 实际效果

1. **累积数据**: 可以分多次运行实验，每次积累更多数据点
2. **精确统计**: 图表清楚显示基于多少次总运行的数据
3. **耗时监控**: 完整的耗时统计帮助优化实验流程
4. **灵活运行**: 数据收集和绘图可以分开进行

## 📝 测试脚本

- **`example_usage.py`**: 完整的使用示例
- **`test_new_features.py`**: 新功能的快速测试演示
- **`README_zealot_morality_analysis.md`**: 详细使用指南

现在您可以：
1. 多次运行数据收集，逐步积累更精确的结果
2. 清楚了解每个阶段的耗时
3. 在图表中看到基于多少次运行的准确信息
4. 灵活地进行实验设计和数据分析 