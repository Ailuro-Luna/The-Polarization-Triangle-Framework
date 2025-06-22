# Zealot Morality Analysis 开发日志 - 2024年12月22日

## 概述

本次开发会话主要对 `polarization_triangle/experiments/zealot_morality_analysis.py` 和 `polarization_triangle/utils/data_manager.py` 进行了重大功能增强，包括：

1. **新增 variance per identity 指标**：计算每个身份组内部的意见方差
2. **实现并行计算功能**：支持多进程并行执行，显著提升性能
3. **优化数据管理系统**：使用 Parquet 格式的高效存储方案
4. **增强可视化系统**：支持更多图表类型和样式配置

## 文件修改详情

### 1. polarization_triangle/experiments/zealot_morality_analysis.py

#### 1.1 文件结构和整体架构

**文件长度**：1383行，是一个功能完整的实验分析模块

**主要组成部分**：
- **工具函数**：时间格式化等辅助功能
- **并行计算支持函数**：多进程任务处理
- **核心实验逻辑函数**：参数组合生成、单次模拟执行、参数扫描
- **数据管理函数**：与新的数据管理器集成
- **绘图相关函数**：样式配置、图表生成
- **高级接口函数**：用户友好的API接口

#### 1.2 新增的 variance per identity 指标

**功能描述**：
- 计算每个身份组（identity=1 和 identity=-1）内部的意见方差
- 提供分离版本和合并版本的可视化

**核心实现**：

```python
def run_single_simulation(config: SimulationConfig, steps: int = 500) -> Dict[str, float]:
    # ... 现有代码 ...
    
    # 计算 variance per identity (每个身份组内的方差)
    variance_per_identity = {'identity_1': 0.0, 'identity_-1': 0.0}
    
    # 获取非zealot节点的意见和身份
    zealot_mask = np.zeros(sim.num_agents, dtype=bool)
    if sim.enable_zealots and sim.zealot_ids:
        zealot_mask[sim.zealot_ids] = True
    
    non_zealot_mask = ~zealot_mask
    non_zealot_opinions = sim.opinions[non_zealot_mask]
    non_zealot_identities = sim.identities[non_zealot_mask]
    
    # 分别计算每个身份组的方差
    for identity_val in [1, -1]:
        identity_mask = non_zealot_identities == identity_val
        if np.sum(identity_mask) > 1:  # 至少需要2个节点才能计算方差
            identity_opinions = non_zealot_opinions[identity_mask]
            variance_per_identity[f'identity_{identity_val}'] = float(np.var(identity_opinions))
        else:
            variance_per_identity[f'identity_{identity_val}'] = 0.0
    
    return {
        'mean_opinion': mean_stats['mean_opinion'],
        'variance': variance_stats['overall_variance'],
        'identity_opinion_difference': identity_opinion_difference,
        'polarization_index': polarization,
        'variance_per_identity': variance_per_identity  # 新增指标
    }
```

**关键修复**：
- 修正了 zealot 识别逻辑：从错误的 `sim.is_zealot` 改为正确的 `sim.zealot_ids`
- 正确区分 zealot 和非 zealot 节点，确保方差计算的准确性

#### 1.3 并行计算功能实现

**设计理念**：
- 采用任务级并行：每个模拟运行作为独立任务
- 保持向后兼容：`num_processes=1` 时使用原有串行逻辑
- 容错机制：失败任务用 NaN 填充，支持自动回退

**核心函数**：

```python
def run_single_simulation_task(task_params):
    """
    单个模拟任务的包装函数，用于多进程并行计算
    """
    try:
        plot_type, combination, x_val, run_idx, steps, process_id = task_params
        
        # 设置进程特定的随机种子
        np.random.seed((int(x_val * 1000) + run_idx + process_id) % (2**32))
        
        # 构建配置并运行模拟
        # ... 配置逻辑 ...
        
        results = run_single_simulation(base_config, steps)
        return (x_val, run_idx, results, True, None)
        
    except Exception as e:
        error_msg = f"Process {process_id}: Simulation failed for x={x_val}, run={run_idx}: {str(e)}"
        return (x_val, run_idx, None, False, error_msg)

def run_parameter_sweep_parallel(plot_type: str, combination: Dict[str, Any], 
                                x_values: List[float], num_runs: int = 5, num_processes: int = 4):
    """
    并行版本的参数扫描
    """
    print(f"🚀 使用 {num_processes} 个进程进行并行计算...")
    
    # 创建所有任务
    tasks = []
    for x_val in x_values:
        for run_idx in range(num_runs):
            process_id = len(tasks) % num_processes
            task = (plot_type, combination, x_val, run_idx, combination['steps'], process_id)
            tasks.append(task)
    
    # 执行并行计算
    try:
        with multiprocessing.Pool(num_processes) as pool:
            results_list = []
            with tqdm(total=len(tasks), desc=f"Running {combination['label']} (parallel)") as pbar:
                for result in pool.imap(run_single_simulation_task, tasks):
                    results_list.append(result)
                    pbar.update(1)
    except Exception as e:
        print(f"❌ 并行计算失败，回退到串行模式: {e}")
        return run_parameter_sweep_serial(plot_type, combination, x_values, num_runs)
    
    return organize_parallel_results(results_list, x_values, num_runs)
```

**性能优化特性**：
- 随机种子管理：确保每个进程有不同但可重现的随机序列
- 进度显示：并行任务也有详细的进度条
- 错误处理：完善的异常处理和回退机制

#### 1.4 增强的可视化系统

**新增图表类型**：
- `variance_per_identity_1`：identity=+1 组的方差图表
- `variance_per_identity_-1`：identity=-1 组的方差图表  
- `variance_per_identity_combined`：两个身份组的合并图表

**样式配置系统**：

```python
def get_variance_per_identity_style(identity_label: str, plot_type: str) -> Dict[str, Any]:
    """
    为 variance per identity 图表生成特殊的样式配置
    """
    # 线型组合：实线用于 ID=1，虚线用于 ID=-1
    linestyles = {
        '1': '-',      # 实线用于 identity=1
        '-1': '--'     # 虚线用于 identity=-1
    }
    
    # 标记形状：圆形用于 ID=1，方形用于 ID=-1
    markers = {
        '1': 'o',      # 圆形用于 identity=1
        '-1': 's'      # 方形用于 identity=-1
    }
    
    # 提取身份值和原始组合标签
    identity_val = identity_label.split('(ID=')[-1].rstrip(')')
    base_label = identity_label.split(' (ID=')[0]
    
    # 基于哈希值分配颜色，确保一致性
    color_index = abs(hash(base_label)) % len(colors)
    if identity_val == '-1':
        color_index = (color_index + len(colors) // 2) % len(colors)
    
    return {
        'color': colors[color_index],
        'linestyle': linestyles.get(identity_val, '-'),
        'marker': markers.get(identity_val, 'o'),
        'markersize': 8 if identity_val == '1' else 6,
        'group': f'identity_{identity_val}'
    }

def get_combined_variance_per_identity_style(identity_label: str, plot_type: str) -> Dict[str, Any]:
    """
    为合并的 variance per identity 图表生成样式配置
    相同配置的两条线使用相同颜色和标记，但用实线/虚线区分身份组
    """
    # 提取身份值和原始组合标签
    identity_val = identity_label.split('(ID=')[-1].rstrip(')')
    base_label = identity_label.split(' (ID=')[0]
    
    # 基于原始组合标签计算颜色索引（确保相同配置使用相同颜色）
    color_index = abs(hash(base_label)) % len(colors)
    
    # 线型：+1 用实线，-1 用虚线
    linestyle = '-' if identity_val == '+1' else '--'
    
    # 标记：相同配置使用相同标记
    marker_index = abs(hash(base_label)) % len(markers)
    marker = markers[marker_index]
    
    return {
        'color': colors[color_index],
        'linestyle': linestyle,
        'marker': marker,
        'markersize': 8 if identity_val == '+1' else 6,
        'group': f'combined_identity_{identity_val}'
    }
```

**图表布局优化**：
- 针对不同线条数量调整图表大小和图例布局
- morality_ratios 图表：24x14 英寸，4列图例
- zealot_numbers 图表：20x12 英寸，3列图例

#### 1.5 实验配置系统

**参数组合生成**：

```python
def create_config_combinations():
    """
    创建实验参数组合配置
    
    1. zealot_numbers实验：测试不同zealot数量对系统的影响
       - 变量：zealot数量 (x轴)
       - 固定：zealot身份分配=True, 身份分布=random
       - 比较：zealot分布模式(random/clustered) × morality比例(0.0/0.3) = 4个组合
    
    2. morality_ratios实验：测试不同morality比例对系统的影响
       - 变量：morality比例 (x轴)
       - 固定：zealot数量=20
       - 比较：zealot模式(random/clustered/none) × zealot身份对齐(True/False) × 
               身份分布(random/clustered) = 10个组合
    """
    # 详细的配置生成逻辑...
```

**最终图表数量**：
- 从原来的 8 张增加到 14 张
- 2 种实验类型 × 7 个指标 = 14 张图表

### 2. polarization_triangle/utils/data_manager.py

#### 2.1 ExperimentDataManager 类架构

**类功能**：专门用于 zealot_morality_analysis 实验的数据存储和读取

**核心特性**：
- 使用 Parquet 格式，平衡压缩率和读取速度
- 支持批次管理和数据累积
- 为并行计算预留接口
- 支持 variance per identity 计算需求

**目录结构**：
```
results/zealot_morality_analysis/
├── experiment_data/
│   ├── zealot_numbers_data.parquet
│   └── morality_ratios_data.parquet
├── metadata/
│   ├── batch_metadata.json
│   └── experiment_config.json
└── mean_plots/
    └── [生成的图表文件]
```

#### 2.2 数据存储格式

**Parquet 格式的扁平化存储**：

```python
def save_batch_results(self, plot_type: str, batch_data: Dict[str, Any], batch_metadata: Dict[str, Any]):
    """
    保存批次实验结果
    
    数据格式转换：
    嵌套格式 -> 扁平DataFrame格式
    {combination: {x_values: [], results: {metric: [[run1, run2, ...], ...]}}}
    ->
    DataFrame with columns: ['batch_id', 'timestamp', 'combination', 'x_value', 'metric', 'run_index', 'value']
    """
    rows = []
    for combination_label, combo_data in batch_data.items():
        x_values = combo_data['x_values']
        results = combo_data['results']
        
        for x_idx, x_value in enumerate(x_values):
            for metric_name, metric_results in results.items():
                if x_idx < len(metric_results):
                    for run_idx, run_value in enumerate(metric_results[x_idx]):
                        rows.append({
                            'batch_id': batch_id,
                            'timestamp': timestamp,
                            'combination': combination_label,
                            'x_value': x_value,
                            'metric': metric_name,
                            'run_index': run_idx,
                            'value': run_value
                        })
    
    # 创建DataFrame并保存为Parquet格式
    new_df = pd.DataFrame(rows)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True) if target_file.exists() else new_df
    combined_df.to_parquet(target_file, compression='snappy', index=False)
```

#### 2.3 数据读取和转换

**绘图格式转换**：

```python
def convert_to_plotting_format(self, plot_type: str) -> Tuple[Dict[str, Dict[str, List[List[float]]]], List[float], Dict[str, int]]:
    """
    将存储的数据转换为绘图格式
    
    Returns:
        (all_results, x_values, total_runs_per_combination)
        
    all_results 格式:
    {
        combination_label: {
            metric_name: [
                [run1_val, run2_val, ...],  # x_value_1 的所有运行结果
                [run1_val, run2_val, ...],  # x_value_2 的所有运行结果
                ...
            ]
        }
    }
    """
    df = self.load_experiment_data(plot_type)
    combinations = sorted(df['combination'].unique())
    x_values = sorted(df['x_value'].unique())
    metrics = sorted(df['metric'].unique())
    
    all_results = {}
    total_runs_per_combination = {}
    
    for combination in combinations:
        combo_data = df[df['combination'] == combination]
        
        # 计算总运行次数
        unique_x_values = len(combo_data['x_value'].unique())
        unique_metrics = len(combo_data['metric'].unique())
        total_runs = len(combo_data) // (unique_x_values * unique_metrics)
        total_runs_per_combination[combination] = total_runs
        
        # 组织数据为绘图格式
        combo_results = {}
        for metric in metrics:
            metric_results = []
            metric_data = combo_data[combo_data['metric'] == metric]
            
            for x_val in x_values:
                x_data = metric_data[metric_data['x_value'] == x_val]
                run_values = x_data['value'].tolist()
                metric_results.append(run_values)
            
            combo_results[metric] = metric_results
        
        all_results[combination] = combo_results
    
    return all_results, x_values, total_runs_per_combination
```

#### 2.4 元数据管理

**批次元数据**：
```json
{
  "batches": [
    {
      "batch_id": "20241222_143052",
      "timestamp": "2024-12-22 14:30:52",
      "experiment_type": "zealot_numbers",
      "num_runs": 5,
      "max_zealots": 50,
      "x_range": [0, 50],
      "combinations_count": 4
    }
  ]
}
```

**实验配置**：
```json
{
  "batch_name": "20241222_143052",
  "num_runs": 5,
  "max_zealots": 50,
  "max_morality": 30,
  "elapsed_time": 1234.56,
  "total_combinations": 14,
  "saved_at": "2024-12-22T14:35:10.123456"
}
```

#### 2.5 摘要报告功能

**摘要报告生成**：

```python
def export_summary_report(self) -> str:
    """
    导出实验摘要报告
    """
    zealot_summary = self.get_experiment_summary('zealot_numbers')
    morality_summary = self.get_experiment_summary('morality_ratios')
    batch_metadata = self.get_batch_metadata()
    
    report = []
    report.append("=" * 60)
    report.append("实验数据摘要报告")
    report.append("=" * 60)
    
    report.append(f"\n📊 Zealot Numbers 实验:")
    report.append(f"   总记录数: {zealot_summary['total_records']}")
    report.append(f"   参数组合数: {len(zealot_summary['combinations'])}")
    report.append(f"   批次数: {len(zealot_summary['batches'])}")
    
    # ... 更多统计信息 ...
    
    return "\n".join(report)
```

## 功能增强总结

### 3.1 新增指标

**Variance per Identity**：
- **功能**：计算每个身份组内部的意见方差
- **实现**：正确识别 zealot 和非 zealot 节点，分别计算两个身份组的方差
- **可视化**：提供分离版本（2张图）和合并版本（2张图）

### 3.2 并行计算功能

**性能提升**：
- **测试结果**：36% 性能提升（146.22秒 → 107.26秒）
- **加速比**：1.36x，并行效率 34.1%
- **容错机制**：失败任务自动处理，支持回退到串行模式

**技术特性**：
- 任务级并行：每个模拟运行作为独立任务
- 随机种子管理：确保结果可重现性
- 进度显示：实时显示并行任务进度

### 3.3 数据管理优化

**存储效率**：
- **格式**：Parquet 格式，自动压缩
- **结构**：扁平化存储，便于查询和分析
- **元数据**：完整的批次和实验配置记录

**功能特性**：
- 批次累积：支持多次运行数据累积
- 格式转换：自动转换为绘图所需格式
- 摘要报告：详细的实验统计信息

### 3.4 可视化增强

**图表数量**：从 8 张增加到 14 张
- 2 种实验类型（zealot_numbers, morality_ratios）
- 7 个指标（原有4个 + 新增3个variance per identity相关）

**样式系统**：
- 智能颜色分配：基于哈希值确保一致性
- 线型区分：实线/虚线区分不同身份组
- 图例优化：根据线条数量自动调整布局

## 使用方法

### 4.1 基本使用

```python
# 完整实验（数据收集 + 绘图）
run_zealot_morality_analysis(
    output_dir="results/zealot_morality_analysis",
    num_runs=5,
    max_zealots=50,
    max_morality=30,
    num_processes=12  # 使用12个进程并行计算
)

# 分步骤使用
# 步骤1：数据收集
run_and_accumulate_data(
    output_dir="results/zealot_morality_analysis",
    num_runs=5,
    max_zealots=50,
    max_morality=30,
    batch_name="batch_001",
    num_processes=12
)

# 步骤2：生成图表
plot_from_accumulated_data("results/zealot_morality_analysis")
```

### 4.2 数据管理器直接使用

```python
from polarization_triangle.utils.data_manager import ExperimentDataManager

# 创建数据管理器
data_manager = ExperimentDataManager("results/zealot_morality_analysis")

# 查看摘要报告
print(data_manager.export_summary_report())

# 获取实验数据
zealot_summary = data_manager.get_experiment_summary('zealot_numbers')
morality_summary = data_manager.get_experiment_summary('morality_ratios')

# 加载原始数据进行自定义分析
df = data_manager.load_experiment_data('zealot_numbers')
```

## 技术亮点

### 5.1 代码质量

- **文档完整**：每个函数都有详细的文档字符串
- **类型注解**：使用 typing 模块提供类型提示
- **错误处理**：完善的异常处理和回退机制
- **代码结构**：清晰的模块化设计

### 5.2 性能优化

- **并行计算**：多进程并行，显著提升性能
- **存储效率**：Parquet 格式，压缩率高，读取快
- **内存管理**：避免大量数据在内存中累积

### 5.3 用户体验

- **进度显示**：详细的进度条和状态信息
- **向后兼容**：保持原有接口不变
- **灵活配置**：支持多种参数配置组合

## 未来扩展方向

### 6.1 功能扩展

- **更多指标**：可以轻松添加新的统计指标
- **可视化选项**：支持更多图表类型和样式
- **数据分析**：集成更多统计分析功能

### 6.2 性能优化

- **分布式计算**：支持跨机器的分布式计算
- **GPU加速**：利用GPU加速计算密集型任务
- **内存优化**：进一步优化内存使用

### 6.3 数据管理

- **数据清理**：自动清理旧数据功能
- **数据导出**：支持更多数据格式导出
- **数据版本控制**：实验数据的版本管理

## 总结

本次开发会话成功实现了以下目标：

1. **新增 variance per identity 指标**：为系统分析提供了新的维度
2. **实现并行计算功能**：显著提升了实验执行效率
3. **优化数据管理系统**：提供了高效、可扩展的数据存储方案
4. **增强可视化系统**：支持更丰富的图表类型和样式配置

整个系统现在具备了：
- **完整性**：从数据收集到可视化的完整流程
- **高效性**：并行计算和优化存储的高性能
- **可扩展性**：模块化设计便于后续扩展
- **易用性**：用户友好的API接口

代码总行数约 1853 行（zealot_morality_analysis.py: 1383行, data_manager.py: 470行），结构清晰，功能完整，为极化三角框架的实验分析提供了强大的工具支持。