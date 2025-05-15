# Polarization Triangle 可视化指南

本文档详细说明了项目中各类关键可视化图表的生成代码和输出路径，便于理解和修改。

## 1. 意见分布热图 (2D Heatmap)

### 单次实验热图

**生成函数**: `draw_opinion_distribution_heatmap` 
**源文件**: `polarization_triangle/visualization/opinion_viz.py`

**调用位置**:
- 在 `polarization_triangle/experiments/zealot_experiment.py` 中的 `run_simulation_and_generate_results` 函数中调用
- 文件路径: `{results_dir}/{mode_name.lower().replace(' ', '_')}_heatmap.png`
- 例如: `results/zealot_experiment/with_clustered_zealots_heatmap.png`

**关键代码片段**:
```python
# zealot_experiment.py 中的调用
draw_opinion_distribution_heatmap(
    opinion_history, 
    f"Opinion Evolution {mode_name}", 
    f"{results_dir}/{mode_name.lower().replace(' ', '_')}_heatmap.png"
)
```

### 多次实验平均热图

**生成函数**: `generate_average_heatmaps` 
**源文件**: `polarization_triangle/experiments/multi_zealot_experiment.py`

**调用位置**:
- 在 `multi_zealot_experiment.py` 的 `run_multi_zealot_experiment` 函数中调用
- 在 `zealot_parameter_sweep.py` 的 `run_zealot_parameter_experiment` 函数中调用
- 文件路径: `{output_dir}/average_results/avg_{mode.lower().replace(' ', '_')}_heatmap.png`
- 例如: `results/multi_zealot_experiment/average_results/avg_with_clustered_zealots_heatmap.png`

**关键代码片段**:
```python
# multi_zealot_experiment.py 中的函数
def generate_average_heatmaps(all_opinion_histories, mode_names, output_dir):
    # ...
    for mode in mode_names:
        # ...
        avg_history = calculate_average_opinion_history(mode_histories)
        heatmap_file = os.path.join(output_dir, f"avg_{mode.lower().replace(' ', '_')}_heatmap.png")
        draw_opinion_distribution_heatmap(
            avg_history,
            f"Average Opinion Evolution {mode} (Multiple Runs)",
            heatmap_file,
            bins=40,
            log_scale=True
        )
```

## 2. 平均方差图表

### 全体 Agent 方差 (非 Zealot)

**生成函数**: 
- 单次实验: `generate_opinion_statistics` 计算数据，`plot_comparative_statistics` 绘制图表
- 多次实验: `average_stats` 计算平均数据，`plot_average_statistics` 绘制平均图表

**源文件**:
- `polarization_triangle/experiments/zealot_experiment.py`
- `polarization_triangle/experiments/multi_zealot_experiment.py`

**输出文件**:
- 单次实验比较图: `{results_dir}/statistics/comparison_non_zealot_variance.png`
- 多次实验平均图: `{output_dir}/statistics/avg_non_zealot_variance.png`

**关键代码片段**:
```python
# zealot_experiment.py 中计算方差数据
for step_opinions in trajectory:
    non_zealot_opinions = np.delete(step_opinions, zealot_ids) if zealot_ids else step_opinions
    non_zealot_var.append(np.var(non_zealot_opinions))

# multi_zealot_experiment.py 中绘制平均方差图
plt.figure(figsize=(12, 7))
for i, mode in enumerate(mode_names):
    plt.plot(step_values, avg_stats[mode]["non_zealot_variance"], 
            label=f'{mode}', 
            color=colors[i], linestyle='-')
plt.xlabel('Step')
plt.ylabel('Variance')
plt.title('Average Opinion Variance (Excluding Zealots) across Different Simulations')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(stats_dir, "avg_non_zealot_variance.png"), dpi=300)
```

### 社区内部方差 (Cluster Variance)

**生成函数**: 
- 单次实验: `generate_opinion_statistics` 计算数据和 `plot_comparative_statistics` 绘制比较图
- 多次实验: `average_stats` 计算平均数据，`plot_average_statistics` 绘制平均图表

**源文件**:
- `polarization_triangle/experiments/zealot_experiment.py`
- `polarization_triangle/experiments/multi_zealot_experiment.py`

**输出文件**:
- 单次实验比较图: `{results_dir}/statistics/comparison_cluster_variance.png`
- 多次实验平均图: `{output_dir}/statistics/avg_cluster_variance.png`
- 每个社区单独方差: `{results_dir}/statistics/{file_prefix}_community_variances.png`

**关键代码片段**:
```python
# zealot_experiment.py 中计算社区内部方差
for community_id, members in communities.items():
    community_non_zealots = [m for m in members if m not in zealot_ids]
    if community_non_zealots:
        community_opinions = step_opinions[community_non_zealots]
        community_var = np.var(community_opinions)
        step_cluster_vars.append(community_var)
```

## 3. 平均意见和绝对意见图表

**生成函数**:
- 单次实验: `generate_opinion_statistics` 计算数据，`plot_comparative_statistics` 绘制图表
- 多次实验: `average_stats` 计算平均数据，`plot_average_statistics` 绘制平均图表

**源文件**:
- `polarization_triangle/experiments/zealot_experiment.py`
- `polarization_triangle/experiments/multi_zealot_experiment.py`

**输出文件**:
- 单次实验平均意见比较图: `{results_dir}/statistics/comparison_mean_opinions.png`
- 单次实验绝对意见比较图: `{results_dir}/statistics/comparison_mean_abs_opinions.png`
- 多次实验平均意见图: `{output_dir}/statistics/avg_mean_opinions.png`
- 多次实验绝对意见图: `{output_dir}/statistics/avg_mean_abs_opinions.png`

**关键代码片段**:
```python
# zealot_experiment.py 中计算平均意见数据
for step_opinions in trajectory:
    mean_opinions.append(np.mean(step_opinions))
    mean_abs_opinions.append(np.mean(np.abs(step_opinions)))

# multi_zealot_experiment.py 中绘制平均意见图
plt.figure(figsize=(12, 7))
for i, mode in enumerate(mode_names):
    plt.plot(step_values, avg_stats[mode]["mean_opinions"], 
            label=f'{mode} - Mean Opinion', 
            color=colors[i], linestyle='-')
# ...
plt.savefig(os.path.join(stats_dir, "avg_mean_opinions.png"), dpi=300)
```

## 4. 负面意见统计图表

**生成函数**:
- 单次实验: `generate_opinion_statistics` 计算数据
- 多次实验: `plot_average_statistics` 绘制平均图表

**源文件**:
- `polarization_triangle/experiments/zealot_experiment.py`
- `polarization_triangle/experiments/multi_zealot_experiment.py`

**输出文件**:
- 负面意见数量比较图: `{results_dir}/statistics/comparison_negative_counts.png`
- 负面意见均值比较图: `{results_dir}/statistics/comparison_negative_means.png`
- 多次实验负面意见数量图: `{output_dir}/statistics/avg_negative_counts.png`
- 多次实验负面意见均值图: `{output_dir}/statistics/avg_negative_means.png`

**关键代码片段**:
```python
# zealot_experiment.py 中计算负面意见数据
for step_opinions in trajectory:
    # 获取非zealot的意见
    non_zealot_opinions = np.delete(step_opinions, zealot_ids) if zealot_ids else step_opinions
    
    # 统计负面意见
    negative_mask = non_zealot_opinions < 0
    negative_opinions = non_zealot_opinions[negative_mask]
    negative_count = len(negative_opinions)
    negative_counts.append(negative_count)
    negative_means.append(np.mean(negative_opinions) if negative_count > 0 else 0)
```

## 5. 交互类型统计图表

**生成函数**: `generate_rule_usage_plots` 
**源文件**: `polarization_triangle/experiments/zealot_experiment.py`

**调用位置**:
- 在 `run_simulation_and_generate_results` 函数中调用

**输出文件**:
- 规则使用统计图: `{results_dir}/{title_prefix}_interaction_types.png`
- 规则累积使用统计图: `{results_dir}/{title_prefix}_interaction_types_cumulative.png`
- 规则使用统计文本文件: `{results_dir}/{title_prefix}_interaction_types_stats.txt`

**关键代码片段**:
```python
# zealot_experiment.py 中的函数
def generate_rule_usage_plots(sim, title_prefix, output_dir):
    # 绘制规则使用统计图
    rule_usage_path = os.path.join(output_dir, f"{title_prefix}_interaction_types.png")
    draw_interaction_type_usage(
        sim.rule_counts_history,
        f"Interaction Types over Time\n{title_prefix}",
        rule_usage_path
    )
    
    # 绘制规则累积使用统计图
    rule_cumulative_path = os.path.join(output_dir, f"{title_prefix}_interaction_types_cumulative.png")
    draw_interaction_type_cumulative_usage(
        sim.rule_counts_history,
        f"Cumulative Interaction Types\n{title_prefix}",
        rule_cumulative_path
    )
```

## 文件命名规则与路径总结

1. **单次实验**:
   - 基本输出目录: `results/zealot_experiment/`
   - 热图: `{mode_name.lower().replace(' ', '_')}_heatmap.png`
   - 统计图表: `statistics/{comparison|file_prefix}_{metric_name}.png`

2. **多次实验**:
   - 基本输出目录: `results/multi_zealot_experiment/`
   - 平均结果目录: `average_results/`
   - 平均热图: `avg_{mode.lower().replace(' ', '_')}_heatmap.png`
   - 平均统计图表: `statistics/avg_{metric_name}.png`

3. **参数扫描实验**:
   - 基本输出目录: `results/zealot_parameter_sweep/{parameter_combination}/`
   - 每次运行子目录: `run_{i+1}/`
   - 平均结果目录: `average_results/`
   - 命名规则同多次实验

## 自定义或修改可视化的建议

如需修改或自定义可视化图表:

1. **修改颜色或样式**:
   - 在相应的绘图函数中修改颜色、线型、标记等参数
   - 例如在 `plot_average_statistics` 中修改 `colors` 和 `linestyles` 数组

2. **添加新的统计指标**:
   - 在 `generate_opinion_statistics` 函数中添加新的统计计算
   - 在相应的绘图函数中添加新的图表绘制代码

3. **更改输出路径**:
   - 修改各实验函数中的 `output_dir` 参数或文件命名逻辑 