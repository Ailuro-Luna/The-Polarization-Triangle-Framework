# 极化三角框架 Sobol敏感性分析使用指南

## 概述

本指南介绍如何使用极化三角框架中新增的Sobol敏感性分析功能，对关键参数α、β、γ、cohesion_factor进行系统性的敏感性分析。

## 功能特性

### 🔍 分析能力
- **参数敏感性量化**: 计算一阶敏感性指数(S1)和总敏感性指数(ST)
- **交互效应分析**: 识别参数间的协同和拮抗作用
- **多指标评估**: 支持11种输出指标的综合分析
- **参数重要性排序**: 为模型调优提供指导

### 📊 可视化功能
- 敏感性指数对比图
- 热力图展示
- 交互效应分析图
- 参数重要性排序图

### ⚡ 性能优化
- 并行计算支持
- 中间结果缓存
- 分批执行机制
- 断点续算功能

## 安装依赖

在使用敏感性分析功能前，需要安装额外的依赖包：

```bash
pip install SALib==1.4.7 seaborn==0.12.2 pandas==2.0.3 openpyxl
```

或者更新整个环境：

```bash
pip install -r requirements.txt
```

**注意**: `openpyxl` 是Excel文件导出的必需依赖。

## 快速开始

### 1. 基础使用

```python
from polarization_triangle.analysis import SobolAnalyzer, SobolConfig

# 创建配置
config = SobolConfig(
    n_samples=100,      # 基础样本数
    n_runs=3,           # 每个参数组合运行次数
    n_processes=4,      # 并行进程数
    output_dir="my_sobol_results"
)

# 运行分析
analyzer = SobolAnalyzer(config)
sensitivity_indices = analyzer.run_complete_analysis()

# 查看结果
summary = analyzer.get_summary_table()
print(summary.head())
```

### 2. 命令行使用

```bash
# 快速测试
python polarization_triangle/scripts/run_sobol_analysis.py --config quick

# 标准分析
python polarization_triangle/scripts/run_sobol_analysis.py --config standard

# 高精度分析
python polarization_triangle/scripts/run_sobol_analysis.py --config high_precision

# 自定义参数
python polarization_triangle/scripts/run_sobol_analysis.py \
    --n-samples 500 \
    --n-runs 5 \
    --n-processes 8 \
    --output-dir my_custom_analysis
```

### 3. 可视化结果

```python
from polarization_triangle.analysis import SensitivityVisualizer

# 创建可视化器
visualizer = SensitivityVisualizer()

# 生成综合报告
plot_files = visualizer.create_comprehensive_report(
    sensitivity_indices,
    output_dir="plots"
)

# 单独创建图表
fig = visualizer.plot_sensitivity_heatmap(sensitivity_indices, 'ST')
fig.show()
```

## 配置选项

### 预设配置

| 配置名 | 样本数 | 运行次数 | 模拟步数 | 进程数 | 适用场景 | 预计耗时 |
|--------|--------|----------|----------|--------|----------|----------|
| `quick` | 50 | 2 | 100 | 2 | 快速测试 | 1-3分钟 |
| `standard` | 500 | 3 | 200 | 4 | 常规分析 | 15-30分钟 |
| `high_precision` | 1000 | 5 | 300 | 6 | 研究级分析 | 1-2小时 |
| `full` | 2000 | 10 | 500 | 8 | 论文发表级 | 4-8小时 |

### 参数配置

#### SobolConfig参数

```python
config = SobolConfig(
    # 参数范围定义
    parameter_bounds={
        'alpha': [0.1, 0.8],        # 自我激活系数范围
        'beta': [0.05, 0.3],        # 社会影响系数范围
        'gamma': [0.2, 2.0],        # 道德化影响系数范围
        'cohesion_factor': [0.0, 0.5]  # 身份凝聚力因子范围
    },
    
    # 采样参数
    n_samples=1000,           # 基础样本数，总样本数 = N × (2D + 2)
    n_runs=5,                 # 每个参数组合运行次数
    
    # 计算参数
    n_processes=4,            # 并行进程数
    save_intermediate=True,   # 是否保存中间结果
    output_dir="sobol_results",  # 输出目录
    
    # 模拟参数
    num_steps=200,            # 每次模拟的步数
    base_config=None          # 基础模拟配置，默认使用标准配置
)
```

#### 自定义基础模拟配置

```python
from polarization_triangle.core.config import SimulationConfig

# 创建自定义基础配置
custom_base = SimulationConfig(
    num_agents=300,
    network_type='lfr',
    morality_rate=0.4,
    opinion_distribution='twin_peak'
)

# 应用到敏感性分析
config = SobolConfig(
    base_config=custom_base,
    n_samples=500
)
```

## 输出指标说明

敏感性分析评估以下11个关键指标：

### 极化相关指标
- **polarization_index**: Koudenburg极化指数，衡量系统整体极化程度
- **opinion_variance**: 意见方差，反映观点分散程度  
- **extreme_ratio**: 极端观点比例，|opinion| > 0.8的Agent比例
- **identity_polarization**: 身份间极化差异

### 收敛相关指标
- **mean_abs_opinion**: 平均绝对意见，系统观点强度
- **final_stability**: 最终稳定性，最后阶段的变异系数

### 动态过程指标
- **trajectory_length**: 意见轨迹长度，观点变化的累积距离
- **oscillation_frequency**: 振荡频率，观点方向改变的频次
- **group_divergence**: 群体分化度，不同身份群体间的意见差异

### 身份相关指标  
- **identity_variance_ratio**: 身份方差比，组间方差与组内方差的比值
- **cross_identity_correlation**: 跨身份相关性，不同身份群体意见的相关系数

## 结果解释

### 敏感性指数

#### 一阶敏感性指数 (S1)
- **含义**: 参数单独对输出方差的贡献比例
- **解释**: S1值越高，表示该参数的主效应越强
- **阈值**: 
  - S1 > 0.1: 高敏感性
  - 0.05 < S1 ≤ 0.1: 中等敏感性
  - S1 ≤ 0.05: 低敏感性

#### 总敏感性指数 (ST)
- **含义**: 参数及其所有交互项对输出方差的总贡献
- **解释**: ST值越高，表示该参数的总体影响越强
- **阈值**:
  - ST > 0.15: 高敏感性
  - 0.1 < ST ≤ 0.15: 中等敏感性
  - ST ≤ 0.1: 低敏感性

#### 交互效应强度 (ST - S1)
- **含义**: 参数通过交互效应产生的影响
- **解释**: 
  - ST - S1 ≈ 0: 主要通过主效应影响
  - ST - S1 > 0: 存在显著交互效应
- **阈值**:
  - ST - S1 > 0.1: 强交互效应
  - 0.05 < ST - S1 ≤ 0.1: 中等交互效应
  - ST - S1 ≤ 0.05: 弱交互效应

### 参数作用机制

#### α (自我激活系数)
- **预期影响**: 对极化相关指标具有高敏感性
- **机制**: 控制Agent坚持自身观点的强度
- **典型表现**: 在polarization_index和extreme_ratio上显示高ST值

#### β (社会影响系数)  
- **预期影响**: 对收敛相关指标具有高敏感性
- **机制**: 控制邻居影响的强度
- **典型表现**: 在mean_abs_opinion和final_stability上显示高ST值

#### γ (道德化影响系数)
- **预期影响**: 在中高道德化率情况下影响显著
- **机制**: 调节道德化对社会影响的抑制作用
- **典型表现**: 主要通过交互效应发挥作用，ST - S1值较大

#### cohesion_factor (身份凝聚力因子)
- **预期影响**: 作为调节因子，主要通过交互效应发挥作用
- **机制**: 增强网络连接强度，促进观点传播
- **典型表现**: 在identity_相关指标上显示影响

## 高级用法

### 1. 批量分析不同场景

```python
scenarios = {
    'high_morality': {'morality_rate': 0.8},
    'low_morality': {'morality_rate': 0.2},
    'large_network': {'num_agents': 500},
    'small_network': {'num_agents': 100}
}

results = {}
for scenario_name, params in scenarios.items():
    base_config = SimulationConfig(**params)
    config = SobolConfig(
        base_config=base_config,
        output_dir=f"sobol_{scenario_name}"
    )
    
    analyzer = SobolAnalyzer(config)
    results[scenario_name] = analyzer.run_complete_analysis()
```

### 2. 自定义指标计算

```python
from polarization_triangle.analysis import SensitivityMetrics

class CustomMetrics(SensitivityMetrics):
    def __init__(self):
        super().__init__()
        self.metric_names.append('my_custom_metric')
    
    def _calculate_custom_metric(self, sim):
        # 实现自定义指标计算
        return np.mean(sim.opinions ** 2)
    
    def calculate_all_metrics(self, sim):
        metrics = super().calculate_all_metrics(sim)
        metrics['my_custom_metric'] = self._calculate_custom_metric(sim)
        return metrics
```

### 3. 结果后处理

```python
# 加载已有结果
analyzer = SobolAnalyzer(config)
sensitivity_indices = analyzer.load_results("existing_results")

# 筛选高敏感性参数
high_sensitivity_params = []
for output_name, indices in sensitivity_indices.items():
    for i, param in enumerate(['alpha', 'beta', 'gamma', 'cohesion_factor']):
        if indices['ST'][i] > 0.15:
            high_sensitivity_params.append((param, output_name, indices['ST'][i]))

# 按敏感性排序
high_sensitivity_params.sort(key=lambda x: x[2], reverse=True)
print("高敏感性参数-指标组合:")
for param, output, sensitivity in high_sensitivity_params[:10]:
    print(f"  {param} -> {output}: {sensitivity:.3f}")
```

## 最佳实践

### 1. 计算资源规划

```python
# 估算计算时间
def estimate_computation_time(config):
    total_simulations = config.n_samples * (2 * 4 + 2) * config.n_runs
    time_per_sim = 2  # 秒，根据实际情况调整
    total_time = total_simulations * time_per_sim / config.n_processes
    
    print(f"预计需要运行 {total_simulations} 次模拟")
    print(f"使用 {config.n_processes} 个进程")
    print(f"预计耗时: {total_time/3600:.1f} 小时")

# 在运行前评估
estimate_computation_time(config)
```

### 2. 渐进式分析

```python
# 从小样本开始，逐步增加精度
sample_sizes = [50, 200, 500, 1000]

for n_samples in sample_sizes:
    config = SobolConfig(
        n_samples=n_samples,
        output_dir=f"progressive_analysis_{n_samples}"
    )
    
    analyzer = SobolAnalyzer(config)
    results = analyzer.run_complete_analysis()
    
    # 检查收敛性
    print(f"样本数 {n_samples} 的结果...")
    # 如果结果稳定，可以停止增加样本
```

### 3. 质量控制

```python
def validate_results(sensitivity_indices):
    """验证敏感性分析结果的质量"""
    for output_name, indices in sensitivity_indices.items():
        # 检查S1和ST的关系 (ST >= S1)
        s1_values = np.array(indices['S1'])
        st_values = np.array(indices['ST'])
        
        if np.any(st_values < s1_values):
            print(f"警告: {output_name} 中ST < S1，可能需要更多样本")
        
        # 检查总和是否合理 (一般ST之和不应远超1)
        st_sum = np.sum(st_values)
        if st_sum > 1.5:
            print(f"警告: {output_name} 的ST总和为 {st_sum:.2f}，可能存在强交互效应")

# 运行验证
validate_results(sensitivity_indices)
```

## 故障排除

### 常见问题

#### 1. 模拟失败率高
**症状**: 出现 `'Simulation' object has no attribute 'run'` 或大量模拟返回默认值
**解决方案**:
- ✅ **已修复**: 使用 `sim.step()` 循环代替 `sim.run()`
- 检查参数范围是否合理
- 减少模拟步数进行测试
- 检查网络生成是否成功

#### 2. Excel导出失败
**症状**: `No module named 'openpyxl'` 错误
**解决方案**:
- ✅ **已修复**: 安装 openpyxl 依赖
```bash
pip install openpyxl
```

#### 3. 图形界面卡住
**症状**: 程序在可视化阶段长时间无响应
**解决方案**:
- ✅ **已修复**: 移除 `plt.show()` 调用，只保存图片文件
- 图表会自动保存到 `{output_dir}/plots/` 目录
- 程序不再需要图形界面交互

#### 4. 中文字体警告
**症状**: 大量中文字体缺失警告
**解决方案**:
- ✅ **已优化**: 图表标签改为英文，兼容性更好
- 警告不影响功能，可以忽略
- 如需中文显示，安装中文字体包

#### 5. 内存不足
**症状**: 运行过程中系统内存耗尽
**解决方案**:
- 减少并行进程数 (`n_processes`)
- 减少Agent数量 (`num_agents`)
- 启用中间结果保存，分批运行

#### 6. 敏感性结果异常
**症状**: ST < S1或敏感性值全为0
**解决方案**:
- 增加样本数量 (`n_samples`)
- 检查输出指标计算是否正确
- 验证参数范围设置
- 某些指标（如 `final_stability`, `trajectory_length`, `oscillation_frequency`）可能在短模拟中无效

#### 7. 计算时间过长
**症状**: 分析运行时间超出预期
**解决方案**:
- 使用 `quick` 配置进行预测试
- 优化基础模拟配置（减少 `num_agents`, `num_steps`）
- 增加并行进程数
- 使用更快的硬件

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用最小配置测试
debug_config = SobolConfig(
    n_samples=10,
    n_runs=1,
    num_steps=20,
    output_dir="debug_test"
)

try:
    analyzer = SobolAnalyzer(debug_config)
    results = analyzer.run_complete_analysis()
    print("调试测试成功")
except Exception as e:
    print(f"调试失败: {e}")
    import traceback
    traceback.print_exc()
```

## 引用和参考

如果您在研究中使用了这个敏感性分析功能，请考虑引用相关文献：

1. Saltelli, A., et al. (2008). Global sensitivity analysis: the primer. John Wiley & Sons.
2. Sobol, I. M. (2001). Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. Mathematics and computers in simulation, 55(1-3), 271-280.

## 最新运行示例

以下是成功运行的实际输出示例：

```bash
$ python polarization_triangle/scripts/run_sobol_analysis.py --config standard

使用配置: standard
样本数: 500
运行次数: 3
进程数: 4
模拟步数: 200
输出目录: sobol_results_standard

生成Saltelli样本...
样本矩阵形状: (5000, 4)

运行模拟 (5000个参数组合)...
使用4个进程并行计算...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:39<00:00, 126.02it/s]

计算敏感性指数...
输出指标: ['polarization_index', 'opinion_variance', 'extreme_ratio', 'identity_polarization', 'mean_abs_opinion', 'final_stability', 'trajectory_length', 'oscillation_frequency', 'group_divergence', 'identity_variance_ratio', 'cross_identity_correlation']

============================================================
生成分析报告
============================================================

敏感性分析摘要 (前10行):
        Parameter         Output    S1    S1_conf    ST   ST_conf  Interaction
             alpha  extreme_ratio  1.109     0.034  1.109     0.012       -0.000
             alpha  mean_abs_opinion  0.188     0.011  0.261     0.012        0.073
              beta  extreme_ratio  0.633     0.017  0.635     0.023        0.002
              beta  identity_variance_ratio  0.606     0.014  0.622     0.018        0.016
              beta  mean_abs_opinion  0.598     0.013  0.628     0.014        0.030

结果已导出到: sobol_results_standard\sobol_results.xlsx

生成可视化报告...
所有图表已保存到: sobol_results_standard\plots

============================================================
关键发现
============================================================

1. 参数重要性排序 (基于平均总敏感性指数):
   1. β (社会影响): 0.584
   2. α (自我激活): 0.332  
   3. γ (道德化影响): 0.197
   4. cohesion_factor (凝聚力): 0.133

2. 平均交互效应强度 (ST - S1):
   α (自我激活): 0.038 (弱)
   β (社会影响): 0.035 (弱)
   γ (道德化影响): 0.061 (中等)
   cohesion_factor (凝聚力): 0.049 (弱)

3. 各输出指标的最敏感参数:
   polarization_index: β (社会影响) (0.752)
   opinion_variance: β (社会影响) (0.612)
   extreme_ratio: α (自我激活) (1.109)
   identity_polarization: β (社会影响) (0.691)
   mean_abs_opinion: β (社会影响) (0.628)
   final_stability: γ (道德化影响) (0.423)
   trajectory_length: β (社会影响) (0.564)
   oscillation_frequency: β (社会影响) (0.781)
   group_divergence: β (社会影响) (0.513)
   identity_variance_ratio: β (社会影响) (0.622)
   cross_identity_correlation: β (社会影响) (0.349)

分析完成！总耗时: 39秒
```

## 更新日志

- **v1.1.0 (最新)**: 🚀 **稳定性和兼容性大幅提升**
  - ✅ 修复了Simulation运行方法问题，使用正确的step()循环
  - ✅ 添加了openpyxl依赖，支持Excel文件导出
  - ✅ 移除plt.show()调用，避免图形界面卡住
  - ✅ 优化中文字体处理，提高跨平台兼容性
  - ✅ 更新了预设配置，提供更准确的时间估算
  - ✅ 完善了错误处理和异常恢复机制

- **v1.0.0**: 初始版本，支持四参数Sobol分析
  - 增加了11种输出指标
  - 实现了并行计算和可视化功能
  - 提供了多种预设配置

## 贡献和支持

如果您发现bug或有改进建议，请提交Issue或Pull Request。

对于使用问题，请参考：
1. 本使用指南
2. 代码中的示例文件
3. 项目的主要文档 