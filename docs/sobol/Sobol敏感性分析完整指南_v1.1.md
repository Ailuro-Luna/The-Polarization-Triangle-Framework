# 极化三角框架 Sobol敏感性分析完整指南 v1.1

## 📋 目录

1. [概述](#概述)
2. [系统要求与安装](#系统要求与安装)
3. [核心功能](#核心功能)
4. [快速开始](#快速开始)
5. [详细配置](#详细配置)
6. [输出指标详解](#输出指标详解)
7. [可视化功能](#可视化功能)
8. [高级用法](#高级用法)
9. [性能优化](#性能优化)
10. [故障排除](#故障排除)
11. [最佳实践](#最佳实践)
12. [API参考](#api参考)
13. [更新日志](#更新日志)

## 概述

极化三角框架Sobol敏感性分析工具是一个功能完整的全局敏感性分析系统，专门用于分析极化动力学模型中四个关键参数（α、β、γ、cohesion_factor）的影响程度和相互作用。

### 🌟 核心特性

- **🔍 全局敏感性分析**: 基于Sobol方法的参数重要性量化
- **🚀 高性能计算**: 多进程并行，支持大规模样本分析
- **📊 丰富的输出指标**: 14种涵盖极化、收敛、动态、身份的关键指标
- **🎨 专业可视化**: 多类型图表，支持高质量输出
- **🛠️ 易于使用**: 命令行和编程接口双重支持
- **💾 可靠存储**: 自动保存中间结果，支持断点续算
- **🔧 高度可配置**: 预设配置 + 自定义参数组合

### 🎯 分析目标

1. **参数重要性排序**: 确定哪个参数对模型行为影响最大
2. **交互效应识别**: 发现参数间的协同或拮抗作用
3. **模型行为理解**: 深化对极化三角框架机制的认识
4. **参数调优指导**: 为实际应用提供参数设置建议

## 系统要求与安装

### 环境要求

- **Python**: 3.8+ (推荐 3.9 或 3.10)
- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **内存**: 最少8GB，推荐16GB+
- **CPU**: 多核处理器（4核+）推荐用于并行计算

### 安装步骤

#### 1. 基础依赖安装

```bash
# 安装核心依赖
pip install SALib==1.4.7 seaborn==0.12.2 pandas==2.0.3 openpyxl

# 或者更新整个环境
pip install -r requirements.txt
```

#### 2. 验证安装

```bash
python -c "from polarization_triangle.analysis import SobolAnalyzer; print('✅ 安装成功')"
```

#### 3. 快速测试

```bash
python polarization_triangle/scripts/run_sobol_analysis.py --config quick
```

### 🆕 v1.1新增依赖

- **openpyxl**: Excel文件导出支持
- 优化的字体处理（无需额外中文字体包）

## 核心功能

### 参数敏感性分析

分析四个关键参数的影响：

| 参数 | 符号 | 作用机制 | 取值范围 | 预期影响 |
|------|------|----------|----------|----------|
| 自我激活系数 | α | 控制Agent观点坚持强度 | [0.1, 0.8] | 高值促进极化 |
| 社会影响系数 | β | 控制邻居影响强度 | [0.05, 0.3] | 高值促进收敛 |
| 道德化影响系数 | γ | 调节道德化抑制效应 | [0.2, 2.0] | 高值增强抵抗 |
| 身份凝聚力因子 | cohesion_factor | 增强网络连接强度 | [0.0, 0.5] | 高值促进传播 |

### 敏感性指数计算

- **一阶敏感性指数 (S1)**: 参数单独对输出方差的贡献
- **总敏感性指数 (ST)**: 参数及其所有交互项的总贡献
- **交互效应强度 (ST-S1)**: 参数间相互作用的强度

### 输出指标体系

涵盖极化动力学的多个维度：

1. **极化指标** (4个)
2. **收敛指标** (2个)  
3. **动态指标** (3个)
4. **身份指标** (2个)
5. **Variance Per Identity指标** (3个) - 新增

## 快速开始

### 方法1: 命令行使用（推荐）

```bash
# 快速测试 (1-3分钟)
python polarization_triangle/scripts/run_sobol_analysis.py --config quick

# 标准分析 (15-30分钟)
python polarization_triangle/scripts/run_sobol_analysis.py --config standard

# 高精度分析 (1-2小时)
python polarization_triangle/scripts/run_sobol_analysis.py --config high_precision

# 论文发表级 (4-8小时)
python polarization_triangle/scripts/run_sobol_analysis.py --config full
```

### 方法2: Python编程接口

```python
from polarization_triangle.analysis import SobolAnalyzer, SobolConfig

# 创建配置
config = SobolConfig(
    n_samples=200,         # 样本数
    n_runs=3,             # 重复运行次数
    n_processes=4,        # 并行进程数
    output_dir="my_analysis"
)

# 运行分析
analyzer = SobolAnalyzer(config)
sensitivity_indices = analyzer.run_complete_analysis()

# 查看结果摘要
summary = analyzer.get_summary_table()
print(summary.head(10))
```

### 方法3: 使用示例脚本

```bash
# 基础敏感性分析示例
python polarization_triangle/examples/sobol_sensitivity_example.py

# Variance Per Identity 专门示例 (新增)
python polarization_triangle/examples/variance_per_identity_example.py
```

## 详细配置

### 预设配置对比

| 配置名 | 样本数 | 运行次数 | 模拟步数 | 进程数 | 总模拟次数 | 适用场景 | 预计耗时 |
|--------|--------|----------|----------|--------|------------|----------|----------|
| `quick` | 50 | 2 | 100 | 2 | 1,000 | 快速测试验证 | 1-3分钟 |
| `standard` | 500 | 3 | 200 | 4 | 15,000 | 常规分析研究 | 15-30分钟 |
| `high_precision` | 1000 | 5 | 300 | 6 | 50,000 | 高精度研究 | 1-2小时 |
| `full` | 2000 | 10 | 500 | 8 | 200,000 | 论文发表级 | 4-8小时 |

### 自定义配置

```python
# 创建自定义基础模拟配置
from polarization_triangle.core.config import SimulationConfig

custom_base = SimulationConfig(
    num_agents=300,                    # Agent数量
    network_type='lfr',               # 网络类型
    morality_rate=0.6,                # 道德化率
    opinion_distribution='twin_peak',  # 初始意见分布
    network_params={
        'tau1': 3, 'tau2': 1.5, 'mu': 0.1,
        'average_degree': 6, 'min_community': 15
    }
)

# 创建敏感性分析配置
config = SobolConfig(
    base_config=custom_base,
    n_samples=800,
    n_runs=4,
    parameter_bounds={              # 自定义参数范围
        'alpha': [0.2, 0.7],
        'beta': [0.08, 0.25],
        'gamma': [0.5, 1.8],
        'cohesion_factor': [0.1, 0.4]
    },
    output_dir="custom_analysis"
)
```

### 高级配置选项

```python
config = SobolConfig(
    # 核心参数
    n_samples=1000,
    n_runs=5,
    n_processes=6,
    
    # 模拟控制
    num_steps=250,
    save_intermediate=True,        # 保存中间结果
    
    # 输出控制
    output_dir="advanced_analysis",
    export_raw_data=True,         # 导出原始数据
    
    # 质量控制
    validate_results=True,        # 结果验证
    confidence_level=0.95,        # 置信区间
    
    # 性能调优
    chunk_size=100,               # 批处理大小
    memory_limit=8,               # 内存限制(GB)
)
```

## 输出指标详解

### 极化相关指标

#### 1. polarization_index (Koudenburg极化指数)
- **计算方法**: 基于5类意见分布的标准化极化度量
- **取值范围**: [0, ∞)，0表示完全一致，值越大极化越严重
- **解释**: 反映系统整体的意见两极化程度

#### 2. opinion_variance (意见方差)
- **计算方法**: `np.var(opinions)`
- **取值范围**: [0, 1]
- **解释**: 直接度量意见分散程度，值越大分歧越大

#### 3. extreme_ratio (极端观点比例)
- **计算方法**: `|opinion| > 0.8` 的Agent比例
- **取值范围**: [0, 1]
- **解释**: 持有极端观点的Agent占比

#### 4. identity_polarization (身份间极化差异)
- **计算方法**: 不同身份群体间意见差异的方差
- **解释**: 反映身份认同对极化的影响

### 收敛相关指标

#### 5. mean_abs_opinion (平均绝对意见)
- **计算方法**: `np.mean(np.abs(opinions))`
- **取值范围**: [0, 1]
- **解释**: 系统整体的意见强度，反映"沉默螺旋"现象

#### 6. final_stability (最终稳定性)
- **计算方法**: 最后10%步数内的变异系数
- **解释**: 系统达到平衡状态的稳定程度

### 动态过程指标

#### 7. trajectory_length (意见轨迹长度)
- **计算方法**: 所有Agent意见变化距离的累积
- **解释**: 反映系统动态变化的复杂程度

#### 8. oscillation_frequency (振荡频率)
- **计算方法**: 意见方向改变次数的平均值
- **解释**: 衡量系统的动态不稳定性

#### 9. group_divergence (群体分化度)
- **计算方法**: 不同群体意见的KL散度
- **解释**: 量化群体间的意见分化程度

### 身份相关指标

#### 10. identity_variance_ratio (身份方差比)
- **计算方法**: 组间方差 / 组内方差
- **解释**: 身份认同对意见形成的影响强度

#### 11. cross_identity_correlation (跨身份相关性)
- **计算方法**: 不同身份群体意见的相关系数
- **解释**: 身份边界的渗透性

### Variance Per Identity 指标 (新增)

#### 12. variance_per_identity_1 (身份群体1方差)
- **计算方法**: `np.var(identity_1_opinions)` (排除zealot)
- **取值范围**: [0, 1]
- **解释**: identity=1群体内部的意见分化程度，高值表示群体内部意见分歧较大

#### 13. variance_per_identity_neg1 (身份群体-1方差)
- **计算方法**: `np.var(identity_neg1_opinions)` (排除zealot)
- **取值范围**: [0, 1]
- **解释**: identity=-1群体内部的意见分化程度，高值表示群体内部意见分歧较大

#### 14. variance_per_identity_mean (身份群体平均方差)
- **计算方法**: `(variance_identity_1 + variance_identity_neg1) / 2`
- **取值范围**: [0, 1]
- **解释**: 两个身份群体方差的均值，反映系统整体的群体内部分化水平

## 可视化功能

### 自动生成的图表类型

#### 1. 敏感性指数对比图
- **文件名**: `sensitivity_comparison_*.png`
- **内容**: S1和ST的条形图对比
- **用途**: 直观比较参数的主效应和总效应

#### 2. 敏感性热力图
- **文件名**: `sensitivity_heatmap_*.png`
- **内容**: 参数-指标敏感性矩阵
- **用途**: 识别参数-指标的敏感性模式

#### 3. 交互效应分析图
- **文件名**: `interaction_effects.png`
- **内容**: ST-S1的可视化展示
- **用途**: 识别强交互效应的参数

#### 4. 参数重要性排序图
- **文件名**: `parameter_ranking_*.png`
- **内容**: 按敏感性排序的参数重要性
- **用途**: 为参数调优提供优先级指导

### 可视化自定义

```python
from polarization_triangle.analysis import SensitivityVisualizer

visualizer = SensitivityVisualizer()

# 创建单个图表
fig = visualizer.plot_sensitivity_comparison(
    sensitivity_indices, 
    'polarization_index',
    save_path='custom_comparison.png',
    figsize=(12, 8),
    dpi=300
)

# 批量生成报告
plot_files = visualizer.create_comprehensive_report(
    sensitivity_indices,
    param_samples,
    simulation_results,
    output_dir='custom_plots'
)
```

## 高级用法

### 批量场景分析

```python
scenarios = {
    'high_morality': {'morality_rate': 0.8},
    'low_morality': {'morality_rate': 0.2},
    'large_network': {'num_agents': 500},
    'dense_network': {'network_params': {'average_degree': 8}}
}

results = {}
for scenario_name, params in scenarios.items():
    base_config = SimulationConfig(**params)
    config = SobolConfig(
        base_config=base_config,
        n_samples=300,
        output_dir=f"scenario_{scenario_name}"
    )
    
    analyzer = SobolAnalyzer(config)
    results[scenario_name] = analyzer.run_complete_analysis()

# 比较不同场景的敏感性模式
compare_scenarios(results)
```

### 自定义指标

```python
from polarization_triangle.analysis import SensitivityMetrics

class ExtendedMetrics(SensitivityMetrics):
    def __init__(self):
        super().__init__()
        self.metric_names.extend(['custom_metric1', 'custom_metric2'])
    
    def _calculate_custom_metric1(self, sim):
        """自定义指标1: 意见集中度"""
        return 1 - np.std(sim.opinions)
    
    def _calculate_custom_metric2(self, sim):
        """自定义指标2: 网络极化度"""
        return calculate_network_polarization(sim)
    
    def calculate_all_metrics(self, sim):
        metrics = super().calculate_all_metrics(sim)
        metrics['custom_metric1'] = self._calculate_custom_metric1(sim)
        metrics['custom_metric2'] = self._calculate_custom_metric2(sim)
        return metrics

# 使用自定义指标
config.custom_metrics_class = ExtendedMetrics
```

### 结果后处理与分析

```python
# 加载已有结果
analyzer = SobolAnalyzer(config)
sensitivity_indices = analyzer.load_results("existing_results")

# 统计分析
import numpy as np

# 识别高敏感性参数
high_sensitivity_params = []
for output_name, indices in sensitivity_indices.items():
    for i, param in enumerate(['alpha', 'beta', 'gamma', 'cohesion_factor']):
        if indices['ST'][i] > 0.2:  # 敏感性阈值
            high_sensitivity_params.append((param, output_name, indices['ST'][i]))

# 参数重要性聚类分析
from sklearn.cluster import KMeans

# 提取敏感性矩阵
sensitivity_matrix = np.array([indices['ST'] for indices in sensitivity_indices.values()])

# 聚类分析找出相似的敏感性模式
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(sensitivity_matrix.T)  # 对参数聚类

print("参数聚类结果:")
param_names = ['alpha', 'beta', 'gamma', 'cohesion_factor']
for i, cluster in enumerate(clusters):
    print(f"{param_names[i]}: 聚类{cluster}")
```

## 性能优化

### 计算资源优化

```python
# 根据系统配置自动调整
import multiprocessing as mp
import psutil

def optimize_config(base_config: SobolConfig) -> SobolConfig:
    """根据系统资源自动优化配置"""
    
    # CPU优化
    cpu_count = mp.cpu_count()
    base_config.n_processes = min(cpu_count - 1, base_config.n_processes)
    
    # 内存优化
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    if available_memory < 8:
        base_config.n_processes = max(2, base_config.n_processes // 2)
        base_config.chunk_size = 50
    
    # 批次大小优化
    total_samples = base_config.n_samples * 10  # Saltelli采样
    if total_samples > 10000:
        base_config.save_intermediate = True
    
    return base_config

# 应用优化
config = optimize_config(config)
```

### 内存管理

```python
# 大规模分析的内存优化策略
config = SobolConfig(
    n_samples=5000,
    chunk_size=200,           # 分批处理
    save_intermediate=True,   # 保存中间结果
    clear_cache=True,         # 清理缓存
    memory_limit=12,          # 内存限制
)

# 手动内存管理
import gc

def memory_efficient_analysis(config):
    analyzer = SobolAnalyzer(config)
    
    # 分阶段执行
    param_samples = analyzer.generate_samples()
    gc.collect()  # 强制垃圾回收
    
    simulation_results = analyzer.run_simulations_batch(param_samples)
    del param_samples
    gc.collect()
    
    sensitivity_indices = analyzer.calculate_sensitivity(simulation_results)
    del simulation_results
    gc.collect()
    
    return sensitivity_indices
```

### 分布式计算

```python
# 为超大规模分析准备的分布式方案
from concurrent.futures import ProcessPoolExecutor, as_completed

def distributed_analysis(config, n_workers=None):
    """分布式敏感性分析"""
    
    if n_workers is None:
        n_workers = min(8, mp.cpu_count())
    
    # 将样本分割为多个子任务
    total_samples = config.n_samples
    samples_per_worker = total_samples // n_workers
    
    tasks = []
    for i in range(n_workers):
        worker_config = config.copy()
        worker_config.n_samples = samples_per_worker
        worker_config.output_dir = f"{config.output_dir}_worker_{i}"
        tasks.append(worker_config)
    
    # 并行执行
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_worker_analysis, task) for task in tasks]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    # 合并结果
    return merge_sensitivity_results(results)

def run_worker_analysis(config):
    """单个工作进程的分析任务"""
    analyzer = SobolAnalyzer(config)
    return analyzer.run_complete_analysis()
```

## 故障排除

### 常见问题及解决方案

#### ✅ 已修复的问题

##### 1. 模拟运行错误
**症状**: `'Simulation' object has no attribute 'run'`
**解决方案**: 已修复，现使用 `sim.step()` 循环

##### 2. Excel导出失败
**症状**: `No module named 'openpyxl'`
**解决方案**: 已添加到依赖，运行 `pip install openpyxl`

##### 3. 图形界面卡住
**症状**: 程序在可视化阶段长时间无响应
**解决方案**: 已移除 `plt.show()`，只保存图片文件

##### 4. 中文字体警告
**症状**: 大量字体警告信息
**解决方案**: 图表标签改为英文，提高兼容性

#### 🔧 当前可能的问题

##### 5. 内存不足
**症状**: `MemoryError` 或系统内存耗尽
**解决方案**:
```python
# 减少资源消耗
config = SobolConfig(
    n_samples=200,        # 减少样本数
    n_processes=2,        # 减少并行进程
    chunk_size=50,        # 分批处理
    save_intermediate=True
)

# 基础配置优化
base_config = SimulationConfig(
    num_agents=150,       # 减少Agent数量
    num_steps=150         # 减少模拟步数
)
```

##### 6. 计算时间过长
**症状**: 分析时间远超预期
**诊断与解决**:
```python
# 时间估算函数
def estimate_time(config):
    """估算计算时间"""
    total_sims = config.n_samples * 10 * config.n_runs
    time_per_sim = 2  # 秒，根据系统调整
    estimated_hours = (total_sims * time_per_sim) / (3600 * config.n_processes)
    
    print(f"预计运行时间: {estimated_hours:.1f} 小时")
    print(f"总模拟次数: {total_sims:,}")
    print(f"并行进程数: {config.n_processes}")
    
    return estimated_hours

# 使用前评估
estimate_time(config)

# 如果时间过长，使用渐进式分析
progressive_configs = [
    SobolConfig(n_samples=50, output_dir="test_50"),
    SobolConfig(n_samples=200, output_dir="test_200"),
    SobolConfig(n_samples=500, output_dir="test_500")
]

for config in progressive_configs:
    if estimate_time(config) < 1:  # 小于1小时
        analyzer = SobolAnalyzer(config)
        results = analyzer.run_complete_analysis()
        # 检查结果稳定性再决定是否继续
```

##### 7. 敏感性结果异常
**症状**: ST < S1, 全零值, 或异常高值
**诊断步骤**:
```python
def diagnose_results(sensitivity_indices):
    """诊断敏感性结果"""
    issues = []
    
    for output_name, indices in sensitivity_indices.items():
        s1_values = np.array(indices['S1'])
        st_values = np.array(indices['ST'])
        
        # 检查ST >= S1关系
        if np.any(st_values < s1_values - 0.05):  # 允许小误差
            issues.append(f"{output_name}: ST < S1")
        
        # 检查零值
        if np.all(s1_values < 0.01) and np.all(st_values < 0.01):
            issues.append(f"{output_name}: 所有敏感性接近零")
        
        # 检查异常值
        if np.any(st_values > 2):
            issues.append(f"{output_name}: 异常高敏感性值")
    
    if issues:
        print("发现的问题:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\n建议解决方案:")
        print("  1. 增加样本数量")
        print("  2. 检查参数范围设置")
        print("  3. 验证基础模拟配置")
        print("  4. 增加运行次数减少随机性")
    else:
        print("✅ 敏感性结果通过基础检查")

# 使用诊断工具
diagnose_results(sensitivity_indices)
```

### 调试模式

```python
# 详细调试配置
debug_config = SobolConfig(
    n_samples=10,             # 最小样本
    n_runs=1,                 # 单次运行
    num_steps=20,             # 短模拟
    n_processes=1,            # 单进程便于调试
    output_dir="debug_test",
    verbose=True              # 详细输出
)

# 启用详细日志
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 逐步调试
try:
    print("1. 创建分析器...")
    analyzer = SobolAnalyzer(debug_config)
    
    print("2. 生成样本...")
    param_samples = analyzer.generate_samples()
    print(f"   样本形状: {param_samples.shape}")
    
    print("3. 运行单个模拟测试...")
    test_params = {
        'alpha': 0.4, 'beta': 0.12, 
        'gamma': 1.0, 'cohesion_factor': 0.2
    }
    test_result = analyzer.run_single_simulation(test_params)
    print(f"   测试结果: {test_result}")
    
    print("4. 运行完整分析...")
    results = analyzer.run_complete_analysis()
    print("✅ 调试成功")
    
except Exception as e:
    print(f"❌ 调试失败: {e}")
    import traceback
    traceback.print_exc()
```

## 最佳实践

### 分析流程建议

#### 1. 准备阶段
```python
# 环境检查
def check_environment():
    """检查运行环境"""
    import sys, psutil, multiprocessing as mp
    
    print(f"Python版本: {sys.version}")
    print(f"可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"CPU核心数: {mp.cpu_count()}")
    
    # 检查关键依赖
    try:
        import SALib, seaborn, pandas, openpyxl
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")

check_environment()
```

#### 2. 探索性分析
```python
# 第一步：快速探索
quick_config = SobolConfig(
    n_samples=50,
    n_runs=2,
    output_dir="exploration"
)

analyzer = SobolAnalyzer(quick_config)
initial_results = analyzer.run_complete_analysis()

# 查看结果模式
print_key_findings(initial_results)
```

#### 3. 深入分析
```python
# 基于初步结果设计深入分析
if looks_promising(initial_results):
    standard_config = SobolConfig(
        n_samples=500,
        n_runs=5,
        output_dir="detailed_analysis"
    )
    
    analyzer = SobolAnalyzer(standard_config)
    detailed_results = analyzer.run_complete_analysis()
```

#### 4. 验证与发布
```python
# 高精度验证
final_config = SobolConfig(
    n_samples=1000,
    n_runs=10,
    output_dir="final_results"
)

final_analyzer = SobolAnalyzer(final_config)
final_results = final_analyzer.run_complete_analysis()

# 生成完整报告
final_analyzer.export_results(include_confidence_intervals=True)
```

### 参数设置指南

#### 样本数选择
- **n_samples = 50**: 快速测试和调试
- **n_samples = 200-500**: 一般研究和分析
- **n_samples = 1000**: 高精度研究
- **n_samples = 2000+**: 论文发表级分析

#### 运行次数选择
- **n_runs = 2**: 调试和快速测试
- **n_runs = 3-5**: 标准分析
- **n_runs = 10+**: 高精度分析，减少随机性

#### 进程数优化
```python
import multiprocessing as mp

# 保守策略：留一个核心给系统
n_processes = max(1, mp.cpu_count() - 1)

# 内存受限策略
available_memory_gb = psutil.virtual_memory().available / (1024**3)
if available_memory_gb < 8:
    n_processes = min(n_processes, 2)
elif available_memory_gb < 16:
    n_processes = min(n_processes, 4)

config.n_processes = n_processes
```

### 结果解释指南

#### 敏感性等级划分
```python
def interpret_sensitivity(st_value):
    """解释敏感性等级"""
    if st_value > 0.3:
        return "极高敏感性 - 关键参数"
    elif st_value > 0.15:
        return "高敏感性 - 重要参数"
    elif st_value > 0.05:
        return "中等敏感性 - 次要参数"
    else:
        return "低敏感性 - 可忽略参数"

# 应用到结果
for output_name, indices in sensitivity_indices.items():
    print(f"\n{output_name}:")
    param_names = ['α', 'β', 'γ', 'cohesion_factor']
    for i, (param, st) in enumerate(zip(param_names, indices['ST'])):
        level = interpret_sensitivity(st)
        print(f"  {param}: {st:.3f} - {level}")
```

#### 交互效应分析
```python
def analyze_interactions(sensitivity_indices):
    """分析参数交互效应"""
    param_names = ['α', 'β', 'γ', 'cohesion_factor']
    
    interaction_matrix = []
    for output_name, indices in sensitivity_indices.items():
        interactions = np.array(indices['ST']) - np.array(indices['S1'])
        interaction_matrix.append(interactions)
    
    mean_interactions = np.mean(interaction_matrix, axis=0)
    
    print("平均交互效应强度:")
    for param, interaction in zip(param_names, mean_interactions):
        if interaction > 0.1:
            level = "强"
        elif interaction > 0.05:
            level = "中等"
        else:
            level = "弱"
        print(f"  {param}: {interaction:.3f} ({level})")

analyze_interactions(sensitivity_indices)
```

### 质量保证

#### 结果验证
```python
def validate_analysis_quality(analyzer, sensitivity_indices):
    """验证分析质量"""
    checks = []
    
    # 1. 样本充足性检查
    if analyzer.config.n_samples < 100:
        checks.append("警告: 样本数量可能不足")
    
    # 2. 收敛性检查
    # （需要多次运行比较，这里简化）
    
    # 3. 一致性检查
    for output_name, indices in sensitivity_indices.items():
        s1_sum = sum(indices['S1'])
        if s1_sum > 1.2:  # 允许一些误差
            checks.append(f"警告: {output_name}的S1总和过高 ({s1_sum:.2f})")
    
    # 4. 置信区间检查
    if hasattr(analyzer, 'confidence_intervals'):
        # 检查置信区间宽度
        pass
    
    if checks:
        print("质量检查发现的问题:")
        for check in checks:
            print(f"  - {check}")
    else:
        print("✅ 分析质量检查通过")

validate_analysis_quality(analyzer, sensitivity_indices)
```

## API参考

### SobolConfig类

```python
class SobolConfig:
    """Sobol敏感性分析配置类"""
    
    def __init__(self,
                 parameter_bounds: Dict[str, List[float]] = None,
                 n_samples: int = 1000,
                 n_runs: int = 5,
                 n_processes: int = 4,
                 num_steps: int = 200,
                 output_dir: str = "sobol_results",
                 save_intermediate: bool = True,
                 base_config: SimulationConfig = None):
        """
        参数:
            parameter_bounds: 参数取值范围字典
            n_samples: Saltelli采样的基础样本数
            n_runs: 每个参数组合的重复运行次数
            n_processes: 并行进程数
            num_steps: 每次模拟的步数
            output_dir: 结果输出目录
            save_intermediate: 是否保存中间结果
            base_config: 基础模拟配置
        """
```

### SobolAnalyzer类

```python
class SobolAnalyzer:
    """Sobol敏感性分析器"""
    
    def __init__(self, config: SobolConfig):
        """初始化分析器"""
    
    def generate_samples(self) -> np.ndarray:
        """生成Saltelli样本"""
    
    def run_single_simulation(self, params: Dict[str, float]) -> Dict[str, float]:
        """运行单次模拟"""
    
    def run_simulations(self, param_samples: np.ndarray) -> List[Dict[str, float]]:
        """并行运行多次模拟"""
    
    def calculate_sensitivity(self, simulation_results: List[Dict[str, float]]) -> Dict[str, Dict]:
        """计算敏感性指数"""
    
    def run_complete_analysis(self) -> Dict[str, Dict]:
        """运行完整分析流程"""
    
    def get_summary_table(self) -> pd.DataFrame:
        """获取结果摘要表"""
    
    def export_results(self, filename: str = None) -> str:
        """导出Excel结果"""
    
    def save_results(self, filename: str = None) -> str:
        """保存结果到pickle文件"""
    
    def load_results(self, filename: str = None) -> Dict[str, Dict]:
        """加载已保存的结果"""
```

### SensitivityVisualizer类

```python
class SensitivityVisualizer:
    """敏感性分析可视化器"""
    
    def plot_sensitivity_comparison(self, 
                                  sensitivity_indices: Dict,
                                  output_name: str,
                                  save_path: str = None) -> plt.Figure:
        """绘制敏感性指数对比图"""
    
    def plot_sensitivity_heatmap(self,
                               sensitivity_indices: Dict,
                               metric_type: str = 'ST',
                               save_path: str = None) -> plt.Figure:
        """绘制敏感性热力图"""
    
    def plot_interaction_effects(self,
                               sensitivity_indices: Dict,
                               save_path: str = None) -> plt.Figure:
        """绘制交互效应图"""
    
    def plot_parameter_ranking(self,
                             sensitivity_indices: Dict,
                             metric_type: str = 'ST',
                             save_path: str = None) -> plt.Figure:
        """绘制参数重要性排序图"""
    
    def create_comprehensive_report(self,
                                  sensitivity_indices: Dict,
                                  param_samples: np.ndarray,
                                  simulation_results: List,
                                  output_dir: str) -> List[str]:
        """创建完整的可视化报告"""
```

## 更新日志

### v1.1.0 (2024年当前) - 🚀 稳定性和兼容性大幅提升

#### 🔧 核心修复
- ✅ **模拟运行修复**: 修复了`'Simulation' object has no attribute 'run'`错误，改用 `sim.step()` 循环
- ✅ **Excel导出支持**: 添加openpyxl依赖，完全支持Excel结果导出
- ✅ **图形界面优化**: 移除所有 `plt.show()` 调用，避免程序卡住，只保存图片文件
- ✅ **字体兼容性**: 优化中文字体处理，图表标签改为英文，提高跨平台兼容性

#### 📊 功能增强
- 🆕 **新增指标**: 添加3个 Variance Per Identity 指标，分析不同身份群体内部的意见分化
  - `variance_per_identity_1`: identity=1群体的意见方差
  - `variance_per_identity_neg1`: identity=-1群体的意见方差
  - `variance_per_identity_mean`: 两个群体方差的均值
- 🆕 **预设配置更新**: 更准确的时间估算和资源配置
- 🆕 **错误诊断工具**: 新增结果质量验证和问题诊断功能
- 🆕 **性能监控**: 添加进度条和详细的执行时间报告
- 🆕 **内存优化**: 改进大规模分析的内存管理

#### 🎨 可视化改进
- 🎨 **高质量图表**: 默认300 DPI输出，适合论文发表
- 🎨 **英文标签**: 所有图表标签改为英文，提高国际化兼容性
- 🎨 **自动布局**: 改进图表布局和颜色方案

#### 📚 文档完善
- 📖 **完整示例**: 添加真实运行输出示例和新指标专门示例
- 📖 **故障排除**: 详细的问题诊断和解决方案
- 📖 **最佳实践**: 全面的使用建议和优化指南
- 📖 **新指标文档**: 详细说明 Variance Per Identity 指标的含义和应用

### v1.0.0 (初始版本)
- 🎯 基础Sobol敏感性分析功能
- 📊 11种核心输出指标
- ⚡ 并行计算支持
- 🎨 基础可视化功能
- 🔧 4种预设配置

---

## 联系与支持

如果您在使用过程中遇到问题或有改进建议，请：

1. 📖 首先查阅本文档的故障排除部分
2. 🧪 使用调试模式进行问题诊断  
3. 💬 查看项目的Issue页面
4. 🚀 提交详细的Bug报告或功能请求

---

**文档版本**: v1.1.0  
**最后更新**: 2024年12月  
**兼容性**: Python 3.8+, 极化三角框架 v1.1+ 