# Zealot功能集成指南

## 概述

Zealot功能现已完全集成到Polarization Triangle Framework的核心代码中。您可以通过配置参数轻松启用和管理zealots，而无需编写额外的代码。

## 主要特性

1. **完全集成**: Zealot功能现在是`Simulation`类的内置功能
2. **配置驱动**: 通过`SimulationConfig`轻松配置zealot参数
3. **多种模式**: 支持random、degree和clustered zealot选择模式
4. **自动管理**: Zealot意见在每个仿真步骤后自动重置
5. **向后兼容**: 所有现有的实验代码都可以正常工作
6. **正确缩放**: 修复了zealot意见被初始缩放因子错误影响的问题 ✅

## 配置参数

在`SimulationConfig`中添加了以下zealot相关参数：

```python
zealot_count: int = 0           # zealot的数量，0表示不使用zealot
zealot_mode: str = "random"     # 选择模式：random, clustered, degree
zealot_opinion: float = 1.0     # zealot固定的意见值
enable_zealots: bool = False    # 是否启用zealot功能
zealot_morality: bool = False   # zealot是否全部设置为moralizing (morality=1)
```

## 使用方法

### 基本使用

```python
from polarization_triangle.core.config import SimulationConfig
from polarization_triangle.core.simulation import Simulation

# 创建带有zealot的配置
config = SimulationConfig(
    num_agents=100,
    zealot_count=10,           # 10个zealots
    zealot_mode='random',      # 随机选择
    enable_zealots=True,       # 启用zealot功能
    zealot_opinion=1.0,        # zealot意见固定为1.0
    zealot_morality=True       # zealots都是moralizing
)

# 创建仿真
sim = Simulation(config)

# 获取zealot信息
print(f"Zealot IDs: {sim.get_zealot_ids()}")

# 运行仿真
for step in range(100):
    sim.step()  # zealot意见会自动重置
```

### Zealot选择模式

#### 1. Random模式
```python
config = SimulationConfig(
    zealot_count=10,
    zealot_mode='random'  # 随机选择10个agents作为zealots
)
```

#### 2. Degree模式
```python
config = SimulationConfig(
    zealot_count=10,
    zealot_mode='degree'  # 选择度数最高的10个nodes作为zealots
)
```

#### 3. Clustered模式
```python
config = SimulationConfig(
    zealot_count=10,
    zealot_mode='clustered',   # 尽量在同一社区内选择zealots
    network_type='lfr'         # 需要具有社区结构的网络
)
```

### 动态管理Zealots

```python
# 手动添加zealots
sim.add_zealots([1, 2, 3], opinion=0.8)

# 移除特定zealots
sim.remove_zealots([1, 2])

# 移除所有zealots
sim.remove_zealots()

# 获取当前zealot列表
current_zealots = sim.get_zealot_ids()
```

## 实验代码

所有现有的实验代码都可以正常工作：

### 单次实验
```python
from polarization_triangle.experiments.zealot_experiment import run_zealot_experiment

result = run_zealot_experiment(
    steps=500,
    num_zealots=50,
    zealot_mode="clustered",
    morality_rate=0.2,
    zealot_morality=True
)
```

### 多次运行实验
```python
from polarization_triangle.experiments.multi_zealot_experiment import run_multi_zealot_experiment

avg_stats = run_multi_zealot_experiment(
    runs=10,
    steps=500,
    zealot_count=20,
    zealot_mode="random"
)
```

### 参数扫描实验
```python
from polarization_triangle.experiments.zealot_parameter_sweep import run_parameter_sweep

run_parameter_sweep(
    runs_per_config=5,
    steps=200
)
```

## 技术细节

### 实现原理

1. **初始化**: 在`Simulation.__init__()`中根据配置选择zealots
2. **意见固定**: 在`Simulation.step()`中自动调用`set_zealot_opinions()`重置zealot意见
3. **缩放处理**: 在`zealot_experiment.py`中，意见缩放后会自动重新设置zealot意见
4. **统计兼容**: 所有统计函数都会正确处理zealots（如计算非zealot方差）

### 性能优化

- Zealot选择在初始化时一次性完成
- 意见重置使用高效的数组操作
- 与现有的numba加速动力学计算兼容

### 🔧 重要修复

**问题**: 在之前版本中，zealot的意见会被`initial_scale`错误缩放，导致zealot影响力减弱。

**解决方案**: 
1. 在意见缩放后立即重新设置zealot意见
2. 在每个simulation step后自动重置zealot意见
3. 确保zealot意见始终保持为配置中指定的值

```python
# 修复前的问题代码：
sim.opinions *= initial_scale  # zealot意见也被缩放了！

# 修复后的正确代码：
sim.opinions *= initial_scale
sim.set_zealot_opinions()  # 重新设置zealot意见，避免被缩放
```

## 迁移指南

如果您之前使用了外部zealot函数，现在可以简化代码：

### 之前的代码
```python
# 旧方式
sim = Simulation(base_config)
zealot_ids = initialize_zealots(sim, 50, "random")
for step in range(steps):
    set_zealot_opinions(sim, zealot_ids)
    sim.step()
```

### 现在的代码
```python
# 新方式
config = SimulationConfig(zealot_count=50, zealot_mode="random", enable_zealots=True)
sim = Simulation(config)
for step in range(steps):
    sim.step()  # zealot意见自动重置
```

## 注意事项

1. **网络兼容性**: clustered模式需要具有社区结构的网络（如LFR网络）
2. **参数验证**: 系统会自动验证zealot_count不超过总agent数量
3. **意见范围**: zealot_opinion应在[-1, 1]范围内
4. **道德化**: 如果zealot_morality=True，所有zealots的morality会设置为1
5. **缩放兼容**: 系统会自动处理意见缩放对zealot的影响 ✅

## 测试验证

### 基本功能测试
```python
from polarization_triangle.core.config import SimulationConfig
from polarization_triangle.core.simulation import Simulation

config = SimulationConfig(
    num_agents=50,
    zealot_count=5,
    enable_zealots=True,
    network_type="random",
    network_params={"p": 0.1}
)

sim = Simulation(config)
print(f"Zealots: {sim.get_zealot_ids()}")

# 运行几步验证zealot意见固定
for i in range(3):
    sim.step()
    zealot_opinions = [sim.opinions[zid] for zid in sim.get_zealot_ids()]
    print(f"Step {i+1}: {zealot_opinions}")
```

### 缩放修复验证
```python
# 测试缩放修复
sim = Simulation(config)
initial_scale = 0.1

# 应用缩放
sim.opinions *= initial_scale
sim.set_zealot_opinions()  # 重新设置zealot意见

# 验证zealot意见是否正确
zealot_opinions = [sim.opinions[zid] for zid in sim.get_zealot_ids()]
assert all(abs(op - 1.0) < 1e-10 for op in zealot_opinions), "Zealot意见未正确重置"
print("✅ 缩放修复验证通过")
```

所有zealot意见应该保持为配置中指定的值（默认1.0），不受初始缩放因子影响。 