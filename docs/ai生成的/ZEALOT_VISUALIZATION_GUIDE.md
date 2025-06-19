# Zealot网络可视化指南

## 概述

`network_viz.py`中的`draw_network`函数现已支持zealot节点的特殊可视化标识。Zealot节点会以**金色边框**突出显示，并且比普通节点稍大，使其在网络图中易于识别。

## 新增功能特性

### 1. Zealot节点标识
- **金色边框**: 所有zealot节点都有3像素宽的金色边框
- **较大尺寸**: Zealot节点大小为40像素，普通节点为20像素
- **自动检测**: 函数会自动检测仿真中是否存在zealot

### 2. 图例增强
- 在所有可视化模式中都会显示zealot数量
- 图例中包含zealot的特殊标识说明
- 支持多种信息的组合显示

### 3. 兼容性
- 向后兼容：没有zealot时功能正常
- 支持所有现有的可视化模式：opinion、identity、morality
- 与所有zealot选择模式兼容：random、degree、clustered

## 使用方法

### 基本使用
```python
from polarization_triangle.core.config import SimulationConfig
from polarization_triangle.core.simulation import Simulation
from polarization_triangle.visualization.network_viz import draw_network

# 创建带有zealot的仿真
config = SimulationConfig(
    num_agents=50,
    zealot_count=8,
    zealot_mode='random',
    enable_zealots=True,
    zealot_opinion=1.0,
    network_type="random",
    network_params={"p": 0.15}
)

sim = Simulation(config)

# 绘制不同类型的网络图
draw_network(sim, "opinion", "Opinion Network with Zealots", "opinion_zealots.png")
draw_network(sim, "identity", "Identity Network with Zealots", "identity_zealots.png")
draw_network(sim, "morality", "Morality Network with Zealots", "morality_zealots.png")
```

### 在实验中的使用

Zealot可视化功能已自动集成到所有实验代码中：

```python
from polarization_triangle.experiments.zealot_experiment import run_zealot_experiment

# 运行实验，生成的所有网络图都会自动标识zealot
result = run_zealot_experiment(
    steps=500,
    num_zealots=20,
    zealot_mode="clustered",
    output_dir="my_experiment"
)
```

生成的网络图文件中，zealot节点会自动显示金色边框。

## 可视化效果说明

### Opinion模式
- **节点颜色**: 基于意见值的冷暖色谱（蓝色=负面，红色=正面）
- **Zealot标识**: 金色边框 + 较大尺寸
- **图例**: 包含意见色谱 + zealot数量信息
- **位置**: Zealot图例显示在左上角

### Identity模式
- **节点颜色**: 红色(identity=1) vs 蓝色(identity=-1)
- **Zealot标识**: 金色边框 + 较大尺寸
- **图例**: 包含身份分类 + zealot数量信息
- **位置**: 组合图例显示在右上角

### Morality模式
- **节点颜色**: 绿色(morality=1) vs 红色(morality=0)
- **Zealot标识**: 金色边框 + 较大尺寸
- **图例**: 包含道德化分类 + zealot数量信息
- **位置**: 组合图例显示在右上角

## 技术实现细节

### 实现原理
1. **自动检测**: 通过`sim.get_zealot_ids()`获取zealot列表
2. **分层绘制**: 先绘制所有节点，再为zealot添加边框
3. **尺寸差异**: 动态创建节点大小数组
4. **边框效果**: 使用`nx.draw_networkx_nodes`的`edgecolors`参数

### 性能优化
- 只在存在zealot时进行额外绘制
- 使用高效的NetworkX绘图函数
- 边框绘制不影响主要节点颜色

### 代码结构
```python
# 获取zealot信息
zealot_ids = sim.get_zealot_ids() if hasattr(sim, 'get_zealot_ids') else []
has_zealots = len(zealot_ids) > 0

# 创建节点大小数组
node_sizes = []
for i in range(sim.num_agents):
    if i in zealot_ids:
        node_sizes.append(zealot_node_size)  # 40像素
    else:
        node_sizes.append(base_node_size)    # 20像素

# 各模式绘制主要节点（opinion、identity、morality）
nx.draw(sim.graph, pos=sim.pos, node_color=node_colors,
        node_size=node_sizes, ...)

# 统一处理zealot边框绘制（所有模式共用）
if has_zealots:
    zealot_pos = {i: sim.pos[i] for i in zealot_ids}
    nx.draw_networkx_nodes(sim.graph, pos=zealot_pos, nodelist=zealot_ids,
                         node_color='none', 
                         edgecolors='gold', 
                         linewidths=3)

# 统一处理图例（所有模式共用）
if has_zealots:
    zealot_patch = Patch(facecolor='none', edgecolor='gold', linewidth=3, 
                       label=f'Zealots (n={len(zealot_ids)})')
    
    if mode_patches:  # identity和morality模式
        all_patches = mode_patches + [zealot_patch]
    else:  # opinion模式
        all_patches = [zealot_patch]
    
    ax.legend(handles=all_patches, loc=legend_loc, title=legend_title)
```

### 代码优化特点

1. **消除重复**: zealot边框绘制和图例处理代码从三个模式中提取为公共逻辑
2. **统一处理**: 所有zealot相关功能在模式处理完成后统一执行
3. **可维护性**: 修改zealot可视化效果只需在一个地方更改
4. **逻辑清晰**: 模式特定逻辑与zealot逻辑分离，代码结构更清晰

## 示例输出

生成的网络图将包含：
- 清晰的节点分类（基于选择的模式）
- 金色边框标识的zealot节点
- 详细的图例说明
- 高质量的PNG输出（300 DPI）

## 注意事项

1. **兼容性**: 需要仿真对象实现`get_zealot_ids()`方法
2. **性能**: 对于大型网络（>1000节点），可视化可能较慢
3. **颜色**: 金色边框在所有背景下都清晰可见
4. **文件格式**: 输出为PNG格式，支持透明背景
5. **图例位置**: 会根据内容自动调整，避免遮挡网络结构

## 自定义选项

如果需要自定义zealot的可视化效果，可以修改以下参数：

```python
# 在 network_viz.py 中可以调整的参数：
base_node_size = 20        # 普通节点大小
zealot_node_size = 40      # zealot节点大小
edgecolors = 'gold'        # 边框颜色
linewidths = 3             # 边框宽度
```

## 故障排除

### 常见问题
1. **没有显示边框**: 检查仿真是否启用了zealot（`enable_zealots=True`）
2. **图例重叠**: 尝试调整图例位置参数
3. **节点太小**: 增加`figsize`参数或调整节点大小

### 调试技巧
```python
# 检查zealot是否正确配置
print(f"Zealot IDs: {sim.get_zealot_ids()}")
print(f"Zealot count: {len(sim.get_zealot_ids())}")
print(f"Enable zealots: {sim.enable_zealots}")
```

这个增强的可视化功能使得在网络分析中能够直观地识别和追踪zealot节点的影响，为理解极化动力学提供重要的视觉线索。 