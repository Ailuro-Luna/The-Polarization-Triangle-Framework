# Zealot-Morality Analysis 图表分析文档

## 概述

本文档详细介绍了`zealot_morality_analysis.py`实验中生成的图表含义，以及Polarization Triangle Framework模型中的核心概念。该实验主要分析zealot数量和morality比例对系统各种指标的影响，通过多维度的可视化图表揭示社会极化过程中的复杂动态关系。

## 核心概念说明

### 1. Opinion (意见)

**定义**: Opinion ($z_i$) 表示agent $i$ 在某个特定议题上的观点立场，取值范围为 $[-1, 1]$。

**含义**:
- `+1`: 强烈支持某个观点
- `0`: 中性态度或无明确立场
- `-1`: 强烈反对某个观点

**动态演化**: Opinion通过以下动力学方程演化：
$$\dot{z_i} = -\delta_i z_i + u_i \cdot \tanh\left(\alpha_i \sigma_{ii} + \frac{\beta}{1 + \gamma_i m_i} \sum_{j \neq i} A_{ij} \sigma_{ij}\right)$$

其中包含：
- **自然衰减项** ($-\delta_i z_i$): 意见向中性状态回归的趋势
- **激活项**: 包括自我激活和社会影响两个组成部分

### 2. Identity (身份认同)

**定义**: Identity ($l_i$) 表示agent $i$ 的社会身份认同，取值为 `±1`。

**含义**:
- `+1`: 归属于身份群体A (例如：政治倾向、社会阶层、文化背景等)
- `-1`: 归属于身份群体B

**作用机制**:
- **群体内聚性**: 相同身份的agents倾向于相互影响和支持
- **群体间对立**: 不同身份的agents之间可能产生冲突或抵制
- **关系系数影响**: Identity直接影响agents之间的关系强度 $A_{ij}$

### 3. Morality (道德化)

**定义**: Morality ($m_i$) 表示agent $i$ 对当前议题的道德化程度，取值为 `0` 或 `1`。

**含义**:
- `1`: 该agent将议题视为道德问题，持有强烈的价值判断
- `0`: 该agent将议题视为实用问题，更注重事实和效果

**影响机制**:
1. **感知意见计算**: 影响agent如何感知他人的观点
   $$\sigma_{ij} = \begin{cases}
   \frac{z_j}{|z_j|}, & \text{如果 } z_j \neq 0 \text{ 且 } (m_i = 1 \text{ 或 } m_j = 1) \\
   0, & \text{如果 } z_j = 0 \\
   z_j, & \text{如果 } m_i = 0 \text{ 且 } m_j = 0
   \end{cases}$$

2. **社会影响强度**: 道德化agent的影响力会被调节：$\frac{\beta}{1 + \gamma_i m_i}$

### 4. Zealot (狂热分子)

**定义**: Zealot是具有固定、不变观点的特殊agents，其opinion值在整个模拟过程中保持恒定。

**特征**:
- **观点固定**: Zealot的opinion不会因社会影响而改变
- **影响他人**: Zealot仍然可以影响其邻居agents的观点
- **极化驱动**: 作为极化过程的"种子"或"锚点"

**配置参数**:
- **数量控制**: `zealot_count` - 系统中zealot的数量
- **分布模式**: `zealot_mode` - random(随机分布) 或 clustered(聚集分布)
- **身份对齐**: `zealot_identity_allocation` - zealot是否与特定身份群体对齐

## 实验设计

该实验生成两类图表，每类包含4种不同的Y轴指标：

### 图表类型分类

#### 类型1: Zealot Numbers Analysis (X轴：Zealot数量)
- **目的**: 分析不同zealot数量对系统的影响
- **X轴范围**: 0 到 50 个zealots (步长为2)
- **比较维度**: 
  - Zealot分布模式 (Random vs Clustered)
  - Morality比例 (0.0 vs 0.3)

#### 类型2: Morality Ratio Analysis (X轴：Morality比例)
- **目的**: 分析不同morality比例对系统的影响  
- **X轴范围**: 0% 到 30% (步长为2%)
- **比较维度**:
  - Zealot模式 (None/Random/Clustered)
  - Zealot身份对齐 (True/False)
  - Identity分布 (Random/Clustered)

### 参数组合详细表格

#### 类型1: Zealot Numbers 参数组合

| 组合编号 | Zealot分布模式 | Morality比例 | 身份对齐 | 身份分布 | 标签说明 |
|---------|--------------|-------------|---------|---------|----------|
| 1 | Random | 0.0 | True | Random | Random Zealots, Morality=0.0 |
| 2 | Random | 0.3 | True | Random | Random Zealots, Morality=0.3 |
| 3 | Clustered | 0.0 | True | Random | Clustered Zealots, Morality=0.0 |
| 4 | Clustered | 0.3 | True | Random | Clustered Zealots, Morality=0.3 |

#### 类型2: Morality Ratios 参数组合

| 组合编号 | Zealot模式 | Zealot数量 | 身份对齐 | 身份分布 | 标签说明 |
|---------|-----------|----------|---------|---------|----------|
| 1 | None | 0 | - | Random | None, ID-cluster=False |
| 2 | None | 0 | - | Clustered | None, ID-cluster=True |
| 3 | Random | 20 | True | Random | Random, ID-align=True, ID-cluster=False |
| 4 | Random | 20 | True | Clustered | Random, ID-align=True, ID-cluster=True |
| 5 | Random | 20 | False | Random | Random, ID-align=False, ID-cluster=False |
| 6 | Random | 20 | False | Clustered | Random, ID-align=False, ID-cluster=True |
| 7 | Clustered | 20 | True | Random | Clustered, ID-align=True, ID-cluster=False |
| 8 | Clustered | 20 | True | Clustered | Clustered, ID-align=True, ID-cluster=True |
| 9 | Clustered | 20 | False | Random | Clustered, ID-align=False, ID-cluster=False |
| 10 | Clustered | 20 | False | Clustered | Clustered, ID-align=False, ID-cluster=True |

### Y轴指标详解

#### 1. Mean Opinion (平均意见)
**计算方法**: 所有非zealot agents的opinion平均值
$$\text{Mean Opinion} = \frac{1}{N_{non-zealot}} \sum_{i \in \text{non-zealots}} z_i$$

**解释意义**:
- **数值范围**: $[-1, 1]$
- **含义解读**:
  - 接近 `+1`: 系统整体倾向于支持立场
  - 接近 `0`: 系统保持中性或分化平衡
  - 接近 `-1`: 系统整体倾向于反对立场
- **分析价值**: 反映系统的整体观点倾向和偏向程度

#### 2. Variance (意见方差)
**计算方法**: 所有非zealot agents opinion的方差
$$\text{Variance} = \frac{1}{N_{non-zealot}} \sum_{i \in \text{non-zealots}} (z_i - \bar{z})^2$$

**解释意义**:
- **数值范围**: $[0, 1]$ (理论最大值)
- **含义解读**:
  - 接近 `0`: 意见高度一致，系统达成共识
  - 中等数值: 意见存在分歧但未极化
  - 接近 `1`: 意见极度分化，可能出现两极化
- **分析价值**: 衡量系统内部意见分歧的程度

#### 3. Variance per Identity (身份间方差)
**计算方法**: 
- 优先使用不同身份群体间的平均意见差异的绝对值
- 后备方案：各身份群体内部方差的平均值

$$\text{Variance per Identity} = |\text{Mean}_{identity=+1} - \text{Mean}_{identity=-1}|$$

**解释意义**:
- **数值范围**: $[0, 2]$ (理论最大值)
- **含义解读**:
  - 接近 `0`: 不同身份群体持有相似观点
  - 中等数值: 身份群体间存在观点差异
  - 接近 `2`: 身份群体完全对立 (+1 vs -1)
- **分析价值**: 反映身份认同对观点分化的影响程度

#### 4. Polarization Index (极化指数)
**计算方法**: 采用Koudenburg极化指数公式，基于意见分布的跨中性对立测量

**具体步骤**:
1. **意见分类**: 将连续意见值离散化为5个类别
   - Category 1: `opinion < -0.6` (强烈反对)
   - Category 2: `-0.6 ≤ opinion < -0.2` (反对)
   - Category 3: `-0.2 ≤ opinion ≤ 0.2` (中性)
   - Category 4: `0.2 < opinion ≤ 0.6` (支持)
   - Category 5: `opinion > 0.6` (强烈支持)

2. **跨中性对立计算**: 
$$\text{Polarization Index} = \frac{2.14 \times n_2 \times n_4 + 2.70 \times (n_1 \times n_4 + n_2 \times n_5) + 3.96 \times n_1 \times n_5}{0.0099 \times N^2}$$

其中 $n_i$ 表示第 $i$ 类别的agent数量，$N$ 为总agent数量。

**解释意义**:
- **数值范围**: 通常 $[0, 100+]$
- **权重含义**:
  - `2.14`: 相邻对立类别的权重 (反对↔支持)
  - `2.70`: 中等对立类别的权重 (强烈反对↔支持, 反对↔强烈支持)
  - `3.96`: 极端对立类别的权重 (强烈反对↔强烈支持)
- **含义解读**:
  - `0-20`: 低极化，意见相对一致
  - `20-50`: 中等极化，存在明显分歧
  - `50-80`: 高极化，阵营对立明显
  - `80+`: 极端极化，社会严重分裂
- **分析价值**: 重点测量跨越中性立场的对立程度，忽略同侧内部分歧

## 图表类型说明

该实验生成4种不同类型的可视化图表，每种都有其特定的分析价值：

### 1. Error Bar Plots (误差条图表)
- **特征**: 显示均值 ± 标准差
- **适用**: 比较不同条件下的平均表现和稳定性
- **解读**: 误差条越小说明结果越稳定，越大说明随机性影响更强

### 2. Scatter Plots (散点图)
- **特征**: 显示所有原始数据点
- **适用**: 观察数据分布、识别异常值和模式
- **解读**: 点的分散程度反映结果的一致性

### 3. Mean Line Plots (均值曲线图)
- **特征**: 只显示平均值的连线
- **适用**: 清晰比较不同条件的趋势
- **解读**: 曲线走势反映参数变化的影响规律

### 4. Combined Plots (组合图)
- **特征**: 同时显示散点和均值曲线
- **适用**: 综合分析趋势和数据分布
- **解读**: 提供最完整的信息视角

## 典型模式与解释

### Zealot数量效应典型模式

1. **无Zealot → 少量Zealot**:
   - Mean Opinion: 从中性向极端倾斜
   - Variance: 可能先增加后稳定
   - Polarization Index: 显著上升

2. **适量Zealot → 大量Zealot**:
   - 系统被Zealot"拖拽"向固定方向
   - 非Zealot agents的自主性减弱
   - 可能出现"饱和"效应

### Morality比例效应典型模式

1. **低Morality (实用导向)**:
   - 更容易达成妥协和共识
   - 意见变化相对温和
   - 身份对立效应较弱

2. **高Morality (道德导向)**:
   - 观点更加极化和固化
   - 群体间冲突加剧
   - 更难达成妥协

### 交互效应

**Zealot × Morality 交互作用**:
- 高Morality环境中，少量Zealot可能产生更强的极化效应
- 低Morality环境中，需要更多Zealot才能驱动显著变化
- Clustered Zealot在高Morality环境中可能产生"回音室"效应

## 实际应用与启示

### 社会政策层面
1. **意见领袖影响**: Zealot分析有助于理解关键意见领袖的作用
2. **道德议题处理**: Morality分析揭示道德化议题的特殊动态
3. **社会和谐维护**: 理解极化形成机制，制定预防措施

### 媒体与传播
1. **信息传播策略**: 基于Zealot分布优化信息传播路径
2. **舆论引导**: 理解道德化程度对舆论形成的影响
3. **极化预警**: 通过监测指标变化预警极化风险

### 组织管理
1. **团队动态**: 分析团队中的意见领袖和价值观分歧
2. **决策过程**: 优化群体决策机制，平衡不同观点
3. **文化建设**: 理解组织文化对个体观点的影响

## 使用建议

### 分析步骤
1. **基础分析**: 先观察Mean Line Plots了解总体趋势
2. **稳定性检验**: 通过Error Bar Plots评估结果可靠性
3. **异常识别**: 用Scatter Plots发现特殊模式或异常值
4. **综合判断**: 结合Combined Plots进行最终解释

### 参数调优
1. **增加运行次数**: 如果误差条较大，考虑增加每个参数点的运行次数
2. **扩展参数范围**: 如果趋势不明显，考虑扩大X轴范围
3. **细化参数步长**: 如果变化剧烈，考虑减小参数步长

### 结果解释注意事项
1. **因果关系**: 图表显示相关性，不等同于因果关系
2. **模型限制**: 结果受模型假设和参数设置影响
3. **现实适用性**: 从模拟结果到实际应用需要谨慎外推

---

*本文档基于Polarization Triangle Framework v1.0，如有疑问请参考项目README.md或相关技术文档。* 