# 极化三角框架 (Polarization Triangle Framework) Agent与邻居的交互计算

## 模型基本变量

### Agent 属性
- $l_i$ — agent $i$ 的 identity (±1)
- $m_i$ — agent $i$ 的 moralization (0或1)
- $z_i$ — agent $i$ 的 opinion (∈ [-1,1])
- $\sigma_{ij}$ — agent $j$ 的 opinion 被 agent $i$ "perceived" 的程度 (seen by agent $i$)，是一个 signal value，取值只有 -1、0、1

### 感知意见表示 (Perceived Opinion)
$\sigma_{ij}$ 的计算公式:

$$\sigma_{ij} = 
\begin{cases}
\frac{z_j}{|z_j|}, & \text{如果 } z_j \neq 0 \text{ 且 } (m_i = 1 \text{ 或 } m_j = 1) \\
0, & \text{如果 } z_j = 0 \\
z_j, & \text{如果 } m_i = 0 \text{ 且 } m_j = 0
\end{cases}$$

### Agents 之间的关系系数
$A_{ij}$ 的定义，其中 $a_{ij}$ 是 adjacency matrix 的值:

$$A_{ij} = 
\begin{cases}
-a_{ij}, & \text{如果 } l_i = -l_j \text{ 且 } m_i, m_j = 1 \text{ 且 } \sigma_{ij} \cdot \sigma_{ji} < 0 \\
\frac{a_{ij}}{\sigma_{ji}} \tilde{\sigma}_{sameIdentity}, & \text{如果 } l_i = l_j \text{ 且 } m_i, m_j = 1 \text{ 且 } \sigma_{ij} \cdot \sigma_{ji} < 0 \\
a_{ij}, & \text{其他情况}
\end{cases}$$

## 意见动态方程 (Opinion Dynamics)

模型的核心动力学方程:

$$
\dot{z_i} = -\delta_i z_i + u_i \cdot \tanh\left(\alpha_i \sigma_{ii} + \frac{\beta}{1 + \gamma_i m_i} \sum_{j \neq i} A_{ij} \sigma_{ij}\right)
$$


这个方程由两个主要部分组成:

1. **回归中性意见** (Relaxation to neutral opinion): $-\delta_i z_i$
   - 这部分表示意见随时间自然衰减回归到中性状态的趋势

2. **意见激活** (Opinion activation): $u_i \cdot \tanh\left(\alpha_i \sigma_{ii} + \frac{\beta}{1 + \gamma_i m_i} \sum_{j \neq i} A_{ij} \sigma_{ij}\right)$
   - 进一步分为:
     - **自我激活** (Self activation): $\alpha_i \sigma_{ii}$
     - **他人意见影响** (Opinion of others): $\frac{\beta}{1 + \gamma_i m_i} \sum_{j \neq i} A_{ij} \sigma_{ij}$

## 模型特性

这个模型通过整合 identity、opinion 和 moralization 三个维度，模拟了社会网络中的极化动态过程。关键特点包括:

1. **Identity 因素** ($l_i$): 代表 agent 的社会身份认同，决定了与其他 agents 的互动方式
2. **Moralization 因素** ($m_i$): 代表 agent 对议题的道德化程度，影响感知意见的强度
3. **意见动态**: 在 self-activation 和社会影响的双重作用下，agent 的意见会随时间演变
4. **社交网络结构**: 通过 $A_{ij}$ 系数捕捉网络中的结构关系和 agent 之间的互动模式

这个框架旨在模拟和分析意见距离、道德冲突和结构对齐如何共同作用，影响社会网络中的群体协调和民主成果。 