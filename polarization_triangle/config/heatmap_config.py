"""
热力图可视化配置示例

此文件展示了如何自定义热力图的颜色映射、尺度范围和其他视觉参数。
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm
import numpy as np

# ==================== 颜色映射方案 ====================

# 常用的颜色映射方案
COLORMAPS = {
    # 单色渐变
    'viridis': 'viridis',       # 默认，绿-蓝-紫
    'plasma': 'plasma',         # 紫-粉-黄
    'inferno': 'inferno',       # 黑-红-黄
    'magma': 'magma',           # 黑-紫-粉-白
    'cividis': 'cividis',       # 色盲友好的蓝-黄
    
    # 双色映射
    'coolwarm': 'coolwarm',     # 蓝-白-红
    'RdBu': 'RdBu',            # 红-白-蓝
    'RdYlBu': 'RdYlBu',        # 红-黄-蓝
    'seismic': 'seismic',       # 红-白-蓝（地震图常用）
    
    # 经典映射
    'hot': 'hot',               # 黑-红-黄-白
    'jet': 'jet',               # 蓝-青-绿-黄-红
    'rainbow': 'rainbow',       # 彩虹色
    'spectral': 'Spectral',     # 光谱色
    
    # 特殊用途
    'binary': 'binary',         # 黑白
    'gray': 'gray',             # 灰度
}

# ==================== 预设配置 ====================

# 1. 高对比度配置（用于突出极值）
HIGH_CONTRAST_CONFIG = {
    'cmap': 'hot',
    'log_scale': True,
    'vmin': 1,        # 最小值设为1（避免log(0)）
    'vmax': 100,      # 最大值设为100
}

# 2. 对称配置（用于强调正负对称的数据）
SYMMETRIC_CONFIG = {
    'cmap': 'RdBu',
    'log_scale': True,
    'vmin': 0,        # 最小值
    'vmax': 50,       # 最大值
}

# 3. 细节强调配置（线性尺度，突出细微变化）
DETAIL_CONFIG = {
    'cmap': 'viridis',
    'log_scale': True,
    'vmin': 0,
    'vmax': 20,       # 较小的最大值，放大细节
}

# 4. 色盲友好配置
COLORBLIND_FRIENDLY_CONFIG = {
    'cmap': 'cividis',
    'log_scale': True,
    'vmin': 1,
    'vmax': 50,
}

# 5. 论文发表配置（黑白友好）
PUBLICATION_CONFIG = {
    'cmap': 'gray',
    'log_scale': True,
    'vmin': 0,
    'vmax': 30,
}

# ==================== 自定义标准化函数 ====================

def create_custom_power_norm(gamma=0.5, vmin=0, vmax=100):
    """
    创建幂律标准化（用于强调中等数值）
    
    参数:
    gamma -- 幂指数，< 1 强调小值，> 1 强调大值
    vmin -- 最小值
    vmax -- 最大值
    """
    return PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

def create_symlog_norm(linthresh=10, vmin=0, vmax=1000):
    """
    创建对称对数标准化（用于跨越多个数量级的数据）
    
    参数:
    linthresh -- 线性阈值，小于此值使用线性尺度
    vmin -- 最小值  
    vmax -- 最大值
    """
    return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

# ==================== 使用示例 ====================

def get_heatmap_config(config_name='default'):
    """
    获取预设的热力图配置
    
    参数:
    config_name -- 配置名称：'high_contrast', 'symmetric', 'detail', 'colorblind', 'publication'
    
    返回:
    dict -- 包含热力图参数的字典
    """
    configs = {
        'default': {
            'cmap': 'viridis',
            'log_scale': True,
            'vmin': None,
            'vmax': None,
        },
        'high_contrast': HIGH_CONTRAST_CONFIG,
        'symmetric': SYMMETRIC_CONFIG,
        'detail': DETAIL_CONFIG,
        'colorblind': COLORBLIND_FRIENDLY_CONFIG,
        'publication': PUBLICATION_CONFIG,
    }
    
    return configs.get(config_name, configs['default'])

# ==================== 热力图参数说明 ====================

HEATMAP_PARAMS_GUIDE = """
热力图参数设置指南:

1. cmap (颜色映射):
   - 'viridis': 默认，适合大多数情况
   - 'hot': 高对比度，突出极值
   - 'coolwarm': 双极性数据，有明确的中心值
   - 'RdBu': 红蓝对比，适合差异显示
   - 'gray': 黑白打印友好

2. log_scale (对数尺度):
   - True: 适合跨越多个数量级的数据
   - False: 适合数值范围较小的数据

3. vmin/vmax (数值范围):
   - 设置具体数值可以：
     * 保持多张图的尺度一致
     * 突出特定数值范围
     * 忽略异常值的影响

4. 特殊标准化:
   - PowerNorm: 强调特定数值范围
   - SymLogNorm: 结合线性和对数尺度
   - 自定义函数: 完全控制颜色映射

使用示例:
```python
# 方法1: 使用预设配置
config = get_heatmap_config('high_contrast')
draw_opinion_distribution_heatmap(data, title, filename, **config)

# 方法2: 自定义设置
draw_opinion_distribution_heatmap(
    data, title, filename,
    cmap='hot',
    log_scale=True,
    vmin=1,
    vmax=100
)

# 方法3: 使用自定义标准化
custom_norm = create_custom_power_norm(gamma=0.5, vmin=0, vmax=50)
draw_opinion_distribution_heatmap(
    data, title, filename,
    custom_norm=custom_norm,
    cmap='viridis'
)
```
"""

if __name__ == "__main__":
    print(HEATMAP_PARAMS_GUIDE) 