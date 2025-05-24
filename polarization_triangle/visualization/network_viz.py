import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Patch
import numpy as np


def draw_network(sim, mode, title, filename):
    """
    绘制网络图，支持zealot节点的特殊标识
    
    参数:
    sim -- simulation实例
    mode -- 绘制模式：'opinion', 'identity', 'morality'
    title -- 图表标题
    filename -- 输出文件名
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 获取zealot信息
    zealot_ids = sim.get_zealot_ids() if hasattr(sim, 'get_zealot_ids') else []
    has_zealots = len(zealot_ids) > 0
    
    # 为所有节点设置基本样式
    base_node_size = 20
    zealot_node_size = 40  # zealot节点稍大
    
    # 创建节点大小数组
    node_sizes = []
    for i in range(sim.num_agents):
        if i in zealot_ids:
            node_sizes.append(zealot_node_size)
        else:
            node_sizes.append(base_node_size)
    
    # 存储模式特定的图例patches
    mode_patches = []
    legend_title = ""
    legend_loc = "upper right"
    
    if mode == "opinion":
        cmap = cm.coolwarm
        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        node_colors = [cmap(norm(op)) for op in sim.opinions]
        
        # 绘制所有节点
        nx.draw(sim.graph, pos=sim.pos, node_color=node_colors,
                with_labels=False, node_size=node_sizes, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Opinion")
        
        # Opinion模式的图例设置
        legend_title = "Special Nodes"
        legend_loc = "upper left"
            
    elif mode == "identity":
        node_colors = ['#e41a1c' if iden == 1 else '#377eb8' for iden in sim.identities]
        
        # 绘制所有节点
        nx.draw(sim.graph, pos=sim.pos, node_color=node_colors,
                with_labels=False, node_size=node_sizes, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        
        # Identity模式的图例patches
        mode_patches = [
            Patch(color='#e41a1c', label='Identity: 1'),
            Patch(color='#377eb8', label='Identity: -1')
        ]
        legend_title = "Identity & Special Nodes"
        
    elif mode == "morality":
        node_colors = ['#1a9850' if m == 1 else '#d73027' for m in sim.morals]
        
        # 绘制所有节点
        nx.draw(sim.graph, pos=sim.pos, node_color=node_colors,
                with_labels=False, node_size=node_sizes, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        
        # Morality模式的图例patches
        mode_patches = [
            Patch(color='#1a9850', label='Morality: 1'),
            Patch(color='#d73027', label='Morality: 0')
        ]
        legend_title = "Morality & Special Nodes"
    
    # 统一处理zealot边框绘制（所有模式共用）
    if has_zealots:
        zealot_pos = {i: sim.pos[i] for i in zealot_ids}
        nx.draw_networkx_nodes(sim.graph, pos=zealot_pos, nodelist=zealot_ids,
                             node_color='none', node_size=[zealot_node_size + 10] * len(zealot_ids),
                             edgecolors='gold', linewidths=1.5, alpha=1.0, ax=ax)
    
    # 统一处理图例（所有模式共用）
    if has_zealots:
        # 创建zealot图例patch
        zealot_patch = Patch(facecolor='none', edgecolor='gold', linewidth=3, 
                           label=f'Zealots (n={len(zealot_ids)})')
        
        # 组合图例patches
        if mode_patches:  # identity和morality模式
            all_patches = mode_patches + [zealot_patch]
        else:  # opinion模式
            all_patches = [zealot_patch]
        
        ax.legend(handles=all_patches, loc=legend_loc, title=legend_title)
    elif mode_patches:  # 有模式特定图例但没有zealots
        ax.legend(handles=mode_patches, loc=legend_loc, title=legend_title.replace(" & Special Nodes", ""))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()