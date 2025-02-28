# visualization.py
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import Patch

def draw_network(sim, mode, title, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    if mode == "opinion":
        cmap = cm.coolwarm
        norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        node_colors = [cmap(norm(op)) for op in sim.opinions]
        nx.draw(sim.G, pos=sim.pos, node_color=node_colors,
                with_labels=False, node_size=20, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Opinion")
    elif mode == "identity":
        node_colors = ['#e41a1c' if iden == 1 else '#377eb8' for iden in sim.identities]
        nx.draw(sim.G, pos=sim.pos, node_color=node_colors,
                with_labels=False, node_size=20, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        patches = [
            Patch(color='#e41a1c', label='Identity: 1'),
            Patch(color='#377eb8', label='Identity: -1')
        ]
        ax.legend(handles=patches, loc='upper right', title="Identity")
    elif mode == "morality":
        node_colors = ['#1a9850' if m == 1 else '#d73027' for m in sim.morals]
        nx.draw(sim.G, pos=sim.pos, node_color=node_colors,
                with_labels=False, node_size=20, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        patches = [
            Patch(color='#1a9850', label='Morality: 1'),
            Patch(color='#d73027', label='Morality: 0')
        ]
        ax.legend(handles=patches, loc='upper right', title="Morality")
    plt.savefig(filename)
    plt.close()
