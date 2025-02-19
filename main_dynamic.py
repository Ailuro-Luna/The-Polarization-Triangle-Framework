import matplotlib
matplotlib.use("TkAgg")  # 使用 TkAgg 后端，避免 PyCharm 内置显示问题
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.widgets import Button
from matplotlib.patches import Patch  # 用于离散图例
from simulation import Simulation
from config import model_params
import threading
import time

# 全局变量
display_mode = "opinion"
cb = None  # 用于保存颜色条对象
running = False  # 控制仿真是否进行
def update_plot():
    global cb
    ax.clear()
    node_colors = update_node_colors(sim, cmap, norm)
    nx.draw(
        sim.G, pos=sim.pos, node_color=node_colors,
        with_labels=False, node_size=50, edge_color="#AAAAAA", ax=ax
    )
    ax.set_title(f"Step {current_step}  Display: {display_mode}")
    ax.set_aspect('equal', 'box')

    # 清空 colorbar 轴内容
    if cb is not None:
        cb.ax.clear()
        cb = None

    ax.legend_ = None
    update_color_legend()
    fig.canvas.draw_idle()

def update_color_legend():
    global cb
    cbar_ax.clear()
    cbar_ax.set_visible(True)

    if display_mode == "opinion":
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
    elif display_mode == "identity":
        cbar_ax.set_visible(False)
        patches = [
            Patch(color='#e41a1c', label='Identity: 1'),
            Patch(color='#377eb8', label='Identity: -1')
        ]
        ax.legend(handles=patches, loc='upper right', title="Identity")
    elif display_mode == "morality":
        cbar_ax.set_visible(False)
        patches = [
            Patch(color='#1a9850', label='Morality: 1'),
            Patch(color='#ffff33', label='Morality: 0'),
            Patch(color='#d73027', label='Morality: -1')
        ]
        ax.legend(handles=patches, loc='upper right', title="Morality")

def switch_display_mode(event):
    global display_mode
    if display_mode == "opinion":
        display_mode = "identity"
    elif display_mode == "identity":
        display_mode = "morality"
    else:
        display_mode = "opinion"
    print("Switched display mode to:", display_mode)
    update_plot()

def update_node_colors(sim, cmap, norm):
    if display_mode == "opinion":
        return [cmap(norm(op)) for op in sim.opinions]
    elif display_mode == "identity":
        return ['#e41a1c' if iden == 1 else '#377eb8' for iden in sim.identities]
    elif display_mode == "morality":
        return ['#1a9850' if m == 1 else '#d73027' if m == -1 else '#ffff33' for m in sim.morals]
    else:
        return [cmap(norm(op)) for op in sim.opinions]

def run_simulation():
    global current_step, running
    while running and current_step < num_steps:
        sim.step()
        current_step += 1
        update_plot()
        fig.canvas.flush_events()
        # time.sleep(0.2)  # 控制更新速度

def start_simulation(event):
    global running, simulation_thread, current_step
    if not running:
        running = True
        simulation_thread = threading.Thread(target=run_simulation)
        simulation_thread.start()

def pause_simulation(event):
    global running
    running = False

def reset_simulation(event):
    global sim, current_step, running
    running = False
    sim = Simulation(**model_params)  # 重新初始化仿真
    current_step = 0
    update_plot()

def main():
    global sim, fig, ax, cbar_ax, cmap, norm, current_step, num_steps

    sim = Simulation(**model_params)
    num_steps = 500
    current_step = 0

    plt.ion()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    cbar_ax = fig.add_axes([0.82, 0.25, 0.02, 0.5])

    button_ax1 = fig.add_axes([0.05, 0.05, 0.2, 0.075])
    start_button = Button(button_ax1, 'Start')
    start_button.on_clicked(start_simulation)

    button_ax2 = fig.add_axes([0.3, 0.05, 0.2, 0.075])
    pause_button = Button(button_ax2, 'Pause')
    pause_button.on_clicked(pause_simulation)

    button_ax3 = fig.add_axes([0.55, 0.05, 0.2, 0.075])
    reset_button = Button(button_ax3, 'Reset')
    reset_button.on_clicked(reset_simulation)

    button_ax4 = fig.add_axes([0.75, 0.05, 0.2, 0.075])
    mode_button = Button(button_ax4, 'Switch Mode')
    mode_button.on_clicked(switch_display_mode)

    norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = cm.coolwarm

    update_plot()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()