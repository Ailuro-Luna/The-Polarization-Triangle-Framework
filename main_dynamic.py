# main_dynamic.py
import matplotlib
matplotlib.use("TkAgg")  # 使用 TkAgg 后端，避免 PyCharm 内置显示问题
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.widgets import Button
from matplotlib.patches import Patch
from simulation import Simulation
from config import model_params
import threading

class SimulationApp:
    def __init__(self, sim_params, num_steps=500):
        # 初始化仿真和可视化参数
        self.sim = Simulation(**sim_params)
        self.num_steps = num_steps
        self.current_step = 0
        self.running = False
        self.display_mode = "opinion"  # 可选："opinion"、"identity"、"morality"
        self.simulation_thread = None

        # 设置颜色映射
        self.cmap = cm.coolwarm
        self.norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        self.cb = None  # 保存颜色条对象

        # 创建图形界面
        self.setup_plot()

    def setup_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 12))
        self.ax = self.fig.add_axes([0.1, 0.1, 0.7, 0.8])
        self.cbar_ax = self.fig.add_axes([0.82, 0.25, 0.02, 0.5])

        # 设置各个按钮
        btn_start = Button(self.fig.add_axes([0.05, 0.05, 0.2, 0.075]), 'Start')
        btn_start.on_clicked(lambda event: self.start_simulation())

        btn_pause = Button(self.fig.add_axes([0.3, 0.05, 0.2, 0.075]), 'Pause')
        btn_pause.on_clicked(lambda event: self.pause_simulation())

        btn_reset = Button(self.fig.add_axes([0.55, 0.05, 0.2, 0.075]), 'Reset')
        btn_reset.on_clicked(lambda event: self.reset_simulation())

        btn_mode = Button(self.fig.add_axes([0.75, 0.05, 0.2, 0.075]), 'Switch Mode')
        btn_mode.on_clicked(lambda event: self.switch_display_mode())

        self.update_plot()
        plt.ioff()
        plt.show()

    def update_plot(self):
        self.ax.clear()
        node_colors = self.get_node_colors()
        nx.draw(
            self.sim.G,
            pos=self.sim.pos,
            node_color=node_colors,
            with_labels=False,
            node_size=20,
            alpha=0.8,
            edge_color="#AAAAAA",
            ax=self.ax
        )
        self.ax.set_title(f"Step {self.current_step}  Display: {self.display_mode}")
        self.ax.set_aspect('equal', 'box')

        if self.display_mode == "opinion":
            # 如果已有 colorbar，则更新；否则创建新的
            if self.cb is None:
                sm = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
                sm.set_array([])
                self.cb = self.fig.colorbar(sm, cax=self.cbar_ax)
            else:
                # 更新现有 colorbar 的映射（可选）
                self.cb.update_normal(cm.ScalarMappable(norm=self.norm, cmap=self.cmap))
        else:
            # 切换到其他模式时移除 colorbar
            if self.cb is not None:
                self.cb.remove()
                self.cb = None
            if self.display_mode == "identity":
                patches = [
                    Patch(color='#e41a1c', label='Identity: 1'),
                    Patch(color='#377eb8', label='Identity: -1')
                ]
                self.ax.legend(handles=patches, loc='upper right', title="Identity")
            elif self.display_mode == "morality":
                patches = [
                    Patch(color='#1a9850', label='Morality: 1'),
                    Patch(color='#d73027', label='Morality: 0')
                ]
                self.ax.legend(handles=patches, loc='upper right', title="Morality")
            self.cbar_ax.clear()
            self.cbar_ax.set_visible(False)

        self.fig.canvas.draw_idle()

    def update_color_legend(self):
        self.cbar_ax.clear()
        self.cbar_ax.set_visible(True)
        if self.display_mode == "opinion":
            sm = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
            sm.set_array([])
            self.cb = self.fig.colorbar(sm, cax=self.cbar_ax)
        elif self.display_mode == "identity":
            self.cbar_ax.set_visible(False)
            patches = [
                Patch(color='#e41a1c', label='Identity: 1'),
                Patch(color='#377eb8', label='Identity: -1')
            ]
            self.ax.legend(handles=patches, loc='upper right', title="Identity")
        elif self.display_mode == "morality":
            self.cbar_ax.set_visible(False)
            patches = [
                Patch(color='#1a9850', label='Morality: 1'),
                Patch(color='#d73027', label='Morality: 0')
            ]
            self.ax.legend(handles=patches, loc='upper right', title="Morality")

    def get_node_colors(self):
        if self.display_mode == "opinion":
            return [self.cmap(self.norm(op)) for op in self.sim.opinions]
        elif self.display_mode == "identity":
            return ['#e41a1c' if iden == 1 else '#377eb8' for iden in self.sim.identities]
        elif self.display_mode == "morality":
            return ['#1a9850' if m == 1 else '#d73027' for m in self.sim.morals]
        else:
            return [self.cmap(self.norm(op)) for op in self.sim.opinions]

    def switch_display_mode(self):
        if self.display_mode == "opinion":
            self.display_mode = "identity"
        elif self.display_mode == "identity":
            self.display_mode = "morality"
        else:
            self.display_mode = "opinion"
        print("Switched display mode to:", self.display_mode)
        self.update_plot()

    def run_simulation(self):
        while self.running and self.current_step < self.num_steps:
            self.sim.step()
            self.current_step += 1
            self.update_plot()
            self.fig.canvas.flush_events()
            # time.sleep(0.2)  # 可根据需要控制更新速度

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.simulation_thread = threading.Thread(target=self.run_simulation)
            self.simulation_thread.start()

    def pause_simulation(self):
        self.running = False

    def reset_simulation(self):
        self.running = False
        self.sim = Simulation(**model_params)  # 重新初始化仿真
        self.current_step = 0
        self.update_plot()

def main():
    app = SimulationApp(model_params)

if __name__ == "__main__":
    main()
