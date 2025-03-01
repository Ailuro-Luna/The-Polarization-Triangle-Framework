# main.py
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.widgets import Button
from matplotlib.patches import Patch
import threading
import sys
from simulation import Simulation
from config import config
from trajectory import draw_opinion_trajectory
from visualization import draw_network, draw_opinion_distribution


sys.setrecursionlimit(1500)

class SimulationApp:
    def __init__(self, config, num_steps=500):
        self.config = config
        self.sim = Simulation(config)
        self.num_steps = num_steps
        self.current_step = 0
        self.running = False
        self.display_mode = "opinion"  # 可选： "opinion", "identity", "morality"
        self.simulation_thread = None
        self.opinion_history = []
        self.cmap = cm.coolwarm
        self.norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        self.cb = None
        self.setup_plot()

    def setup_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 12))
        self.ax = self.fig.add_axes([0.1, 0.1, 0.7, 0.8])
        self.cbar_ax = self.fig.add_axes([0.82, 0.25, 0.02, 0.5])
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
        if self.cb is not None:
            self.cb.ax.clear()
            self.cb = None
        self.ax.legend_ = None
        self.update_color_legend()
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
        self.opinion_history = []
        self.opinion_history.append(self.sim.opinions.copy())
        while self.running and self.current_step < self.num_steps:
            self.sim.step()
            self.current_step += 1
            self.opinion_history.append(self.sim.opinions.copy())
            self.update_plot()
            self.fig.canvas.flush_events()
        trajectory_filename = "opinion_trajectory.png"
        draw_opinion_trajectory(self.opinion_history,
                                f"Opinion Trajectories (Steps 0-{self.current_step})",
                                trajectory_filename)
        print("Trajectory saved to", trajectory_filename)

        # 绘制opinion分布图
        distribution_filename = "opinion_distribution.png"
        draw_opinion_distribution(self.sim,
                                  f"Opinion Distribution at Step {self.current_step}",
                                  distribution_filename)
        print("Distribution saved to", distribution_filename)

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.simulation_thread = threading.Thread(target=self.run_simulation)
            self.simulation_thread.start()

    def pause_simulation(self):
        self.running = False

    def reset_simulation(self):
        self.running = False
        self.sim = Simulation(self.config)
        self.current_step = 0
        self.update_plot()

def main():
    app = SimulationApp(config)

if __name__ == "__main__":
    main()
