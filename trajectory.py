# trajectory.py
import numpy as np
import matplotlib.pyplot as plt

def run_simulation_with_trajectory(sim, steps=500):
    history = []
    history.append(sim.opinions.copy())
    for _ in range(steps):
        sim.step()
        history.append(sim.opinions.copy())
    return history

def draw_opinion_trajectory(history, title, filename):
    history = np.array(history)
    total_steps = history.shape[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(total_steps)
    for i in range(history.shape[1]):
        ax.plot(x, history[:, i], color='gray', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Opinion")
    ax.set_ylim(-1, 1)
    plt.savefig(filename)
    plt.close()
