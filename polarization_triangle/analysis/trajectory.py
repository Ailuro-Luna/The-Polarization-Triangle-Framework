def run_simulation_with_trajectory(sim, steps=500):
    history = []
    history.append(sim.opinions.copy())
    for _ in range(steps):
        sim.step()
        history.append(sim.opinions.copy())
    return history