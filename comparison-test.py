# comparison_test.py
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from config import SimulationConfig, lfr_config
from visualization import draw_network, draw_opinion_distribution, draw_opinion_distribution_heatmap
from trajectory import run_simulation_with_trajectory, draw_opinion_trajectory


def create_folder_structure():
    """Create the folder structure for organizing comparison results"""
    base_dir = "comparison_results"
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create subdirectories for each visualization type
    vis_types = ["trajectories", "heatmaps", "distributions", "networks", "start_state", "end_state"]
    
    for vis_type in vis_types:
        folder_path = os.path.join(base_dir, vis_type)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
    return base_dir


def generate_test_configs():
    """Generate test configurations with varying morality rates and identity-issue mappings"""
    # Base configuration (copy of lfr_config)
    base_config = copy.deepcopy(lfr_config)
    base_config.num_agents = 500
    
    # Parameters to vary
    morality_rates = [0.0, 0.25, 0.5, 0.75, 1.0]  # Low, medium, high moralization
    
    # Different identity-issue mappings
    identity_mappings = [
        # No effect (neutral)
        {1: 0.0, -1: 0.0},
        
        # Weak alignment
        {1: 0.2, -1: -0.2},
        
        # Strong alignment 
        {1: 0.7, -1: -0.7},
        
        # Asymmetric alignment
        {1: 0.7, -1: -0.2}
    ]
    
    # Names for the mappings (for file naming)
    mapping_names = ["neutral", "weak", "strong", "asymm"]
    
    configs = []
    for mor_rate in morality_rates:
        for i, mapping in enumerate(identity_mappings):
            config = copy.deepcopy(base_config)
            config.morality_rate = mor_rate
            config.identity_issue_mapping = mapping
            
            # Create a descriptive name for this configuration
            config_name = f"mor{int(mor_rate*100):03d}_id_{mapping_names[i]}"
            
            configs.append((config, config_name))
    
    return configs


def run_comparison_tests(steps=250):
    """Run simulations with different configurations and save comparative visualizations"""
    base_dir = create_folder_structure()
    configs = generate_test_configs()
    
    # Dictionaries to store results for each visualization type
    trajectories = {}
    heatmaps = {}
    
    for config, config_name in configs:
        print(f"Running simulation with configuration: {config_name}")
        
        # Create the simulation
        sim = Simulation(config)
        
        # Save initial state
        draw_network(sim, "opinion", f"Initial Opinion - {config_name}", 
                    os.path.join(base_dir, "start_state", f"opinion_{config_name}.png"))
        draw_network(sim, "morality", f"Initial Morality - {config_name}", 
                    os.path.join(base_dir, "start_state", f"morality_{config_name}.png"))
        draw_network(sim, "identity", f"Identity - {config_name}", 
                    os.path.join(base_dir, "start_state", f"identity_{config_name}.png"))
        
        # Run simulation and record trajectory
        trajectory = run_simulation_with_trajectory(sim, steps=steps)
        trajectories[config_name] = trajectory
        
        # Save individual trajectory
        draw_opinion_trajectory(trajectory, f"Opinion Trajectory - {config_name}", 
                              os.path.join(base_dir, "trajectories", f"trajectory_{config_name}.png"))
        
        # Save heatmap
        draw_opinion_distribution_heatmap(
            trajectory, 
            f"Opinion Distribution Over Time - {config_name}",
            os.path.join(base_dir, "heatmaps", f"heatmap_{config_name}.png"),
            bins=40,
            log_scale=True
        )
        
        # Save final opinion distribution
        draw_opinion_distribution(
            sim,
            f"Final Opinion Distribution - {config_name}",
            os.path.join(base_dir, "distributions", f"dist_{config_name}.png")
        )
        
        # Save final network state
        draw_network(sim, "opinion", f"Final Opinion - {config_name}", 
                    os.path.join(base_dir, "end_state", f"opinion_{config_name}.png"))
        draw_network(sim, "morality", f"Final Morality - {config_name}", 
                    os.path.join(base_dir, "end_state", f"morality_{config_name}.png"))
    
    # Create combined trajectory plots for comparison
    create_combined_trajectories(trajectories, base_dir)
    
    # Create combined distribution plots for comparison
    create_final_distribution_comparison(configs, base_dir, steps)


def create_combined_trajectories(trajectories, base_dir):
    """Create combined trajectory plots for comparing different configurations"""
    # Group by morality rate
    morality_groups = {}
    for config_name, trajectory in trajectories.items():
        mor_rate = config_name.split('_')[0]
        if mor_rate not in morality_groups:
            morality_groups[mor_rate] = []
        morality_groups[mor_rate].append((config_name, trajectory))
    
    # Plot trajectories grouped by morality rate
    for mor_rate, group in morality_groups.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for config_name, trajectory in group:
            # Calculate the average trajectory (mean opinion)
            avg_trajectory = np.mean(trajectory, axis=1)
            id_mapping = config_name.split('_')[-1]
            
            # Plot with different colors/styles based on identity mapping
            if id_mapping == "neutral":
                ax.plot(avg_trajectory, label=f"{id_mapping}", color='blue', linewidth=2)
            elif id_mapping == "weak":
                ax.plot(avg_trajectory, label=f"{id_mapping}", color='green', linewidth=2)
            elif id_mapping == "strong":
                ax.plot(avg_trajectory, label=f"{id_mapping}", color='red', linewidth=2)
            elif id_mapping == "asymm":
                ax.plot(avg_trajectory, label=f"{id_mapping}", color='purple', linewidth=2)
        
        ax.set_title(f"Average Opinion Trajectories - Morality Rate: {mor_rate}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Average Opinion")
        ax.set_ylim(-1, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "trajectories", f"combined_{mor_rate}.png"))
        plt.close(fig)


def create_final_distribution_comparison(configs, base_dir, steps):
    """Create side-by-side comparisons of final opinion distributions"""
    # Group by identity mapping
    mapping_groups = {}
    for config, config_name in configs:
        id_mapping = config_name.split('_')[-1]
        if id_mapping not in mapping_groups:
            mapping_groups[id_mapping] = []
        mapping_groups[id_mapping].append((config, config_name))
    
    # For each identity mapping, create a comparison of different morality rates
    for id_mapping, group in mapping_groups.items():
        fig, axs = plt.subplots(1, len(group), figsize=(5*len(group), 5), sharey=True)
        
        for i, (config, config_name) in enumerate(group):
            # Create a new simulation
            sim = Simulation(config)
            # Run it for the specified number of steps
            for _ in range(steps):
                sim.step()
            
            # Plot the final distribution
            mor_rate = int(config_name.split('_')[0][3:]) / 100
            axs[i].hist(sim.opinions, bins=20, range=(-1, 1), color='blue', alpha=0.7, edgecolor='black')
            axs[i].set_title(f"Morality Rate: {mor_rate}")
            axs[i].set_xlabel("Opinion")
            if i == 0:
                axs[i].set_ylabel("Number of Agents")
        
        fig.suptitle(f"Final Opinion Distributions - Identity Mapping: {id_mapping}")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "distributions", f"comparison_{id_mapping}.png"))
        plt.close(fig)


if __name__ == "__main__":
    run_comparison_tests()
