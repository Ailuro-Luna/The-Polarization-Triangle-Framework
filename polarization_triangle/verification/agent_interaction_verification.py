import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
from pathlib import Path

from polarization_triangle.core.config import SimulationConfig, high_polarization_config
from polarization_triangle.core.dynamics import (
    calculate_perceived_opinion_func,
    calculate_relationship_coefficient_func,
    calculate_same_identity_sigma_func
)

class TwoAgentVerification:
    """
    A simplified Simulation class for verifying interactions between two agents
    """
    def __init__(self, config=None):
        """Initialize verification environment"""
        # Use high polarization config
        self.config = config or high_polarization_config
        
        # Create a simple network with two agents
        self.num_agents = 2
        self.graph = nx.Graph()
        self.graph.add_nodes_from([0, 1])
        self.graph.add_edge(0, 1)
        
        # Adjacency matrix
        self.adj_matrix = np.array([[0, 1], [1, 0]])
        
        # Get model parameters from config
        self.delta = self.config.delta  # Opinion decay rate
        self.u = np.ones(self.num_agents) * self.config.u  # Opinion activation coefficient
        self.alpha = np.ones(self.num_agents) * self.config.alpha  # Self-activation coefficient
        self.beta = self.config.beta  # Social influence coefficient
        self.gamma = np.ones(self.num_agents) * self.config.gamma  # Moralization impact coefficient
        
        # Store neighbor lists for each agent
        self.neighbors_list = [[1], [0]]
        
        # Neighbor CSR format - simplified version
        self.neighbors_indices = np.array([1, 0], dtype=np.int32)
        self.neighbors_indptr = np.array([0, 1, 2], dtype=np.int32)
        
        # Initialize trajectory storage
        self.opinion_trajectory = []
        self.self_activation_history = []
        self.social_influence_history = []
        
        print(f"Model Parameters: delta={self.delta}, u={self.u[0]}, alpha={self.alpha[0]}, beta={self.beta}, gamma={self.gamma[0]}")
    
    def setup_scenario(self, focal_opinion, other_opinion, focal_identity, other_identity, 
                       focal_moral, other_moral):
        """Set up a specific scenario state"""
        # Initialize attributes
        self.opinions = np.array([focal_opinion, other_opinion], dtype=np.float64)
        self.identities = np.array([focal_identity, other_identity], dtype=np.int32)
        self.morals = np.array([focal_moral, other_moral], dtype=np.int32)
        
        # Record initial state
        self.initial_opinions = self.opinions.copy()
        
        # Reset trajectory storage
        self.opinion_trajectory = [self.opinions.copy()]
        self.self_activation_history = []
        self.social_influence_history = []
    
    def calculate_perceived_opinion(self, i, j):
        """Calculate how agent i perceives agent j's opinion"""
        return calculate_perceived_opinion_func(self.opinions, self.morals, i, j)
    
    def calculate_relationship_coefficient(self, i, j):
        """Calculate relationship coefficient between agent i and agent j"""
        # Calculate average perceived opinion of neighbors with same identity
        # In the two-agent case, if identities are the same, the average is just the other agent's perceived opinion
        if self.identities[i] == self.identities[j]:
            sigma_same_identity = self.calculate_perceived_opinion(i, j)
        else:
            sigma_same_identity = 0.0
        
        return calculate_relationship_coefficient_func(
            self.adj_matrix, 
            self.identities, 
            self.morals, 
            self.opinions, 
            i, j, 
            sigma_same_identity
        )
    
    def step(self):
        """Execute one time step, calculate new opinion values"""
        # Initialize opinion changes
        opinion_changes = np.zeros(self.num_agents)
        self_activation_values = np.zeros(self.num_agents)
        social_influence_values = np.zeros(self.num_agents)
        
        for i in range(self.num_agents):
            # Self-perception
            sigma_ii = np.sign(self.opinions[i]) if self.opinions[i] != 0 else 0
            
            # Calculate neighbor influence
            neighbor_influence = 0
            for j in self.neighbors_list[i]:
                A_ij = self.calculate_relationship_coefficient(i, j)
                sigma_ij = self.calculate_perceived_opinion(i, j)
                neighbor_influence += A_ij * sigma_ij
            
            # Calculate self-activation and social influence
            self_activation = self.alpha[i] * sigma_ii
            social_influence = (self.beta / (1 + self.gamma[i] * self.morals[i])) * neighbor_influence
            
            # Save values for later analysis
            self_activation_values[i] = self_activation
            social_influence_values[i] = social_influence
            
            # Calculate opinion change rate
            regression_term = -self.delta * self.opinions[i]
            activation_term = self.u[i] * np.tanh(self_activation + social_influence)
            
            # Total change
            opinion_changes[i] = regression_term + activation_term
        
        # Apply opinion changes
        step_size = self.config.influence_factor
        self.opinions += step_size * opinion_changes
        self.opinions = np.clip(self.opinions, -1, 1)
        
        # Store history
        self.opinion_trajectory.append(self.opinions.copy())
        self.self_activation_history.append(self_activation_values.copy())
        self.social_influence_history.append(social_influence_values.copy())
        
        # Return focal agent's opinion change and related values
        return {
            "opinion_change": self.opinions[0] - self.initial_opinions[0],
            "final_opinion": self.opinions[0],
            "self_activation": self_activation_values[0],
            "social_influence": social_influence_values[0]
        }
    
    def run_simulation(self, num_steps=1):
        """Run simulation for multiple steps"""
        results_over_time = []
        
        for step_num in range(num_steps):
            step_result = self.step()
            
            # Add step number to result
            step_result["step"] = step_num + 1
            results_over_time.append(step_result)
        
        # Calculate final results
        final_result = {
            "opinion_change": self.opinions[0] - self.initial_opinions[0],
            "final_opinion": self.opinions[0],
            "num_steps": num_steps,
            "trajectory": np.array(self.opinion_trajectory),
            "self_activation": np.array(self.self_activation_history),
            "social_influence": np.array(self.social_influence_history)
        }
        
        return final_result, results_over_time
    
    def get_trajectory_dataframe(self):
        """Convert trajectory to DataFrame for easier analysis"""
        steps = len(self.opinion_trajectory)
        
        data = {
            "step": list(range(steps)),
            "focal_opinion": [op[0] for op in self.opinion_trajectory],
            "other_opinion": [op[1] for op in self.opinion_trajectory],
        }
        
        # Add self-activation and social influence if available
        if self.self_activation_history:
            data["focal_self_activation"] = [sa[0] for sa in self.self_activation_history]
            data["other_self_activation"] = [sa[1] for sa in self.self_activation_history]
            # Pad with initial values for step 0
            data["focal_self_activation"].insert(0, 0)
            data["other_self_activation"].insert(0, 0)
            
        if self.social_influence_history:
            data["focal_social_influence"] = [si[0] for si in self.social_influence_history]
            data["other_social_influence"] = [si[1] for si in self.social_influence_history]
            # Pad with initial values for step 0
            data["focal_social_influence"].insert(0, 0)
            data["other_social_influence"].insert(0, 0)
            
        return pd.DataFrame(data)

def run_verification_tests(num_steps=1):
    """Run 16 verification rules, return results"""
    # Create verification instance
    verifier = TwoAgentVerification()
    
    # Define verification rules
    rules = [
        # Rule 1-4: Same Opinion Direction, Same Identity
        {"name": "Rule 1", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": 1, "focal_m": 0, "other_m": 0, "expected": "High convergence"},
        {"name": "Rule 2", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": 1, "focal_m": 0, "other_m": 1, "expected": "Moderate pull toward extremes"},
        {"name": "Rule 3", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": 1, "focal_m": 1, "other_m": 0, "expected": "Moderate pull of other toward extremes"},
        {"name": "Rule 4", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": 1, "focal_m": 1, "other_m": 1, "expected": "High polarization"},
        
        # Rule 5-8: Same Opinion Direction, Different Identity
        {"name": "Rule 5", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": -1, "focal_m": 0, "other_m": 0, "expected": "Moderate convergence"},
        {"name": "Rule 6", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": -1, "focal_m": 0, "other_m": 1, "expected": "Low pull toward extremes"},
        {"name": "Rule 7", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": -1, "focal_m": 1, "other_m": 0, "expected": "Low pull of other toward extremes"},
        {"name": "Rule 8", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": -1, "focal_m": 1, "other_m": 1, "expected": "Moderate polarization"},
        
        # Rule 9-12: Different Opinion Direction, Same Identity
        {"name": "Rule 9", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": 1, "focal_m": 0, "other_m": 0, "expected": "Very high convergence"},
        {"name": "Rule 10", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": 1, "focal_m": 0, "other_m": 1, "expected": "Moderate convergence/pull"},
        {"name": "Rule 11", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": 1, "focal_m": 1, "other_m": 0, "expected": "Low resistance and position holding"},
        {"name": "Rule 12", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": 1, "focal_m": 1, "other_m": 1, "expected": "Low mutual polarization"},
        
        # Rule 13-16: Different Opinion Direction, Different Identity
        {"name": "Rule 13", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": -1, "focal_m": 0, "other_m": 0, "expected": "Low convergence"},
        {"name": "Rule 14", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": -1, "focal_m": 0, "other_m": 1, "expected": "High pull toward other side"},
        {"name": "Rule 15", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": -1, "focal_m": 1, "other_m": 0, "expected": "High resistance and movement to extremes"},
        {"name": "Rule 16", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": -1, "focal_m": 1, "other_m": 1, "expected": "Very high mutual polarization"},
    ]
    
    # Store results
    results = []
    trajectory_data = {}
    
    # Test each rule
    for rule in rules:
        # Set up scenario
        verifier.setup_scenario(
            focal_opinion=rule["focal_op"],
            other_opinion=rule["other_op"],
            focal_identity=rule["focal_id"],
            other_identity=rule["other_id"],
            focal_moral=rule["focal_m"],
            other_moral=rule["other_m"]
        )
        
        # Execute simulation for specified number of steps
        final_result, _ = verifier.run_simulation(num_steps)
        
        # Get trajectory data
        trajectory_df = verifier.get_trajectory_dataframe()
        trajectory_data[rule["name"]] = trajectory_df
        
        # Store result
        results.append({
            "rule": rule["name"],
            "expected_effect": rule["expected"],
            "opinion_change": final_result["opinion_change"],
            "final_opinion": final_result["final_opinion"],
            "focal_op": rule["focal_op"],
            "other_op": rule["other_op"],
            "focal_id": rule["focal_id"],
            "other_id": rule["other_id"],
            "focal_m": rule["focal_m"],
            "other_m": rule["other_m"]
        })
    
    return pd.DataFrame(results), trajectory_data

def save_verification_results(results, output_dir, trajectory_data=None):
    """Save verification results to file"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV
    results.to_csv(os.path.join(output_dir, 'verification_results.csv'), index=False)
    
    # Save trajectory data if available
    if trajectory_data:
        trajectory_dir = os.path.join(output_dir, 'trajectories')
        os.makedirs(trajectory_dir, exist_ok=True)
        
        for rule_name, df in trajectory_data.items():
            # Clean rule name for filename
            filename = f"{rule_name.replace(' ', '_')}_trajectory.csv"
            df.to_csv(os.path.join(trajectory_dir, filename), index=False)
    
    # Return file path
    return os.path.join(output_dir, 'verification_results.csv')

def main(output_dir='results/verification/agent_interaction_verification', num_steps=10):
    """Main function"""
    print("Running agent interaction verification tests...")
    
    # Run verification tests with specified steps
    results, trajectory_data = run_verification_tests(num_steps=num_steps)
    
    # Print results
    print(results[["rule", "expected_effect", "opinion_change", "final_opinion"]])
    
    # Save results using the provided output directory
    result_path = save_verification_results(results, output_dir, trajectory_data)
    print(f"Results saved to: {result_path}")

    from polarization_triangle.visualization.verification_visualizer import plot_verification_results
    plot_verification_results(results, output_dir) 
    
    return results, trajectory_data

if __name__ == "__main__":
    main() 