import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import os
from pathlib import Path

from polarization_triangle.core.config import SimulationConfig, lfr_config
from polarization_triangle.core.dynamics import (
    calculate_perceived_opinion_func,
    calculate_relationship_coefficient_func,
    calculate_same_identity_sigma_func
)
from polarization_triangle.visualization.network_viz import draw_network

class MultiAgentVerification:
    """
    A simplified Simulation class for verifying interactions between agents
    with configurable number of neighbors for each main agent
    """
    def __init__(self, k=1, config=None):
        """
        Initialize verification environment
        
        Parameters:
        k -- Number of neighbors for each main agent (focal and neighbor)
        config -- Simulation configuration
        """
        # Use LFR config
        self.config = config or lfr_config
        
        # Set number of neighbors per main agent
        self.k = k
        
        # Calculate total number of agents (2 main agents + k neighbors for each)
        self.num_agents = 2 + 2 * k
        
        # Create network
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_agents))
        
        # Add edges: 
        # - Edge between main agents (0-1)
        # - Edges from focal agent to its neighbors (0 to 2, 3, ..., k+1)
        # - Edges from neighbor agent to its neighbors (1 to k+2, k+3, ..., 2k+1)
        
        # Main agents connection
        edges = [(0, 1)]
        
        # Focal agent to its neighbors
        for i in range(k):
            edges.append((0, 2 + i))
            
        # Neighbor agent to its neighbors
        for i in range(k):
            edges.append((1, 2 + k + i))
            
        self.graph.add_edges_from(edges)
        
        # Create adjacency matrix
        self.adj_matrix = np.zeros((self.num_agents, self.num_agents))
        for i, j in edges:
            self.adj_matrix[i, j] = 1
            self.adj_matrix[j, i] = 1
        
        # Create node positions for visualization
        self.pos = {}
        
        # Main agents
        self.pos[0] = np.array([-2, 0])  # focal agent
        self.pos[1] = np.array([2, 0])   # neighbor agent
        
        # Focal agent's neighbors (arrange in arc on left side)
        for i in range(k):
            angle = (i / k - 0.5) * np.pi  # Spread evenly in a half-circle
            radius = 2
            self.pos[2 + i] = np.array([-2, 0]) + radius * np.array([np.cos(angle), np.sin(angle)])
            
        # Neighbor agent's neighbors (arrange in arc on right side)
        for i in range(k):
            angle = (i / k - 0.5) * np.pi  # Spread evenly in a half-circle
            radius = 2
            self.pos[2 + k + i] = np.array([2, 0]) + radius * np.array([np.cos(angle + np.pi), np.sin(angle + np.pi)])
        
        # Get model parameters from config
        self.delta = self.config.delta  # Opinion decay rate
        self.u = np.ones(self.num_agents) * self.config.u  # Opinion activation coefficient
        self.alpha = np.ones(self.num_agents) * self.config.alpha  # Self-activation coefficient
        self.beta = self.config.beta  # Social influence coefficient
        self.gamma = np.ones(self.num_agents) * self.config.gamma  # Moralization impact coefficient
        
        # Store neighbor lists for each agent
        self.neighbors_list = [[] for _ in range(self.num_agents)]
        
        # Main agents: focal and neighbor
        self.neighbors_list[0] = [1] + list(range(2, 2 + k))
        self.neighbors_list[1] = [0] + list(range(2 + k, 2 + 2*k))
        
        # Focal agent's neighbors (only connected to focal)
        for i in range(k):
            self.neighbors_list[2 + i] = [0]
            
        # Neighbor agent's neighbors (only connected to neighbor)
        for i in range(k):
            self.neighbors_list[2 + k + i] = [1]
        
        # Create CSR format neighbor representation
        neighbors_indices = []
        neighbors_indptr = [0]
        
        for i in range(self.num_agents):
            neighbors_indices.extend(self.neighbors_list[i])
            neighbors_indptr.append(len(neighbors_indices))
            
        self.neighbors_indices = np.array(neighbors_indices, dtype=np.int32)
        self.neighbors_indptr = np.array(neighbors_indptr, dtype=np.int32)
        
        # Initialize trajectory storage
        self.opinion_trajectory = []
        self.self_activation_history = []
        self.social_influence_history = []
        
        print(f"Model Parameters: delta={self.delta}, u={self.u[0]}, alpha={self.alpha[0]}, beta={self.beta}, gamma={self.gamma[0]}")
        print(f"Network: 2 main agents + {k} neighbors each = {self.num_agents} total agents")
    
    def setup_scenario(self, focal_opinion, other_opinion, focal_identity, other_identity, 
                       focal_moral, other_moral):
        """Set up a specific scenario state with multiple agents"""
        # Main agents (focal and neighbor)
        focal_agent_opinion = focal_opinion
        neighbor_agent_opinion = other_opinion
        focal_agent_identity = focal_identity
        neighbor_agent_identity = other_identity
        focal_agent_moral = focal_moral
        neighbor_agent_moral = other_moral
        
        # Initialize arrays
        self.opinions = np.zeros(self.num_agents, dtype=np.float64)
        self.identities = np.zeros(self.num_agents, dtype=np.int32)
        self.morals = np.zeros(self.num_agents, dtype=np.int32)
        
        # Set main agents' attributes
        self.opinions[0] = focal_agent_opinion
        self.opinions[1] = neighbor_agent_opinion
        self.identities[0] = focal_agent_identity
        self.identities[1] = neighbor_agent_identity
        self.morals[0] = focal_agent_moral
        self.morals[1] = neighbor_agent_moral
        
        # Set focal agent's neighbors' attributes (copy from focal agent)
        for i in range(self.k):
            self.opinions[2 + i] = focal_agent_opinion
            self.identities[2 + i] = focal_agent_identity
            self.morals[2 + i] = focal_agent_moral
            
        # Set neighbor agent's neighbors' attributes (copy from neighbor agent)
        for i in range(self.k):
            self.opinions[2 + self.k + i] = neighbor_agent_opinion
            self.identities[2 + self.k + i] = neighbor_agent_identity
            self.morals[2 + self.k + i] = neighbor_agent_moral
        
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
        # Calculate same identity sigma 
        sigma_same_identity = calculate_same_identity_sigma_func(
            self.opinions,
            self.morals, 
            self.identities,
            self.neighbors_indices,
            self.neighbors_indptr,
            i
        )
        
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
        
        # Prepare results dictionary
        result = {
            "focal_opinion_change": self.opinions[0] - self.initial_opinions[0],
            "focal_final_opinion": self.opinions[0],
            "focal_self_activation": self_activation_values[0],
            "focal_social_influence": social_influence_values[0],
            "neighbor_opinion_change": self.opinions[1] - self.initial_opinions[1],
            "neighbor_final_opinion": self.opinions[1],
            "neighbor_self_activation": self_activation_values[1],
            "neighbor_social_influence": social_influence_values[1],
        }
        
        # Add data for all neighbors
        for i in range(self.k):
            focal_neighbor_idx = 2 + i
            result[f"focal_neighbor_{i}_opinion_change"] = self.opinions[focal_neighbor_idx] - self.initial_opinions[focal_neighbor_idx]
            result[f"focal_neighbor_{i}_final_opinion"] = self.opinions[focal_neighbor_idx]
            
        for i in range(self.k):
            neighbor_neighbor_idx = 2 + self.k + i
            result[f"neighbor_neighbor_{i}_opinion_change"] = self.opinions[neighbor_neighbor_idx] - self.initial_opinions[neighbor_neighbor_idx]
            result[f"neighbor_neighbor_{i}_final_opinion"] = self.opinions[neighbor_neighbor_idx]
        
        return result
    
    def run_simulation(self, num_steps=1):
        """Run simulation for multiple steps"""
        results_over_time = []
        
        # Record initial state for all agents
        initial_opinions = self.initial_opinions.copy()
        
        for step_num in range(num_steps):
            step_result = self.step()
            
            # Add step number to result
            step_result["step"] = step_num + 1
            results_over_time.append(step_result)
        
        # Calculate final results
        final_result = {
            "focal_opinion_change": self.opinions[0] - initial_opinions[0],
            "focal_final_opinion": self.opinions[0],
            "neighbor_opinion_change": self.opinions[1] - initial_opinions[1],
            "neighbor_final_opinion": self.opinions[1],
            "num_steps": num_steps,
            "trajectory": np.array(self.opinion_trajectory),
            "self_activation": np.array(self.self_activation_history),
            "social_influence": np.array(self.social_influence_history)
        }
        
        # Add data for all neighbors
        for i in range(self.k):
            focal_neighbor_idx = 2 + i
            final_result[f"focal_neighbor_{i}_opinion_change"] = self.opinions[focal_neighbor_idx] - initial_opinions[focal_neighbor_idx]
            final_result[f"focal_neighbor_{i}_final_opinion"] = self.opinions[focal_neighbor_idx]
            
        for i in range(self.k):
            neighbor_neighbor_idx = 2 + self.k + i
            final_result[f"neighbor_neighbor_{i}_opinion_change"] = self.opinions[neighbor_neighbor_idx] - initial_opinions[neighbor_neighbor_idx]
            final_result[f"neighbor_neighbor_{i}_final_opinion"] = self.opinions[neighbor_neighbor_idx]
        
        return final_result, results_over_time
    
    def get_trajectory_dataframe(self):
        """Convert trajectory to DataFrame for easier analysis"""
        steps = len(self.opinion_trajectory)
        
        data = {
            "step": list(range(steps)),
            "focal_opinion": [op[0] for op in self.opinion_trajectory],
            "neighbor_opinion": [op[1] for op in self.opinion_trajectory],
        }
        
        # Add data for all neighbors
        for i in range(self.k):
            focal_neighbor_idx = 2 + i
            data[f"focal_neighbor_{i}_opinion"] = [op[focal_neighbor_idx] for op in self.opinion_trajectory]
            
        for i in range(self.k):
            neighbor_neighbor_idx = 2 + self.k + i
            data[f"neighbor_neighbor_{i}_opinion"] = [op[neighbor_neighbor_idx] for op in self.opinion_trajectory]
        
        # Add self-activation and social influence if available
        if self.self_activation_history:
            data["focal_self_activation"] = [sa[0] for sa in self.self_activation_history]
            data["neighbor_self_activation"] = [sa[1] for sa in self.self_activation_history]
            
            # Add data for all neighbors
            for i in range(self.k):
                focal_neighbor_idx = 2 + i
                data[f"focal_neighbor_{i}_self_activation"] = [sa[focal_neighbor_idx] for sa in self.self_activation_history]
                
            for i in range(self.k):
                neighbor_neighbor_idx = 2 + self.k + i
                data[f"neighbor_neighbor_{i}_self_activation"] = [sa[neighbor_neighbor_idx] for sa in self.self_activation_history]
            
            # Pad with initial values for step 0
            data["focal_self_activation"].insert(0, 0)
            data["neighbor_self_activation"].insert(0, 0)
            
            for i in range(self.k):
                data[f"focal_neighbor_{i}_self_activation"].insert(0, 0)
            for i in range(self.k):
                data[f"neighbor_neighbor_{i}_self_activation"].insert(0, 0)
            
        if self.social_influence_history:
            data["focal_social_influence"] = [si[0] for si in self.social_influence_history]
            data["neighbor_social_influence"] = [si[1] for si in self.social_influence_history]
            
            # Add data for all neighbors
            for i in range(self.k):
                focal_neighbor_idx = 2 + i
                data[f"focal_neighbor_{i}_social_influence"] = [si[focal_neighbor_idx] for si in self.social_influence_history]
                
            for i in range(self.k):
                neighbor_neighbor_idx = 2 + self.k + i
                data[f"neighbor_neighbor_{i}_social_influence"] = [si[neighbor_neighbor_idx] for si in self.social_influence_history]
            
            # Pad with initial values for step 0
            data["focal_social_influence"].insert(0, 0)
            data["neighbor_social_influence"].insert(0, 0)
            
            for i in range(self.k):
                data[f"focal_neighbor_{i}_social_influence"].insert(0, 0)
            for i in range(self.k):
                data[f"neighbor_neighbor_{i}_social_influence"].insert(0, 0)
            
        return pd.DataFrame(data)
    
    def visualize_network(self, output_dir, rule_name):
        """Visualize the agent network with different attributes"""
        os.makedirs(output_dir, exist_ok=True)
        viz_dir = os.path.join(output_dir, 'network_viz')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 获取绝对文件路径
        opinion_path = os.path.join(os.path.abspath(viz_dir), f"{rule_name}_opinion_network.png")
        identity_path = os.path.join(os.path.abspath(viz_dir), f"{rule_name}_identity_network.png")
        morality_path = os.path.join(os.path.abspath(viz_dir), f"{rule_name}_morality_network.png")
        
        # 使用matplotlib直接创建可视化，而不是依赖draw_network函数
        
        # 1. 可视化意见
        fig, ax = plt.subplots(figsize=(10, 8))
        # 创建颜色映射
        cmap = plt.cm.coolwarm
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        node_colors = [cmap(norm(op)) for op in self.opinions]
        
        # 使用self.pos定位
        nx.draw(self.graph, pos=self.pos, node_color=node_colors,
                with_labels=False, node_size=600, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        
        # 添加节点标签
        labels = {}
        labels[0] = f"Focal\n{self.opinions[0]:.2f}"
        labels[1] = f"Neighbor\n{self.opinions[1]:.2f}"
        
        for i in range(self.k):
            focal_neighbor_idx = 2 + i
            labels[focal_neighbor_idx] = f"F-N{i}\n{self.opinions[focal_neighbor_idx]:.2f}"
            
        for i in range(self.k):
            neighbor_neighbor_idx = 2 + self.k + i
            labels[neighbor_neighbor_idx] = f"N-N{i}\n{self.opinions[neighbor_neighbor_idx]:.2f}"
        
        nx.draw_networkx_labels(self.graph, pos=self.pos, labels=labels, font_size=8)
        
        ax.set_title(f"Rule {rule_name}: Agent Opinions")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Opinion")
        plt.savefig(opinion_path)
        plt.close()
        
        # 2. 可视化身份
        fig, ax = plt.subplots(figsize=(10, 8))
        node_colors = ['#e41a1c' if iden == 1 else '#377eb8' for iden in self.identities]
        nx.draw(self.graph, pos=self.pos, node_color=node_colors,
                with_labels=False, node_size=600, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        
        # 添加节点标签
        labels = {}
        labels[0] = f"Focal\nID: {self.identities[0]}"
        labels[1] = f"Neighbor\nID: {self.identities[1]}"
        
        for i in range(self.k):
            focal_neighbor_idx = 2 + i
            labels[focal_neighbor_idx] = f"F-N{i}\nID: {self.identities[focal_neighbor_idx]}"
            
        for i in range(self.k):
            neighbor_neighbor_idx = 2 + self.k + i
            labels[neighbor_neighbor_idx] = f"N-N{i}\nID: {self.identities[neighbor_neighbor_idx]}"
        
        nx.draw_networkx_labels(self.graph, pos=self.pos, labels=labels, font_size=8)
        
        ax.set_title(f"Rule {rule_name}: Agent Identities")
        patches = [
            mpatches.Patch(color='#e41a1c', label='Identity: 1'),
            mpatches.Patch(color='#377eb8', label='Identity: -1')
        ]
        ax.legend(handles=patches, loc='upper right', title="Identity")
        plt.savefig(identity_path)
        plt.close()
        
        # 3. 可视化道德性
        fig, ax = plt.subplots(figsize=(10, 8))
        node_colors = ['#1a9850' if m == 1 else '#d73027' for m in self.morals]
        nx.draw(self.graph, pos=self.pos, node_color=node_colors,
                with_labels=False, node_size=600, alpha=0.8,
                edge_color="#AAAAAA", ax=ax)
        
        # 添加节点标签
        labels = {}
        labels[0] = f"Focal\nM: {self.morals[0]}"
        labels[1] = f"Neighbor\nM: {self.morals[1]}"
        
        for i in range(self.k):
            focal_neighbor_idx = 2 + i
            labels[focal_neighbor_idx] = f"F-N{i}\nM: {self.morals[focal_neighbor_idx]}"
            
        for i in range(self.k):
            neighbor_neighbor_idx = 2 + self.k + i
            labels[neighbor_neighbor_idx] = f"N-N{i}\nM: {self.morals[neighbor_neighbor_idx]}"
        
        nx.draw_networkx_labels(self.graph, pos=self.pos, labels=labels, font_size=8)
        
        ax.set_title(f"Rule {rule_name}: Agent Moralities")
        patches = [
            mpatches.Patch(color='#1a9850', label='Morality: 1'),
            mpatches.Patch(color='#d73027', label='Morality: 0')
        ]
        ax.legend(handles=patches, loc='upper right', title="Morality")
        plt.savefig(morality_path)
        plt.close()
        
        print(f"Network visualizations for Rule {rule_name} saved to {viz_dir}")
        return viz_dir

def run_verification_tests(num_steps=1, k=1):
    """
    Run 16 verification rules, return results
    
    Parameters:
    num_steps -- Number of simulation steps to run
    k -- Number of neighbors for each main agent
    """
    # Create verification instance
    verifier = MultiAgentVerification(k=k)
    
    # Define verification rules
    rules = [
        # Rule 1-4: Same Opinion Direction, Same Identity
        {"name": "Rule 1", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": 1, "focal_m": 0, "other_m": 0, "expected": "High Convergence"},
        {"name": "Rule 2", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": 1, "focal_m": 0, "other_m": 1, "expected": "Moderate Radicalize"},
        {"name": "Rule 3", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": 1, "focal_m": 1, "other_m": 0, "expected": "no change"},
        {"name": "Rule 4", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": 1, "focal_m": 1, "other_m": 1, "expected": "High Radicalize"},
        
        # Rule 5-8: Same Opinion Direction, Different Identity
        {"name": "Rule 5", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": -1, "focal_m": 0, "other_m": 0, "expected": "Moderate Convergence"},
        {"name": "Rule 6", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": -1, "focal_m": 0, "other_m": 1, "expected": "Low Radicalize"},
        {"name": "Rule 7", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": -1, "focal_m": 1, "other_m": 0, "expected": "No Change"},
        {"name": "Rule 8", "focal_op": 0.25, "other_op": 0.75, "focal_id": 1, "other_id": -1, "focal_m": 1, "other_m": 1, "expected": "Moderate Radicalize"},
        
        # Rule 9-12: Different Opinion Direction, Same Identity
        {"name": "Rule 9", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": 1, "focal_m": 0, "other_m": 0, "expected": "Very High Convergence"},
        {"name": "Rule 10", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": 1, "focal_m": 0, "other_m": 1, "expected": "Moderate Convergence"},
        {"name": "Rule 11", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": 1, "focal_m": 1, "other_m": 0, "expected": "Resist"},
        {"name": "Rule 12", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": 1, "focal_m": 1, "other_m": 1, "expected": "Low Radicalize"},
        
        # Rule 13-16: Different Opinion Direction, Different Identity
        {"name": "Rule 13", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": -1, "focal_m": 0, "other_m": 0, "expected": "Low Convergence"},
        {"name": "Rule 14", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": -1, "focal_m": 0, "other_m": 1, "expected": "Moderate Convergence"},
        {"name": "Rule 15", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": -1, "focal_m": 1, "other_m": 0, "expected": "Resist"},
        {"name": "Rule 16", "focal_op": 0.25, "other_op": -0.75, "focal_id": 1, "other_id": -1, "focal_m": 1, "other_m": 1, "expected": "Very High Radicalize"},
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
        
        # Visualize network for this rule
        viz_dir = verifier.visualize_network(output_dir='results/verification/agent_interaction_verification', rule_name=rule["name"])
        
        # Prepare result data
        result_data = {
            "rule": rule["name"],
            "expected_effect": rule["expected"],
            "focal_opinion_change": final_result["focal_opinion_change"],
            "focal_final_opinion": final_result["focal_final_opinion"],
            "neighbor_opinion_change": final_result["neighbor_opinion_change"],
            "neighbor_final_opinion": final_result["neighbor_final_opinion"],
            "focal_op": rule["focal_op"],
            "other_op": rule["other_op"],
            "focal_id": rule["focal_id"],
            "other_id": rule["other_id"],
            "focal_m": rule["focal_m"],
            "other_m": rule["other_m"]
        }
        
        # Add data for all neighbors
        for i in range(verifier.k):
            result_data[f"focal_neighbor_{i}_opinion_change"] = final_result[f"focal_neighbor_{i}_opinion_change"]
            result_data[f"focal_neighbor_{i}_final_opinion"] = final_result[f"focal_neighbor_{i}_final_opinion"]
            
        for i in range(verifier.k):
            result_data[f"neighbor_neighbor_{i}_opinion_change"] = final_result[f"neighbor_neighbor_{i}_opinion_change"]
            result_data[f"neighbor_neighbor_{i}_final_opinion"] = final_result[f"neighbor_neighbor_{i}_final_opinion"]
        
        results.append(result_data)
    
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

def main(k=1, output_dir='results/verification/agent_interaction_verification', num_steps=1):
    """
    Main function
    
    Parameters:
    k -- Number of neighbors for each main agent
    output_dir -- Directory to save results
    num_steps -- Number of simulation steps to run
    """
    print(f"Running agent interaction verification tests with {2*(k+1)}-agent system ({k} neighbors per main agent)...")
    
    # Run verification tests with specified steps and neighbors
    results, trajectory_data = run_verification_tests(num_steps=num_steps, k=k)
    
    # Print results
    print(results[["rule", "expected_effect", "focal_opinion_change", "neighbor_opinion_change", "focal_final_opinion", "neighbor_final_opinion"]])
    
    # Save results using the provided output directory
    result_path = save_verification_results(results, output_dir, trajectory_data)
    print(f"Results saved to: {result_path}")

    from polarization_triangle.visualization.verification_visualizer import plot_verification_results
    plot_verification_results(results, output_dir, k=k) 
    
    return results, trajectory_data

if __name__ == "__main__":
    # Default k=1 (1 neighbor per main agent, total 4 agents)
    # Can be overridden with command line arguments
    import sys
    k = 1  # default
    if len(sys.argv) > 1:
        try:
            k = int(sys.argv[1])
            print(f"Using k={k} neighbors per main agent")
        except ValueError:
            print(f"Invalid k value: {sys.argv[1]}, using default k=1")
    
    main(k=k) 