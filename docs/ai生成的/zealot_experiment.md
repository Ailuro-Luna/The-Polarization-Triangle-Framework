# Zealot Experiment Documentation

## Overview

The `zealot_experiment.py` is a simulation module designed to study the influence of zealots (agents with fixed extreme opinions) on opinion dynamics in social networks. It allows for comparing different zealot distribution strategies and analyzing their impact on opinion formation, polarization, and community dynamics.

## Experimental Setup

The module implements a framework to compare four different zealot distribution strategies:

1. **No Zealots**: Baseline simulation with no zealots
2. **Clustered Zealots**: Zealots concentrated within the same communities
3. **Random Zealots**: Zealots distributed randomly across the network
4. **High-Degree Zealots**: Zealots placed at the most connected nodes in the network

## Parameters

The main experiment function `run_zealot_experiment()` accepts the following parameters:

- `steps`: Number of simulation steps (default: 500)
- `initial_scale`: Initial opinion scaling factor (default: 0.1) - simulates neutral attitudes towards a new topic
- `num_zealots`: Number of zealots to introduce (default: 50)
- `seed`: Random seed for reproducibility (default: 42)
- `output_dir`: Results output directory (default: None, uses "results/zealot_experiment")
- `morality_rate`: Proportion of non-zealot agents that are moralized (default: 0.0)
- `zealot_morality`: Whether all zealots are moralized (default: False)
- `identity_clustered`: Whether to cluster agents by identity during initialization (default: False)
- `zealot_mode`: Zealot initialization mode - "none", "clustered", "random", "high-degree" (default: None, runs all modes)

## Generated Visualizations

The experiment generates a comprehensive set of visualizations stored in the `results/zealot_experiment/` directory:

### Network Visualizations

- **Network Visualizations**: Network graphs showing the distribution of zealots and opinions
  - `{mode}_network.png`: Network visualization with zealots highlighted in red
  - `{mode}_opinion_network.png`: Opinion distribution across the network

### Opinion Distribution Visualizations

- **Opinion Heatmaps**: Shows the evolution of opinion distribution over time
  - `{mode}_heatmap.png`: Heatmap of opinion distribution over time

### Interaction Rule Visualizations

- **Rule Usage Statistics**: Visualizations of interaction rule usage
  - `{mode}_interaction_types.png`: Rule usage over time
  - `{mode}_interaction_types_cumulative.png`: Cumulative rule usage
  - `{mode}_interaction_types_stats.txt`: Detailed statistics about rule usage

### Activation Component Visualizations

Located in the `activation_components/` subdirectory:

- **Activation Components**: Visualizations related to self-activation and social influence
  - `{mode}_activation_components.png`: Scatter plot of activation components
  - `{mode}_activation_history.png`: Activation components over time
  - `{mode}_activation_heatmap.png`: Heatmap of activation components
  - `{mode}_activation_trajectory.png`: Trajectory of activation for selected agents
  - `{mode}_activation_data.csv`: Raw data of activation components

### Opinion Statistics Visualizations

Located in the `statistics/` subdirectory:

- **Individual Statistics**:
  - `{mode}_mean_opinions.png`: Mean opinion and absolute opinion over time
  - `{mode}_non_zealot_variance.png`: Variance of non-zealot opinions
  - `{mode}_cluster_variance.png`: Mean variance within clusters
  - `{mode}_negative_opinion_stats.png`: Statistics for negative opinions
  - `{mode}_positive_opinion_stats.png`: Statistics for positive opinions
  - `{mode}_community_variances.png`: Opinion variance for each community

- **Comparative Statistics**:
  - `comparison_mean_opinions.png`: Comparison of mean opinions across simulations
  - `comparison_mean_abs_opinions.png`: Comparison of mean absolute opinions
  - `comparison_non_zealot_variance.png`: Comparison of non-zealot variance
  - `comparison_cluster_variance.png`: Comparison of cluster variance
  - `comparison_negative_counts.png`: Comparison of negative opinion counts
  - `comparison_negative_means.png`: Comparison of negative opinion means
  - `comparison_positive_counts.png`: Comparison of positive opinion counts
  - `comparison_positive_means.png`: Comparison of positive opinion means

## Data Statistics and Analysis

The experiment collects and analyzes the following data:

### Opinion Statistics
- Mean opinion and mean absolute opinion over time
- Opinion variance excluding zealots
- Mean opinion variance within clusters
- Community-level opinion variance
- Counts and means of positive and negative opinions

### Interaction Statistics
- Usage frequency of each interaction rule
- Percentage of each interaction type
- Cumulative rule usage over time

### Activation Component Statistics
- Self-activation components
- Social influence components
- Total activation levels

## CSV Data Files

Raw data is exported to CSV files for further analysis:

- `{mode}_opinion_stats.csv`: All opinion statistics over time
- `{mode}_community_variances.csv`: Community-level variance data
- `comparison_opinion_stats.csv`: Combined statistics for all simulations
- `{mode}_activation_data.csv`: Activation component data

## Example Usage

To run the default zealot experiment:

```python
from polarization_triangle.experiments.zealot_experiment import run_zealot_experiment

# Run with default parameters
run_zealot_experiment()

# Run with custom parameters
run_zealot_experiment(
    steps=1000,           # Run for 1000 steps
    initial_scale=0.2,    # Initial opinions scaled to 20%
    num_zealots=20,       # 20 zealots
    seed=42,              # Fixed random seed
    morality_rate=0.3,    # 30% of non-zealots are moralized
    zealot_morality=True, # All zealots are moralized
    identity_clustered=True, # Cluster agents by identity
    zealot_mode="clustered"  # Only run clustered zealot mode
)

# Run only specific zealot modes
run_zealot_experiment(
    zealot_mode="random",  # Only run random zealot distribution
    num_zealots=30,
    steps=500
)
```

## Implementation Details

The simulation implements zealots by fixing their opinions to extreme values (1.0) and updating the network at each time step. The experiment leverages the Polarization Triangle Framework's core components including:

- Network generation and community detection
- Opinion dynamics simulation
- Identity and morality interactions
- Multi-level statistical analysis
- Comprehensive visualization toolkit

This experiment helps understand how different zealot distribution strategies affect opinion dynamics, polarization processes, and community structures in social networks. 