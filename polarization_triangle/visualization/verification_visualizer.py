import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from matplotlib.patches import Patch

def plot_verification_results(results, output_dir='results/verification/agent_interaction_verification', k=1, cohesion_factor=None):
    """
    Plot verification results showing paired opinion changes for focal and neighbor agents.
    
    Parameters:
    results -- DataFrame with verification results including focal and neighbor changes
    output_dir -- Output directory
    k -- Number of neighbors for each main agent
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    
    plt.figure(figsize=(20, 10))
    
    # Set positions for bars
    num_rules = len(results)
    x = np.arange(num_rules)
    width = 0.35
    
    # Draw bar chart for focal and neighbor agents
    rects1 = plt.bar(x - width/2, results['focal_opinion_change'], width, label='Focal Agent', color='royalblue')
    # Add hatch for neighbor agent for distinction in this plot as well
    rects2 = plt.bar(x + width/2, results['neighbor_opinion_change'], width, label='Neighbor Agent (Hatched)', color='lightcoral', hatch='//')
    
    # Add rule labels and expected effect
    plt.xticks(x, results['rule'], rotation=60, ha='right')
    
    # Add y=0 horizontal line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Set title and labels
    if cohesion_factor is not None:
        plt.title(f'Agent Interaction Verification: Paired Opinion Changes (k={k} neighbors, cohesion_factor={cohesion_factor})', fontsize=16)
    else:
        plt.title(f'Agent Interaction Verification: Paired Opinion Changes (k={k} neighbors)', fontsize=16)
    plt.xlabel('Verification Rules', fontsize=14)
    plt.ylabel('Opinion Change', fontsize=14)
    
    # Add value labels to each bar
    def add_labels(rects, offset=0.01):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 
                     height + offset if height >= 0 else height - offset,
                     f'{height:.3f}',
                     ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            
    add_labels(rects1)
    add_labels(rects2)
    
    # Add expected effect text above each pair
    # Recalculate y-limits after plotting bars and labels
    y_min, y_max = plt.gca().get_ylim()
    # Adjust vertical position and font size for effect text
    text_y_position = y_max - (y_max - y_min) * 0.02 # Position text near the top
    
    for i, effect in enumerate(results['expected_effect']):
        plt.text(x[i], text_y_position, effect, 
                 ha='center', va='top', fontsize=5, rotation=0, # Use va='top', smaller font
                 bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.6))
                 
        # 添加数值标签在expected effect正下方
        focal_change = results['focal_opinion_change'].iloc[i]
        neighbor_change = results['neighbor_opinion_change'].iloc[i]
        plt.text(x[i], text_y_position - (y_max - y_min) * 0.06, 
                 f"F: {focal_change:.3f}\nN: {neighbor_change:.3f}", 
                 ha='center', va='top', fontsize=5, rotation=0,
                 bbox=dict(boxstyle="round,pad=0.2", fc="lightcyan", alpha=0.6))

    plt.legend(loc='upper left')
    
    # Add a bit more room at the top for text
    plt.ylim(y_min, y_max + (y_max - y_min) * 0.1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect slightly if needed
    
    # Save chart
    plot_path = os.path.join(output_dir, 'verification_results_paired_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Print save path
    print(f"Paired chart saved to: {plot_path}")
    
    # Create charts grouped by categories
    plot_by_categories(results, output_dir, k, cohesion_factor)

def plot_by_categories(results, output_dir, k=1, cohesion_factor=None):
    """Plot results grouped by different categories with paired bars"""
    # Group data
    groups = {
        'Same Opinion, Same Identity': results.iloc[0:4],
        'Same Opinion, Different Identity': results.iloc[4:8],
        'Different Opinion, Same Identity': results.iloc[8:12],
        'Different Opinion, Different Identity': results.iloc[12:16]
    }
    
    # 计算所有数据的y轴范围
    all_focal_changes = results['focal_opinion_change']
    all_neighbor_changes = results['neighbor_opinion_change']
    all_changes = pd.concat([all_focal_changes, all_neighbor_changes])
    global_ymin = all_changes.min()
    global_ymax = all_changes.max()
    
    # 为标签和文本留出额外空间
    y_padding = (global_ymax - global_ymin) * 0.15
    global_ymin = global_ymin - y_padding * 0.5
    global_ymax = global_ymax + y_padding
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 15)) # Slightly taller figure
    axs = axs.flatten()
    
    group_keys = list(groups.keys())
    moralization_legend_elements = [] # Collect unique moralization patches
    seen_moralization_keys = set()
    
    for i in range(4):
        ax = axs[i]
        group_name = group_keys[i]
        data = groups[group_name]
        
        ax.set_title(group_name, fontsize=14)
        category_legend = plot_category(ax, data)
        
        # 设置统一的y轴范围
        ax.set_ylim(global_ymin, global_ymax)
        
        # Collect unique legend patches for moralization status
        for handle in category_legend:
            key = (handle.get_facecolor(), handle.get_label())
            if key not in seen_moralization_keys:
                 moralization_legend_elements.append(handle)
                 seen_moralization_keys.add(key)

    # Add main title
    if cohesion_factor is not None:
        plt.suptitle(f'Agent Interaction Verification: Paired Opinion Changes by Scenario Category (k={k} neighbors, cohesion_factor={cohesion_factor})', fontsize=16)
    else:
        plt.suptitle(f'Agent Interaction Verification: Paired Opinion Changes by Scenario Category (k={k} neighbors)', fontsize=16)
    
    # Create separate legends: one for agent type, one for moralization color code
    focal_patch = Patch(facecolor='gray', label='Focal Agent')
    neighbor_patch = Patch(facecolor='gray', hatch='//', label='Neighbor Agent')
    agent_legend_handles = [focal_patch, neighbor_patch]
    
    # Place legends: Agent type top-right, Moralization code bottom-center
    fig.legend(handles=agent_legend_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    # Use unique moralization patches collected from subplots
    fig.legend(handles=moralization_legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), title="Moralization (Focal - Neighbor)")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout for legends/title
    
    # Save chart
    plot_path = os.path.join(output_dir, 'verification_results_by_category_paired.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Print save path
    print(f"Category paired chart saved to: {plot_path}")

def plot_category(ax, data):
    """Plot a category of data with paired bars on the given axis, using hatches for neighbors."""
    # Set positions for bars
    num_rules = len(data)
    x = np.arange(num_rules)
    width = 0.35
    
    # Draw bar chart for focal (no hatch) and neighbor (with hatch initially)
    rects1 = ax.bar(x - width/2, data['focal_opinion_change'], width, label='Focal')
    # Initialize neighbor bars with hatch, color will be set in loop
    rects2 = ax.bar(x + width/2, data['neighbor_opinion_change'], width, label='Neighbor', hatch='//') 
    
    # Add rule labels
    ax.set_xticks(x)
    ax.set_xticklabels(data['rule'], rotation=45, ha='right')
    
    # Add y=0 horizontal line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Color bars based on moralization status (same color for the pair)
    colors = {
        (0, 0): 'lightblue', (0, 1): 'lightgreen',
        (1, 0): 'orange', (1, 1): 'pink'
    }
    legend_labels = {
        (0, 0): 'NM - NM', (0, 1): 'NM - M',
        (1, 0): 'M - NM', (1, 1): 'M - M'
    }
    current_legend_elements = []
    seen_keys_in_category = set()
    
    for i in range(num_rules):
        focal_m = data.iloc[i]['focal_m']
        other_m = data.iloc[i]['other_m']
        color_key = (focal_m, other_m)
        bar_color = colors.get(color_key, 'gray')
        
        # Set color for focal bar
        rects1[i].set_color(bar_color)
        rects1[i].set_edgecolor('black') # Add consistent edge color
        
        # Set color, explicitly set hatch, and set edge color for neighbor bar
        rects2[i].set_color(bar_color)
        rects2[i].set_hatch('//') # Re-apply hatch just in case
        rects2[i].set_edgecolor('black') # Ensure hatch lines are visible
        
        # Add legend info if not already seen in this category
        if color_key not in seen_keys_in_category and color_key in legend_labels:
            # Use a representative patch for the legend (color only)
            current_legend_elements.append(Patch(facecolor=bar_color, label=legend_labels[color_key]))
            seen_keys_in_category.add(color_key)
            
    # Add value labels
    def add_cat_labels(rects, offset=0.01):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 
                    height + offset if height >= 0 else height - offset, 
                    f'{height:.3f}', 
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
            
    # Comment out the calls to remove value labels from category plots
    # add_cat_labels(rects1)
    # add_cat_labels(rects2)

    # Add expected effect text above each pair
    # 获取当前y轴范围用于定位文本
    y_min, y_max = ax.get_ylim()
    # Adjust vertical position and font size
    text_y_position = y_max - (y_max - y_min) * 0.02 # Position text near the top
    
    for i, effect in enumerate(data['expected_effect']):
        ax.text(x[i], text_y_position, effect, 
                 ha='center', va='top', fontsize=11, rotation=0, # Increased font size from 6.5 to 16
                 bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.6))
                 
        # 添加数值标签在expected effect正下方
        focal_change = data['focal_opinion_change'].iloc[i]
        neighbor_change = data['neighbor_opinion_change'].iloc[i]
        ax.text(x[i], text_y_position - (y_max - y_min) * 0.07, 
                f"F: {focal_change:.3f}\nN: {neighbor_change:.3f}", 
                ha='center', va='top', fontsize=9, rotation=0,
                bbox=dict(boxstyle="round,pad=0.2", fc="lightcyan", alpha=0.6))

    # Set y-axis label
    ax.set_ylabel('Opinion Change', fontsize=12)
    
    # Return legend elements specific to this category for global legend creation
    return current_legend_elements

def plot_trajectory(trajectory_data, output_dir='results/verification/trajectories'):
    """
    Plot opinion trajectories for all rules
    
    Parameters:
    trajectory_data -- Dictionary with trajectory DataFrames for each rule
    output_dir -- Output directory for trajectory plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Group rules by categories
    rule_categories = {
        'same_op_same_id': ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4'],
        'same_op_diff_id': ['Rule 5', 'Rule 6', 'Rule 7', 'Rule 8'],
        'diff_op_same_id': ['Rule 9', 'Rule 10', 'Rule 11', 'Rule 12'],
        'diff_op_diff_id': ['Rule 13', 'Rule 14', 'Rule 15', 'Rule 16']
    }
    
    category_titles = {
        'same_op_same_id': 'Same Opinion Direction, Same Identity',
        'same_op_diff_id': 'Same Opinion Direction, Different Identity',
        'diff_op_same_id': 'Different Opinion Direction, Same Identity',
        'diff_op_diff_id': 'Different Opinion Direction, Different Identity'
    }
    
    # Plot trajectories by category
    for category, rules in rule_categories.items():
        plt.figure(figsize=(12, 8))
        
        for rule in rules:
            if rule in trajectory_data:
                df = trajectory_data[rule]
                plt.plot(df['step'], df['focal_opinion'], label=rule)
        
        plt.title(f'Opinion Trajectories: {category_titles[category]}', fontsize=16)
        plt.xlabel('Step', fontsize=14)
        plt.ylabel('Focal Agent Opinion', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'trajectory_{category}.png'), dpi=300)
        plt.close()
        
        print(f"Trajectory plot saved to: {os.path.join(output_dir, f'trajectory_{category}.png')}")
    
    # Plot all trajectories in one figure
    plt.figure(figsize=(15, 10))
    
    for rule, df in trajectory_data.items():
        plt.plot(df['step'], df['focal_opinion'], label=rule)
    
    plt.title('Opinion Trajectories: All Rules', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Focal Agent Opinion', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_all_rules.png'), dpi=300)
    plt.close()
    
    print(f"Combined trajectory plot saved to: {os.path.join(output_dir, 'trajectory_all_rules.png')}")

def visualize_from_file(result_file='results/verification/verification_results.csv', trajectory_dir='results/verification/trajectories', k=1):
    """
    Load results from file and visualize
    
    Parameters:
    result_file -- Path to the results CSV file
    trajectory_dir -- Directory containing trajectory CSV files
    k -- Number of neighbors for each main agent
    """
    # Check if file exists
    if not os.path.exists(result_file):
        print(f"Error: Result file does not exist: {result_file}")
        return
    
    # Load results
    results = pd.read_csv(result_file)
    
    # Output directory
    output_dir = os.path.dirname(result_file)
    
    # Plot charts
    plot_verification_results(results, output_dir, k=k)
    
    # Load and plot trajectories if available
    if os.path.exists(trajectory_dir):
        trajectory_data = {}
        
        for file in os.listdir(trajectory_dir):
            if file.endswith('_trajectory.csv'):
                rule_name = file.replace('_trajectory.csv', '').replace('_', ' ')
                trajectory_data[rule_name] = pd.read_csv(os.path.join(trajectory_dir, file))
        
        if trajectory_data:
            plot_trajectory(trajectory_data, output_dir=trajectory_dir)

def main():
    """Main function"""
    # Default result file path
    result_file = 'results/verification/verification_results.csv'
    trajectory_dir = 'results/verification/trajectories'
    k = 1  # Default number of neighbors
    
    # Use command line argument if provided
    import sys
    if len(sys.argv) > 1:
        result_file = sys.argv[1]
    
    # Parse k value if provided as second argument
    if len(sys.argv) > 2:
        try:
            k = int(sys.argv[2])
            print(f"Using k={k} neighbors per main agent")
        except ValueError:
            print(f"Invalid k value: {sys.argv[2]}, using default k=1")
    
    # Visualize results
    visualize_from_file(result_file, trajectory_dir, k=k)

if __name__ == "__main__":
    main() 