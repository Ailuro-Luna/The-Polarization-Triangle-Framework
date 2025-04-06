import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def plot_verification_results(results, output_dir='results/verification'):
    """
    Plot verification results
    
    Parameters:
    results -- DataFrame with verification results
    output_dir -- Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a bar chart with categories
    plt.figure(figsize=(14, 8))
    
    # Set positions for bars
    x = np.arange(len(results))
    width = 0.8
    
    # Set colors based on opinion_change values
    colors = []
    for change in results['opinion_change']:
        if change > 0.05:  # Significant positive change
            colors.append('green')
        elif change < -0.05:  # Significant negative change
            colors.append('red')
        else:  # Small change
            colors.append('blue')
    
    # Draw bar chart
    bars = plt.bar(x, results['opinion_change'], width, color=colors)
    
    # Add rule labels
    plt.xticks(x, results['rule'], rotation=45, ha='right')
    
    # Add y=0 horizontal line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Set title and labels
    plt.title('Agent Interaction Verification: Opinion Changes of Focal Agent', fontsize=16)
    plt.xlabel('Verification Rules', fontsize=14)
    plt.ylabel('Opinion Change', fontsize=14)
    
    # Add value labels to each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                 height if height >= 0 else height - 0.02,
                 f'{height:.3f}',
                 ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Positive Change (More Extreme)'),
        Patch(facecolor='blue', label='Minor Change'),
        Patch(facecolor='red', label='Negative Change (More Neutral)')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    # Save chart
    plt.savefig(os.path.join(output_dir, 'verification_results_plot.png'), dpi=300)
    plt.close()
    
    # Print save path
    print(f"Chart saved to: {os.path.join(output_dir, 'verification_results_plot.png')}")
    
    # Create charts grouped by categories
    plot_by_categories(results, output_dir)

def plot_by_categories(results, output_dir):
    """Plot results grouped by different categories"""
    # Group data
    same_op_same_id = results.iloc[0:4]  # Rule 1-4
    same_op_diff_id = results.iloc[4:8]  # Rule 5-8
    diff_op_same_id = results.iloc[8:12]  # Rule 9-12
    diff_op_diff_id = results.iloc[12:16]  # Rule 13-16
    
    # Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # Set subplot titles
    axs[0, 0].set_title('Same Opinion Direction, Same Identity', fontsize=14)
    axs[0, 1].set_title('Same Opinion Direction, Different Identity', fontsize=14)
    axs[1, 0].set_title('Different Opinion Direction, Same Identity', fontsize=14)
    axs[1, 1].set_title('Different Opinion Direction, Different Identity', fontsize=14)
    
    # Draw subplots
    plot_category(axs[0, 0], same_op_same_id)
    plot_category(axs[0, 1], same_op_diff_id)
    plot_category(axs[1, 0], diff_op_same_id)
    plot_category(axs[1, 1], diff_op_diff_id)
    
    # Add main title
    plt.suptitle('Agent Interaction Verification: Opinion Changes by Scenario Category', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
    
    # Save chart
    plt.savefig(os.path.join(output_dir, 'verification_results_by_category.png'), dpi=300)
    plt.close()
    
    # Print save path
    print(f"Category chart saved to: {os.path.join(output_dir, 'verification_results_by_category.png')}")

def plot_category(ax, data):
    """Plot a category of data on the given axis"""
    # Set positions for bars
    x = np.arange(len(data))
    width = 0.6
    
    # Draw bar chart
    bars = ax.bar(x, data['opinion_change'], width)
    
    # Add rule labels
    ax.set_xticks(x)
    ax.set_xticklabels(data['rule'], rotation=45, ha='right')
    
    # Add y=0 horizontal line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Color each bar based on moralization status
    for i, bar in enumerate(bars):
        focal_m = data.iloc[i]['focal_m']
        other_m = data.iloc[i]['other_m']
        
        if focal_m == 0 and other_m == 0:
            bar.set_color('lightblue')
        elif focal_m == 0 and other_m == 1:
            bar.set_color('lightgreen')
        elif focal_m == 1 and other_m == 0:
            bar.set_color('orange')
        else:  # focal_m == 1 and other_m == 1
            bar.set_color('pink')
        
        # Add value labels
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 
                height if height >= 0 else height - 0.02,
                f'{height:.3f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Non-moralized - Non-moralized'),
        Patch(facecolor='lightgreen', label='Non-moralized - Moralized'),
        Patch(facecolor='orange', label='Moralized - Non-moralized'),
        Patch(facecolor='pink', label='Moralized - Moralized')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    
    # Set y-axis label
    ax.set_ylabel('Opinion Change', fontsize=12)

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

def visualize_from_file(result_file='results/verification/verification_results.csv', trajectory_dir='results/verification/trajectories'):
    """Load results from file and visualize"""
    # Check if file exists
    if not os.path.exists(result_file):
        print(f"Error: Result file does not exist: {result_file}")
        return
    
    # Load results
    results = pd.read_csv(result_file)
    
    # Output directory
    output_dir = os.path.dirname(result_file)
    
    # Plot charts
    plot_verification_results(results, output_dir)
    
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
    
    # Use command line argument if provided
    import sys
    if len(sys.argv) > 1:
        result_file = sys.argv[1]
    
    # Visualize results
    visualize_from_file(result_file, trajectory_dir)

if __name__ == "__main__":
    main() 