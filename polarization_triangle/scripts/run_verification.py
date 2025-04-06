import os
import argparse
from pathlib import Path
import sys
sys.path.append("../..")  # Add parent directory to path

from polarization_triangle.verification.agent_interaction_verification import run_verification_tests, save_verification_results
from polarization_triangle.visualization.verification_visualizer import plot_verification_results, plot_trajectory

def main():
    """Run verification tests and visualize results"""
    parser = argparse.ArgumentParser(description='Run agent interaction verification tests')
    parser.add_argument('--output-dir', type=str, default='results/verification',
                       help='Output directory for results')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of steps to simulate for each rule')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only visualize existing results without running simulations')
    args = parser.parse_args()
    
    # Ensure output directory exists (create full path including parent)
    output_dir = args.output_dir
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if args.visualize_only:
        results_file = os.path.join(output_dir, 'verification_results.csv')
        trajectories_dir = os.path.join(output_dir, 'trajectories')
        
        if os.path.exists(results_file):
            import pandas as pd
            results = pd.read_csv(results_file)
            
            # Load trajectory data
            trajectory_data = {}
            if os.path.exists(trajectories_dir):
                for file in os.listdir(trajectories_dir):
                    if file.endswith('_trajectory.csv'):
                        rule_name = file.replace('_trajectory.csv', '').replace('_', ' ')
                        trajectory_data[rule_name] = pd.read_csv(os.path.join(trajectories_dir, file))
            
            # Visualize results
            plot_verification_results(results, output_dir)
            if trajectory_data:
                plot_trajectory(trajectory_data, trajectories_dir)
                
            print(f"Visualizations updated in {output_dir}")
        else:
            print(f"Error: Results file not found at {results_file}")
        
        return
    
    print(f"Running agent interaction verification tests with {args.steps} steps")
    print(f"Results will be saved to: {output_dir}")
    
    # Run verification tests with specified number of steps
    results, trajectory_data = run_verification_tests(num_steps=args.steps)
    
    # Print results
    print("\nVerification Results Summary:")
    print(results[["rule", "expected_effect", "opinion_change"]].to_string(index=False))
    
    # Save results
    result_path = save_verification_results(results, trajectory_data, output_dir)
    print(f"Results saved to: {result_path}")
    
    # Visualize results
    plot_verification_results(results, output_dir)
    trajectories_dir = os.path.join(output_dir, 'trajectories')
    plot_trajectory(trajectory_data, trajectories_dir)
    
    print(f"Verification tests completed with {args.steps} steps!")
    return results, trajectory_data

if __name__ == "__main__":
    main() 