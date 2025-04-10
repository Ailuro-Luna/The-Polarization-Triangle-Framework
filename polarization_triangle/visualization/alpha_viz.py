import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


class AlphaVisualizer:
    """
    Visualizer for alpha verification analysis.
    Handles visualization of dzdt functions, time evolution, bifurcation diagrams,
    and critical alpha value analysis.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        output_dir : str
            Output directory for visualizations
        """
        self.output_dir = output_dir or os.path.join(
            Path(__file__).parent.parent.parent, 'results', 'verification'
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_dzdt(self, z_values, dzdt_values_by_alpha, alpha_values, equilibria_by_alpha):
        """
        Visualize dzdt function for different alpha values
        
        Parameters:
        -----------
        z_values : array
            Array of z values used for calculations
        dzdt_values_by_alpha : dict
            Dictionary mapping alpha values to dzdt values
        alpha_values : array
            Array of alpha values to visualize
        equilibria_by_alpha : dict
            Dictionary mapping alpha values to equilibrium points
            
        Returns:
        --------
        fig, ax : matplotlib figure objects
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for alpha in alpha_values:
            dzdt_values = dzdt_values_by_alpha[alpha]
            ax.plot(z_values, dzdt_values, label=f'alpha = {alpha:.1f}')
            
            # Mark equilibrium points
            if alpha in equilibria_by_alpha:
                for eq in equilibria_by_alpha[alpha]:
                    ax.plot(eq, 0, 'o', markersize=6)
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('z (Opinion Value)', fontsize=14)
        ax.set_ylabel('dz/dt', fontsize=14)
        ax.set_title('dz/dt = -z + tanh(alpha*z) for Different alpha Values', fontsize=16)
        ax.legend()
        ax.grid(True)
        
        return fig, ax
    
    def visualize_time_evolution(self, time_evolutions, alpha_values, equilibria_by_alpha):
        """
        Visualize time evolution of z for different alpha values
        
        Parameters:
        -----------
        time_evolutions : dict
            Dictionary mapping (alpha, z0) to solution trajectories
        alpha_values : array
            Array of alpha values to visualize
        equilibria_by_alpha : dict
            Dictionary mapping alpha values to equilibrium points
            
        Returns:
        --------
        fig, axes : matplotlib figure objects
        """
        z0_values = list(set([z0 for _, z0 in time_evolutions.keys()]))
        z0_values.sort()
        
        fig, axes = plt.subplots(len(alpha_values), 1, figsize=(10, 12), sharex=True)
        if len(alpha_values) == 1:
            axes = [axes]
        
        for i, alpha in enumerate(alpha_values):
            ax = axes[i]
            
            for z0 in z0_values:
                if (alpha, z0) in time_evolutions:
                    sol = time_evolutions[(alpha, z0)]
                    ax.plot(sol.t, sol.y[0], label=f'z0 = {z0:.1f}')
            
            # Mark equilibrium points
            if alpha in equilibria_by_alpha:
                for eq in equilibria_by_alpha[alpha]:
                    ax.axhline(y=eq, color='r', linestyle='--', alpha=0.5)
            
            ax.set_ylabel('z(t)', fontsize=12)
            ax.set_title(f'alpha = {alpha:.1f}', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True)
        
        axes[-1].set_xlabel('Time', fontsize=14)
        fig.suptitle('Time Evolution of z for Different alpha Values: dz/dt = -z + tanh(alpha*z)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig, axes
    
    def visualize_bifurcation(self, equilibria_data):
        """
        Visualize bifurcation diagram
        
        Parameters:
        -----------
        equilibria_data : list of tuples
            List of (alpha, equilibrium) points
            
        Returns:
        --------
        fig, ax : matplotlib figure objects
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if equilibria_data:
            alpha_eq, z_eq = zip(*equilibria_data)
            ax.scatter(alpha_eq, z_eq, s=10, color='b', alpha=0.7)
        
        ax.set_xlabel('alpha (Self-Activation Coefficient)', fontsize=14)
        ax.set_ylabel('Equilibrium Point z*', fontsize=14)
        ax.set_title('Bifurcation Diagram: dz/dt = -z + tanh(alpha*z)', fontsize=16)
        ax.grid(True)
        
        return fig, ax
    
    def visualize_critical_alpha(self, z_values, critical_alpha_data, derivative_data=None):
        """
        Visualize behavior around critical alpha value
        
        Parameters:
        -----------
        z_values : array
            Array of z values
        critical_alpha_data : dict
            Dictionary mapping alpha values to dzdt values
        derivative_data : tuple, optional
            Tuple of (alpha_fine, derivative_at_zero) for derivative plot
            
        Returns:
        --------
        fig, axes : matplotlib figure objects
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot dzdt functions
        ax1 = axes[0]
        alpha_values = list(critical_alpha_data.keys())
        alpha_values.sort()
        
        for alpha in alpha_values:
            dzdt_values = critical_alpha_data[alpha]
            ax1.plot(z_values, dzdt_values, label=f'alpha = {alpha:.2f}')
            
            # Mark equilibrium points where dzdt crosses zero
            for i in range(len(z_values) - 1):
                if dzdt_values[i] * dzdt_values[i+1] <= 0:
                    # Approximately find where the line crosses zero
                    z1, z2 = z_values[i], z_values[i+1]
                    dzdt1, dzdt2 = dzdt_values[i], dzdt_values[i+1]
                    if dzdt2 - dzdt1 != 0:  # Avoid division by zero
                        z_eq = z1 - dzdt1 * (z2 - z1) / (dzdt2 - dzdt1)
                        ax1.plot(z_eq, 0, 'o', markersize=6)
        
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_xlabel('z (Opinion Value)', fontsize=14)
        ax1.set_ylabel('dz/dt', fontsize=14)
        ax1.set_title('dz/dt Function Near Critical alpha: dz/dt = -z + tanh(alpha*z)', fontsize=16)
        ax1.legend()
        ax1.grid(True)
        
        # Analyze derivative at z = 0 if data is provided
        ax2 = axes[1]
        
        if derivative_data:
            alpha_fine, derivative_at_zero = derivative_data
            ax2.plot(alpha_fine, derivative_at_zero)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax2.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, 
                      label='Critical Value alpha = 1')
            
            ax2.set_xlabel('alpha (Self-Activation Coefficient)', fontsize=14)
            ax2.set_ylabel('Derivative at z=0 (∂(dz/dt)/∂z)', fontsize=14)
            ax2.set_title('Stability Analysis at z=0', fontsize=16)
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        return fig, axes

    def visualize_equilibrium_data(self, df):
        """
        Visualize equilibrium data from DataFrame
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with equilibrium data including 'alpha', 'equilibrium', 'stability'
            
        Returns:
        --------
        fig : matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Split data by stability
        stable = df[df['stability'] == 'Stable']
        unstable = df[df['stability'] == 'Unstable']
        
        # Plot points
        if not stable.empty:
            ax.scatter(stable['alpha'], stable['equilibrium'], c='g', label='Stable', alpha=0.7)
        if not unstable.empty:
            ax.scatter(unstable['alpha'], unstable['equilibrium'], c='r', label='Unstable', alpha=0.7)
        
        ax.set_xlabel('alpha (Self-Activation Coefficient)', fontsize=14)
        ax.set_ylabel('Equilibrium Point z*', fontsize=14)
        ax.set_title('Equilibrium Points and Stability', fontsize=16)
        ax.axvline(x=1.0, color='k', linestyle='--', label='Critical alpha = 1')
        ax.grid(True)
        ax.legend()
        
        return fig

    def save_visualizations(self, figures, names):
        """
        Save multiple visualizations
        
        Parameters:
        -----------
        figures : list
            List of matplotlib figures
        names : list
            List of filenames to save each figure
        """
        assert len(figures) == len(names), "Number of figures must match number of names"
        
        for fig, name in zip(figures, names):
            filepath = os.path.join(self.output_dir, name)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved visualization to: {filepath}")

    def visualize_from_files(self, equilibrium_file='equilibrium_data.csv'):
        """
        Generate visualizations from data files
        
        Parameters:
        -----------
        equilibrium_file : str
            Path to equilibrium data CSV file
        """
        # Check if file exists
        equilibrium_path = os.path.join(self.output_dir, equilibrium_file)
        if not os.path.exists(equilibrium_path):
            print(f"Warning: Equilibrium data file not found: {equilibrium_path}")
            return
            
        # Load equilibrium data
        df = pd.read_csv(equilibrium_path)
        
        # Generate and save visualization
        fig = self.visualize_equilibrium_data(df)
        filepath = os.path.join(self.output_dir, 'equilibrium_visualization.png')
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved equilibrium visualization to: {filepath}")


def main():
    """
    Main function to test visualizer
    """
    visualizer = AlphaVisualizer()
    visualizer.visualize_from_files()


if __name__ == "__main__":
    main() 