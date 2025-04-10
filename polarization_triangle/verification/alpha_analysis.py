import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
from pathlib import Path
import pandas as pd
from polarization_triangle.visualization.alpha_viz import AlphaVisualizer

class AlphaVerification:
    """
    Verification of system behavior when beta=0, delta=1, u=1:
    dz/dt = -z + tanh(alpha*z)
    
    Analyzes system dynamics as alpha varies between -1 and 2.
    """
    
    def __init__(self, alpha_range=None, z_range=None, time_range=None, output_dir=None):
        """
        Initialize verification class
        
        Parameters:
        -----------
        alpha_range : tuple
            Range of alpha for analysis, default is (-1, 2)
        z_range : tuple
            Range of z for analysis, default is (-2, 2)
        time_range : tuple
            Time evolution range, default is (0, 10)
        output_dir : str
            Output directory, default is './results'
        """
        self.alpha_range = alpha_range or (-1, 2)
        self.z_range = z_range or (-1, 1)
        self.time_range = time_range or (0, 10)
        
        self.output_dir = output_dir or os.path.join(
            Path(__file__).parent.parent.parent, 'results', 'verification'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = AlphaVisualizer(output_dir=self.output_dir)
    
    def dzdt(self, z, alpha):
        """
        Calculate the dzdt function when beta=0, delta=1, u=1
        
        Parameters:
        -----------
        z : float or array
            Opinion value
        alpha : float
            Self-activation coefficient
            
        Returns:
        --------
        float or array
            dzdt value
        """
        return -z + np.tanh(alpha * z)
    
    def solve_ode(self, alpha, z0, t_span, t_eval):
        """
        Solve ODE for given alpha value and initial conditions
        
        Parameters:
        -----------
        alpha : float
            Self-activation coefficient
        z0 : float
            Initial opinion value
        t_span : tuple
            Solution time range
        t_eval : array
            Evaluation time points
            
        Returns:
        --------
        sol : OdeSolution
            ODE solution result
        """
        def system(t, z):
            return self.dzdt(z, alpha)
        
        sol = solve_ivp(
            system, 
            t_span=t_span, 
            y0=[z0], 
            t_eval=t_eval,
            method='RK45'
        )
        return sol
    
    def find_equilibrium_points(self, alpha, z_range=(-2, 2), points=1000):
        """
        Find equilibrium points for given alpha value
        
        Parameters:
        -----------
        alpha : float
            Self-activation coefficient
        z_range : tuple
            Search range
        points : int
            Number of search points
            
        Returns:
        --------
        list
            List of equilibrium points
        """
        z_values = np.linspace(z_range[0], z_range[1], points)
        dzdt_values = self.dzdt(z_values, alpha)
        
        # Look for points where dzdt crosses zero
        equilibria = []
        for i in range(len(z_values) - 1):
            if dzdt_values[i] * dzdt_values[i+1] <= 0:
                # Linear interpolation to find more precise equilibrium point
                z1, z2 = z_values[i], z_values[i+1]
                dzdt1, dzdt2 = dzdt_values[i], dzdt_values[i+1]
                
                # Linear interpolation formula
                if dzdt2 - dzdt1 != 0:  # Avoid division by zero
                    z_eq = z1 - dzdt1 * (z2 - z1) / (dzdt2 - dzdt1)
                    equilibria.append(z_eq)
        
        return equilibria
    
    def analyze_dzdt(self, alpha_values=None, z_values=None):
        """
        Analyze dzdt function for different alpha values
        
        Parameters:
        -----------
        alpha_values : array
            List of alpha values to analyze
        z_values : array
            List of z values to analyze
            
        Returns:
        --------
        fig, ax : matplotlib figure objects
        """
        if alpha_values is None:
            alpha_values = np.linspace(self.alpha_range[0], self.alpha_range[1], 7)
        
        if z_values is None:
            z_values = np.linspace(self.z_range[0], self.z_range[1], 1000)
        
        # Calculate dzdt values for each alpha
        dzdt_values_by_alpha = {}
        equilibria_by_alpha = {}
        
        for alpha in alpha_values:
            dzdt_values_by_alpha[alpha] = self.dzdt(z_values, alpha)
            equilibria_by_alpha[alpha] = self.find_equilibrium_points(alpha)
        
        # Use visualizer to create the plot
        return self.visualizer.visualize_dzdt(z_values, dzdt_values_by_alpha, alpha_values, equilibria_by_alpha)
    
    def analyze_time_evolution(self, alpha_values=None, z0_values=None, t_max=None):
        """
        Analyze time evolution of z for different alpha values
        
        Parameters:
        -----------
        alpha_values : array
            List of alpha values to analyze
        z0_values : array
            List of initial z values
        t_max : float
            Maximum time
            
        Returns:
        --------
        fig, axes : matplotlib figure objects
        """
        if alpha_values is None:
            alpha_values = np.linspace(self.alpha_range[0], self.alpha_range[1], 4)
        
        if z0_values is None:
            z0_values = [-1.5, -0.5, 0.5, 1.5]
        
        if t_max is None:
            t_max = self.time_range[1]
        
        t_eval = np.linspace(0, t_max, 1000)
        
        # Calculate time evolutions for each alpha and z0
        time_evolutions = {}
        equilibria_by_alpha = {}
        
        for alpha in alpha_values:
            equilibria_by_alpha[alpha] = self.find_equilibrium_points(alpha)
            
            for z0 in z0_values:
                sol = self.solve_ode(alpha, z0, (0, t_max), t_eval)
                time_evolutions[(alpha, z0)] = sol
        
        # Use visualizer to create the plot
        return self.visualizer.visualize_time_evolution(time_evolutions, alpha_values, equilibria_by_alpha)
    
    def analyze_bifurcation(self, z0_values=None, num_alphas=100, t_max=None):
        """
        Analyze bifurcation behavior due to change in alpha values
        
        Parameters:
        -----------
        z0_values : array
            List of initial z values
        num_alphas : int
            Number of alpha values
        t_max : float
            Maximum time
            
        Returns:
        --------
        fig, ax : matplotlib figure objects
        """
        if z0_values is None:
            z0_values = [-1.5, -0.5, 0.5, 1.5]
        
        if t_max is None:
            t_max = self.time_range[1]
        
        alpha_values = np.linspace(self.alpha_range[0], self.alpha_range[1], num_alphas)
        
        equilibria_data = []
        
        for alpha in alpha_values:
            # Find equilibrium points
            equilibria = self.find_equilibrium_points(alpha)
            
            # Store equilibrium points for each alpha value
            for eq in equilibria:
                equilibria_data.append((alpha, eq))
        
        # Use visualizer to create the plot
        return self.visualizer.visualize_bifurcation(equilibria_data)
    
    def analyze_critical_alpha(self):
        """
        Detailed analysis of behavior around critical alpha value
        
        In theory, for the equation dz/dt = -z + tanh(alpha*z):
        - When alpha < 1, the system has only one stable equilibrium point at z = 0
        - When alpha > 1, z = 0 becomes unstable, and two new stable equilibrium points emerge
        
        Returns:
        --------
        fig, axes : matplotlib figure objects
        """
        # Focus on alpha values near the critical point
        alpha_values = [0.8, 0.9, 0.99, 1.0, 1.01, 1.1, 1.2]
        z_values = np.linspace(-2, 2, 1000)
        
        # Calculate dzdt values for each alpha
        critical_alpha_data = {}
        
        for alpha in alpha_values:
            critical_alpha_data[alpha] = self.dzdt(z_values, alpha)
        
        # Calculate derivative at z = 0
        alpha_fine = np.linspace(0.5, 1.5, 1000)
        derivative_at_zero = np.array([-1 + alpha for alpha in alpha_fine])
        
        # Use visualizer to create the plot
        return self.visualizer.visualize_critical_alpha(z_values, critical_alpha_data, (alpha_fine, derivative_at_zero))
        
    def generate_analysis_report(self):
        """
        Generate analysis report summarizing verification results
        
        Returns:
        --------
        report_path : str
            Path to saved report
        """
        # Analyze equilibrium points for different alpha values
        alpha_values = np.linspace(self.alpha_range[0], self.alpha_range[1], 20)
        equilibrium_data = []
        
        for alpha in alpha_values:
            equilibria = self.find_equilibrium_points(alpha)
            # For each equilibrium point, calculate its stability
            for eq in equilibria:
                # Calculate derivative at equilibrium point
                h = 1e-6
                dzdt_plus = self.dzdt(eq + h, alpha)
                dzdt_minus = self.dzdt(eq - h, alpha)
                derivative = (dzdt_plus - dzdt_minus) / (2*h)
                
                stability = "Stable" if derivative < 0 else "Unstable"
                
                equilibrium_data.append({
                    "alpha": alpha,
                    "equilibrium": eq,
                    "stability": stability,
                    "derivative": derivative
                })
        
        # Create DataFrame
        df = pd.DataFrame(equilibrium_data)
        
        # Calculate critical alpha value
        critical_alpha = 1.0  # Theoretical value
        
        # Generate report text
        report = "# Polarization Triangle Alpha Verification Analysis Report\n\n"
        report += "## Analysis Overview\n\n"
        report += "This report analyzes the simplified model equation when beta=0, delta=1, u=1:\n\n"
        report += "$$\\frac{dz}{dt} = -z + \\tanh(\\alpha z)$$\n\n"
        report += "and its behavior as alpha varies in the range [{}, {}].\n\n".format(
            self.alpha_range[0], self.alpha_range[1]
        )
        
        report += "## Theoretical Analysis\n\n"
        report += "For the equation dz/dt = -z + tanh(alpha*z), stability analysis shows:\n\n"
        report += "1. Equilibrium points satisfy: z = tanh(alpha*z)\n"
        report += "2. At z=0, the system derivative is -1+alpha\n"
        report += "3. Critical value alpha=1:\n"
        report += "   - When alpha < 1, z=0 is the only stable equilibrium point\n"
        report += "   - When alpha = 1, z=0 is critically stable\n"
        report += "   - When alpha > 1, z=0 becomes unstable, and two new stable equilibrium points emerge\n\n"
        
        report += "## Numerical Analysis Results\n\n"
        
        # Analyze different alpha ranges
        alpha_below_1 = df[df['alpha'] < 0.99]
        alpha_around_1 = df[(df['alpha'] >= 0.99) & (df['alpha'] <= 1.01)]
        alpha_above_1 = df[df['alpha'] > 1.01]
        
        report += "### Case: alpha < 1\n\n"
        if not alpha_below_1.empty:
            report += "- Number of equilibrium points: {}\n".format(
                alpha_below_1.groupby('alpha').size().mean()
            )
            stable_count = (alpha_below_1['stability'] == "Stable").sum()
            report += "- Proportion of stable equilibrium points: {:.2f}%\n".format(
                100 * stable_count / len(alpha_below_1)
            )
            report += "- Average equilibrium point positions: {}\n\n".format(
                alpha_below_1.groupby('stability')['equilibrium'].mean().to_dict()
            )
        else:
            report += "No alpha values analyzed in this range\n\n"
        
        report += "### Case: alpha ≈ 1\n\n"
        if not alpha_around_1.empty:
            report += "- Number of equilibrium points: {}\n".format(
                alpha_around_1.groupby('alpha').size().mean()
            )
            stable_count = (alpha_around_1['stability'] == "Stable").sum()
            report += "- Proportion of stable equilibrium points: {:.2f}%\n".format(
                100 * stable_count / len(alpha_around_1) if len(alpha_around_1) > 0 else 0
            )
            report += "- Average equilibrium point positions: {}\n\n".format(
                alpha_around_1.groupby('stability')['equilibrium'].mean().to_dict()
            )
        else:
            report += "No alpha values analyzed in this range\n\n"
        
        report += "### Case: alpha > 1\n\n"
        if not alpha_above_1.empty:
            report += "- Number of equilibrium points: {}\n".format(
                alpha_above_1.groupby('alpha').size().mean()
            )
            stable_count = (alpha_above_1['stability'] == "Stable").sum()
            report += "- Proportion of stable equilibrium points: {:.2f}%\n".format(
                100 * stable_count / len(alpha_above_1)
            )
            report += "- Average equilibrium point positions: {}\n\n".format(
                alpha_above_1.groupby('stability')['equilibrium'].mean().to_dict()
            )
        else:
            report += "No alpha values analyzed in this range\n\n"
        
        report += "## Conclusions\n\n"
        report += "1. Bifurcation behavior: The system undergoes bifurcation at alpha=1, transitioning from one stable equilibrium point to one unstable and two stable equilibrium points\n"
        report += "2. Critical behavior: alpha=1 is the critical point of the system, where the stability of the z=0 equilibrium point changes\n"
        report += "3. Polarization behavior explanation:\n"
        report += "   - When alpha < 1, the system tends to return to the neutral state (z=0)\n"
        report += "   - When alpha > 1, the system tends to polarize (stabilizing at non-zero equilibrium points)\n"
        report += "   - The alpha parameter can be interpreted as the degree of self-reinforcement of an agent's own opinion\n"
        report += "   - When self-reinforcement is strong enough (alpha > 1), it may lead to polarization\n\n"
        
        # Save report
        report_path = os.path.join(self.output_dir, 'alpha_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save equilibrium point data
        csv_path = os.path.join(self.output_dir, 'equilibrium_data.csv')
        df.to_csv(csv_path, index=False)
        
        # Create additional visualization for equilibrium data
        fig = self.visualizer.visualize_equilibrium_data(df)
        fig.savefig(os.path.join(self.output_dir, 'equilibrium_stability.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return report_path
        
    def run(self):
        """
        Run all analyses
        """
        # Analyze dzdt
        fig1, _ = self.analyze_dzdt()
        plt.savefig(os.path.join(self.output_dir, 'alpha_dzdt_analysis.png'), dpi=300, bbox_inches='tight')
        
        # Analyze time evolution
        fig2, _ = self.analyze_time_evolution()
        plt.savefig(os.path.join(self.output_dir, 'alpha_time_evolution.png'), dpi=300, bbox_inches='tight')
        
        # Analyze bifurcation diagram
        fig3, _ = self.analyze_bifurcation()
        plt.savefig(os.path.join(self.output_dir, 'alpha_bifurcation.png'), dpi=300, bbox_inches='tight')
        
        # Analyze critical alpha
        fig4, _ = self.analyze_critical_alpha()
        plt.savefig(os.path.join(self.output_dir, 'alpha_critical_analysis.png'), dpi=300, bbox_inches='tight')
        
        # Generate analysis report
        report_path = self.generate_analysis_report()
        print(f"Analysis report saved to: {report_path}")
        
        plt.close('all')
        print(f"Analysis results saved to: {self.output_dir}")


if __name__ == "__main__":
    verification = AlphaVerification()
    verification.run() 