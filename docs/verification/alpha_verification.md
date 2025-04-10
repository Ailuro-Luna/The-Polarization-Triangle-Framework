# Alpha Verification Module

## Overview

The Alpha Verification module is designed to analyze the behavior of a simplified opinion dynamics equation when the social influence parameter (beta) is set to zero. This module focuses on studying how the self-activation coefficient (alpha) affects the stability and evolution of opinions in the absence of social influence.

The core equation under study is:

$$\frac{dz}{dt} = -z + \tanh(\alpha z)$$

Where:
- $z$ represents the opinion value
- $\alpha$ (alpha) is the self-activation coefficient
- $\delta$ (delta) is set to 1 (opinion decay rate)
- $u$ is set to 1 (opinion activation coefficient)
- $\beta$ (beta) is set to 0 (social influence coefficient)

This simplified equation allows us to focus specifically on how an agent's self-reinforcement mechanism impacts opinion dynamics.

## Module Structure

The Alpha Verification module is structured with clear separation of concerns:

### Core Files
- **`polarization_triangle/verification/alpha_analysis.py`**: Contains the `AlphaVerification` class that implements the core analysis functionality
- **`polarization_triangle/visualization/alpha_viz.py`**: Contains the `AlphaVisualizer` class responsible for creating visualizations
- **`polarization_triangle/scripts/run_alpha_verification.py`**: CLI script for running the verification

### Output
- All results are saved to the `results/verification` directory by default
- Generated files include:
  - `alpha_dzdt_analysis.png`: Plot of dz/dt function for different alpha values
  - `alpha_time_evolution.png`: Plot of z(t) trajectories for different alpha values
  - `alpha_bifurcation.png`: Bifurcation diagram showing equilibrium points vs alpha
  - `alpha_critical_analysis.png`: Analysis of behavior near the critical alpha value
  - `alpha_analysis_report.md`: Detailed report of findings
  - `equilibrium_data.csv`: Raw data of equilibrium points for different alpha values
  - `equilibrium_stability.png`: Visualization of equilibrium points with stability information

## Configuration Options

When running the verification, several parameters can be configured:

- **`alpha_range`**: Range of alpha values to analyze (default: -1 to 2)
- **`z_range`**: Range of opinion values to analyze (default: -1 to 1)
- **`time_range`**: Time duration for evolution analysis (default: 0 to 10)
- **`output_dir`**: Directory to save results (default: "results/verification")

## Running the Verification

The alpha verification can be run in several ways:

### Using the Command Line Interface

```bash
# Basic usage
python -m polarization_triangle.main --test-type verification --verification-type alpha

# With custom alpha range
python -m polarization_triangle.main --test-type verification --verification-type alpha --alpha-min -1 --alpha-max 2

# With custom output directory
python -m polarization_triangle.main --test-type verification --verification-type alpha --output-dir results/custom_dir
```

### Using the Dedicated Script

```bash
# Basic usage
python -m polarization_triangle.scripts.run_alpha_verification

# With custom parameters
python -m polarization_triangle.scripts.run_alpha_verification --alpha-min -1 --alpha-max 2 --output-dir results/custom_dir
```

### Programmatic Usage

```python
from polarization_triangle.verification.alpha_analysis import AlphaVerification

# Create verification instance with custom parameters
verification = AlphaVerification(
    alpha_range=(-1, 2),
    z_range=(-1, 1),
    time_range=(0, 10),
    output_dir="results/custom_dir"
)

# Run all analyses
verification.run()

# Or run specific analyses
fig, ax = verification.analyze_bifurcation()
```

## Verification Logic

The verification performs several key analyses:

1. **dzdt Analysis**: Examines how the rate of change of opinion (dz/dt) varies with opinion (z) for different alpha values.

2. **Time Evolution Analysis**: Simulates the evolution of opinions over time starting from different initial conditions.

3. **Bifurcation Analysis**: Identifies how the equilibrium points change as alpha varies, revealing the critical value at which the system's behavior fundamentally changes.

4. **Critical Alpha Analysis**: Focuses on behavior near the critical alpha value (alpha = 1), examining the stability of equilibrium points.

5. **Equilibrium Stability Analysis**: Classifies equilibrium points as stable or unstable based on the derivative of dz/dt at those points.

The core logic involves solving the ODE dz/dt = -z + tanh(alpha*z) and analyzing its fixed points and stability properties.

## Key Findings and Conclusions

The verification results show that:

1. **Critical Bifurcation at alpha = 1**: 
   - For alpha < 1: The system has a single stable equilibrium point at z = 0
   - For alpha = 1: The equilibrium at z = 0 is critically stable
   - For alpha > 1: The equilibrium at z = 0 becomes unstable, and two new stable equilibrium points emerge (one positive, one negative)

2. **Implications for Opinion Dynamics**:
   - When alpha < 1, all opinions eventually converge to neutral (z = 0)
   - When alpha > 1, opinions diverge to polarized states (positive or negative)
   - The transition at alpha = 1 represents a "phase transition" from consensus to polarization

3. **Interpretation in Social Context**:
   - Alpha represents the strength of self-reinforcement of one's opinion
   - When self-reinforcement is weak (alpha < 1), external factors lead opinions toward neutral
   - When self-reinforcement is strong (alpha > 1), opinions tend toward extremes (polarization)
   - This suggests that even without social influence (beta = 0), internal self-reinforcement alone can drive polarization

These findings provide insights into how the self-activation parameter affects opinion dynamics, showing that polarization can emerge even in the absence of social influence when self-reinforcement exceeds a critical threshold.

## Implementation Details

### Core Computation Functions

- **`dzdt(z, alpha)`**: Calculates the rate of change of opinion based on the equation
- **`solve_ode(alpha, z0, t_span, t_eval)`**: Solves the ODE for given parameters
- **`find_equilibrium_points(alpha, z_range, points)`**: Finds equilibrium points for a given alpha value

### Analysis Functions

- **`analyze_dzdt()`**: Analyzes the dzdt function for different alpha values
- **`analyze_time_evolution()`**: Analyzes how opinions evolve over time
- **`analyze_bifurcation()`**: Analyzes how equilibrium points change with alpha
- **`analyze_critical_alpha()`**: Detailed analysis around the critical alpha value
- **`generate_analysis_report()`**: Generates a comprehensive analysis report

### Visualization Functions

The visualizer provides dedicated methods for creating each type of plot:
- **`visualize_dzdt()`**: Creates plots of dz/dt functions
- **`visualize_time_evolution()`**: Creates plots of opinion trajectories
- **`visualize_bifurcation()`**: Creates bifurcation diagrams
- **`visualize_critical_alpha()`**: Creates plots focusing on critical alpha behavior
- **`visualize_equilibrium_data()`**: Visualizes equilibrium points and their stability

## Integration with Main Framework

The Alpha Verification module is integrated with the main Polarization Triangle Framework and can be accessed through the main CLI interface. This provides a consistent interface for running different types of verifications and analyses within the framework.

## Future Extensions

Potential future extensions to this verification module include:
- Adding noise to the dynamics to model random fluctuations
- Exploring scenarios with multiple agents with varying alpha values
- Introducing small social influence (non-zero beta) to study the interplay between self-activation and social influence
- Comparing numerical results with analytical predictions for more complex scenarios

## References

1. Castellano, C., Fortunato, S., & Loreto, V. (2009). Statistical physics of social dynamics. Reviews of Modern Physics, 81(2), 591-646.
2. Strogatz, S. H. (2018). Nonlinear dynamics and chaos: With applications to physics, biology, chemistry, and engineering. CRC Press.
3. Lorenz, J. (2007). Continuous opinion dynamics under bounded confidence: A survey. International Journal of Modern Physics C, 18(12), 1819-1838. 