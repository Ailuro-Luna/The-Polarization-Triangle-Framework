# Verification Module for Polarization Triangle Framework

This directory contains documentation and analysis related to the verification tests for the Polarization Triangle Framework.

## Contents

### 验证模块文档

- [Agent Interaction Verification](agent_interaction_consolidated.md): 详细的双agent验证过程文档，包括方法论、规则和预期结果
- [Alpha Verification](alpha_verification.md): Alpha参数验证模块文档，分析当beta=0时自我激活系数对意见动态的影响
- [Alpha-Beta Verification](alpha_beta_verification.md): Alpha-Beta参数验证模块文档，分析不同参数组合对极化的影响

### 验证结果分析

验证模块包含三个主要组件，用于测试极化三角框架的不同方面：

1. **Agent Interaction Verification**: 测试16种基本的双agent交互规则
2. **Alpha Verification**: 分析简化方程 dz/dt = -z + tanh(alpha*z) 的行为
3. **Alpha-Beta Verification**: 测试不同alpha和beta参数组合对极化的影响

## Implementation

The verification module has been implemented with the following components:

### Agent Interaction Verification
1. **Core verification class**: `MultiAgentVerification` in `polarization_triangle/verification/agent_interaction_verification.py`
2. **Result visualization**: `polarization_triangle/visualization/verification_visualizer.py`
3. **Execution script**: `polarization_triangle/scripts/run_verification.py`

### Alpha Verification  
1. **Core verification class**: `AlphaVerification` in `polarization_triangle/verification/alpha_analysis.py`
2. **Visualization**: `AlphaVisualizer` in `polarization_triangle/visualization/alpha_viz.py`
3. **Execution script**: `polarization_triangle/scripts/run_alpha_verification.py`

### Alpha-Beta Verification
1. **Core verification class**: `AlphaBetaVerification` in `polarization_triangle/verification/alphabeta_analysis.py`
2. **Execution script**: `polarization_triangle/scripts/run_alphabeta_verification.py`

## Running the Verification Tests

### Agent Interaction Verification

To run the agent interaction verification tests, use the following command:

```bash
python -m polarization_triangle.scripts.run_verification --steps 30 --output-dir results/verification
```

Parameters:
- `--steps`: Number of steps to simulate for each rule (default: 10)
- `--output-dir`: Directory where results will be saved (default: results/verification)
- `--visualize-only`: Flag to only generate visualizations from existing results

### Alpha Verification

To run the alpha verification analysis:

```bash
# Basic usage
python -m polarization_triangle.main --test-type verification --verification-type alpha

# With custom parameters
python -m polarization_triangle.scripts.run_alpha_verification --alpha-min -1 --alpha-max 2 --output-dir results/verification/alpha
```

### Alpha-Beta Verification

To run the alpha-beta verification analysis:

```bash
# Basic usage  
python -m polarization_triangle.main --test-type verification --verification-type alphabeta

# With custom parameters
python -m polarization_triangle.scripts.run_alphabeta_verification --output-dir results/verification/alphabeta --beta-min 0.1 --beta-max 2.0 --beta-steps 10 --morality-rate 0.3 --num-runs 10
```

### Run All Verification Tests

To run all verification tests at once:

```bash
python -m polarization_triangle.main --test-type verification --verification-type all --output-dir results/verification
```

## Results

### Agent Interaction Verification Results

The agent interaction verification tests generate the following outputs:

1. `verification_results.csv`: Summary of results for all 16 verification rules
2. Individual trajectory CSV files for each rule
3. Visualizations:
   - Overall opinion change bar chart
   - Category-based opinion change comparisons
   - Trajectory plots by category
   - Combined trajectory plot

### Alpha Verification Results

The alpha verification generates:

1. `alpha_dzdt_analysis.png`: Plot of dz/dt function for different alpha values
2. `alpha_time_evolution.png`: Plot of z(t) trajectories for different alpha values  
3. `alpha_bifurcation.png`: Bifurcation diagram showing equilibrium points vs alpha
4. `alpha_critical_analysis.png`: Analysis of behavior near the critical alpha value
5. `alpha_analysis_report.md`: Detailed report of findings
6. `equilibrium_data.csv`: Raw data of equilibrium points for different alpha values
7. `equilibrium_stability.png`: Visualization of equilibrium points with stability information

### Alpha-Beta Verification Results

The alpha-beta verification generates:

1. `alphabeta_results.csv`: Summary of polarization metrics for all parameter combinations
2. `alphabeta_comparison_plots.png`: Comparison plots of polarization metrics
3. Individual opinion distribution and heatmap plots for each parameter combination
4. Statistical analysis of parameter effects on polarization

These results allow us to validate the core mechanics of the Polarization Triangle Framework and ensure that the model behaves as expected under different conditions of opinion direction, identity, and moralization status.

## Future Work

Potential future extensions to the verification module include:

1. Additional parameter sensitivity tests
2. More complex network structures beyond two agents
3. Detailed component analysis of the opinion change equation
4. Verification of long-term equilibrium states 