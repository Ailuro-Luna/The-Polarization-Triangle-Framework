# Verification Module for Polarization Triangle Framework

This directory contains documentation and analysis related to the verification tests for the Polarization Triangle Framework.

## Contents

- [Agent Interaction Verification](agent_interaction_verification.md): Detailed documentation about the two-agent verification process, including methodologies, rules, and expected outcomes.
- [Agent Interaction Results](agent_interaction_results.md): Analysis of the verification results, including key findings and theoretical implications.

## Implementation

The verification module has been implemented with the following components:

1. **Core verification class**: `TwoAgentVerification` in `polarization_triangle/verification/agent_interaction_verification.py`
2. **Result visualization**: `polarization_triangle/visualization/verification_visualizer.py`
3. **Execution script**: `polarization_triangle/scripts/run_verification.py`

## Running the Verification Tests

To run the verification tests, use the following command:

```bash
python -m polarization_triangle.scripts.run_verification --steps 30 --output-dir results/verification
```

Parameters:
- `--steps`: Number of steps to simulate for each rule (default: 10)
- `--output-dir`: Directory where results will be saved (default: results/verification)
- `--visualize-only`: Flag to only generate visualizations from existing results

## Results

The verification tests generate the following outputs:

1. `verification_results.csv`: Summary of results for all 16 verification rules
2. Individual trajectory CSV files for each rule
3. Visualizations:
   - Overall opinion change bar chart
   - Category-based opinion change comparisons
   - Trajectory plots by category
   - Combined trajectory plot

These results allow us to validate the core mechanics of the Polarization Triangle Framework and ensure that the model behaves as expected under different conditions of opinion direction, identity, and moralization status.

## Future Work

Potential future extensions to the verification module include:

1. Additional parameter sensitivity tests
2. More complex network structures beyond two agents
3. Detailed component analysis of the opinion change equation
4. Verification of long-term equilibrium states 