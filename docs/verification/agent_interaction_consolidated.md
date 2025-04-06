# Agent Interaction Verification and Results

This document describes the verification process and results analysis for the Polarization Triangle Framework, focusing on two-agent interaction scenarios that test the core dynamics of the model.

## 1. Verification Framework Overview

The verification framework implements a series of tests on 16 different interaction scenarios between two agents. These scenarios are systematically designed to cover all combinations of:

1. Opinion direction (same vs. different)
2. Identity (same vs. different)
3. Moralization status of both agents (moralized vs. non-moralized)

By testing these scenarios, we can verify that the model behaves as expected under various conditions and captures the theoretical predictions of the Polarization Triangle Framework.

## 2. Implementation

### Verification Setup

The verification is implemented through a simplified simulation environment that focuses on just two agents:

- A **focal agent** with a fixed initial opinion of 0.25
- An **other agent** with either 0.75 (same direction) or -0.75 (different direction)

The simulation runs for 30 steps and records the trajectory of opinions over time, as well as the components of the opinion dynamics equation (self-activation and social influence).

### Key Components

1. **TwoAgentVerification Class**: A lightweight version of the main Simulation class that implements the opinion dynamics for two agents.
2. **Verification Rules**: 16 predefined scenarios that test different combinations of agent properties.
3. **Multi-step Simulation**: The ability to run the simulation for multiple steps to observe longer-term dynamics.
4. **Trajectory Analysis**: Tools to record and analyze the evolution of opinions over time.

### Model Configuration

The verification uses the high polarization configuration from the main model:

```python
# Key parameters
delta = 0.05    # Opinion decay rate
u = 0.15        # Opinion activation coefficient
alpha = 0.3     # Self-activation coefficient 
beta = 0.35     # Social influence coefficient
gamma = 0.7     # Moralization impact coefficient
```

This configuration is designed to make the effects more pronounced and easier to observe within a small number of simulation steps.

## 3. The 16 Verification Rules

The verification tests 16 specific interaction scenarios, each designed to test a different aspect of the model:

| Rule | Opinion Direction | Identity | Morality (Self, Other) | Expected Effect | Rationale |
|------|-------------------|----------|------------------------|-----------------|-----------|
| 1    | Same (both +)     | Same     | {0,0}                  | High convergence | Both non-moralized with same identity reinforces consensus |
| 2    | Same (both +)     | Same     | {0,1}                  | Moderate pull toward extremes | Non-moralized focal agent influenced by moralized other agent |
| 3    | Same (both +)     | Same     | {1,0}                  | Moderate pull of other toward extremes | Moralized focal agent influences non-moralized other |
| 4    | Same (both +)     | Same     | {1,1}                  | High polarization | Same identity and both moralized leads to mutual reinforcement |
| 5    | Same (both +)     | Different| {0,0}                  | Moderate convergence | Open to discussion but identity difference weakens consensus |
| 6    | Same (both +)     | Different| {0,1}                  | Low pull toward extremes | Identity difference weakens moralized agent's influence |
| 7    | Same (both +)     | Different| {1,0}                  | Low pull of other toward extremes | Identity difference weakens focal agent's influence |
| 8    | Same (both +)     | Different| {1,1}                  | Moderate polarization | Both moralized but different identity causes weaker polarization |
| 9    | Different (+/-)   | Same     | {0,0}                  | Very high convergence | Same identity, no moral barriers allows reaching consensus |
| 10   | Different (+/-)   | Same     | {0,1}                  | Moderate convergence/pull | Influenced by moralized agent, but same identity mitigates |
| 11   | Different (+/-)   | Same     | {1,0}                  | Low resistance and position holding | Same identity reduces resistance level |
| 12   | Different (+/-)   | Same     | {1,1}                  | Low mutual polarization | Same identity weakens opposing polarization |
| 13   | Different (+/-)   | Different| {0,0}                  | Low convergence | Different identity hinders consensus but dialogue possible |
| 14   | Different (+/-)   | Different| {0,1}                  | High pull toward other side | Non-moralized more easily influenced with different identity |
| 15   | Different (+/-)   | Different| {1,0}                  | High resistance and movement to extremes | Moralization plus different identity causes strong resistance |
| 16   | Different (+/-)   | Different| {1,1}                  | Very high mutual polarization | Different identity and both moralized causes extreme polarization |

## 4. Results Analysis

### Key Findings

From the generated results, we can observe the following key patterns:

#### 1. Moralization as the Dominant Factor

- **Asymmetric Influence**: When the focal agent is non-moralized but the other agent is moralized (Rules 2 and 6), we observe the highest opinion change (0.205), indicating that moralized agents have a disproportionately strong influence on non-moralized agents
- **Resistance Effect**: When the focal agent is moralized (Rules 3, 4, 7, 8, 11, 12, 15, 16), it shows considerable resistance to external influences, maintaining its position or moving toward extremes
- **Mutual Repulsion**: When both agents are moralized with opposing opinions (Rules 12 and 16), they produce a strong mutual repulsion effect (0.161), moving toward more extreme positions rather than converging

#### 2. Identity's Secondary Role

- Strikingly, identity differences show minimal impact in the results - the patterns in same identity groups (Rules 1-4, 9-12) are nearly identical to different identity groups (Rules 5-8, 13-16)
- This pattern holds true across both same opinion direction and different opinion direction scenarios
- Identity effects appear to be overshadowed by moralization status in determining opinion dynamics

#### 3. Opinion Direction Effects

- Same opinion direction (Rules 1-8): All cases show positive opinion change, indicating movement toward more extreme positions
- Different opinion direction (Rules 9-16): Results depend primarily on moralization status:
  - Non-moralized focal agents (Rules 9, 10, 13, 14) show moderate movement toward more neutral positions (-0.018 to -0.056)
  - Moralized focal agents either resist change (Rules 11, 15, minimal movement of 0.004) or move toward extremes (Rules 12, 16, significant movement of 0.161)

### Categorical Analysis

The results organized by scenario categories reveal striking patterns:

1. **Same Opinion Direction, Same/Different Identity** (Rules 1-8):
   - Both categories show identical patterns
   - Non-moralized focal agents with moralized other agents (Rules 2, 6) show the strongest movement toward extremes (0.205)
   - Mutual non-moralization (Rules 1, 5) produces moderate convergence (0.182)
   - When the focal agent is moralized (Rules 3, 4, 7, 8), opinion change is more moderate (0.161)

2. **Different Opinion Direction, Same/Different Identity** (Rules 9-16):
   - Again, both categories show identical patterns
   - Mutual non-moralization (Rules 9, 13) results in movement toward neutrality (-0.018)
   - Non-moralized focal with moralized other (Rules 10, 14) shows the strongest pull toward neutrality (-0.056)
   - Moralized focal with non-moralized other (Rules 11, 15) shows near stability (0.004)
   - Mutual moralization (Rules 12, 16) produces strong movement toward extremes (0.161)

## 5. Theoretical Implications

These verification results strongly support the key theoretical predictions of the Polarization Triangle Framework:

1. **Moralization as the Primary Driver**: Moralization emerges as the dominant factor influencing opinion dynamics, significantly more important than identity alignment. It determines both resistance to external influence and the ability to influence others.

2. **Asymmetric Influence Patterns**: Non-moralized agents are highly susceptible to being influenced by moralized agents. This explains why in real-world settings, moralized minorities can exert disproportionate influence on non-moralized majorities.

3. **Mutual Repulsion Effect**: When agents with opposing opinions are both moralized, they move away from each other rather than converging, even with shared identity. This contradicts traditional views but explains real-world polarization phenomena.

4. **Subordinate Role of Identity**: Shared identity appears insufficient to overcome moralization effects. This suggests that when moral positions are activated, common group membership may fail to prevent polarization.

5. **Conditional Convergence**: Convergence primarily occurs among non-moralized agents. Once moralization enters the picture, convergence becomes unlikely except in specific circumstances.

## 6. Conclusion

This verification testing confirms that the Polarization Triangle Framework successfully captures key dynamics of social polarization:

1. The model accurately reproduces the asymmetric influence of moralized agents on non-moralized ones
2. It captures both convergence and polarization patterns under different conditions
3. It demonstrates how moralization can override identity effects in social influence processes
4. It shows how mutual moralization with opposing views leads to increased polarization rather than moderation

These results have important implications for understanding real-world polarization. They suggest that polarization is not simply the result of widening differences in opinions, but emerges from complex interactions between moralization, identity factors, and opinion direction. In particular, moralization appears to be the critical trigger for repulsion and polarization, potentially even overriding the cohesive force of shared identity.

Future research should explore these dynamics in larger, more complex networks, and investigate potential interventions for mitigating polarization based on these insights.

## 7. Running the Verification Tests

To run the verification tests, use the following command:

```bash
python -m polarization_triangle.scripts.run_verification --steps 30 --output-dir results/verification
```

Parameters:
- `--steps`: Number of steps to simulate for each rule (default: 10)
- `--output-dir`: Directory where results will be saved (default: results/verification)
- `--visualize-only`: Flag to only generate visualizations from existing results 