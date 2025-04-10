# Alpha-Beta Parameter Verification in the Polarization Triangle Framework

## Overview

This document presents the verification experiments conducted to investigate the effects of the Alpha (self-activation coefficient) and Beta (social influence coefficient) parameters in the Polarization Triangle Framework under zero moralization conditions. The experiments specifically focus on testing two hypotheses:

1. For low Alpha values, increasing Beta can potentially produce effects similar to high Alpha
2. For high Alpha values, increasing Beta might only reinforce existing effects

## Methodology

### Parameter Setup

The verification experiments used the following parameter ranges:
- **Low Alpha**: 0.5
- **Medium Alpha**: 0.9, 1.0, 1.1
- **High Alpha**: 1.5
- **Beta Range**: 0.1 to 2.0 (at equal intervals)
- **Moralization Rate**: 0 (no moralization)

### Experimental Design

For each Alpha-Beta parameter combination:
- Multiple simulations (typically 10) were run to ensure statistical reliability
- Each simulation used the LFR network configuration with 500 agents
- Standard parameters were maintained across all simulations except Alpha and Beta
- The results were averaged to minimize the impact of random variations

### Metrics

Four key metrics were used to evaluate polarization:
1. **Mean Extremity**: Average absolute value of opinions, indicating overall extremism
2. **Extreme Ratio**: Proportion of agents with opinion magnitude greater than 0.7
3. **Bimodality**: Measure of the dual-peak nature of opinion distribution
4. **Variance**: Overall dispersion of opinions

## Results

### Mean Extremity

![Mean Extremity](../../results/verification/alphabeta/visualizations/comparison_mean_extremity无道德化.png)

The Mean Extremity graph demonstrates:
- Low Alpha (0.5) starts with minimal extremity (~0.27) when Beta is low, but rapidly increases with Beta
- Medium Alpha values (0.9-1.1) show intermediate initial extremity levels
- High Alpha (1.5) maintains high extremity (~0.89) even at minimal social influence
- All curves converge to near-maximum extremity (close to 1.0) when Beta reaches approximately 1.0
- Beta's impact is most pronounced for the Alpha=0.5 curve (steepest slope)

### Extreme Ratio

![Extreme Ratio](../../results/verification/alphabeta/visualizations/comparison_extreme_ratio无道德化.png)

The Extreme Ratio analysis reveals:
- Almost no extreme opinions exist for Alpha=0.5 with low Beta (~0.02)
- Alpha=0.9 shows moderate extreme ratio (~0.4) at low Beta
- Alpha=1.1 presents substantial extreme opinions (~0.8) even at minimal social influence
- Alpha=1.5 maintains a high proportion of extreme opinions (~0.95) across the entire Beta range
- All curves reach saturation (>0.95) when Beta approaches 1.0

### Bimodality

![Bimodality](../../results/verification/alphabeta/visualizations/comparison_bimodality无道德化.png)

The Bimodality measure shows a strikingly different pattern:
- Bimodality decreases as Beta increases for all Alpha values
- Alpha=1.1 demonstrates the highest bimodality (~7.2) at low Beta
- Alpha=1.0 shows substantial bimodality (~6.0) at low Beta
- Alpha=0.9 exhibits moderate bimodality (~4.5) at low Beta
- Alpha=0.5 presents much lower bimodality (~1.2) even at low Beta
- Alpha=1.5 shows minimal bimodality throughout the Beta range
- All curves converge to near-zero bimodality when Beta exceeds 1.0

### Variance

![Variance](../../results/verification/alphabeta/visualizations/comparison_variance无道德化.png)

The Variance metric indicates:
- Low Alpha (0.5) begins with minimal variance (~0.10) but rapidly increases with Beta
- High Alpha (1.5) maintains high variance (0.75-0.95) across the entire Beta range
- All curves peak around Beta=1.0, with slight decreases at higher Beta values
- Medium Alpha values (0.9-1.1) show steeper slopes than high Alpha

## Analysis

### Parameter Interaction Effects

1. **Alpha-Beta Substitution Effect**
   - For low Alpha (0.5), increasing Beta above 0.5 can achieve polarization levels similar to high Alpha values
   - The substitution effect is most evident in Mean Extremity and Extreme Ratio metrics
   - However, the qualitative polarization pattern (as indicated by Bimodality) differs significantly

2. **Critical Thresholds**
   - Beta≈1.0 appears to be a critical threshold for all metrics
   - Beyond this threshold, further increases in Beta have minimal effect
   - For Bimodality, Beta≈0.5 marks a significant transition point for medium Alpha values

3. **Alpha-Beta Parameter Space**
   - Low Alpha + Low Beta: Minimal polarization
   - Low Alpha + High Beta: High polarization, low bimodality
   - Medium Alpha + Low Beta: Moderate polarization, high bimodality
   - Medium Alpha + High Beta: High polarization, low bimodality
   - High Alpha + Any Beta: High polarization, low bimodality

4. **Role of Bimodality**
   - Medium Alpha values (0.9-1.1) with low Beta create the strongest bimodal distributions
   - High Beta eliminates bimodality regardless of Alpha value
   - Alpha=1.5 shows minimal bimodality across all Beta values

### Theoretical Implications

1. **Mechanistic Differences**
   - Self-activation (Alpha) and social influence (Beta) drive polarization through distinct mechanisms
   - Alpha primarily influences the initial formation of polarized opinions
   - Beta transforms the opinion distribution from bimodal to unimodal extreme consensus

2. **Polarization Without Moralization**
   - Even without moralization, the system can reach high polarization
   - The polarization pattern differs from moralized systems, particularly in bimodality behavior

3. **Parameter Phase Transitions**
   - Clear evidence of phase transitions in the parameter space
   - Medium Alpha values (0.9-1.1) represent a critical region where bimodality is maximized at low Beta

## Conclusions

1. **Hypothesis Validation**
   - **Hypothesis 1: Confirmed** - For low Alpha (0.5), sufficiently high Beta (>0.5) can indeed produce polarization levels similar to high Alpha systems in terms of extremity and variance.
   - **Hypothesis 2: Confirmed** - For high Alpha (1.5), Beta has minimal effect on overall polarization metrics.

2. **Novel Findings**
   - The unexpected bimodality behavior reveals important dynamics:
     - Medium Alpha (0.9-1.1) with low Beta creates the strongest opinion division
     - High Beta eliminates bimodality and pushes the system toward uniform extremism
     - High Alpha (1.5) skips the bimodal stage even at low Beta

3. **Practical Implications**
   - In non-moralized social systems:
     - Controlling both self-activation and social influence is necessary to prevent polarization
     - Moderate self-activation with low social influence leads to divided but stable opinion clusters
     - High social influence invariably leads to extreme consensus regardless of self-activation levels

## Future Research Directions

1. Compare these findings with moralized conditions (morality_rate > 0)
2. Explore different network structures and their impact on Alpha-Beta dynamics
3. Investigate the temporal aspects of polarization formation under different parameter regimes
4. Examine parameter combinations around identified critical thresholds
5. Develop intervention strategies based on the understanding of Alpha-Beta parameter effects 