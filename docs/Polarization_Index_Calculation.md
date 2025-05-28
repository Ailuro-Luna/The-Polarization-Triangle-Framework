# Polarization Index Calculation in The Polarization Triangle Framework

## Overview

Our simulation implements the **Koudenburg Polarization Index**, a quantitative measure that captures the degree of opinion polarization within a population of agents. This index provides a standardized metric ranging from 0 to 100+ to assess how divided opinions are across the population.

## Methodology

### Step 1: Opinion Categorization

The continuous opinion values (ranging from -1 to +1) are discretized into **5 distinct categories**:

| Category | Opinion Range | Description |
|----------|---------------|-------------|
| **Category 1** | `opinion < -0.6` | **Strongly Disagree** |
| **Category 2** | `-0.6 ≤ opinion < -0.2` | **Disagree** |
| **Category 3** | `-0.2 ≤ opinion ≤ 0.2` | **Neutral** |
| **Category 4** | `0.2 < opinion ≤ 0.6` | **Agree** |
| **Category 5** | `opinion > 0.6` | **Strongly Agree** |

### Step 2: Agent Count Extraction

For each category, we count the number of agents:
- `n₁` = Number of agents in Category 1 (Strongly Disagree)
- `n₂` = Number of agents in Category 2 (Disagree)  
- `n₃` = Number of agents in Category 3 (Neutral)
- `n₄` = Number of agents in Category 4 (Agree)
- `n₅` = Number of agents in Category 5 (Strongly Agree)

Where `N = n₁ + n₂ + n₃ + n₄ + n₅` is the total number of agents.

### Step 3: Polarization Index Application

The polarization index is calculated by:

```
Polarization Index = Numerator / Denominator
```

**Numerator (Cross-Neutral Weighted Interactions):**
```
Numerator = 2.14 × (n₂ × n₄) + 2.70 × (n₁ × n₄ + n₂ × n₅) + 3.96 × (n₁ × n₅)
```

**Denominator (Normalization Factor):**
```
Denominator = 0.0099 × N²
```

## Key Features

### 1. **Cross-Neutral Focus**
The formula specifically measures interactions that **cross the neutral point** (Category 3), emphasizing disagreements between opposing sides rather than internal consensus.

### 2. **Distance-Based Weighting**
- **2.14**: Weight for adjacent opposing categories (Disagree ↔ Agree)
- **2.70**: Weight for moderate-extreme opposing pairs (Strongly Disagree ↔ Agree, Disagree ↔ Strongly Agree)
- **3.96**: Weight for extreme opposing categories (Strongly Disagree ↔ Strongly Agree)

### 3. **Quadratic Normalization**
The `N²` term in the denominator accounts for all possible pairwise interactions in the population, providing scale-invariant results.

## Implementation Details

```python
def calculate_polarization_index(self):
    # 1. Discretize opinions into 5 categories
    category_counts = np.zeros(5, dtype=np.int32)
    
    for opinion in self.opinions:
        if opinion < -0.6:
            category_counts[0] += 1      # Strongly Disagree
        elif opinion < -0.2:
            category_counts[1] += 1      # Disagree
        elif opinion <= 0.2:
            category_counts[2] += 1      # Neutral
        elif opinion <= 0.6:
            category_counts[3] += 1      # Agree
        else:
            category_counts[4] += 1      # Strongly Agree
    
    # 2. Extract category counts
    n1, n2, n3, n4, n5 = category_counts
    N = self.num_agents
    
    # 3. Apply Koudenburg formula
    numerator = (2.14 * n2 * n4 + 
                2.70 * (n1 * n4 + n2 * n5) + 
                3.96 * n1 * n5)
    
    denominator = 0.0099 * (N ** 2)
    
    return numerator / denominator if denominator > 0 else 0.0
```

## Interpretation

- **Low Values (0-20)**: Minimal polarization, opinions are relatively consensus-oriented
- **Medium Values (20-50)**: Moderate polarization, some division but not extreme
- **High Values (50-80)**: Significant polarization, clear opposing camps
- **Very High Values (80+)**: Extreme polarization, population is highly divided

## Integration with Simulation

The polarization index is calculated at **every simulation step** and stored in `polarization_history`, allowing for:
- **Temporal analysis** of polarization evolution
- **Comparative studies** across different parameter settings
- **Real-time monitoring** of system dynamics

This metric provides a robust, standardized way to quantify the polarization outcomes of our agent-based model within the Polarization Triangle Framework. 