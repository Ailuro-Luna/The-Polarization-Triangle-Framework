# The Polarization Triangle Framework
## Agent-Neighbor Interaction Model

---

## Basic Model Variables

### Agent Attributes
- **Identity ($l_i$)**: Agent $i$'s identity (±1)
- **Moralization ($m_i$)**: Agent $i$'s moralization (0 or 1)
- **Opinion ($z_i$)**: Agent $i$'s opinion (∈ [-1,1])
- **Signal ($\sigma_{ij}$)**: How agent $j$'s opinion is perceived by agent $i$ (-1, 0, or 1)

### Current Implementation Values
```python
# Agent parameters
self.identities = np.empty(self.num_agents, dtype=np.int32)  # Either 1 or -1
self.morals = np.empty(self.num_agents, dtype=np.int32)      # Either 0 or 1
self.opinions = np.empty(self.num_agents, dtype=np.float64)  # Range [-1, 1]
```

---

## Perceived Opinion Calculation

$$\sigma_{ij} = 
\begin{cases}
\frac{z_j}{|z_j|}, & \text{if } z_j \neq 0 \text{ and } (m_i = 1 \text{ or } m_j = 1) \\
0, & \text{if } z_j = 0 \\
z_j, & \text{if } m_i = 0 \text{ and } m_j = 0
\end{cases}$$

### Implementation
```python
def calculate_perceived_opinion(self, i, j):
    z_j = self.opinions[j]
    m_i = self.morals[i]
    m_j = self.morals[j]
    
    if z_j == 0:
        return 0
    elif (m_i == 1 or m_j == 1):
        return np.sign(z_j)  # Returns sign of z_j (1 or -1)
    else:
        return z_j  # Returns actual value
```

---

## Relationship Coefficient

$$A_{ij} = 
\begin{cases}
-a_{ij}, & \text{if } l_i = -l_j \text{ and } m_i, m_j = 1 \text{ and } \sigma_{ij} \cdot \sigma_{ji} < 0 \\
\frac{a_{ij}}{\sigma_{ji}} \tilde{\sigma}_{sameIdentity}, & \text{if } l_i = l_j \text{ and } m_i, m_j = 1 \text{ and } \sigma_{ij} \cdot \sigma_{ji} < 0 \\
a_{ij}, & \text{otherwise}
\end{cases}$$

Where $a_{ij}$ is the adjacency matrix value and $\tilde{\sigma}_{sameIdentity}$ is the average perceived opinion of neighbors with same identity.

---

## Opinion Dynamics Equation

$$
\dot{z_i} = -\delta_i z_i + u_i \cdot \tanh\left(\alpha_i \sigma_{ii} + \frac{\beta}{1 + \gamma_i m_i} \sum_{j \neq i} A_{ij} \sigma_{ij}\right)
$$

### Components:
1. **Relaxation to neutral opinion**: $-\delta_i z_i$
2. **Opinion activation**: $u_i \cdot \tanh\left(\alpha_i \sigma_{ii} + \frac{\beta}{1 + \gamma_i m_i} \sum_{j \neq i} A_{ij} \sigma_{ij}\right)$
   - **Self-activation**: $\alpha_i \sigma_{ii}$
   - **Social influence**: $\frac{\beta}{1 + \gamma_i m_i} \sum_{j \neq i} A_{ij} \sigma_{ij}$

---

## Current Parameter Values

From `simulation.py`:
```python
# Model parameters
self.delta = 0.1    # Opinion decay rate
self.u = np.ones(self.num_agents) * 0.1  # Opinion activation coefficient
self.alpha = np.ones(self.num_agents) * 0.5  # Self-activation coefficient
self.beta = 0.5     # Social influence coefficient
self.gamma = np.ones(self.num_agents) * 0.5  # Moralization impact coefficient
```

---

## Configuration Parameters

From `config.py` (using `lfr_config`):
```python
# Network parameters
network_type = "lfr"
network_params = {
    "tau1": 3,
    "tau2": 1.5,
    "mu": 0.1,
    "average_degree": 5,
    "min_community": 10
}

# Opinion parameters
opinion_distribution = "twin_peak"
coupling = "partial"
extreme_fraction = 0.1
moral_correlation = "partial"
morality_rate = 0.5  # Moralization rate
influence_factor = 0.1  # Used as step size
```

---

## Implementation of Step Function

```python
def step(self):
    # Initialize opinion changes
    opinion_changes = np.zeros(self.num_agents)
    
    for i in range(self.num_agents):
        # Self-perception
        sigma_ii = np.sign(self.opinions[i]) if self.opinions[i] != 0 else 0
        
        # Calculate neighbor influence
        neighbor_influence = 0
        for j in self.neighbors_list[i]:
            A_ij = self.calculate_relationship_coefficient(i, j)
            sigma_ij = self.calculate_perceived_opinion(i, j)
            neighbor_influence += A_ij * sigma_ij
        
        # Calculate opinion change rate
        regression_term = -self.delta * self.opinions[i]
        activation_term = self.u[i] * np.tanh(
            self.alpha[i] * sigma_ii + 
            (self.beta / (1 + self.gamma[i] * self.morals[i])) * neighbor_influence
        )
        
        # Total change
        opinion_changes[i] = regression_term + activation_term
    
    # Apply opinion changes
    step_size = self.config.influence_factor
    self.opinions += step_size * opinion_changes
    self.opinions = np.clip(self.opinions, -1, 1)
```

---

## Key Model Features

1. **Identity Factor** ($l_i$): Represents social identity, determines interaction patterns
2. **Moralization Factor** ($m_i$): Represents moral intensity of issues, affects perception strength
3. **Opinion Dynamics**: Evolve under dual influence of self-activation and social influence
4. **Network Structure**: Captures structural relationships and interaction patterns

---

## Significance of the Model

The framework aims to simulate and analyze how:
- Opinion distance
- Moral conflict
- Structural alignment

These factors jointly affect group coordination and democratic outcomes in social networks.

---

## Next Steps & Future Development

- Parameter sensitivity analysis
- Testing different network topologies
- Incorporating real-world data
- Adding external influence factors
- Exploring intervention strategies to reduce polarization 