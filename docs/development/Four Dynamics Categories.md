# Mathematical Formulation of Opinion Dynamics 

## Parameters and Variables

- $o_i$: Opinion of agent $i$ (ranges from -1 to 1)
- $o_j$: Opinion of neighbor $j$ (ranges from -1 to 1)
- $m_i$: Morality state of agent $i$ (0 for non-moralized, 1 for moralized)
- $m_j$: Morality state of neighbor $j$ (0 for non-moralized, 1 for moralized)
- $I_{same}$: Identity similarity indicator (1 if same identity, 0 if different)
- $\alpha$: Influence factor (controls magnitude of opinion change)
- $R$: Resistance factor = $1 - |o_i|^3$ (makes reaching extreme values harder)

## General Update Rule

$$o_i(t+1) = o_i(t) + \Delta o_i$$

Where $\Delta o_i$ depends on the interaction type determined by opinion direction, identity, and morality states.

## Specific Dynamics

### 1. Convergence (Rules 1, 5, 9, 13)
$$\Delta o_i = \alpha \cdot (o_j - o_i)$$

* Rule 1: Same opinion, same identity, {0,0}
* Rule 5: Same opinion, different identity, {0,0}
* Rule 9: Different opinion, same identity, {0,0}
* Rule 13: Different opinion, different identity, {0,0}

### 2. Pulling (Rules 2, 6, 10, 14)
$$\Delta o_i = \alpha \cdot \beta \cdot (o_j + \gamma \cdot \text{sign}(o_j) \cdot (1-|o_j|) - o_i)$$

Where $\beta$ varies from 1.0-1.5 and $\gamma$ from 0.1-0.2 depending on the rule

* Rule 2: Same opinion, same identity, {0,1}
* Rule 6: Same opinion, different identity, {0,1}
* Rule 10: Different opinion, same identity, {0,1}
* Rule 14: Different opinion, different identity, {0,1}

### 3. Polarization (Rules 3, 4, 7, 8, 12, 16)
$$\Delta o_i = \alpha \cdot \delta \cdot \text{sign}(o_i) \cdot (1-|o_i|) \cdot R$$

Where $\delta$ varies from 0.6-1.5 based on the rule

* Rule 3, 7: Same opinion, {1,0}, agent pulls other toward extremes
* Rule 4: Same opinion, same identity, {1,1}, high polarization
* Rule 8: Same opinion, different identity, {1,1}, moderate polarization
* Rule 12: Different opinion, same identity, {1,1}, low polarization
* Rule 16: Different opinion, different identity, {1,1}, very high polarization

### 4. Resistance
#### 4.1 Low Resistance (Rule 11)
$$\Delta o_i = \alpha \cdot 0.5 \cdot (\text{sign}(o_j) - o_i)$$
* Rule 11: Different opinion, same identity, {1,0}

#### 4.2 High Resistance (Rule 15)
$$\Delta o_i = \alpha \cdot 1.2 \cdot \text{sign}(o_i) \cdot (1-|o_i|) \cdot R$$
* Rule 15: Different opinion, different identity, {1,0}

## Probability Parameters

Each rule has an associated probability parameter determining how likely the interaction will occur:

* Convergence: $p_{conv} \in \{0.3, 0.5, 0.7, 0.9\}$ (low to very high)
* Pulling: $p_{pull} \in \{0.4, 0.6, 0.8\}$ (low to high)
* Polarization: $p_{polar} \in \{0.3, 0.5, 0.7, 0.9\}$ (low to very high)
* Resistance: $p_{resist} \in \{0.4, 0.8\}$ (low to high)
