# Probabilistic Diffusion Distillation (PDD)

Strictly following **Shortcut Models** with **Temporal Memory** and **Relative Coordinates**.

## 1. Input Processing & Temporal Memory

### A. Relative Transform (Invariant Coordinates)
Every position $P_{t-k} = [x_{t-k}, y_{t-k}, \theta_{t-k}]$ in the history is transformed relative to the **current robot pose** $P_t$:
- $\Delta P_{t-k} = R(-\theta_t) \cdot (P_{t-k} - P_t)$
- This ensures the policy is translationally and rotationally invariant.

### B. Temporal Buffer (Tokens over Time)
Maintain a history window of $H$ steps.
- **Latent History**: Each step produces $K$ stochastic tokens $\mathcal{Z} = \{ (\mu, \sigma) \}$.
- **Total Conditioning**: Matrix of $H \times K$ tokens + relative poses.

## 2. Model Architecture

### A. Teacher and Student
- **Teacher**: processes map $74 \times 74 \to K$ tokens.
- **Student**: processes scan line $1 \times 64 \to K$ tokens $(\mu, \sigma)$.

### B. Shortcut Policy w/ Temporal Attention
- **Query (Q)**: `MLP([a_noisy, t_emb, d_emb, goal_rel, action_prev])`.
- **Key/Value (K, V)**: Buffer of $H \times K$ tokens + relative position of each token.

## 3. Algorithms (Shortcut Training)

### Algorithm 1: Training Loop
1. Sample $a_t = (1-t)a_0 + t a_1$.
2. **Case 1: Flow-Matching ($d=0$):**
   - $\mathcal{L} = \|s_\theta(a_t, t, 0, ...) - (a_1 - a_0)\|^2$.
3. **Case 2: Shortcut Consistency ($d > 0$):**
   - $s_{target} = \text{StopGrad}(s_{\theta_{ema}}(t, d) + s_{\theta_{ema}}(t+d, d))/2$
   - $\mathcal{L} = \|s_\theta(a_t, t, 2d, ...) - s_{target}\|^2$.

## 4. Usage
Run from project root:
```bash
python -m pdd.train --dataset data/distillation_v1
```
Test architecture:
```bash
python -m pdd.tests
```
