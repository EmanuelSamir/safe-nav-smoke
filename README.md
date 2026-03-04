# Risk-Aware Navigation in Dynamic Smoke Environments via Probabilistic Fourier Neural Operators

This repository contains the codebase for the paper **"Risk-Aware Navigation in Dynamic Smoke Environments via Probabilistic Fourier Neural Operators"**. It provides a novel framework that integrates Probabilistic Fourier Neural Operators (PFNO) to forecast the dynamics of dense smoke environments and a risk-aware Model Predictive Path Integral (MPPI) control strategy that evaluates these probabilistic predictions to navigate safely through stochastic fluid hazards.


## Repository Structure

- `agents/`: Agent definitions (Unicycle, Dubins).
- `configs/`: Hydra configuration files for models, environments, and experiments.
- `media/`: Images and visualization recordings.
- `scripts/`: Standalone scripts for generating rollouts, benchmarking, and analyzing metrics.
- `simulator/`: Dynamic fluid smoke simulation built using PhiFlow.
- `src/`: Core logic containing experimental pipelines, the MPPI-CVaR controller, mathematical models (PFNO, ConvLSTM), and custom environment wrappers.
- `run_experiment.py`: Main entry point for evaluating path planners.
- `scripts/analyze_metrics.py`: Script to parse rollout trajectories to reproduce paper box/bar plots.

## Installation

1. Clone the repository and navigate to its root:
   ```bash
   git clone https://github.com/yourusername/safe-nav-smoke.git
   cd safe-nav-smoke
   ```

2. Establish the virtual environment and install the required torch, fourier-neural-operator, and phiflow dependencies using the provided bash script:
   ```bash
   bash setup.sh
   # This will create a local `.env/` and install requirements.txt automatically.
   ```

## Pipeline Outline

Our navigational framework follows four primary stages: Data Generation, Physics/Behavior Prediction, Forecast Uncertainty Quantification, and finally Navigation and Control evaluation.

### 1. Data Generation

To train the physical neural operators, we simulate completely randomized fluid boundaries using our `simulator/dynamic_smoke.py`. For convenience, the `scripts/generate_smoke_data.py` script manages the generation of random fluid episodes to build testing and validation trajectories.

### 2. Behavior Prediction (Model Training)

The core dynamics map $\rho_{t} \to (\mu, \sigma)_{t+N}$ is learned across the randomized datasets using the **PFNO (FNO3D)** model. We benchmark its structural reconstruction capacity against a temporal **ConvLSTM**.

To automatically launch the training curriculum for the predictive models, invoke the bash script:
```bash
bash src/training/run_all_training.sh
```

*(Note: Ensure paths inside the `.sh` correspond correctly to your local environment context. Results and checkpoints will automatically populate to an `outputs/` directory.)*

### 3. Uncertainty Quantification (Forecasting Rollouts)

We utilize the trained generative boundaries to evaluate pure $N$-step spatiotemporal prediction metrics. 
The forecasting modules generate autoregressive probabilistic density rollouts, which evaluate both explicit spread and associated structural uncertainties ($\sigma$).

You can run batched multi-sample model forecasting (or evaluate a pre-trained model checkpoint) through the evaluation script:
```bash
# Ensure you update the checkpoint paths CKPT_* inside this script before running.
bash scripts/run_rollouts.sh
```

### 4. Policy Planning and Risk Navigation

We map the $N$-step output distributions $(\mu, \sigma)$ to a structural cost map through a Conditional Value-at-Risk (CVaR) conversion mechanism, driving an information-theoretic MPPI approach. 

The baseline experiments evaluated in the paper—and the primary script for dynamically rendering episodes tracking the agent through the environment—is controlled through `run_experiment.py`. It integrates seamlessly with the configurations via hydra overrides:

```bash
# 1. PFNO-MPPI (Our Proposed Probabilistic Framework)
python run_experiment.py --config-name behavior_prediction

# 2. HOCBF Baseline (High-Order Control Barrier Functions)
python run_experiment.py --config-name cbf

# 3. Static MPPI Baseline (Persistence Assumption)
python run_experiment.py --config-name persistent

# 4. Risk-Agnostic MPPI Baseline (Idealized shortest-path reference)
python run_experiment.py --config-name no_risk
```
Experiment rollouts and metrics are saved dynamically to `outputs/`.

### 5. Evaluation and Paper Metrics (Figure Reproduction)

Once the configurations have been evaluated across various randomized test cases, the summary path planning results—which include bounding evaluations of **Max / Mean Smoke Exposure** and **Flight Navigation Time**—can be plotted precisely corresponding to the paper artifacts via the analysis script:

```bash
python scripts/analyze_metrics.py
```
Outputs are routed to `.pdf` and `.png` versions within the central logging directory.

