# Safe Navigation in Smoke Environments via Multi-Agent Drones

This repository contains the infrastructure and algorithms for a research project focused on the **safe navigation of multiple drones (2D) in simulated smoke environments**.

The main objective is to enable multiple agents to navigate safely (avoiding collisions and minimizing risk when crossing hazardous zones) using environment behavior prediction and predictive control algorithms.

## 🚀 Key Features

1. **Smoke Forecasting:**
   - We train models to predict smoke propagation over time.
   - **Primary Proposal:** Fourier Neural Operator (FNO).
   - **Baseline:** ConvLSTM.
   
2. **Predictive Control & Navigation:**
   - We use multiple variants of **MPPI** (Model Predictive Path Integral).
   - We implemented a **Dual Guard** controller, which inherits from MPPI but incorporates a safety function (Safety Shield / Control Barrier Functions / Hamilton-Jacobi Reachability) during the rollout to theoretically guarantee safety.

3. **Controlled Simulation (Playback):**
   - Since simulating fluid dynamics for smoke is computationally expensive, the environment uses a **Playback** system.
   - Playback reads previously saved simulations to guarantee fair 1-to-1 comparisons between different controllers and models under identical conditions.

---

## 📂 Architecture

The codebase is mainly divided into core modules (`src/`) and the execution pipelines (`projects/`).

* `src/env/`: Simulation environments, including `playback` logic to load pre-calculated smoke episodes.
* `src/models/` and `src/training/`: Neural network architectures (FNO, ConvLSTM) in PyTorch Lightning and their training logic.
* `src/controllers/`: Navigation controllers (MPPI, Dual Guard, CBF, HJ).
* `src/agents/`: Robot dynamics and parameters.
* `src/wrappers/`: Adapters to couple smoke predictions with the environment.
* `projects/`: Self-contained project folders for execution pipelines.

## 🏗️ Configuration Philosophy

The project has completely migrated away from Hydra and now uses **Strict Pydantic** for configuration management. This architecture promotes clean, reproducible experiments with the following design rules:

* **Separation of Concerns:** Core code (`src/`) is completely agnostic to execution. It simply defines the logic and Pydantic schemas with sensible defaults.
* **Minimal YAML Configurations:** Project YAMLs are now extremely clean. They only contain the parameters you want to **override**. All other parameters automatically inherit their defaults from the Pydantic schemas.
* **No Ghost Parameters:** We strictly use `model_config = ConfigDict(extra="forbid")` in Pydantic so that any obsolete or misspelled parameter in a YAML will immediately halt execution.
* **Reproducibility (`config_used.yaml`):** Every time an experiment runs, it merges the minimal YAML with the Python defaults and saves the complete, resolved configuration as `config_used.yaml` in the output directory. This serves as the single source of truth for reproducibility.

### Projects Structure (`projects/`)

Experiments and pipelines are grouped in the `projects/` folder. Each project is self-contained and groups its own orchestration logic, compound Pydantic schemas, and local YAML configurations. No global `configs/` folder is needed anymore.

```text
├── src/                             # CORE CODE (execution-agnostic)
│
└── projects/                        # CUMULATIVE EXPERIMENTS & PIPELINES
    │
    ├── 1_data_collection/           # PROJECT 1: Collect smoke and environment data
    │   ├── schema.py                # Schema tailored ONLY for data collection
    │   ├── 0_data_playback...py     # Executable scripts
    │   └── physics_config.yaml      # Minimal YAML overriding specific params
    │
    ├── 2_training/                  # PROJECT 2: Train prediction models
    │   ├── schema.py                # Schema defining FNO and ConvLSTM training settings
    │   ├── run_fno.py               # Executable for training FNO
    │   └── fno_config.yaml          # Minimal YAML for FNO training
    │
    └── 4_controller_comparison/     # PROJECT 4: Benchmark controllers
        ├── schema.py                # Schema tailored for benchmark runs
        ├── benchmark.py             # Script to evaluate controllers
        └── benchmark_config.yaml    # Config listing the controllers to test
```

This structure makes it much easier to iterate on isolated experiments (e.g. testing models, comparing controllers, running the full integration) without breaking past pipelines or carrying over inherited configurations.
