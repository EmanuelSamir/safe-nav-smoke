# Refactoring Complete: Phase 2 (Architecture) & Phase 3 (Cleanup)

The deep refactoring of the experiment logic is complete. The project now follows a modular, configuration-driven architecture using Hydra and a centralized `BaseExperiment` class.

## New Structure

*   **Entry Points (Root):**
    *   `run_experiment.py`: Universal script to run any experiment.
    *   `collect_data.py`: Script for data collection.

*   **Configuration (`config/`):**
    *   `config.yaml`: Global defaults.
    *   `experiment/`: Specific experiment configs (`base.yaml`, `cbf.yaml`, `vaessm.yaml`).
    *   `env/`, `agent/`: Reusable components.

*   **Source Code (`src/`):**
    *   `experiments/`: Contains `BaseExperiment` and specific implementations (`MPPIExperiment`, `CBFExperiment`, `VAESSMExperiment`).
    *   `models/`: Contains `VAESSMWrapper` (handling legacy checkpoints).
    *   `utils/`, `visualization/`: Consolidated utilities.

*   **Legacy (`legacy/`):**
    *   All old scripts from `run/` have been moved here.

## How to Run

### 1. Run MPPI Experiment (Baseline)
```bash
python run_experiment.py experiment=base
```

### 2. Run CBF Experiment
```bash
python run_experiment.py experiment=cbf
```

### 3. Run VAESSM Experiment
```bash
python run_experiment.py experiment=vaessm
```
*Note: This uses the checkpoint `vaessm_3rd.pt`.*

### 4. Collect Data
```bash
python collect_data.py data_collection.num_samples=1000
```

## Key Changes & Improvements

1.  **Deduplication**: Common logic (loop, logging, rendering) is now in `BaseExperiment`.
2.  **Configuration**: All parameters are in YAML files, making experiments reproducible and easy to modify without changing code.
3.  **Modularity**: Experiments are classes. Models are wrapped with standard interfaces.
4.  **Legacy Support**: `VAESSMWrapper` includes a `LegacyScalarFieldVAESSM` to support your existing checkpoints (`vaessm_3rd.pt`) which had a different architecture than the current `learning/vaessm.py`.
5.  **MPS Support**: Added explicit float32 casting for Mac Metal Performance Shaders (MPS) compatibility.

## Next Steps
*   Review `config/` files to tune parameters as needed.
*   Implement new experiments by subclassing `BaseExperiment` and adding a new config file.
