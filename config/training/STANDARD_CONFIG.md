# Standard Configuration Schema for Safe Nav Smoke Models

## Training Configuration Structure

All training configs should follow this standardized structure:

```yaml
training:
  experiment_name: "model_name_experiment"
  seed: 42
  
  # Data configuration
  data:
    buffer_path: "data/raw/replay_buffer.npz"  # For model-free
    # OR
    data_path: "data/raw/global_source_200_100.npz"  # For model-based
    
    batch_size: 32
    train_split: 0.9
    max_samples: null  # null = use all data
    
    # Model-free specific
    sequence_length: 10  # For sequential models (RNP, SNP)
    num_workers: 0
    
    # Model-based specific
    context_frames: 10
    target_frames: 15
    info_ratio_per_frame: 0.2
  
  # Model architecture
  model:
    # Common parameters (use consistent names)
    latent_dim: 128        # z dimension (stochastic)
    hidden_dim: 128        # h dimension (deterministic state)
    embed_dim: 128         # r dimension (representation)
    
    # Network dimensions
    encoder_hidden_dim: 128
    decoder_hidden_dim: 128
    prior_hidden_dim: 128
    posterior_hidden_dim: 128
    
    # Model-specific
    action_dim: 2          # For model-free
    use_actions: false     # For model-based (NEW: make actions optional)
    out_mode: "full"       # For PINN models: "full" or "lite"
    
  # Optimizer configuration
  optimizer:
    lr: 1e-4
    max_epochs: 200
    grad_clip: 1.0
    beta_max: 1.0          # For KL annealing
    min_lr: 1e-6           # For schedulers
  
  # Loss configuration
  loss:
    mse_weight: 1.0
    kl_weight: 1.0
    pde_weight: 1.0        # For PINN models
    pde_type: "blind_discovery"  # or "navier_stokes"
```

## Naming Conventions

### Dimensions
- `latent_dim` or `z_dim`: Stochastic latent dimension
- `hidden_dim` or `deter_dim`: Deterministic state dimension (RNN/LSTM hidden)
- `embed_dim` or `r_dim`: Representation/embedding dimension
- `action_dim`: Action space dimension

### Data
- `buffer_path`: For sequential replay buffer (model-free)
- `data_path`: For pre-generated datasets (model-based)
- `sequence_length`: Temporal window for sequential models
- `context_frames` + `target_frames`: For Neural Process models

### Model Types
- **Model-Free**: RNP, SNP_v1, SNP_v2 (use `buffer_path`, require `action_dim`)
- **Model-Based**: PINN_CNP, PINN_LNP (use `data_path`, `use_actions` optional)
