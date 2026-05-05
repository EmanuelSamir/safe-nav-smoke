import torch
import numpy as np
import torch.nn.functional as F

# Internal imports
from .models import PDDTeacherEncoder, PDDStudentEncoder, PDDShortcutPolicy
from .lightning import PDDShortcutModule

def test_teacher():
    print("Testing Teacher Encoder...")
    batch_size = 4
    model = PDDTeacherEncoder(latent_dim=64)
    dummy_map = torch.randn(batch_size, 1, 74, 74)
    tokens = model(dummy_map)
    assert tokens.shape == (batch_size, 16, 64)
    print("  Teacher OK.\n")

def test_student():
    print("Testing Student Encoder...")
    batch_size = 4
    model = PDDStudentEncoder(latent_dim=64, num_tokens=16)
    dummy_scan = torch.randn(batch_size, 1, 64)
    mu, log_sigma = model(dummy_scan)
    assert mu.shape == (batch_size, 16, 64)
    assert log_sigma.shape == (batch_size, 16, 64)
    print("  Student OK.\n")

def test_one_step_inference():
    print("Testing Policy Forward Pass (One-Step Shortcut Inference)...")
    batch_size = 2
    window_size = 10
    horizon = 15
    module = PDDShortcutModule(latent_dim=64, action_dim=4, horizon=horizon)
    
    obs = torch.randn(batch_size, window_size + 1, 64)
    prev_a = torch.randn(batch_size, 4)
    goal_rel = torch.randn(batch_size, 3)
    heading = torch.randn(batch_size, 2)
    rel_poses = torch.randn(batch_size, window_size + 1, 3)
    
    # Forward uses EMA policy to jump from a0 to a1
    pred_traj = module(obs, None, prev_a, goal_rel, heading, rel_poses)
    print(f"  Predicted Trajectory: {pred_traj.shape} (Expected: (B, 15, 4))")
    assert pred_traj.shape == (batch_size, horizon, 4)
    print("  Inference OK.\n")

def test_training_step():
    print("Testing Algorithm 1 Training Step (Shortcut Distillation)...")
    batch_size = 12 # Ensure it splits 75/25 correctly (9 FM, 3 SC)
    window_size = 5
    horizon = 15
    module = PDDShortcutModule(latent_dim=64, action_dim=4, horizon=horizon, sc_ratio=0.25, M=128)
    
    batch = {
        "obs": torch.randn(batch_size, window_size + 1, 64),
        "local_map": torch.randn(batch_size, window_size + 1, 1, 74, 74),
        "expert_traj": torch.randn(batch_size, horizon, 4),
        "prev_action": torch.randn(batch_size, 4),
        "goal_rel": torch.randn(batch_size, 3),
        "heading": torch.randn(batch_size, 2),
        "rel_poses": torch.randn(batch_size, window_size + 1, 3)
    }
    
    # Should calculate FM loss (9/12) and SC loss (3/12)
    loss = module.training_step(batch, batch_idx=0)
    print(f"  Total Loss: {loss:.4f}")
    assert not torch.isnan(loss)
    print("  Algorithm 1 Training OK.\n")

def test_validation_step():
    print("Testing PDDShortcutModule Validation Step...")
    batch_size = 4
    window_size = 5
    horizon = 15
    module = PDDShortcutModule(latent_dim=64, action_dim=4, horizon=horizon)
    
    batch = {
        "obs": torch.randn(batch_size, window_size + 1, 64),
        "local_map": torch.randn(batch_size, window_size + 1, 1, 74, 74),
        "expert_traj": torch.randn(batch_size, horizon, 4),
        "prev_action": torch.randn(batch_size, 4),
        "goal_rel": torch.randn(batch_size, 3),
        "heading": torch.randn(batch_size, 2),
        "rel_poses": torch.randn(batch_size, window_size + 1, 3)
    }
    
    # This should run forward inside and log val_loss
    loss = module.validation_step(batch, batch_idx=0)
    print(f"  Validation Loss: {loss:.4f}")
    assert not torch.isnan(loss)
    print("  Validation Step OK.\n")

if __name__ == "__main__":
    test_teacher()
    test_student()
    test_one_step_inference()
    test_training_step()
    test_validation_step()
    print("All PDD Model & Loop Tests Passed Successfully!")
