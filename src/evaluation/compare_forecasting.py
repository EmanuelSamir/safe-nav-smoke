import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import os
import logging
from tqdm import tqdm

from src.models.wrappers import SNPWrapper, PINNWrapper, SmokeModelWrapper
from src.models.model_free.snp_v2 import SnpV2, SnpV2Params
from src.models.model_based.pinn_cnp import PINN_CNP
# Import other models as needed

# Assuming we use the Sequential Dataset for evaluation context
from src.models.model_free_utils import SequentialDataset, sequential_collate_fn
from envs.replay_buffer import GenericReplayBuffer

def evaluate_models(models: Dict[str, SmokeModelWrapper], dataset, device='cpu'):
    results = {}
    
    # We will iterate through dataset sequences
    # Evaluation protocol: 
    # Context: First 10 frames
    # Target: Next 15 frames
    context_len = 10
    target_len = 15
    
    # We need a loader that returns sequences of length context + target
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=sequential_collate_fn)
    
    mae_stats = {name: [] for name in models.keys()}
    
    for batch in tqdm(loader, desc="Evaluating"):
        # batch is list of steps. Length should be >= 25
        if len(batch) < context_len + target_len:
            continue
            
        full_seq = batch
        context_seq = full_seq[:context_len]
        target_seq = full_seq[context_len:context_len+target_len]
        
        # Prepare Context Data
        # For SNP: List of Obs
        context_obs = [step['obs'].to(device) for step in context_seq]
        context_actions = torch.stack([step['action'].to(device) for step in context_seq])
        
        # Future Actions (known for control, or assumed)
        future_actions = torch.stack([step['action'].to(device) for step in target_seq])
        
        # Query Points: We want to evaluate at the GT locations of the target frames
        # Each target step has a set of points.
        # Wrappers expect query_locations. 
        # If we evaluate point-wise, we do it per step or aggregated.
        # Let's do step-by-step evaluation if wrapper supports it, or simple loop here.
        
        # The standardized wrapper `predict_future` assumes we give query times and locations.
        # But locations might change per frame in our dataset (sparse scanning).
        # PINN wrapper handles (T, N) grid.
        # SNP wrapper handles (T, N).
        
        # Simplified Eval: For each target frame, predict at its locations.
        
        for name, model in models.items():
            model_mae = 0
            
            # Predict for the whole horizon?
            # Or step by step?
            # Our wrapper `predict_future` does rollout.
            # But if query locations change every frame, we might need a more flexible interface 
            # or call it per frame?
            # Let's assume we can query specific points at specific times.
            
            # For simplicity in this demo, let's just use the wrapper to predict 
            # at the locations of the 13th frame (example) or accumulated.
            
            # Actually, to compute MAE over all points:
            # We can run the wrapper to get a flexible prediction object or loop.
            
            # Let's manually interact with wrapper or expand wrapper capabilities later.
            # For now, implemented wrapper assumes fixed N locations for all T.
            # If dataset has varying N, we might pick a subset or pad.
            pass
            
            # Placeholder for MAE calculation
            # mae_stats[name].append(np.mean(np.abs(pred - gt)))

    return mae_stats

if __name__ == "__main__":
    # Example usage
    pass
