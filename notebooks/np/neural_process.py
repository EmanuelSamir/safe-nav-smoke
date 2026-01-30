import os
# Enable MPS fallback for grid_sampler_2d_backward
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any

# ==========================================
# 1. Dataset Handling
# ==========================================

class ZeroShotMultiMNIST(Dataset):
    """
    Creates "Zero-Shot Multi-MNIST" samples by placing multiple MNIST digits
    onto a larger canvas. This tests the model's ability to generalize to
    scenes with multiple objects (more numbers) despite being trained on single digits.
    """
    def __init__(self, 
                 root: str = './data', 
                 train: bool = False,
                 num_digits: int = 2,
                 canvas_multiplier: int = 2,
                 min_val: float = 0.0,
                 max_val: float = 1.0,
                 val_offset: float = 0.0):
        
        self.mnist = torchvision.datasets.MNIST(
            root=root, 
            train=train, 
            download=True, 
            transform=transforms.ToTensor()
        )
        
        self.num_digits = num_digits
        self.canvas_size = 28 * canvas_multiplier
        self.min_val = min_val
        self.max_val = max_val
        self.val_offset = val_offset
        
        # Grid for the larger canvas
        H, W = self.canvas_size, self.canvas_size
        x_range = torch.linspace(-1, 1, W)
        y_range = torch.linspace(-1, 1, H)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
        self.grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        
    def __len__(self):
        # Arbitrary length, we generate samples on the fly
        return 1000 

    def __getitem__(self, idx):
        # Create blank canvas
        canvas = torch.zeros((self.canvas_size, self.canvas_size))
        
        # Select n random digits
        indices = np.random.choice(len(self.mnist), self.num_digits, replace=False)
        
        for i in indices:
            img, _ = self.mnist[i] # (1, 28, 28)
            img = img.squeeze()
            
            # Random position
            h_rem = self.canvas_size - 28
            w_rem = self.canvas_size - 28
            r = np.random.randint(0, h_rem + 1)
            c = np.random.randint(0, w_rem + 1)
            
            # Place digit (using MAX to handle overlap)
            # canvas[r:r+28, c:c+28] = torch.max(canvas[r:r+28, c:c+28], img)
            
            # Or sum (with clamp later) - Sum preserves intensity info better for regression
            canvas[r:r+28, c:c+28] += img
            
        y = canvas.reshape(-1, 1)
        
        # Offset and Clamp
        if self.val_offset != 0.0:
            y += self.val_offset
            
        y = torch.clamp(y, self.min_val, self.max_val)
        
        # Return Label as 0 (dummy)
        return self.grid, y, 0


class MNISTRegressionDataset(Dataset):
    """
    Wraps MNIST to provide it as a set of (x, y) points for Neural Processes.
    Each image is treated as a function f: [-1, 1]^2 -> [0, 1].
    """
    def __init__(self, 
                 root: str = './data', 
                 train: bool = True, 
                 min_val: float = 0.0,
                 max_val: float = 1.0,
                 add_noise: float = 0.0,
                 avoid_zeros: bool = False,
                 zero_replacement: float = 1e-3,
                 val_offset: float = 0.0):
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.mnist = torchvision.datasets.MNIST(
            root=root, 
            train=train, 
            download=True, 
            transform=self.transform
        )
        
        self.min_val = min_val
        self.max_val = max_val
        self.add_noise = add_noise
        self.avoid_zeros = avoid_zeros
        self.zero_replacement = zero_replacement
        self.val_offset = val_offset
        
        # Precompute grid coordinates (normalized to [-1, 1])
        # MNIST is 28x28
        H, W = 28, 28
        self.H, self.W = H, W
        x_range = torch.linspace(-1, 1, W)
        y_range = torch.linspace(-1, 1, H)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # (N_pixels, 2) -> Each row is (x, y)
        self.grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2) 

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx] # img is (1, 28, 28)
        
        # Flatten image pixels to (N_pixels, 1)
        y = img.reshape(-1, 1)
        
        # 1. Normalization / Modification logic
        if self.avoid_zeros:
            # Replace absolute zeros with a small epsilon
            y[y < 1e-6] = self.zero_replacement
            
        # Optional: Add noise to pixel values
        if self.add_noise > 0:
            y += torch.randn_like(y) * self.add_noise
        
        # Apply Offset
        if self.val_offset != 0.0:
            y += self.val_offset
            
        # Clamp to ensure validity if needed (e.g. 0-1)
        y = torch.clamp(y, self.min_val, self.max_val)
        
        return self.grid, y, label

# ==========================================
# 2. Convolutional Conditional Neural Process
# ==========================================

class RBFInterpolation(nn.Module):
    """
    maps off-grid points to grid using RBF kernel.
    """
    def __init__(self, grid_res, grid_min=-1, grid_max=1, sigma=0.1):
        super().__init__()
        self.grid_res = grid_res
        self.grid_min = grid_min
        self.grid_max = grid_max
        # Learnable RBF width? Or fixed. Fixed is usually enough for ConvCNP
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True)
        
        # Precompute grid mesh
        x = torch.linspace(grid_min, grid_max, grid_res)
        y = torch.linspace(grid_min, grid_max, grid_res)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Shape: (1, H*W, 2)
        self.register_buffer('grid_points', torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2))

    def forward(self, x_c, y_c):
        """
        x_c: (B, N_c, 2)
        y_c: (B, N_c, y_dim)
        Returns: (B, 1+y_dim, H, W)
           channel 0: density (sum of weights)
           channel 1+: weighted signal
        """
        B, N_c, _ = x_c.shape
        grid_points = self.grid_points # (1, M, 2)
        
        # We need distances between x_c and grid_points
        # Typically x_c is sparse (N_c << M), or not. 
        # For full matrix: (B, N_c, 1, 2) - (1, 1, M, 2) is expensive if M is large.
        # But here 28x28 ~ 784, N_c ~ 50-200. It's fine.
        
        # Dist squared: ||x_c - x_grid||^2
        dists = torch.cdist(x_c, grid_points) ** 2 # (B, N_c, M)
        
        # RBF Weights
        weights = torch.exp(-0.5 * dists / (self.sigma ** 2)) # (B, N_c, M)
        
        # Map to grid
        # Density: sum of weights
        density = weights.sum(dim=1).unsqueeze(1) # (B, 1, M)
        
        # Signal: sum of weights * y_c
        # y_c: (B, N_c, y_dim) -> permute to (B, y_dim, N_c)
        y_c_t = y_c.permute(0, 2, 1)
        signal = torch.bmm(y_c_t, weights) # (B, y_dim, M)
        
        # Combine
        # Avoid division by zero in normalization later if we do it
        # ConvCNP typically passes both [density, signal] to CNN.
        # Or [density, signal / (density + eps)]
        
        normalized_signal = signal / (density + 1e-5)
        
        # Reshape to grid
        out = torch.cat([density, normalized_signal], dim=1) # (B, 1+y_dim, M)
        out = out.reshape(B, -1, self.grid_res, self.grid_res)
        
        return out


class CNNBackbone(nn.Module):
    """Simple ResNet or UNet-like processing on grid."""
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2 * in_channels, 3, padding=1) # Output features
        )
    def forward(self, x):
        return self.net(x)

class ConvCNP(nn.Module):
    def __init__(self, 
                 x_dim=2, 
                 y_dim=1, 
                 grid_res=64):
        super().__init__()
        
        self.grid_res = grid_res
        
        # 1. SetConv (Off-grid -> Grid)
        self.set_conv = RBFInterpolation(grid_res=grid_res, sigma=0.1)
        
        # 2. CNN (Grid -> Grid features)
        # Input channels: 1 (density) + y_dim
        self.cnn = CNNBackbone(in_channels=1+y_dim, hidden_channels=128)
        
        # 3. Readout (Grid features -> mu, sigma)
        # We assume CNN output dimension maps to readout
        # Actually standard ConvCNP often projects CNN output to 2*y_dim (mu, sigma)
        # The CNN above outputs 2*(1+y_dim) which is maybe too much/wrong.
        # Let's align:
        # We want final query interpolation to give us mu, sigma.
        # So the grid should contain (mu_grid, sigma_grid).
        
        self.final_conv = nn.Conv2d(2*(1+y_dim), 2 * y_dim, 1)

    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        x_context: (B, N_c, 2)
        y_context: (B, N_c, y_dim)
        x_target:  (B, N_t, 2)
        """
        
        # 1. Discretize Context
        # feature_grid: (B, 1+y_dim, H, W)
        feature_grid = self.set_conv(x_context, y_context)
        
        # 2. Process with CNN
        # h_grid: (B, hid, H, W)
        h_grid = self.cnn(feature_grid)
        
        # 3. Project to Distribution params on grid
        # out_grid: (B, 2*y_dim, H, W)
        out_grid = self.final_conv(h_grid)
        
        # 4. Interpolate at Target locations
        # x_target is in [-1, 1]. grid_sample expects [-1, 1] (y, x) order?
        # torch.grid_sample expects (B, C, H, W) and (B, H_out, W_out, 2) grid.
        # Here x_target is (B, N_t, 2). We can treat N_t as W_out, H_out=1.
        
        # grid_sample expects coord order (x, y)
        # We constructed grid with meshgrid(y, x, indexing='ij') -> (y, x) order in first implementation?
        # Wait: torch.meshgrid default is 'ij' since recent versions? No 'ij' means matrix indexing (row, col) i.e. (y, x).
        # x_range is dim 1, y_range is dim 0.
        # So grid[..., 0] is y, grid[..., 1] is x.
        # My x_target is (x, y). 
        # grid_sample expects (x, y). 
        # I need to be careful with coordinate alignment. 
        # Let's assume standard intuitive (x, y) input.
        
        # Reshape train data matches: stack(grid_x, grid_y) -> (x, y).
        
        target_grid = x_target.unsqueeze(1) # (B, 1, N_t, 2)
        
        # Interpolate
        # sampled: (B, 2*y_dim, 1, N_t)
        sampled = F.grid_sample(out_grid, target_grid, align_corners=True) 
        
        sampled = sampled.squeeze(2).permute(0, 2, 1) # (B, N_t, 2*y_dim)
        
        mu, log_sigma = torch.chunk(sampled, 2, dim=-1)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        
        return mu, sigma

# ==========================================
# 3. Training & Benchmarking Utils
# ==========================================

def get_context_target_split(x, y, min_context=10, max_context=200):
    """
    Randomly splits points into context and target sets.
    Returns: x_c, y_c, x_t, y_t
    """
    B, N, _ = x.shape
    num_context = np.random.randint(min_context, max_context)
    
    indices = torch.randperm(N)
    idx_c = indices[:num_context]
    
    x_c = x[:, idx_c, :]
    y_c = y[:, idx_c, :]
    
    x_t = x
    y_t = y
    
    return x_c, y_c, x_t, y_t

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        
        # Create random context/target split
        # For ZSMM max_context can be higher
        x_c, y_c, x_t, y_t = get_context_target_split(x, y, min_context=10, max_context=x.shape[1]//4)
        
        optimizer.zero_grad()
        mu, sigma = model(x_c, y_c, x_t)
        
        # Negative Log Likelihood
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(y_t).sum(dim=-1).mean() 
        loss = -log_prob
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate_reconstruction(model, dataset, device, num_context_points=100, num_samples=5):
    """Visualizes reconstruction for a few samples."""
    model.eval()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    axes[0,0].set_title("Ground Truth")
    axes[0,1].set_title(f"Context ({num_context_points} pts)")
    axes[0,2].set_title("ConvCNP Prediction")
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            grid, y, _ = dataset[idx]
            grid = grid.unsqueeze(0).to(device) # (1, N, 2)
            y = y.unsqueeze(0).to(device)       # (1, N, 1)
            
            # Create Fixed Context mask
            N = grid.shape[1]
            perm = torch.randperm(N)
            idx_c = perm[:num_context_points]
            
            x_c = grid[:, idx_c, :]
            y_c = y[:, idx_c, :]
            
            # Predict full image
            mu, sigma = model(x_c, y_c, grid)
            
            # --- Plotting ---
            # Determine appropriate H, W from grid length
            L = grid.shape[1]
            S = int(np.sqrt(L))
            
            # 1. Ground Truth
            gt_img = y.cpu().reshape(S, S)
            axes[i, 0].imshow(gt_img, cmap='gray')
            axes[i, 0].axis('off')
            
            # 2. Context (Sparse)
            ctx_viz = torch.zeros(S, S)
            y_flat = torch.zeros(L)
            y_flat[idx_c] = y_c.cpu().squeeze()
            ctx_viz = y_flat.reshape(S, S)
            
            axes[i, 1].imshow(ctx_viz, cmap='gray')
            axes[i, 1].axis('off')
            
            # 3. Prediction
            pred_img = mu.cpu().reshape(S, S)
            axes[i, 2].imshow(pred_img, cmap='gray')
            axes[i, 2].axis('off')
            
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Main Benchmark Script
# ==========================================

if __name__ == "__main__":
    # COnfiguration
    BATCH_SIZE = 16 
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    # Dataset Setup with variable parameters
    dataset_params = {
        'avoid_zeros': False,
        'zero_replacement': 1e-2,
        'add_noise': 0.0,
        'val_offset': 0.0,
    }
    
    # Use ZSMM for Training and Testing now? Or Train MNIST / Test ZSMM?
    # User said "benchmark comparison", "future changes to mnist".
    # User said "add ZSMM as test".
    # User requested ConvCNP update.
    # Standard practice: Train on ZSMM (or MNIST) -> Test ZSMM.
    # Let's train on ZSMM to fully utilize ConvCNP capabilities for larger scenes.
    print("Loading Zero-Shot Multi-MNIST for Training & Testing...")
    train_ds = ZeroShotMultiMNIST(num_digits=2, canvas_multiplier=2, train=True)
    test_ds = ZeroShotMultiMNIST(num_digits=2, canvas_multiplier=2, train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model Setup
    # Grid res roughly matches canvas size 56x56 -> 64 is good power of 2
    model = ConvCNP(x_dim=2, y_dim=1, grid_res=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # Slightly lower LR for Conv
    
    # Benchmarking Loop
    print("Starting Training (ConvCNP)...")
    loss_history = []
    
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        loss_history.append(loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss (NLL): {loss:.4f}")
        
    print("Training Complete.")
    
    # Visualization
    print("Visualizing results...")
    evaluate_reconstruction(model, test_ds, DEVICE, num_context_points=200)
