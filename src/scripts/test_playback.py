import numpy as np
import os
import sys
sys.path.append(os.getcwd())
try:
    from simulator.playback_smoke import PlaybackSmoke, PlaybackSmokeParams
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def main():
    path = "data/smoke_cache.npz"
    if not os.path.exists(path):
        print(f"Data file not found at {path}. Make sure generation finished.")
        return

    print("Initializing PlaybackSmoke...")
    params = PlaybackSmokeParams(data_path=path)
    sim = PlaybackSmoke(params)
    
    print(f"Loaded {sim.num_episodes} episodes, max_steps={sim.max_steps}")
    
    # Test reset
    sim.reset(episode_idx=0)
    print("Reset to episode 0")
    
    # Test get_smoke_map
    grid = sim.get_smoke_map()
    print(f"Grid shape: {grid.shape}")
    
    # Test density interpolation
    # Test points: center of pixel (0.5, 0.5) should match grid[0,0] approximately?
    # Test point (10.5, 10.5)
    pos = np.array([[0.5, 0.5], [10.5, 10.5], [25.0, 25.0]])
    print(f"Sampling at: {pos}")
    den = sim.get_smoke_density(pos)
    print(f"Densities: \n{den}")
    
    assert den.shape == (3, 1), f"Expected shape (3, 1), got {den.shape}"
    
    # Test step
    sim.step()
    print("Stepped 1 frame")
    
    print("Verification successful!")

if __name__ == "__main__":
    main()
