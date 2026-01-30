import numpy as np
from scipy.ndimage import map_coordinates
from dataclasses import dataclass
import os

@dataclass
class PlaybackSmokeParams:
    data_path: str

class PlaybackSmoke:
    def __init__(self, params: PlaybackSmokeParams):
        self.params = params
        if not os.path.exists(params.data_path):
            raise FileNotFoundError(f"Smoke data file not found: {params.data_path}")
            
        print(f"Loading smoke data from {params.data_path}...")
        loaded = np.load(params.data_path)
        self.data = loaded['smoke_data'] # (N_ep, N_steps, H, W)
        self.x_size = float(loaded['x_size'])
        self.y_size = float(loaded['y_size'])
        self.resolution = float(loaded['resolution'])
        
        self.num_episodes = self.data.shape[0]
        self.max_steps = self.data.shape[1]
        
        # Start before first episode so first reset() call sets it to 0
        self.current_episode_idx = -1 
        self.current_step_idx = 0
        
        print(f"Playback ready: {self.num_episodes} episodes, {self.max_steps} steps, {self.data.shape[2:]} grid")
        
    def reset(self, episode_idx=None):
        if episode_idx is not None:
             self.current_episode_idx = episode_idx % self.num_episodes
        else:
             self.current_episode_idx = (self.current_episode_idx + 1) % self.num_episodes
             
        self.current_step_idx = 0
        
    def step(self, dt=None):
        # dt is ignored, we just advance frame
        if self.current_step_idx < self.max_steps - 1:
            self.current_step_idx += 1
            
    def get_smoke_map(self):
        return self.data[self.current_episode_idx, self.current_step_idx]
        
    def get_smoke_density(self, pos: np.ndarray) -> np.ndarray:
        """
        Sample smoke density at given positions using interpolation.
        pos: (N, 2) array of (x, y) coordinates
        Returns: (N, 1) array of densities
        """
        if pos.ndim == 1:
            pos = pos.reshape(1, 2)
            
        # Map (x, y) to grid indices
        # Grid indices correspond to cell centers. 
        # Assuming centered grid covering [0, x_size] x [0, y_size]
        # Index 0 is at 0.5 * resolution
        # coordinate = (index + 0.5) * resolution
        # index = (coordinate / resolution) - 0.5
        
        x_coords = (pos[:, 0] / self.resolution) - 0.5
        y_coords = (pos[:, 1] / self.resolution) - 0.5
        
        # map_coordinates expects coordinates for each dimension: (dim0_coords, dim1_coords)
        # Grid shape is (H, W) -> (y, x)
        coords = np.stack([y_coords, x_coords]) 
        
        # Get current grid
        grid = self.data[self.current_episode_idx, self.current_step_idx]
        
        # Interpolate
        # mode='constant', cval=0.0 assumes zero smoke outside bounds
        values = map_coordinates(grid, coords, order=1, mode='constant', cval=0.0)
        
        return values.reshape(-1, 1)

    def get_smoke_extent(self):
        return [0, self.x_size, 0, self.y_size]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    # Argument parser to easily switch data files if needed
    parser = argparse.ArgumentParser(description="Test PlaybackSmoke")
    parser.add_argument("--data_path", type=str, default="data/global_source_400_100_2nd.npz", help="Path to .npz smoke data")
    args = parser.parse_args()

    # Create parameters
    params = PlaybackSmokeParams(data_path=args.data_path)

    try:
        # Initialize simulator
        sim = PlaybackSmoke(params)
        
        # Reset to start
        sim.reset()
        
        # Visualization setup
        fig, ax = plt.subplots(figsize=(8, 6))
        
        print(f"Starting playback test on file: {args.data_path}")
        print("Press Ctrl+C to stop...")
        
        # Loop through a few steps
        for i in range(50): # Run for 50 frames or until closed
            smoke_map = sim.get_smoke_map()
            extent = sim.get_smoke_extent()
            
            ax.clear()
            im = ax.imshow(smoke_map, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"Episode {sim.current_episode_idx}, Step {sim.current_step_idx}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            
            # Test point sampling
            # Sample a few points along a diagonal
            test_points = np.array([
                [10, 10],
                [20, 20],
                [30, 15]
            ])
            densities = sim.get_smoke_density(test_points)
            
            # Plot test points
            ax.scatter(test_points[:, 0], test_points[:, 1], c='red', s=50, label='Sensors')
            for j, (x, y) in enumerate(test_points):
                ax.text(x+1, y+1, f"{densities[j, 0]:.2f}", color='yellow')
            
            plt.pause(0.1)
            
            sim.step()
            
            # Loop episode if done
            if sim.current_step_idx >= sim.max_steps - 1:
                print("Episode finished, resetting...")
                sim.reset()
                
        plt.show()
        
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
