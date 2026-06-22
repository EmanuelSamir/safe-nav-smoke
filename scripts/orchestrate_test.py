import os
import sys
import time
import torch
import hydra
import numpy as np
from typing import Generator, Dict, Any
from prefect import flow, task
from omegaconf import OmegaConf

# Add project root to python path
sys.path.append(os.getcwd())

from src.env.simulator.smoke import SmokeParams, Smoke, BlobParams
from src.env.simulator.playback import Playback, PlaybackParams
from src.env.simulator.playback_schema import SmokeDataSchema
from src.env.simulator.sensor import GlobalSensor, DownwardsSensor, Camera1DSensor
from src.env.smoke_env import SmokeEnv
from src.env.replay_env import ReplayEnv

# Define output and data paths
TEST_SIM_PATH = "data/test_physics_smoke_slow"
TEST_ENV_OUTPUT_DIR = "outputs"
TEST_TRANSITIONS_PATH = f"{TEST_ENV_OUTPUT_DIR}/env_transitions"


@task(name="Generate Simulation Data")
def generate_simulation_data(
    num_episodes: int = 2,
    episode_steps: int = 100,
    x_size: float = 35.0,
    y_size: float = 30.0,
    resolution: float = 0.2,
    dt: float = 0.1,
    force: bool = False,
):
    print("Step 1: Generating simulation data...")
    if os.path.exists(TEST_SIM_PATH) and not force:
        print(f"Simulation data already exists at {TEST_SIM_PATH}. Skipping generation step.")
        return 0.0
        
    start_time = time.time()

    
    H = int(y_size / resolution)
    W = int(x_size / resolution)
    
    def episode_generator():
        for ep in range(num_episodes):
            # Randomized blob params
            num_blobs = np.random.randint(4, 9)
            episode_blobs = []
            for _ in range(num_blobs):
                x_c = np.random.uniform(2.0, x_size - 2.0)
                y_c = np.random.uniform(2.0, y_size - 2.0)
                episode_blobs.append(
                    BlobParams(
                        x_pos=x_c,
                        y_pos=y_c,
                        intensity=1.0,
                        spread_rate=np.random.uniform(1.0, 3.0),
                    )
                )
            
            params = SmokeParams(
                x_size=x_size,
                y_size=y_size,
                resolution=resolution,
                average_wind_speed=5.0,
                smoke_emission_rate=1.8,
                smoke_diffusion_rate=2.5,
                smoke_decay_rate=1.5,
                buoyancy_factor=1.2,
                inflow_bank_count=5,
            )
            sim = Smoke(params, blob_params_list=episode_blobs)

            
            episode_data = np.zeros((episode_steps, H, W), dtype=np.float32)
            for step in range(episode_steps):
                episode_data[step] = sim.get_smoke_map()
                sim.step(dt=dt)
                
            yield {
                SmokeDataSchema.SMOKE_DATA: episode_data,
                SmokeDataSchema.X_SIZE: float(x_size),
                SmokeDataSchema.Y_SIZE: float(y_size),
                SmokeDataSchema.RESOLUTION: float(resolution),
                SmokeDataSchema.DT: float(dt),
                "episode_id": ep,
            }
            
    import datasets
    from datasets import Dataset
    
    features = datasets.Features(
        {
            SmokeDataSchema.SMOKE_DATA: datasets.Array3D(
                shape=(episode_steps, H, W), dtype="float32"
            ),
            SmokeDataSchema.X_SIZE: datasets.Value("float32"),
            SmokeDataSchema.Y_SIZE: datasets.Value("float32"),
            SmokeDataSchema.RESOLUTION: datasets.Value("float32"),
            SmokeDataSchema.DT: datasets.Value("float32"),
            "episode_id": datasets.Value("int32"),
        }
    )
    
    ds = Dataset.from_generator(episode_generator, features=features, writer_batch_size=50)
    os.makedirs(os.path.dirname(TEST_SIM_PATH), exist_ok=True)
    ds.save_to_disk(TEST_SIM_PATH)
    
    duration = time.time() - start_time
    print(f"Simulation data saved to {TEST_SIM_PATH}. Duration: {duration:.2f}s")
    return duration

@task(name="Run Playback Sensors")
def test_sensors_playback():
    print("\nStep 2: Testing 3 sensors in Playback mode...")
    start_time = time.time()
    
    playback_params = PlaybackParams(data_path=TEST_SIM_PATH)
    sim = Playback(playback_params)
    sim.reset()
    
    sensor_configs = ["global", "downwards", "camera1d"]
    curr_pos = torch.tensor([10.0, 10.0, 0.0])
    
    # Programmatically compose configuration
    from hydra import compose, initialize
    
    # We clear initialize status if already initialized
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
        
    with initialize(version_base=None, config_path="../configs"):
        for stype in sensor_configs:
            composed_cfg = compose(config_name="config", overrides=[f"env/sensors@sensor={stype}"])
            sensor_params = OmegaConf.to_object(composed_cfg.sensor)
            
            if stype == "global":
                sensor_class = GlobalSensor
            elif stype == "downwards":
                sensor_class = DownwardsSensor
            elif stype == "camera1d":
                sensor_class = Camera1DSensor
            else:
                raise ValueError(f"Unknown sensor: {stype}")
                
            sensor = sensor_class(sensor_params)
            res = sensor.read(sim, curr_pos)
            print(f" - {sensor_class.__name__} read shape: {res.readings.shape}")
            
    duration = time.time() - start_time
    print(f"Sensors tested in Playback mode. Duration: {duration:.2f}s")
    return duration

@task(name="Run Playback Environment")
def test_smoke_env_playback():
    print("\nStep 3: Running environment using Playback...")
    start_time = time.time()
    
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
        
    with initialize(version_base=None, config_path="../configs"):
        # Load environment config with default override settings
        cfg = compose(
            config_name="config",
            overrides=[
                "env=smoke_env",
                "env/simulator@simulator=playback",
                f"simulator.data_path={TEST_SIM_PATH}",
                f"hydra.run.dir={TEST_ENV_OUTPUT_DIR}",
            ]
        )
        OmegaConf.resolve(cfg)
        
        # Override for a short test
        cfg.env.num_episodes = 2
        cfg.env.max_steps = 100
        cfg.env.save_transitions = True
        
        env_params = OmegaConf.to_object(cfg.env)
        robot_params = OmegaConf.to_object(cfg.agent)
        sensor_params = OmegaConf.to_object(cfg.sensor)
        playback_params = OmegaConf.to_object(cfg.simulator)
        
        print("Initializing SmokeEnv...")
        env = SmokeEnv(
            env_params=env_params,
            robot_params=robot_params,
            sensor_params=sensor_params,
            simulator_params=playback_params,
        )
        
        obs, _ = env.reset()
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
        env.close()
        
    duration = time.time() - start_time
    print(f"Environment run completed and transitions saved. Duration: {duration:.2f}s")
    return duration

@task(name="Test Replay Environment")
def test_replay_env():
    print("\nStep 4: Running ReplayEnv loader test...")
    start_time = time.time()
    
    if not os.path.exists(TEST_TRANSITIONS_PATH):
        raise FileNotFoundError(f"Transitions dataset not found at {TEST_TRANSITIONS_PATH}")
        
    replay = ReplayEnv.from_hf_dataset(TEST_TRANSITIONS_PATH)
    print(f"ReplayEnv successfully loaded {len(replay)} transitions.")
    
    if len(replay) > 0:
        print(f"Replay buffer sample keys: {list(replay[0].keys())}")
        
    duration = time.time() - start_time
    print(f"ReplayEnv verification complete. Duration: {duration:.2f}s")
    return duration

@task(name="Test Model Architectures")
def test_model_architectures():
    print("\nStep 5: Testing model architectures sanity checks...")
    start_time = time.time()
    import subprocess
    
    python_bin = "/Users/emanuelsamir/Documents/dev/cmu/py-envs/dev/bin/python"
    
    # Run ConvLSTM sanity check
    print("Running ConvLSTM sanity check...")
    res_lstm = subprocess.run([python_bin, "src/models/conv_lstm.py"], capture_output=True, text=True)
    if res_lstm.returncode != 0:
        print(res_lstm.stderr)
        raise RuntimeError(f"ConvLSTM sanity check failed with exit code {res_lstm.returncode}")
    print(" - ConvLSTM sanity check passed.")

    # Run FNO sanity check
    print("Running FNO sanity check...")
    res_fno = subprocess.run([python_bin, "src/models/fno.py"], capture_output=True, text=True)
    if res_fno.returncode != 0:
        print(res_fno.stderr)
        raise RuntimeError(f"FNO sanity check failed with exit code {res_fno.returncode}")
    print(" - FNO sanity check passed.")
    
    duration = time.time() - start_time
    print(f"Model sanity checks verification complete. Duration: {duration:.2f}s")
    return duration


@task(name="Test Training Loops (Test Mode)")
def test_training_loops():
    print("\nStep 6: Testing training loops in test mode...")
    start_time = time.time()
    import subprocess
    
    python_bin = "/Users/emanuelsamir/Documents/dev/cmu/py-envs/dev/bin/python"
    
    # Run ConvLSTM training in test mode
    print("Running ConvLSTM training in test mode...")
    res_lstm = subprocess.run(
        [python_bin, "src/training/train_conv_lstm.py", "+test=True", "data.data_path=data/physics_smoke_dry_run"],
        capture_output=True,
        text=True
    )
    if res_lstm.returncode != 0:
        print(res_lstm.stderr)
        raise RuntimeError(f"ConvLSTM training loop test failed with exit code {res_lstm.returncode}")
    print(" - ConvLSTM training loop test passed.")

    # Run FNO training in test mode
    print("Running FNO training in test mode...")
    res_fno = subprocess.run(
        [python_bin, "src/training/train_fno.py", "+test=True", "data.data_path=data/physics_smoke_dry_run"],
        capture_output=True,
        text=True
    )
    if res_fno.returncode != 0:
        print(res_fno.stderr)
        raise RuntimeError(f"FNO training loop test failed with exit code {res_fno.returncode}")
    print(" - FNO training loop test passed.")
    
    duration = time.time() - start_time
    print(f"Training loops verification complete. Duration: {duration:.2f}s")
    return duration


@flow(name="Smoke Safe Navigation Test Suite")
def test_suite_flow():
    t_sim = generate_simulation_data()
    t_sensors = test_sensors_playback()
    t_env = test_smoke_env_playback()
    t_replay = test_replay_env()
    t_models = test_model_architectures()
    t_train = test_training_loops()
    
    print("\n" + "="*50)
    print("        Flow Execution Timing Summary")
    print("="*50)
    print(f"1. Simulation Generation  : {t_sim:.2f}s")
    print(f"2. Sensors Playback Test  : {t_sensors:.2f}s")
    print(f"3. Environment Playback   : {t_env:.2f}s")
    print(f"4. Replay Environment Test: {t_replay:.2f}s")
    print(f"5. Model Sanity Checks    : {t_models:.2f}s")
    print(f"6. Training Loop Checks   : {t_train:.2f}s")
    print("-"*50)
    print(f"Total Flow Duration      : {t_sim + t_sensors + t_env + t_replay + t_models + t_train:.2f}s")
    print("="*50)

if __name__ == "__main__":
    test_suite_flow()
