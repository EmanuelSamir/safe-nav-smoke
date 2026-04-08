from dataclasses import dataclass
from typing import List


@dataclass
class SmokeDataSchema:
    """Explicit definition of the data structure for Smoke Playback and Replay Buffers."""

    # Primary data keys (Playback & Sequential)
    SMOKE_DATA = "smoke_data"  # (N_episodes, N_steps, H, W)
    X_SIZE = "x_size"  # float (meters)
    Y_SIZE = "y_size"  # float (meters)
    RESOLUTION = "resolution"  # float (meters per pixel)
    DT = "dt"  # float (seconds per step)
    NUM_EPISODES = "num_episodes"  # int
    EPISODE_STEPS = "episode_steps"  # int
    METADATA = "metadata"  # dict (info)

    # Transition keys (Replay Buffer / HF Dataset)
    OBS_LOCATION = "obs_location"  # [x, y]
    OBS_ANGLE = "obs_angle"  # [theta]
    OBS_READINGS = "obs_readings"  # [r1, r2, ...]
    OBS_FULL_MAP = "obs_full_map"  # [[...]] 2D grid
    ACTION = "action"  # [v, w]
    REWARD = "reward"  # float
    NEXT_OBS_LOCATION = "next_obs_location"
    NEXT_OBS_READINGS = "next_obs_readings"
    NEXT_OBS_FULL_MAP = "next_obs_full_map"
    TERMINATED = "terminated"  # bool
    TRUNCATED = "truncated"  # bool

    @classmethod
    def get_playback_keys(cls) -> List[str]:
        return [cls.SMOKE_DATA, cls.X_SIZE, cls.Y_SIZE, cls.RESOLUTION, cls.DT]

    @classmethod
    def get_replay_buffer_keys(cls) -> List[str]:
        return [
            cls.OBS_LOCATION,
            cls.OBS_ANGLE,
            cls.OBS_READINGS,
            cls.OBS_FULL_MAP,
            cls.ACTION,
            cls.REWARD,
            cls.NEXT_OBS_LOCATION,
            cls.NEXT_OBS_READINGS,
            cls.NEXT_OBS_FULL_MAP,
            cls.TERMINATED,
            cls.TRUNCATED,
        ]
