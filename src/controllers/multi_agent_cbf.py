"""Multi-agent CBF safety controller.

Uses ``BaseMultiAgentController._make_filter_fns`` to wire the QP-CBF
safety filter into MPPI in one of three modes:

* ``"filter"``  — Hard QP projection applied only at the output step.
* ``"dual-guard"`` — DualGuard: QP projection during rollout AND output.
* ``"penalty"`` — Soft barrier penalty added to the running cost.
"""

from typing import Any

import torch

from src.controllers.base.cbf_safety import CBFFilter, CBFFilterConfig
from src.controllers.base.mppi import MPPIConfig
from src.controllers.base_multi_agent import AgentMPPI, BaseMultiAgentController
from src.controllers.schemas import MultiAgentCBFConfig


class MultiAgentCBFController(BaseMultiAgentController):
    """Unified Control Barrier Function (CBF) multi-agent controller.

    Args:
        num_agents:  Number of agents.
        robot_config: Robot configuration.
        mppi_config: MPPI hyper-parameters.
        cbf_config:  CBF safety parameters.
        cbf_mode:    Safety mode — ``"filter"``, ``"dual-guard"``, or ``"penalty"``.
        goal_thresh: Distance threshold for goal-reached detection.
        device:      PyTorch device string.
        dtype:       PyTorch floating-point dtype.
    """

    def __init__(
        self,
        num_agents: int,
        robot_config: Any,
        config: MultiAgentCBFConfig,
        goal_thresh: float = 0.1,
        dtype=torch.float32,
    ):
        super().__init__(
            num_agents=num_agents,
            robot_config=robot_config,
            config=config,
            goal_thresh=goal_thresh,
            dtype=dtype,
        )
        self.cbf_mode = config.mode
        self.cbf_filter_config = CBFFilterConfig(
            robot_radius=config.safety.robot_radius,
            safe_margin=config.safety.safe_margin,
            r_sense=config.safety.r_sense,
            dt=config.dt,
            k1=config.k1,
            k2=config.k2,
        )

    def _prepare_agent(self, ego_key: str, ego_ctrl: AgentMPPI, neighbors: list) -> None:
        u_min = ego_ctrl.u_min
        u_max = ego_ctrl.u_max
        cbf = CBFFilter(self.cbf_filter_config)

        self._make_filter_fns(
            ego_ctrl,
            self.cbf_mode,
            safety_fn=lambda state, t=0: cbf.h_function(state, neighbors, t),
            qp_fn=lambda state, u, t=0: cbf.qp_filter(state, u, neighbors, u_min, u_max, t),
            safe_margin=self.cbf_filter_config.safe_margin,
            penalty_weight=self.config.rho,
        )


if __name__ == "__main__":
    import numpy as np

    from src.agents.schemas import RobotConfig
    # RobotParams

    robot_config = RobotConfig(
        name="dubins2d",
        device="cpu",
        action_dim=2,
        state_dim=3,
        action_max=[6.0, 4.0],
        action_min=[0.0, -4.0],
        state_max=[30.0, 30.0, 2 * np.pi],
        state_min=[-30.0, -30.0, 0.0],
        dt=0.1,
    )
    from src.controllers.schemas import SharedSafetyConfig

    mppi_config = MPPIConfig(
        nx=3,
        noise_sigma=torch.eye(2),
        num_samples=10,
        horizon=5,
        device="cpu",
        u_min=torch.tensor([0.0, -4.0]),
        u_max=torch.tensor([6.0, 4.0]),
    )
    shared_safety = SharedSafetyConfig(robot_radius=0.4, safe_margin=0.2, r_sense=8.0)

    for mode in ("filter", "dual-guard", "penalty"):
        config = MultiAgentCBFConfig(
            safety=shared_safety, mppi=mppi_config, mode=mode, k1=1.5, k2=1.5, rho=5.0, dt=0.1
        )
        controller = MultiAgentCBFController(
            num_agents=2,
            robot_config=robot_config,
            config=config,
        )
        controller.set_goals({"agent_0": [10.0, 0.0], "agent_1": [-10.0, 0.0]})
        obs = {
            "agent_0": {"location": [0.0, 0.0], "angle": 0.0},
            "agent_1": {"location": [0.5, 0.0], "angle": 0.0},
        }
        cmds = controller.get_commands(obs)
        print(f"[mode={mode}] Test passed. Commands:", {k: v.tolist() for k, v in cmds.items()})
