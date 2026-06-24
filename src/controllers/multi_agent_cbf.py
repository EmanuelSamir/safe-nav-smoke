"""Multi-agent CBF safety controller.

Uses ``BaseMultiAgentController._make_filter_fns`` to wire the QP-CBF
safety filter into MPPI in one of three modes:

* ``"filter"``  — Hard QP projection applied only at the output step.
* ``"rollout"`` — DualGuard: QP projection during rollout AND output.
* ``"penalty"`` — Soft barrier penalty added to the running cost.
"""

from typing import Any

import torch

from src.controllers.base.cbf_safety import CBFFilter, CBFFilterParams
from src.controllers.base.mppi import MPPIParams
from src.controllers.base_multi_agent import AgentMPPI, BaseMultiAgentController, SafetyMode


class MultiAgentCBFController(BaseMultiAgentController):
    """Unified Control Barrier Function (CBF) multi-agent controller.

    Args:
        num_agents:  Number of agents.
        robot_params: Robot configuration.
        mppi_params: MPPI hyper-parameters.
        cbf_params:  CBF safety parameters.
        cbf_mode:    Safety mode — ``"filter"``, ``"rollout"``, or ``"penalty"``.
        goal_thresh: Distance threshold for goal-reached detection.
        device:      PyTorch device string.
        dtype:       PyTorch floating-point dtype.
        dt:          Simulation timestep (s).
    """

    def __init__(
        self,
        num_agents: int,
        robot_params: Any,
        mppi_params: MPPIParams,
        cbf_params: CBFFilterParams,
        cbf_mode: SafetyMode = "filter",
        goal_thresh: float = 0.1,
        device: str = "cpu",
        dtype=torch.float32,
        dt: float = 0.1,
    ):
        super().__init__(
            num_agents=num_agents,
            robot_params=robot_params,
            mppi_params=mppi_params,
            goal_thresh=goal_thresh,
            device=device,
            dtype=dtype,
            dt=dt,
            r_sense=cbf_params.r_sense,
        )
        self.cbf_params = cbf_params
        self.cbf_mode = cbf_mode

    def _prepare_agent(self, ego_key: str, ego_ctrl: AgentMPPI, neighbors: list) -> None:
        u_min = ego_ctrl.u_min
        u_max = ego_ctrl.u_max
        cbf = CBFFilter(self.cbf_params)

        self._make_filter_fns(
            ego_ctrl,
            self.cbf_mode,
            safety_fn=lambda state, t=0: cbf.h_function(state, neighbors, t),
            qp_fn=lambda state, u, t=0: cbf.qp_filter(state, u, neighbors, u_min, u_max, t),
            safe_margin=0.0,
            penalty_weight=self.cbf_params.rho,
        )


if __name__ == "__main__":
    import numpy as np

    from agents.basic_robot import RobotParams

    robot_params = RobotParams(
        name="dubins",
        action_dim=2,
        state_dim=3,
        action_max=[6.0, 4.0],
        action_min=[0.0, -4.0],
        state_max=[30.0, 30.0, 2 * np.pi],
        state_min=[-30.0, -30.0, 0.0],
        dt=0.1,
    )
    mppi_params = MPPIParams(
        nx=3,
        noise_sigma=torch.eye(2),
        num_samples=10,
        horizon=5,
        device="cpu",
        u_min=torch.tensor([0.0, -4.0]),
        u_max=torch.tensor([6.0, 4.0]),
    )
    cbf_params = CBFFilterParams(d_safe=1.6, k1=2.5, k2=2.5, dt=0.1, r_sense=4.0)

    for mode in ("filter", "rollout", "penalty"):
        controller = MultiAgentCBFController(
            num_agents=2,
            robot_params=robot_params,
            mppi_params=mppi_params,
            cbf_params=cbf_params,
            cbf_mode=mode,
        )
        controller.set_goals({"agent_0": [10.0, 0.0], "agent_1": [-10.0, 0.0]})
        obs = {
            "agent_0": {"location": [0.0, 0.0], "angle": 0.0},
            "agent_1": {"location": [0.5, 0.0], "angle": 0.0},
        }
        cmds = controller.get_commands(obs)
        print(f"[mode={mode}] Test passed. Commands:", {k: v.tolist() for k, v in cmds.items()})
