

self.L = 0.4  # Lookahead distance for safety point

    def cbf_h_function(
        self, state: torch.Tensor, neighbors: list, d_safe: float, k1: float, dt: float, t: int
    ) -> torch.Tensor:
        K = state.shape[0]
        device = state.device
        if len(neighbors) == 0:
            return 1000.0 * torch.ones(K, device=device)

        p_i_center = state[:, [STATE_X, STATE_Y]]
        theta = state[:, STATE_THETA]

        # Ego safety point: p_i_safe = p_i_center + L * [cos(theta), sin(theta)]
        p_i_safe = p_i_center + self.L * torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

        if isinstance(neighbors, list):
            neighbors_tensor = torch.stack(neighbors).to(device)
        else:
            neighbors_tensor = neighbors.to(device)

        p_j_center = neighbors_tensor[:, [STATE_X, STATE_Y]]
        theta_j = neighbors_tensor[:, STATE_THETA]

        # Neighbor safety point: p_j_safe = p_j_center + L * [cos(theta_j), sin(theta_j)]
        p_j_safe = p_j_center + self.L * torch.stack(
            [torch.cos(theta_j), torch.sin(theta_j)], dim=1
        )

        # Dynamic neighbor velocity prediction:
        v_nominal = self.action_max[0] / 2.0
        v_j_x = v_nominal * torch.cos(theta_j)
        v_j_y = v_nominal * torch.sin(theta_j)
        v_j = torch.stack([v_j_x, v_j_y], dim=1)

        p_j_pred = p_j_safe + v_j * (t * dt)
        d_safe_barrier = d_safe + 2.0 * self.L + 0.2
        p_rel = p_i_safe.unsqueeze(1) - p_j_pred.unsqueeze(0)  # (K, N_neigh, 2)

        dist_sq = torch.sum(p_rel**2, dim=2)
        h0 = dist_sq - d_safe_barrier**2

        h_comp, _ = torch.min(h0, dim=1)
        return h_comp

    def cbf_safe_control(
        self,
        state: torch.Tensor,
        neighbors: list,
        d_safe: float,
        k1: float,
        k2: float,
        dt: float,
        t: int,
        u_min: torch.Tensor,
        u_max: torch.Tensor,
    ) -> torch.Tensor:
        K = state.shape[0]
        device = state.device
        if len(neighbors) == 0:
            return torch.zeros(K, 2, device=device)

        p_i_center = state[:, [STATE_X, STATE_Y]]
        theta = state[:, STATE_THETA]
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        v_nominal = float(self.action_max[0] / 2.0)

        # Ego safety point: p_i_safe = p_i_center + L * [cos(theta), sin(theta)]
        p_i_safe = p_i_center + self.L * torch.stack([cos_t, sin_t], dim=1)

        if isinstance(neighbors, list):
            neighbors_tensor = torch.stack(neighbors).to(device)
        else:
            neighbors_tensor = neighbors.to(device)

        p_j_center = neighbors_tensor[:, [STATE_X, STATE_Y]]
        theta_j = neighbors_tensor[:, STATE_THETA]

        # Neighbor safety point: p_j_safe = p_j_center + L * [cos(theta_j), sin(theta_j)]
        p_j_safe = p_j_center + self.L * torch.stack(
            [torch.cos(theta_j), torch.sin(theta_j)], dim=1
        )

        # Dynamic neighbor velocity prediction:
        v_j_x = v_nominal * torch.cos(theta_j)
        v_j_y = v_nominal * torch.sin(theta_j)
        v_j = torch.stack([v_j_x, v_j_y], dim=1)

        p_j_pred = p_j_safe + v_j * (t * dt)
        p_rel = p_i_safe.unsqueeze(1) - p_j_pred.unsqueeze(0)  # (K, N_neigh, 2)
        dist_sq = torch.sum(p_rel**2, dim=2)  # (K, N_neigh)
        d_safe_barrier = d_safe + 2.0 * self.L + 0.2
        h0_all = dist_sq - d_safe_barrier**2  # (K, N_neigh)

        # Find the critical (closest) neighbor for each batch item
        crit_indices = torch.argmin(h0_all, dim=1)
        batch_idx = torch.arange(K, device=device)

        h0_crit = h0_all[batch_idx, crit_indices]
        p_rel_crit = p_rel[batch_idx, crit_indices]
        v_j_crit = v_j[crit_indices]

        # Coefficients for action projection: R(theta, L)
        # Safety point speed = R * u where R = [cos(theta), -L*sin(theta); sin(theta), L*cos(theta)]
        # A = 2 * p_rel^T * R
        A_v = 2.0 * (p_rel_crit[:, 0] * cos_t + p_rel_crit[:, 1] * sin_t)
        A_w = 2.0 * (-self.L * p_rel_crit[:, 0] * sin_t + self.L * p_rel_crit[:, 1] * cos_t)
        A = torch.stack([A_v, A_w], dim=1)  # (K, 2)

        # 1st-degree HOCBF constraint: A * u >= B where B = 2 * p_rel_crit^T * v_j_crit - k1 * h0_crit
        B = 2.0 * torch.sum(p_rel_crit * v_j_crit, dim=1) - k1 * h0_crit

        # Project a preferred nominal forward velocity u_pref = [v_nominal, 0.0] instead of [0.0, 0.0]
        u_pref = torch.zeros(K, 2, device=device)
        u_pref[:, 0] = v_nominal

        # A * u_pref
        A_u_pref = A[:, 0] * u_pref[:, 0]

        # Violation of the preferred control: violation = B - A * u_pref
        violation = B - A_u_pref

        # Projection formula: u_safe = u_pref + (max(0, violation) / ||A||^2) * A
        A_norm_sq = torch.clamp(torch.sum(A**2, dim=1), min=1e-6)
        u_safe = u_pref + (torch.clamp(violation, min=0.0) / A_norm_sq).unsqueeze(-1) * A

        # Clamp analytically computed backup controls to physical bounds
        u_safe_clamped = torch.max(torch.min(u_safe, u_max), u_min)
        return u_safe_clamped