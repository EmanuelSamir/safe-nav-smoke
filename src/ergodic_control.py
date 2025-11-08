import numpy as np 
np.set_printoptions(precision=4)
rng = np.random.default_rng(10)

from scipy.stats import multivariate_normal as mvn
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from io import BytesIO
import imageio
from envs.smoke_env_dyn import DynamicSmokeEnv, EnvParams, RobotParams, SmokeBlobParams, DownwardsSensorParams, DynamicSmokeParams
from src.utils import *


class iLQRErgodicControl:
    def __init__(self, dt, tsteps, x_dim, u_dim, Q_z, R_v) -> None:
        self.dt = dt 
        self.tsteps = tsteps 

        self.x_dim = x_dim 
        self.u_dim = u_dim

        self.Q_z = Q_z 
        self.Q_z_inv = np.linalg.inv(Q_z)
        self.R_v = R_v 
        self.R_v_inv = np.linalg.inv(R_v)

        self.curr_x_traj = None 
        self.curr_y_traj = None

    def dyn(self, xt, ut):
        raise NotImplementedError("Not implemented.")

    def step(self, xt, ut): 
        """RK4 integration"""
        k1 = self.dt * self.dyn(xt, ut)
        k2 = self.dt * self.dyn(xt + k1/2.0, ut)
        k3 = self.dt * self.dyn(xt + k2/2.0, ut)
        k4 = self.dt * self.dyn(xt + k3, ut)

        xt_new = xt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return xt_new 
    
    def traj_sim(self, x0, u_traj):
        x_traj = np.zeros((self.tsteps, self.x_dim))
        xt = x0.copy()
        for t_idx in range(self.tsteps):
            xt = self.step(xt, u_traj[t_idx])
            x_traj[t_idx] = xt.copy()
        return x_traj
    
    def loss(self):
        raise NotImplementedError("Not implemented.")
    
    def get_At_mat(self, t_idx):
        raise NotImplementedError("Not implemented.")
    
    def get_Bt_mat(self, t_idx):
        raise NotImplementedError("Not implemented.")

    def get_at_vec(self, t_idx):
        raise NotImplementedError("Not implemented.")
    
    def get_bt_vec(self, t_idx):
        raise NotImplementedError("Not implemented.")

    # the following functions are utilities for solving the Riccati equation
    def P_dyn_rev(self, Pt, At, Bt, at, bt):
        return Pt @ At + At.T @ Pt - Pt @ Bt @ self.R_v_inv @ Bt.T @ Pt + self.Q_z 
    
    def P_dyn_step(self, Pt, At, Bt, at, bt):
        k1 = self.dt * self.P_dyn_rev(Pt, At, Bt, at, bt)
        k2 = self.dt * self.P_dyn_rev(Pt+k1/2, At, Bt, at, bt)
        k3 = self.dt * self.P_dyn_rev(Pt+k2/2, At, Bt, at, bt)
        k4 = self.dt * self.P_dyn_rev(Pt+k3, At, Bt, at, bt)

        Pt_new = Pt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return Pt_new 
    
    def P_traj_revsim(self, PT, A_traj, B_traj, a_traj, b_traj):
        P_traj_rev = np.zeros((self.tsteps, self.x_dim, self.x_dim))
        P_curr = PT.copy()
        for t in range(self.tsteps):
            At = A_traj[-1-t]
            Bt = B_traj[-1-t]
            at = a_traj[-1-t]
            bt = b_traj[-1-t]

            P_new = self.P_dyn_step(P_curr, At, Bt, at, bt)
            P_traj_rev[t] = P_new.copy()
            P_curr = P_new 
        
        return P_traj_rev

    def r_dyn_rev(self, rt, Pt, At, Bt, at, bt):
        return (At - Bt @ self.R_v_inv @ Bt.T @ Pt).T @ rt + at - Pt @ Bt @ self.R_v_inv @ bt

    def r_dyn_step(self, rt, Pt, At, Bt, at, bt):
        k1 = self.dt * self.r_dyn_rev(rt, Pt, At, Bt, at, bt)
        k2 = self.dt * self.r_dyn_rev(rt+k1/2, Pt, At, Bt, at, bt)
        k3 = self.dt * self.r_dyn_rev(rt+k2/2, Pt, At, Bt, at, bt)
        k4 = self.dt * self.r_dyn_rev(rt+k3, Pt, At, Bt, at, bt)

        rt_new = rt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return rt_new
    
    def r_traj_revsim(self, rT, P_traj, A_traj, B_traj, a_traj, b_traj):
        r_traj_rev = np.zeros((self.tsteps, self.x_dim))
        r_curr = rT
        for t in range(self.tsteps):
            Pt = P_traj[-1-t]
            At = A_traj[-1-t]
            Bt = B_traj[-1-t]
            at = a_traj[-1-t]
            bt = b_traj[-1-t]

            r_new = self.r_dyn_step(r_curr, Pt, At, Bt, at, bt)
            r_traj_rev[t] = r_new.copy()
            r_curr = r_new 

        return r_traj_rev

    def z_dyn(self, zt, Pt, rt, At, Bt, bt):
        return At @ zt + Bt @ self.z2v(zt, Pt, rt, Bt, bt)
    
    def z_dyn_step(self, zt, Pt, rt, At, Bt, bt):
        k1 = self.dt * self.z_dyn(zt, Pt, rt, At, Bt, bt)
        k2 = self.dt * self.z_dyn(zt+k1/2, Pt, rt, At, Bt, bt)
        k3 = self.dt * self.z_dyn(zt+k2/2, Pt, rt, At, Bt, bt)
        k4 = self.dt * self.z_dyn(zt+k3, Pt, rt, At, Bt, bt)

        zt_new = zt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return zt_new

    def z_traj_sim(self, z0, P_traj, r_traj, A_traj, B_traj, b_traj):
        z_traj = np.zeros((self.tsteps, self.x_dim))
        z_curr = z0.copy()

        for t in range(self.tsteps):
            Pt = P_traj[t]
            rt = r_traj[t]
            At = A_traj[t]
            Bt = B_traj[t]
            bt = b_traj[t]

            z_new = self.z_dyn_step(z_curr, Pt, rt, At, Bt, bt)
            z_traj[t] = z_new.copy()
            z_curr = z_new
        
        return z_traj
    
    def z2v(self, zt, Pt, rt, Bt, bt):
        return -self.R_v_inv @ Bt.T @ Pt @ zt - self.R_v_inv @ Bt.T @ rt - self.R_v_inv @ bt

    def get_descent(self, x0, u_traj):
        # forward simulate the trajectory
        x_traj = self.traj_sim(x0, u_traj)
        self.curr_x_traj = x_traj.copy()
        self.curr_u_traj = u_traj.copy()

        # sovle the Riccati equation backward in time
        A_traj = np.zeros((self.tsteps, self.x_dim, self.x_dim))
        B_traj = np.zeros((self.tsteps, self.x_dim, self.u_dim))
        a_traj = np.zeros((self.tsteps, self.x_dim))
        b_traj = np.zeros((self.tsteps, self.u_dim))

        for t_idx in range(self.tsteps):
            A_traj[t_idx] = self.get_At_mat(t_idx)
            B_traj[t_idx] = self.get_Bt_mat(t_idx)
            a_traj[t_idx] = self.get_at_vec(t_idx)
            b_traj[t_idx] = self.get_bt_vec(t_idx)

        # print('a_traj:\n', a_traj)
        
        PT = np.zeros((self.x_dim, self.x_dim))
        P_traj_rev = self.P_traj_revsim(PT, A_traj, B_traj, a_traj, b_traj)
        P_traj = np.flip(P_traj_rev, axis=0)

        rT = np.zeros(self.x_dim)
        r_traj_rev = self.r_traj_revsim(rT, P_traj, A_traj, B_traj, a_traj, b_traj)
        r_traj = np.flip(r_traj_rev, axis=0)

        z0 = np.zeros(self.x_dim)
        z_traj = self.z_traj_sim(z0, P_traj, r_traj, A_traj, B_traj, b_traj)

        # compute the descent direction
        v_traj = np.zeros((self.tsteps, self.u_dim))
        for t in range(self.tsteps):
            zt = z_traj[t]
            Pt = P_traj[t]
            rt = r_traj[t]
            Bt = B_traj[t]
            bt = b_traj[t]
            v_traj[t] = self.z2v(zt, Pt, rt, Bt, bt)
        
        return v_traj

class iLQRErgodic_dubins(iLQRErgodicControl):
    def __init__(self, dt, tsteps, x_dim, u_dim, Q_z, R_v,
                 R, ks, L_list, lamk_list, hk_list, phik_list) -> None:
        super().__init__(dt, tsteps, x_dim, u_dim, Q_z, R_v)
        
        self.R = R 
        self.ks = ks 
        self.L_list = L_list
        self.lamk_list = lamk_list 
        self.hk_list = hk_list 
        self.phik_list = phik_list 

    def dyn(self, xt, ut):
        xdot = np.array([
            ut[0] * np.cos(xt[2]),
            ut[0] * np.sin(xt[2]),
            ut[1]
        ])
        return xdot 
        
    def get_At_mat(self, t_idx):
        xt = self.curr_x_traj[t_idx]
        ut = self.curr_u_traj[t_idx]
        A = np.array([
            [0.0, 0.0, -np.sin(xt[2]) * ut[0]],
            [0.0, 0.0,  np.cos(xt[2]) * ut[0]],
            [0.0, 0.0, 0.0]
        ])
        return A
    
    def get_Bt_mat(self, t_idx):
        xt = self.curr_x_traj[t_idx]
        B = np.array([
            [np.cos(xt[2]), 0.0],
            [np.sin(xt[2]), 0.0],
            [0.0, 1.0]
        ])
        return B

    def get_at_vec(self, t_idx):
        xt = self.curr_x_traj[t_idx][:2]
        x_traj = self.curr_x_traj[:,:2]
        
        dfk_xt_all = np.array([
            -np.pi * self.ks[:,0] / self.L_list[0] * np.sin(np.pi * self.ks[:,0] / self.L_list[0] * xt[0]) * np.cos(np.pi * self.ks[:,1] / self.L_list[1] * xt[1]),
            -np.pi * self.ks[:,1] / self.L_list[1] * np.cos(np.pi * self.ks[:,0] / self.L_list[0] * xt[0]) * np.sin(np.pi * self.ks[:,1] / self.L_list[1] * xt[1]),
        ]) / self.hk_list

        fk_all = np.prod(np.cos(np.pi * self.ks / self.L_list * x_traj[:,None]), axis=2) / self.hk_list
        ck_all = np.sum(fk_all, axis=0) * self.dt / (self.tsteps * self.dt)

        at = np.sum(self.lamk_list * 2.0 * (ck_all - self.phik_list) * dfk_xt_all / (self.tsteps * self.dt), axis=1)
        return np.array([at[0], at[1], 0.0])

    def get_bt_vec(self, t_idx):
        ut = self.curr_u_traj[t_idx]
        return self.R @ ut 
    
    def loss(self, x_traj, u_traj):
        fk_all = np.prod(np.cos(np.pi * self.ks / self.L_list * x_traj[:,:2][:,None]), axis=2) / self.hk_list
        ck_all = np.sum(fk_all, axis=0) * self.dt / (self.tsteps * self.dt)
        erg_metric = np.sum(self.lamk_list * np.square(ck_all - self.phik_list))

        ctrl_cost = np.sum(self.R @ u_traj.T * u_traj.T) * self.dt 
        return erg_metric + ctrl_cost 


if __name__ == "__main__":
    # Define the target distribution: Arbitrary information spots
    world_x_size = 80.0
    world_y_size = 50.0

    num_info_spots = 8

    means = np.array([
        [60.0, 60.0],
        [20.0, 20.0],
        [40.0, 40.0],
        [10.0, 30.0],
        [60.0, 10.0],
        [10.0, 40.0],
        [50.0, 20.0],
        [70.0, 40.0],
    ])

    # means = np.random.uniform(low=0.0, high=np.array([world_x_size, world_y_size]), size=(num_info_spots, 2))
    covs = [np.array([
        [35.0, 15.0],
        [15.0, 35.0]
    ]) for _ in range(num_info_spots)]
    ws = np.ones(num_info_spots) * 0.3

    def pdf(x):
        pdf = np.zeros(x.shape[0])
        for mean, cov, w in zip(means, covs, ws):
            pdf += w * mvn.pdf(x, mean, cov)
        pdf = pdf / np.max(pdf)
        return pdf

    # Define a 1-by-1 2D search space
    L_list = np.array([world_x_size, world_y_size])  # boundaries for each dimension

    # Discretize the search space into 100-by-100 mesh grids
    dx = dy = resolution  = 0.5
    num_rows, num_cols = get_index_bounds(world_x_size, world_y_size, resolution)
    grids_x, grids_y = np.meshgrid(
        np.linspace(0, L_list[0], num_cols),
        np.linspace(0, L_list[1], num_rows)
    )
    grids = np.array([grids_x.ravel(), grids_y.ravel()]).T

    # plt.contourf(grids_x, grids_y, pdf(grids).reshape(grids_x.shape), cmap='Reds')
    # plt.show()

    # Configure the index vectors
    num_k_per_dim = 30 # TODO: Make this a parameter
    ks_dim1, ks_dim2 = np.meshgrid(
        np.arange(num_k_per_dim), np.arange(num_k_per_dim)
    )
    ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T

    # Pre-processing lambda_k and h_k
    lamk_list = np.power(1.0 + np.linalg.norm(ks, axis=1), -3/2.0)
    hk_list = np.zeros(ks.shape[0])

    for i, k_vec in enumerate(ks):
        fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)  
        hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
        hk_list[i] = hk

    # compute the coefficients for the target distribution
    phik_list = np.zeros(ks.shape[0])  
    pdf_vals = pdf(grids)
    for i, (k_vec, hk) in enumerate(zip(ks, hk_list)):
        fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)  
        fk_vals /= hk

        phik = np.sum(fk_vals * pdf_vals) * dx * dy 
        phik_list[i] = phik


    # Define environment
    env_params = EnvParams.load_from_yaml("envs/env_cfg.yaml")

    robot_params = RobotParams.load_from_yaml("agents/dubins_cfg.yaml")
    world_x_size = 60
    world_y_size = 50
    env_params.world_x_size = world_x_size
    env_params.world_y_size = world_y_size
    smoke_blob_params = [
        SmokeBlobParams(x_pos=10, y_pos=40, intensity=1.0, spread_rate=1.0),
        SmokeBlobParams(x_pos=20, y_pos=20, intensity=1.0, spread_rate=3.0),
        SmokeBlobParams(x_pos=15, y_pos=45, intensity=1.0, spread_rate=4.0),
        SmokeBlobParams(x_pos=40, y_pos=40, intensity=1.0, spread_rate=8.0)
    ]
    sensor_params = DownwardsSensorParams(world_x_size=world_x_size, world_y_size=world_y_size)
    smoke_params = DynamicSmokeParams(x_size=world_x_size, y_size=world_y_size, smoke_blob_params=smoke_blob_params, resolution=0.3, fov_sensor_params=sensor_params)

    env = DynamicSmokeEnv(env_params, robot_params, smoke_params)
    initial_state = {"location": np.array([10, 10]), "angle": 0, "smoke_density": 0}
    env.reset(initial_state=initial_state)

    fig, ax = plt.subplots(figsize=(4, 4))
    global_steps = 100

    frames = []

    u_traj = None

    for step in range(global_steps):
        if step % 100 == 0:
            # Define the optimal control problem 
            dt = 0.1
            tsteps_ahead = 300
            R = 10.0*np.diag([0.01, 0.01])
            Q_z = 5.0 * np.diag([0.01, 0.01, 0.001])
            R_v = np.diag([0.002, 0.001])

            state = env._get_obs()
            x0 = np.array([state["location"][0], state["location"][1], state["angle"]])

            u_traj = np.tile(np.array([5.0, 0.0]), reps=(tsteps_ahead,1))

            trajopt_ergodic_dubins = iLQRErgodic_dubins(
                dt, tsteps_ahead, x_dim=3, u_dim=2, Q_z=Q_z, R_v=R_v,
                R=R, ks=ks, L_list=L_list, lamk_list=lamk_list,
                hk_list=hk_list, phik_list=phik_list
            )
            loss_val_list = []
            for iter in range(50):
                step = 0.04
                alpha = 0.5
                v_traj = None
                
                x_traj = trajopt_ergodic_dubins.traj_sim(x0, u_traj)
                for _i in range(3):
                    if v_traj is None:
                        v_traj = trajopt_ergodic_dubins.get_descent(x0, u_traj)
                        loss_val = trajopt_ergodic_dubins.loss(x_traj, u_traj)
                        loss_val_list.append(loss_val)
                    temp_u_traj = u_traj + step * v_traj
                    temp_x_traj = trajopt_ergodic_dubins.traj_sim(x0, temp_u_traj)
                    temp_loss_val = trajopt_ergodic_dubins.loss(temp_x_traj, temp_u_traj)
                    if temp_loss_val < loss_val:
                        break
                    else:
                        step *= alpha
                u_traj += step * v_traj
            fs, axs = plt.subplots(figsize=(4, 4))
            # axs.plot(loss_val_list)
            axs.plot(u_traj[:,0])
            axs.plot(u_traj[:,1])

        action = u_traj[0].copy()
        u_traj = u_traj[1:]

        state, reward, terminated, truncated, info = env.step(action)

        env._render_frame(fig=fig, ax=ax)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = imageio.imread(buf)
        frames.append(image)

    if frames:
        imageio.mimsave(f'smoke_dynamic.gif', frames, duration=10.0)

    env.close()

    # # Define the optimal control problem 
    # dt = 0.1
    # tsteps = 100
    # R = np.diag([0.01, 0.01])
    # Q_z = np.diag([0.01, 0.01, 0.001])
    # R_v = np.diag([0.002, 0.001])

    # x0 = np.array([env.observation_space.sample()["location"][0],
    #                env.observation_space.sample()["location"][1],
    #                env.observation_space.sample()["angle"]])
    # init_u_traj = np.tile(np.array([0.2, 1.2]), reps=(tsteps,1))

    # trajopt_ergodic_dubins = iLQRErgodic_dubins(
    #     dt, tsteps, x_dim=3, u_dim=2, Q_z=Q_z, R_v=R_v,
    #     R=R, ks=ks, L_list=L_list, lamk_list=lamk_list,
    #     hk_list=hk_list, phik_list=phik_list
    # )

    # # Iterative trajectory optimization for ergodic control
    # u_traj = init_u_traj.copy()
    # step = 0.01
    # loss_list = []

    # fig, axes = plt.subplots(1, 3, dpi=70, figsize=(25,5), tight_layout=True)

    # for iter in tqdm(range(50)):
    #     # Simulation
    #     x_traj = trajopt_ergodic_dubins.traj_sim(x0, u_traj)
    #     # TODO: every 10 iterations, update phik_list

    #     # Compute v trajectory
    #     # Simulate future trajectory loss to get best 
    #     step = 0.004
    #     alpha = 0.5
    #     v_traj = None
    #     for _i in range(3):
    #         if v_traj is None:
    #             v_traj = trajopt_ergodic_dubins.get_descent(x0, u_traj)
    #             loss_val = trajopt_ergodic_dubins.loss(x_traj, u_traj)
    #             loss_list.append(loss_val)
    #         temp_u_traj = u_traj + step * v_traj
    #         temp_x_traj = trajopt_ergodic_dubins.traj_sim(x0, temp_u_traj)
    #         temp_loss_val = trajopt_ergodic_dubins.loss(temp_x_traj, temp_u_traj)
    #         if temp_loss_val < loss_val:
    #             break
    #         else:
    #             step *= alpha
    #     u_traj += step * v_traj

    #     # visualize every 10 iterations
    #     if (iter+1) % 5 == 0:
    #         ax1 = axes[0]
    #         ax1.cla()
    #         ax1.set_aspect('equal', adjustable='box')
    #         ax1.set_xlim(0.0, L_list[0])
    #         ax1.set_ylim(0.0, L_list[1])
    #         ax1.set_title('Iteration: {:d}'.format(iter+1))
    #         ax1.set_xlabel('X (m)')
    #         ax1.set_ylabel('Y (m)')
    #         ax1.contourf(grids_x, grids_y, pdf_vals.reshape(grids_x.shape), cmap='Reds')
    #         ax1.plot([x0[0], x_traj[0,0]], [x0[1], x_traj[0,1]], linestyle='-', linewidth=2, color='k', alpha=1.0)
    #         ax1.plot(x_traj[:,0], x_traj[:,1], linestyle='-', marker='o', color='k', linewidth=2, alpha=1.0, label='Optimized trajectory')
    #         ax1.plot(x0[0], x0[1], linestyle='', marker='o', markersize=15, color='C0', alpha=1.0, label='Initial state')
    #         ax1.legend(loc=1)

    #         ax2 = axes[1]
    #         ax2.cla()
    #         ax2.set_title('Control vs. Time')
    #         ax2.set_ylim(-1.5, 2.5)
    #         ax2.plot(np.arange(tsteps)*dt, u_traj[:,0], color='C0', label=r'$u_1$')
    #         ax2.plot(np.arange(tsteps)*dt, u_traj[:,1], color='C1', label=r'$u_2$')
    #         ax2.set_xlabel('Time (s)')
    #         ax2.set_ylabel('Control')
    #         ax2.legend(loc=1)
    #         height = ax1.get_position().height
    #         ax2.set_position([ax2.get_position().x0, ax1.get_position().y0, ax2.get_position().width, height])

    #         ax3 = axes[2]
    #         ax3.cla()
    #         ax3.set_title('Objective vs. Iteration')
    #         ax3.set_xlim(-0.2, 100.2)
    #         ax3.set_ylim(9e-2, 1.1)
    #         ax3.set_xlabel('Iteration')
    #         ax3.set_ylabel('Objective')
    #         ax3.plot(np.arange(iter+1), loss_list, color='C3')
    #         height = ax1.get_position().height
    #         ax3.set_position([ax3.get_position().x0, ax1.get_position().y0, ax3.get_position().width, height])
    #         ax3.set_yscale('log')

    #         # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
    #         display.clear_output(wait=True)
    #         display.display(fig)

    # display.clear_output(wait=True)
    # plt.show()
    # plt.close()