import os
import warnings
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

import scipy.stats as stats
from collections import deque
from simulator.static_smoke import StaticSmoke, SmokeBlobParams
from simulator.dynamic_smoke import DynamicSmoke, DynamicSmokeParams, DownwardsSensorParams

from learning.base_model import BaseModel
from src.utils import *
from tqdm import tqdm
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.kl import kl_divergence

from envs.replay_buffer import GenericReplayBuffer
from envs.smoke_env_dyn import *

from torch.utils.data import Dataset, DataLoader

import math

import torch.optim as optim


@dataclass
class ObsVAESSM:
    xs: torch.Tensor
    ys: torch.Tensor
    values: torch.Tensor = None
    mask: torch.Tensor = None

    def __post_init__(self):
        assert self.xs.shape == self.ys.shape, f"xs and ys must have the same shape, {self.xs.shape} vs {self.ys.shape}"
        assert self.values is None or self.xs.shape == self.values.shape, f"xs and values must have the same shape, {self.xs.shape} vs {self.values.shape}"
        assert self.mask is None or self.mask.shape == self.xs.shape, f"mask must have the same shape as xs, {self.mask.shape} vs {self.xs.shape}"

        self.xs = self.xs.float()
        self.ys = self.ys.float()
        self.values = self.values.float() if self.values is not None else None
        self.mask = self.mask.bool() if self.mask is not None else None

@dataclass
class VAESSMParams:
    r_dim: int = 100
    embed_dim: int = 100
    deter_dim: int = 100
    stoch_dim: int = 100
    x_dim: int = 2
    y_dim: int = 1
    action_dim: int = 2

    encoder_hidden_dim: int = 100

    prior_hidden_dim: int = 100
    posterior_hidden_dim: int = 100

    decoder_hidden_dim: int = 100

class AttentionAggregator(nn.Module):
    def __init__(self, r_dim, h_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(r_dim, h_dim)
        self.key_proj   = nn.Linear(r_dim, h_dim)
        self.value_proj = nn.Linear(r_dim, h_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=h_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.out_proj = nn.Linear(h_dim, r_dim)

    def forward(self, r_i):
        q = r_i.mean(dim=1, keepdim=True)  # (B,1,r_dim)

        Q = self.query_proj(q)             # (B,1,h)
        K = self.key_proj(r_i)             # (B,N,h)
        V = self.value_proj(r_i)           # (B,N,h)

        attn_out, _ = self.attn(Q, K, V)   # (B,1,h)

        r = self.out_proj(attn_out.squeeze(1))  # (B,r_dim)
        return r


class VAEEncoder(nn.Module):
    def __init__(self, vaessm_params: VAESSMParams):
        super(VAEEncoder, self).__init__()

        self.vaessm_params = vaessm_params

        self.x_dim = vaessm_params.x_dim
        self.y_dim = vaessm_params.y_dim
        self.hidden_dim = vaessm_params.encoder_hidden_dim
        self.r_dim = vaessm_params.r_dim
        self.embed_dim = vaessm_params.embed_dim

        layers = [nn.Linear(self.x_dim + self.y_dim, self.hidden_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.hidden_dim, self.r_dim)]

        self.input_to_rs = nn.Sequential(*layers)
        self.rs_to_r = AttentionAggregator(self.r_dim, self.hidden_dim)
        self.r_to_hidden = nn.Sequential(nn.Linear(self.r_dim, self.hidden_dim), nn.ReLU(inplace=True))
        self.hidden_to_embed_mu = nn.Linear(self.hidden_dim, self.embed_dim)
        self.hidden_to_embed_sigma = nn.Sequential(nn.Linear(self.hidden_dim, self.embed_dim), nn.Sigmoid())

    def forward(self, obs: ObsVAESSM):
        pos = torch.stack([obs.xs, obs.ys], dim=-1)
        values = obs.values.unsqueeze(-1)

        x = torch.cat([pos, values], dim=-1)
        rs = self.input_to_rs(x)
        r = self.rs_to_r(rs)

        hidden = self.r_to_hidden(r)
        embed_mu = self.hidden_to_embed_mu(hidden)
        embed_sigma = 0.1 + 0.9 * self.hidden_to_embed_sigma(hidden)
        return Normal(embed_mu, embed_sigma)

class VAEDecoder(nn.Module):
    def __init__(self, vaessm_params: VAESSMParams):
        super(VAEDecoder, self).__init__()

        self.vaessm_params = vaessm_params

        self.x_dim = vaessm_params.x_dim
        self.z_dim = vaessm_params.stoch_dim #+ vaessm_params.deter_dim
        self.h_dim = vaessm_params.decoder_hidden_dim
        self.y_dim = vaessm_params.y_dim

        layers = [nn.Linear(self.z_dim + self.x_dim, self.h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(self.h_dim, self.h_dim),
                  nn.ReLU(inplace=True)]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_y_mu = nn.Linear(self.h_dim, self.y_dim)
        self.hidden_to_y_sigma = nn.Linear(self.h_dim, self.y_dim)

    def forward(self, latents: torch.Tensor, query_obs: ObsVAESSM):
        batch_size, num_points = query_obs.xs.shape
        
        # Embed query positions
        x = torch.stack([query_obs.xs, query_obs.ys], dim=-1)
        
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        latents = latents.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        latents_flat = latents.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, latents_flat), dim=1)
        hidden = self.xz_to_hidden(input_pairs)
        y_mu = self.hidden_to_y_mu(hidden)
        y_sigma = self.hidden_to_y_sigma(hidden)
        # Reshape output into expected shape
        y_mu = y_mu.view(batch_size, num_points, self.y_dim)
        y_sigma = y_sigma.view(batch_size, num_points, self.y_dim)
        y_sigma = 0.9 * F.softplus(y_sigma)+ 0.1 
        return Normal(y_mu, y_sigma)


class ScalarFieldVAESSM(nn.Module):
    def __init__(self, params: VAESSMParams):
        super().__init__()
        self.params = params
        self.encoder = VAEEncoder(params)
        self.decoder = VAEDecoder(params)

        # GRU: h = f(h, z, u)
        self.rnn_input_dim = params.stoch_dim + params.action_dim
        self.gru = nn.GRUCell(self.rnn_input_dim, params.deter_dim)
        
        # Prior Network z ~ p(z| h)
        self.prior_net = nn.Sequential(
            nn.Linear(params.deter_dim, params.prior_hidden_dim), nn.ELU(),
            nn.Linear(params.prior_hidden_dim, params.prior_hidden_dim, nn.ELU())
        )

        self.prior_net_mu = nn.Linear(params.prior_hidden_dim, params.stoch_dim)
        self.prior_net_sigma = nn.Linear(params.prior_hidden_dim, params.stoch_dim)
        
        # Posterior Network z ~ q(z| h, x)
        self.posterior_net = nn.Sequential(
            nn.Linear(params.deter_dim + params.embed_dim, params.posterior_hidden_dim), nn.ELU(),
            nn.Linear(params.posterior_hidden_dim, params.posterior_hidden_dim, nn.ELU())
        )

        self.posterior_net_mu = nn.Linear(params.posterior_hidden_dim, params.stoch_dim)
        self.posterior_net_sigma = nn.Linear(params.posterior_hidden_dim, params.stoch_dim)


    def forward(self, prev_h: torch.Tensor, prev_z: torch.Tensor, prev_action: torch.Tensor, dones: torch.Tensor, obs: ObsVAESSM | None = None, query_obs: ObsVAESSM | None = None):
        # Ajustar dimensiones de dones para broadcasting (B) -> (B, 1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # 0. MÁSCARA DE RESET
        # Si done=1, (1-done)=0 -> El estado se vuelve cero (reset).
        # Si done=0, (1-done)=1 -> El estado se mantiene.
        mask = 1.0 - dones.float()
        
        prev_h = prev_h * mask
        prev_z = prev_z * mask

        # 1. h_t = f(z_t-1, h_t-1, a_t-1)
        # Ahora la GRU recibe ceros si hubo un done, iniciando efectivamente una nueva secuencia
        rnn_input = torch.cat([prev_z, prev_action], dim=-1)
        h = self.gru(rnn_input, prev_h)
        
        # 2. Prior z_t ~ p(z_t | h_t)
        prior_out = self.prior_net(h)
        prior_mu = self.prior_net_mu(prior_out)
        prior_sigma = 0.9*F.softplus(self.prior_net_sigma(prior_out)) + 0.1
        prior_dist = torch.distributions.Normal(prior_mu, prior_sigma)

        if obs is not None:
            embed = self.encoder(obs)
            embed_sample = embed.rsample()
            
            # Posterior z_t ~ q(z_t | h_t, x_t)
            post_in = torch.cat([h, embed_sample], dim=1)
            post_out = self.posterior_net(post_in)
            post_mu = self.posterior_net_mu(post_out)
            post_sigma = 0.9*F.softplus(self.posterior_net_sigma(post_out)) + 0.1
            post_dist = torch.distributions.Normal(post_mu, post_sigma)
            z = post_dist.rsample()
        else:
            post_dist = None
            z = prior_dist.rsample()

        # 3. Decodificar
        decoded_dist = None
        if query_obs is not None:
            # latents = torch.cat([h, z], dim=1)
            # decoded_dist = self.decoder(latents, query_obs)
            decoded_dist = self.decoder(z, query_obs)
            
        return h, z, prior_dist, post_dist, decoded_dist
    

def collate_variable_batch(batch_list: list[ObsVAESSM]):
    """
    Toma una lista de observaciones (obs).
    """
    
    obs_xys = [torch.stack([b.xs, b.ys], dim=-1) for b in batch_list]
    obs_vals = [b.values for b in batch_list if b.values is not None]
    
    # Pad Sequence (obs)
    obs_xy_pad = pad_sequence(obs_xys, batch_first=True, padding_value=0)

    obs_lens = torch.tensor([len(t) for t in obs_xys])
    obs_max_len = obs_xy_pad.shape[1]
    obs_mask = (torch.arange(obs_max_len)[None, :] >= obs_lens[:, None]).bool()

    if obs_vals:
        obs_val_pad = pad_sequence(obs_vals, batch_first=True, padding_value=0)
    else:
        obs_val_pad = None

    batch_obsvaessm = ObsVAESSM(
        xs=obs_xy_pad[:, :, 0],
        ys=obs_xy_pad[:, :, 1],
        values=obs_val_pad,
        mask=obs_mask
    )
    return batch_obsvaessm

class VAESSMSequenceDataset(Dataset):
    def __init__(self, replay_buffer, sequence_length, valid_indices):
        self.buffer = replay_buffer
        self.seq_len = sequence_length + 1 
        self.indices = valid_indices
        
        # DEFINIR LÍMITES (Es bueno tenerlos como constantes)
        self.max_coord = 50.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        
        # Usamos nombres distintos para las listas acumuladoras
        # para no confundirlas con los datos de cada paso
        xs_seq, ys_seq, vals_seq, acts_seq, dones_seq = [], [], [], [], []
        
        for t in range(self.seq_len):
            curr_idx = start_idx + t
            data = self.buffer.get_from_index(curr_idx)
            
            # 1. Obtener datos crudos (Numpy)
            raw_xs = data["smoke_value_positions"][:, 0]
            raw_ys = data["smoke_value_positions"][:, 1]
            raw_vals = data["smoke_values"]

            indxs = np.arange(len(raw_xs))
            np.random.shuffle(indxs)

            # Only keeping 20% of data
            indxs = indxs[:int(len(indxs) * 0.2)]

            raw_xs = raw_xs[indxs]
            raw_ys = raw_ys[indxs]
            raw_vals = raw_vals[indxs]

            raw_done = data["done"] # O "dones" según tu buffer

            v_local, w_local = data["actions"]
            r_theta = data["state"][2] # <--- Asegúrate que esto existe en tu buffer
            r_x, r_y = data["state"][0], data["state"][1]
            
            # Proyectar velocidad al marco global
            global_vx = v_local * np.cos(r_theta)
            global_vy = v_local * np.sin(r_theta)
            
            # Acción efectiva: [vel_x, vel_y]
            action_vec = torch.tensor([global_vx, global_vy], dtype=torch.float32)

            # 3. Normalización (Igual que antes)
            norm_xs = (raw_xs - r_x)
            norm_ys = (raw_ys - r_y)

            # 3. Guardar en las listas (Convirtiendo a Tensor Float)
            xs_seq.append(torch.from_numpy(norm_xs).float())
            ys_seq.append(torch.from_numpy(norm_ys).float())
            
            # Los valores ya están en 0-1, así que entran directo
            # .squeeze() es importante si vienen como (N, 1)
            vals_seq.append(torch.from_numpy(raw_vals).float().squeeze())
            
            dones_seq.append(float(raw_done))
            acts_seq.append(action_vec)
        return xs_seq, ys_seq, vals_seq, torch.stack(acts_seq), torch.tensor(dones_seq)

def vaessm_collate_fn(batch):
    """
    Transforma una lista de secuencias [(seq1), (seq2)...] 
    en un formato batch-major para el VAESSM (Time, Batch, ...).
    """
    # batch es una lista de tuplas (xs, ys, vals, dones) de longitud B
    # Cada elemento de la tupla es una lista de longitud L (la secuencia)
    
    seq_len = len(batch[0][0])
    batch_size = len(batch)
    
    # Estructura final deseada: Una lista de longitud L, donde cada elemento
    # es un ObsVAESSM que contiene el batch (B) combinado.
    
    batched_steps = [] # Lista de L pasos
    
    # Transponer: iterar por tiempo t
    for t in range(seq_len):
        batch_xs_t = []
        batch_ys_t = []
        batch_vals_t = []
        batch_acts_t = [] # Nueva lista
        batch_dones_t = []
        
        for b in range(batch_size):
            # batch[b] es la secuencia b
            # batch[b][0] son las xs de esa secuencia
            # batch[b][0][t] son las xs en el tiempo t
            batch_xs_t.append(batch[b][0][t])
            batch_ys_t.append(batch[b][1][t])
            batch_vals_t.append(batch[b][2][t])
            batch_acts_t.append(batch[b][3][t]) # Agregar acciones
            batch_dones_t.append(batch[b][4][t])
        
        # Ahora usamos tu función 'collate_variable_batch' para este paso de tiempo t
        # Primero creamos objetos ObsVAESSM temporales para usar tu función existente
        # (O idealmente, modificas collate_variable_batch para aceptar listas crudas)
        step_obs_list = []
        for i in range(batch_size):
            step_obs_list.append(ObsVAESSM(
                xs=batch_xs_t[i],
                ys=batch_ys_t[i],
                values=batch_vals_t[i]
            ))
            
        batched_obs_t = collate_variable_batch(step_obs_list)
        batched_dones_t = torch.tensor(batch_dones_t).float().unsqueeze(1) # (B, 1)
        batched_acts_t = torch.stack(batch_acts_t)
        
        batched_steps.append((batched_obs_t, batched_acts_t, batched_dones_t))
        
    return batched_steps

if __name__ == "__main__":
    env_params = EnvParams.load_from_yaml("/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke/envs/env_cfg.yaml")

    robot_type = "dubins2d"
    cfg_file = f"agents/{robot_type}_cfg.yaml"

    sensor_params = DownwardsSensorParams(
        world_x_size=env_params.world_x_size,
        world_y_size=env_params.world_y_size,
        points_in_range=30,
        fov_size_degrees=50
    )

    # ==== DATA LOADING ====
    num_collection_data = 1e4
    replay_buffer = GenericReplayBuffer(buffer_size=num_collection_data, data_keys=[])
    replay_buffer.load_from_file('/Users/emanuelsamir/Documents/dev/cmu/research/experiments/7_safe_nav_smoke/misc/collection_data_static_1_0114_0214/replay_buffer.npz')

    # ==== GP SETUP ====
    datasize = replay_buffer.current_size

    train_split = 0.9
    test_split = 0.1

    train_idx = np.arange(int(datasize * train_split))
    test_idx = np.arange(int(datasize * train_split), int(datasize * (train_split + test_split)))


    # ==== CONFIGURACION ====
    BATCH_SIZE = 24
    EPOCHS = 20
    SEQ_LEN = 10 # Pasos de entrenamiento efectivos

    # CAMBIO 2: Filtramos índices para asegurar tener espacio para SEQ_LEN + 1
    valid_train_indices = train_idx[train_idx < (datasize - (SEQ_LEN + 1))]

    train_dataset = VAESSMSequenceDataset(replay_buffer, SEQ_LEN, valid_train_indices)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=vaessm_collate_fn,
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vaessm = ScalarFieldVAESSM(params=VAESSMParams()).to(device)
    optimizer = optim.Adam(vaessm.parameters(), lr=8e-4)
    global_step = 0

    def beta_schedule(step, warmup=5000, max_beta=0.5):
        return min(max_beta, step / warmup)

    # ==== TRAINING LOOP ====
    for epoch in range(EPOCHS):
        vaessm.train()
        total_loss = 0
        
        # tqdm para barra de progreso
        for batch_sequence in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # batch_sequence es una lista de longitud SEQ_LEN + 1

            beta = beta_schedule(global_step)
            
            current_batch_size = batch_sequence[0][1].shape[0]
            
            prev_h = torch.zeros(current_batch_size, vaessm.params.deter_dim).to(device)
            prev_z = torch.zeros(current_batch_size, vaessm.params.stoch_dim).to(device)
            
            kl_loss = 0
            raw_kl_loss = 0
            ll_loss = 0
            sequence_loss = 0

            prior_dist_mean = []
            prior_dist_var = []
            abs_dists = []
            max_dists = []
            
            # CAMBIO 3: Separar Inputs y Targets
            # inputs:  t=0 ... t=9
            # targets: t=1 ... t=10
            inputs = batch_sequence[:-1]
            targets = batch_sequence[1:]
            
            # Iteramos en pares (Input Actual, Target Futuro)
            for t, ((obs_in, act_in, done_in), (obs_target, _, done_target)) in enumerate(zip(inputs, targets)):
                
                # Mover a GPU
                obs_in.xs, obs_in.ys, obs_in.values = obs_in.xs.to(device), obs_in.ys.to(device), obs_in.values.to(device)
                if obs_in.mask is not None: obs_in.mask = obs_in.mask.to(device)
                
                obs_target.xs, obs_target.ys, obs_target.values = obs_target.xs.to(device), obs_target.ys.to(device), obs_target.values.to(device)
                if obs_target.mask is not None: obs_target.mask = obs_target.mask.to(device)
                
                done_in = done_in.to(device) # (B, 1)
                act_in = act_in.to(device) # [Batch, 2]

                obs_in.values = obs_in.values.squeeze()
                obs_target.values = obs_target.values.squeeze()

                # === FORWARD ===
                # 1. Obs Actual (t) -> Encoder -> Posterior
                # 2. Done Actual (t) -> Resetea estado si t era nuevo episodio
                # 3. Query Obs (t+1) -> Decoder predice valores en las posiciones de t+1
                h, z, prior_dist, post_dist, y_pred_dist = vaessm(
                    prev_h=prev_h,
                    prev_z=prev_z,
                    prev_action=act_in, # <--- AQUI PASAMOS LA ACCION
                    dones=done_in,     # Resetea memoria si el input actual es start de episodio
                    obs=obs_in,        # Codifica t
                    query_obs=obs_target # Decodifica (predice) en coordenadas de t+1
                )

                # === LOSS ===
                # A. KL Divergence (Latent Regularization)
                # Divergencia entre Prior (predicción sin ver obs) y Posterior (tras ver obs)
                raw_kl = kl_divergence(post_dist, prior_dist)
                raw_kl_vec = raw_kl.sum(dim=-1)
                kl_vec = raw_kl_vec
                # kl_vec = raw_kl_vec.clamp(min = 1.0)

                y_target = obs_target.values.unsqueeze(-1)
                log_likelihood = - y_pred_dist.log_prob(y_target).squeeze().mean(dim=-1)

                # Si tienes máscara de padding (obs.mask), aplícala aquí al promedio espacial:
                # if obs_target.mask is not None:
                #     valid_target = (~obs_target.mask).float()
                #     log_likelihood = (log_likelihood * valid_target).sum(dim=1) / valid_target.sum(dim=1).clamp(min=1)
                # else:
                #     log_likelihood = log_likelihood.mean(dim=1)
                
                # === MÁSCARA DE VALIDEZ TEMPORAL (CRUCIAL) ===
                # Si done_in es True, significa que el paso 't' fue terminal.
                # Por tanto, 't+1' pertenece a OTRO episodio. No hay relación causal.
                # Debemos anular la loss de predicción en este caso.
                valid_transition = (1.0 - done_in).squeeze() # (B,)

                # Loss total por sample
                kl_vec_valid = kl_vec * valid_transition
                log_likelihood = log_likelihood * valid_transition
                
                # Promedio final solo sobre los elementos válidos
                step_loss = beta * kl_vec_valid + log_likelihood
                sequence_loss += step_loss.mean()

                raw_kl_vec_valid = raw_kl_vec * valid_transition
                raw_kl_loss += raw_kl_vec_valid.mean()
                kl_loss += kl_vec_valid.mean()
                ll_loss += log_likelihood.mean()

                prior_dist_mean.append(prior_dist.mean.mean().item())
                prior_dist_var.append(prior_dist.variance.mean().item())
                abs_dist = torch.abs(y_pred_dist.loc - y_target)
                mean_abs_dist = abs_dist.squeeze().mean(dim=-1) * valid_transition
                max_abs_dist = abs_dist.squeeze().max(dim=-1).values * valid_transition
                abs_dists.append(mean_abs_dist.mean().item())
                max_dists.append(max_abs_dist.mean().item())
                
                # Actualizar estados para el siguiente paso
                prev_h = h
                prev_z = z

            # Backprop
            global_step += 1
            optimizer.zero_grad()
            sequence_loss.backward()
            torch.nn.utils.clip_grad_norm_(vaessm.parameters(), max_norm=1.0)
            optimizer.step()

            prior_dist_mean = np.mean(prior_dist_mean)
            prior_dist_var = np.mean(prior_dist_var)
            abs_dist = np.mean(abs_dists)
            max_dist = np.mean(max_dists)
            print(f"Prior mean: {prior_dist_mean:.4f}, Prior var: {prior_dist_var:.4f}, Abs dist: {abs_dist:.4f}, Max dist: {max_dist:.4f}")
            print(f"Raw KL: {raw_kl_loss / len(batch_sequence):.4f}, KL: {kl_loss / len(batch_sequence):.4f}, LL: {ll_loss / len(batch_sequence):.4f}")
            
            total_loss += sequence_loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss / len(train_loader):.4f}")

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(vaessm.state_dict(), "checkpoints/vaessm_6th.pt")

    # 1. PREPARACIÓN DATASET TEST
    valid_test_indices = test_idx[test_idx < (datasize - SEQ_LEN)]
    test_dataset = VAESSMSequenceDataset(replay_buffer, SEQ_LEN, valid_test_indices)

    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, # No es necesario mezclar para test
        collate_fn=vaessm_collate_fn,
        num_workers=4
    )

    # 2. BUCLE DE EVALUACIÓN
    vaessm.eval() # Importante: poner en modo evaluación (apaga dropout, etc.)
    test_total_loss = 0
    test_total_mse = 0
    total_steps = 0

    # Guardamos datos del último batch para visualizar después
    batches_data = []

    with torch.no_grad(): # Desactiva el cálculo de gradientes para ahorrar memoria/tiempo
        for batch_idx, batch_sequence in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluando"):
            
            # Inicializar estados
            current_batch_size = batch_sequence[0][1].shape[0]
            prev_h = torch.zeros(current_batch_size, vaessm.params.deter_dim).to(device)
            prev_z = torch.zeros(current_batch_size, vaessm.params.stoch_dim).to(device)
            
            # Listas para reconstruir la secuencia completa (para visualización)
            seq_targets = []
            seq_preds = []
            seq_xs = []
            seq_ys = []
            
            inputs = batch_sequence[:-1]
            targets = batch_sequence[1:]

            for t, ((obs_in, act_in, done_in), (obs_target, _, done_target)) in enumerate(zip(inputs, targets)):
                # Mover a device
                obs_in.xs = obs_in.xs.to(device)
                obs_in.ys = obs_in.ys.to(device)
                obs_in.values = obs_in.values.to(device)
                if obs_in.mask is not None: obs_in.mask = obs_in.mask.to(device)
                act_in = act_in.to(device)
                done_in = done_in.to(device)

                # Forward (Inferencia)
                h, z, prior_dist, post_dist, pred_dist = vaessm(
                    prev_h=prev_h,
                    prev_z=prev_z,
                    prev_action=act_in,
                    dones=done_in,
                    obs=obs_in,
                    query_obs=obs_target 
                )
                
                # Cálculos de Loss (Igual que en train)
                kl_loss = torch.distributions.kl_divergence(post_dist, prior_dist).mean()
                gt_values = obs_target.values.unsqueeze(-1)
                log_likelihood = pred_dist.log_prob(gt_values).mean()
                
                target = obs_target.values
                pred = pred_dist.mean
                
                loss_step = kl_loss - log_likelihood
                
                test_total_loss += loss_step.item()
                test_total_mse += -log_likelihood.item()
                total_steps += 1
                
                # Actualizar estados
                prev_h = h
                prev_z = z
                
                # Guardar datos para visualización (solo del paso t)
                seq_targets.append(target.cpu())
                seq_preds.append(pred.cpu())
                seq_xs.append(obs_target.xs.cpu())
                seq_ys.append(obs_target.ys.cpu())

            # Guardamos la última secuencia completa procesada
            batch_data = {
                'xs': seq_xs, 
                'ys': seq_ys, 
                'targets': seq_targets, 
                'preds': seq_preds
            }
            batches_data.append(batch_data)

    avg_test_loss = test_total_loss / total_steps
    avg_test_mse = test_total_mse / total_steps

    (f"\nRESULTADOS TEST:")
    print(f"Avg Loss (ELBO): {avg_test_loss:.4f}")
    print(f"Avg MSE (Reconstruction Error): {avg_test_mse:.6f}")

    # 3. VISUALIZACIÓN
    # Vamos a graficar un paso de tiempo aleatorio de la última secuencia del último batch
    batch_idx = np.random.randint(0, len(batches_data))
    last_batch_data = batches_data[batch_idx]
    if last_batch_data is not None:
        import matplotlib.pyplot as plt
        
        # Elegimos un paso de tiempo (ej. el paso 5 de la secuencia)
        time_step = min(2, len(last_batch_data['xs']) - 1)
        # Elegimos un elemento del batch (ej. el índice 0)
        batch_elem = 0
        
        # Extraer datos
        # xs es lista de tensores [Batch, N_points]. Queremos [time_step][batch_elem]
        xs = last_batch_data['xs'][time_step][batch_elem].numpy()
        ys = last_batch_data['ys'][time_step][batch_elem].numpy()
        prev_xs = last_batch_data['xs'][time_step-1][batch_elem].numpy()
        prev_ys = last_batch_data['ys'][time_step-1][batch_elem].numpy()
        truth = last_batch_data['targets'][time_step][batch_elem].squeeze().numpy()
        pred = last_batch_data['preds'][time_step][batch_elem].squeeze().numpy()
        
        # Filtrar padding (donde xs es 0 y ys es 0 usualmente, o usar máscara si la guardaste)
        # Asumiendo que padding es 0,0 y el mundo real no tiene 0,0 exacto o visualmente no importa
        mask = (xs != 0) | (ys != 0) 
        
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        
        # Configuración común de scatter
        sc_args = {'s': 50, 'cmap': 'plasma', 'alpha': 0.8}
        
        # Plot Ground Truth
        vmin, vmax = 0, 1.0
        im1 = axes[0].scatter(xs[mask], ys[mask], c=truth[mask], vmin=vmin, vmax=vmax, **sc_args)
        axes[0].set_title(f"Ground Truth (t={time_step})")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot Predicción
        im2 = axes[1].scatter(xs[mask], ys[mask], c=pred[mask], vmin=vmin, vmax=vmax, **sc_args)
        axes[1].set_title(f"VAESSM Reconstruction (t={time_step})")
        plt.colorbar(im2, ax=axes[1])

        # Plot Input
        im3 = axes[2].scatter(prev_xs[mask], prev_ys[mask], c=pred[mask], vmin=vmin, vmax=vmax, **sc_args)
        axes[2].set_title(f"Input (t={time_step-1})")
        plt.colorbar(im3, ax=axes[2])
        
        # Plot Error Absoluto
        error = np.abs(truth - pred)
        im4 = axes[3].scatter(xs[mask], ys[mask], c=error[mask], **sc_args, vmin=vmin, vmax=vmax)
        axes[3].set_title("Absolute Error")
        plt.colorbar(im4, ax=axes[3])

        sample_loss = np.abs(truth - pred) 
        print("Loss sample", sample_loss.mean(), sample_loss.max(), sample_loss.min())
        
        # for ax in axes:
        #     ax.set_xlim(-env_params.world_x_size/2, env_params.world_x_size/2)
        #     ax.set_ylim(-env_params.world_y_size/2, env_params.world_y_size/2)
        #     ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        print("Visualización generada.")
