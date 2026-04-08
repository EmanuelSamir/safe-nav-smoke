import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_features=64, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, det_dim=256, stoch_dim=32, hidden_dim=256, out_features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_features),
            nn.Sigmoid(),
        )

    def forward(self, h, s):
        x = torch.cat([h, s], dim=-1)
        return self.net(x)

class RSSM(nn.Module):
    def __init__(self, act_dim=2, det_dim=256, stoch_dim=32, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim

        # Trans Model: (h_t-1, s_t-1, a_t-1) -> h_t
        self.rnn = nn.GRUCell(stoch_dim + act_dim, det_dim)

        # Prior: h_t -> mean, std
        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )

        # Posterior: (h_t, embed_t) -> mean, std
        self.post_net = nn.Sequential(
            nn.Linear(det_dim + embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )

    def get_stoch_dist(self, stats):
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std) + 0.1
        return torch.distributions.Normal(mean, std)

    def transition(self, h_prev, s_prev, action):
        """h_t = cell(h_t-1, s_t-1, a_t-1). Returns h_t and prior(s_t|h_t)"""
        x = torch.cat([s_prev, action], dim=-1)
        h_next = self.rnn(x, h_prev)
        prior_stats = self.prior_net(h_next)
        prior_dist = self.get_stoch_dist(prior_stats)
        return h_next, prior_dist

    def posterior(self, h_next, embed):
        """Returns posterior(s_t | h_t, e_t)"""
        x = torch.cat([h_next, embed], dim=-1)
        post_stats = self.post_net(x)
        post_dist = self.get_stoch_dist(post_stats)
        return post_dist

class DreamerV1VAE(nn.Module):
    def __init__(self, in_features=64, act_dim=2, det_dim=256, stoch_dim=32):
        super().__init__()
        self.encoder = Encoder(in_features=in_features)
        self.decoder = Decoder(det_dim=det_dim, stoch_dim=stoch_dim, out_features=in_features)
        self.rssm = RSSM(act_dim=act_dim, det_dim=det_dim, stoch_dim=stoch_dim)

    def forward(self, observations, actions, h_prev=None, s_prev=None):
        """Processes a sequence of (obs, acts) to return latent states and distributions"""
        B, T, _ = observations.shape
        device = observations.device

        if h_prev is None:
            h_prev = torch.zeros(B, self.rssm.det_dim, device=device)
        if s_prev is None:
            s_prev = torch.zeros(B, self.rssm.stoch_dim, device=device)

        embeds = self.encoder(observations)

        posteriors = []
        priors = []
        h_states = []
        s_states = []

        curr_h, curr_s = h_prev, s_prev

        for t in range(T):
            # 1. RSSM Transition to get h_t and prior(s_t)
            curr_h, prior_dist = self.rssm.transition(curr_h, curr_s, actions[:, t])
            # 2. Posterior update with e_t
            post_dist = self.rssm.posterior(curr_h, embeds[:, t])
            # 3. Sample s_t using reparameterization
            curr_s = post_dist.rsample()

            h_states.append(curr_h)
            s_states.append(curr_s)
            priors.append(prior_dist)
            posteriors.append(post_dist)

        return torch.stack(h_states, dim=1), torch.stack(s_states, dim=1), priors, posteriors

    def rollout(self, h_start, s_start, actions):
        """Imagine future given starting state and action sequence"""
        h_states = []
        s_states = []
        curr_h, curr_s = h_start, s_start

        for t in range(actions.shape[1]):
            curr_h, prior_dist = self.rssm.transition(curr_h, curr_s, actions[:, t])
            curr_s = prior_dist.rsample()
            h_states.append(curr_h)
            s_states.append(curr_s)

        return torch.stack(h_states, dim=1), torch.stack(s_states, dim=1)
