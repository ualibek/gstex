"""
GaussianNCA: a Neural Cellular Automaton living on the splats of a 3DGS scene.

State layout per splat (channels):
    [ geometry (11, read-only) | SH coefficients (evolving) | hidden (evolving) ]

    geometry = normalized position (3) + log-scale (3) + quaternion (4) + opacity logit (1)
    SH       = 3 channels if sh_mode == "dc", else 3 * K (K = (sh_degree+1)^2)
    hidden   = n_hidden free channels (the automaton's memory)

The update MLP consumes the perception vector and produces a delta for the
evolving channels only, stochastically masked per splat (each splat's delta is
zeroed with probability 1 - fire_rate). Geometry channels never change.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .gaussian_model import GaussianModel
from .graph import build_knn_neighbors, build_neighbor_basis
from .perception import GraphPerception

N_GEOMETRY = 11


class GaussianNCA(nn.Module):
    def __init__(self, gaussians: GaussianModel,
                 sh_mode='dc',
                 n_hidden=16,
                 knn_k=8,
                 basis_degree=2,
                 n_filters=8,
                 hidden_dim=128,
                 fire_rate=0.5,
                 seed_noise_std=0.0):
        """
        :param gaussians: frozen GaussianModel providing geometry + graph
        :param sh_mode: "dc" (view-independent color, K=1) or "full" (all SH bands of the scene)
        :param n_hidden: number of free hidden state channels
        :param knn_k: neighbors per splat in the KNN graph
        :param basis_degree: SH degree of the *edge direction* basis (perception filters)
        :param n_filters: number of learnable perception filters
        :param hidden_dim: width of the update MLP
        :param fire_rate: probability that a splat's delta is applied at each step
                          (the rest are zeroed — stochastic asynchronous updates)
        :param seed_noise_std: std of gaussian noise on hidden channels at seeding
        """
        super().__init__()
        assert sh_mode in ('dc', 'full')
        self.sh_mode = sh_mode
        self.fire_rate = fire_rate
        self.seed_noise_std = seed_noise_std
        self.n_hidden = n_hidden

        n = gaussians.num_gaussians
        self.num_gaussians = n
        self.scene_sh_degree = gaussians.sh_degree
        self.render_sh_degree = 0 if sh_mode == 'dc' else gaussians.sh_degree
        self.sh_k = (self.render_sh_degree + 1) ** 2
        self.n_sh = 3 * self.sh_k

        # ---------------- frozen geometry context ---------------- #
        with torch.no_grad():
            center, radius = gaussians.scene_center_and_radius()
            pos = (gaussians.means - center) / (radius + 1e-8)
            geometry = torch.cat([
                pos,                                    # 3
                gaussians.log_scales,                   # 3
                gaussians.normalized_quats,             # 4
                gaussians.opacity_logits[:, None],      # 1
            ], dim=1)
        self.register_buffer('geometry', geometry)      # [N, 11]

        # ---------------- graph + perception ---------------- #
        neighbors = build_knn_neighbors(gaussians.means, k=knn_k)
        neighbor_basis = build_neighbor_basis(gaussians.means, neighbors, degree=basis_degree)
        self.perception = GraphPerception(neighbors, neighbor_basis, n_filters=n_filters)

        # ---------------- channel bookkeeping ---------------- #
        self.n_evolving = self.n_sh + n_hidden
        self.n_channels = N_GEOMETRY + self.n_evolving
        self.sh_slice = slice(N_GEOMETRY, N_GEOMETRY + self.n_sh)
        self.evolving_slice = slice(N_GEOMETRY, self.n_channels)

        # ---------------- update MLP ---------------- #
        perception_dim = self.perception.out_multiplier * self.n_channels
        self.update_mlp = nn.Sequential(
            nn.Linear(perception_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_evolving),
        )
        # NCA convention: zero-init the last layer so the initial delta is ~0
        nn.init.zeros_(self.update_mlp[-1].weight)
        nn.init.zeros_(self.update_mlp[-1].bias)

    # ------------------------------------------------------------------ #
    def seed(self, batch_size=1, device=None):
        """
        Initial state: geometry context, SH = 0 (neutral gray: rgb = 0.5 at f_dc = 0),
        hidden = 0 (+ optional noise).
        :return: [B, N, C]
        """
        device = device or self.geometry.device
        state = torch.zeros(batch_size, self.num_gaussians, self.n_channels, device=device)
        state[..., :N_GEOMETRY] = self.geometry.to(device)
        if self.seed_noise_std > 0:
            hidden_slice = slice(N_GEOMETRY + self.n_sh, self.n_channels)
            state[..., hidden_slice] = torch.randn_like(state[..., hidden_slice]) * self.seed_noise_std
        return state

    def step(self, state):
        """
        One NCA step. Geometry channels are never modified.
        :param state: [B, N, C]
        :return: new state [B, N, C]
        """
        perception = self.perception(state)                        # [B, N, (F+1)*C]
        delta = self.update_mlp(perception)                        # [B, N, n_evolving]

        # Stochastic asynchronous update: zero each splat's delta w.p. 1 - fire_rate
        if self.fire_rate < 1.0:
            fire_mask = (torch.rand(state.shape[0], state.shape[1], 1,
                                    device=state.device) <= self.fire_rate).float()
            delta = delta * fire_mask

        new_state = state.clone()
        new_state[..., self.evolving_slice] = state[..., self.evolving_slice] + delta
        return new_state

    def forward(self, state, n_steps, checkpoint=False):
        """
        Run n_steps of the automaton.
        :param checkpoint: use gradient checkpointing per step to save memory
        :return: final state [B, N, C]
        """
        for _ in range(n_steps):
            if checkpoint and self.training:
                state = torch.utils.checkpoint.checkpoint(
                    self.step, state, use_reentrant=False)
            else:
                state = self.step(state)
        return state

    # ------------------------------------------------------------------ #
    def decode_colors(self, state):
        """
        Extract the SH color coefficients from the state.
        :param state: [B, N, C]
        :return: colors [B, N, K, 3] ready for gsplat (render with sh_degree=self.render_sh_degree)
        """
        sh = state[..., self.sh_slice]                             # [B, N, 3*K]
        b, n, _ = sh.shape
        return sh.view(b, n, self.sh_k, 3)

    def decode_dc_rest(self, state):
        """
        Split a single state's SH channels into (f_dc [N,3], f_rest [N,K-1,3])
        for GaussianModel.save_ply. Accepts [N, C] or [1, N, C].
        """
        if state.dim() == 3:
            state = state[0]
        colors = self.decode_colors(state[None])[0]                # [N, K, 3]
        return GaussianModel.split_sh(colors)
