"""
Learnable graph-convolution perception layers.

Each perception filter assigns a scalar weight to every (node, neighbor) pair
as a learnable linear combination of the fixed per-neighbor SH basis (the
anisotropic analogue of MeshNCA's directional filters), then aggregates
neighbor states with those weights. The identity perception (the node's own
state) is concatenated last.

Because the KNN neighborhood is dense ([N, k]), aggregation is a single gather
followed by one batched matmul — no scatter ops.
"""

import torch
import torch.nn as nn


class GraphPerception(nn.Module):
    def __init__(self, neighbors, neighbor_basis, n_filters=8):
        """
        :param neighbors: [N, k] — fixed KNN neighbor indices
        :param neighbor_basis: [N, k, D] — fixed per-neighbor SH basis features
        :param n_filters: number of learnable perception filters F
        """
        super().__init__()
        self.register_buffer('neighbors', neighbors)
        self.register_buffer('neighbor_basis', neighbor_basis)
        self.n_filters = n_filters
        self.k = neighbors.shape[1]

        d = neighbor_basis.shape[-1]
        # Learnable filter bank: maps the D-dim neighbor basis to F edge weights
        self.filter_weights = nn.Parameter(torch.randn(d, n_filters) * 0.3)

    @property
    def out_multiplier(self):
        """Output channels = out_multiplier * state channels."""
        return self.n_filters + 1  # + identity

    def forward(self, state):
        """
        :param state: [B, N, C]
        :return: perception vector [B, N, (F+1) * C]
        """
        b, n, c = state.shape

        # Per-neighbor filter weights, [N, k, F] (cheap: recomputed each step
        # so gradients reach filter_weights, but it is a tiny matmul)
        weights = self.neighbor_basis @ self.filter_weights

        # Gather neighbor states: [B, N, k, C]
        messages = state[:, self.neighbors.reshape(-1), :].view(b, n, self.k, c)

        # Weighted aggregation over the k neighbors as one batched matmul:
        # perceived[b, n, f, c] = sum_k weights[n, k, f] * messages[b, n, k, c]
        perceived = torch.einsum('nkf,bnkc->bnfc', weights, messages) / self.k

        return torch.cat([perceived.reshape(b, n, -1), state], dim=-1)  # [B, N, (F+1)*C]
