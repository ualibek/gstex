"""
KNN neighborhood construction over splat centers and the fixed per-neighbor
geometric features (real spherical-harmonics basis of the edge directions).

Because the graph is a fixed-k KNN graph, the neighborhood is stored densely as
a [N, k] index matrix — perception then needs no scatter ops, just a gather and
a batched matmul. Geometry is frozen, so everything here is computed once.
"""

import torch


@torch.no_grad()
def build_knn_neighbors(means, k, chunk_size=8192):
    """
    Dense KNN neighborhood over the splat centers.

    Tries pytorch3d if available, otherwise falls back to a chunked torch.cdist
    implementation (self-contained, a few seconds for ~10^6 points).

    :param means: [N, 3] splat centers
    :param k: number of neighbors per node (excluding self)
    :return: neighbors [N, k] long tensor of neighbor indices
    """
    n = means.shape[0]
    assert k < n, "k must be smaller than the number of splats"

    try:
        from pytorch3d.ops import knn_points
        knn = knn_points(means[None], means[None], K=k + 1)
        return knn.idx[0, :, 1:].contiguous()  # [N, k], drop self
    except ImportError:
        pass

    # Chunked brute-force fallback
    neighbor_chunks = []
    for start in range(0, n, chunk_size):
        chunk = means[start:start + chunk_size]                    # [c, 3]
        c = chunk.shape[0]
        dists = torch.cdist(chunk, means)                          # [c, N]
        # Exclude self-matches before topk
        rows = torch.arange(c, device=means.device)
        dists[rows, rows + start] = float('inf')
        idx = dists.topk(k, dim=-1, largest=False).indices         # [c, k]
        neighbor_chunks.append(idx)
    return torch.cat(neighbor_chunks, dim=0)                       # [N, k]


def sh_basis(dirs, degree=2):
    """
    Real spherical harmonics basis evaluated at unit directions.
    Exact normalization/sign conventions are irrelevant here because learnable
    filter weights follow; we only need a linearly independent directional basis.

    :param dirs: [..., 3] unit vectors
    :param degree: max SH degree (0..3), giving (degree+1)^2 basis functions
    :return: [..., (degree+1)^2]
    """
    assert 0 <= degree <= 3
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    basis = [torch.full_like(x, 0.28209479177387814)]

    if degree >= 1:
        basis += [
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
        ]
    if degree >= 2:
        basis += [
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (3.0 * z * z - 1.0),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
        ]
    if degree >= 3:
        basis += [
            -0.5900435899266435 * y * (3.0 * x * x - y * y),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (5.0 * z * z - 1.0),
            0.3731763325901154 * z * (5.0 * z * z - 3.0),
            -0.4570457994644658 * x * (5.0 * z * z - 1.0),
            1.445305721320277 * z * (x * x - y * y),
            -0.5900435899266435 * x * (x * x - 3.0 * y * y),
        ]
    return torch.stack(basis, dim=-1)


@torch.no_grad()
def build_neighbor_basis(means, neighbors, degree=2, include_distance=True):
    """
    Precompute the fixed per-neighbor feature basis used by the perception layers.

    :param means: [N, 3]
    :param neighbors: [N, k]
    :return: [N, k, D] with D = (degree+1)^2 (+1 if include_distance)
    """
    delta = means[neighbors] - means[:, None, :]                   # [N, k, 3]
    dists = delta.norm(dim=-1)                                     # [N, k]
    dirs = delta / (dists[..., None] + 1e-8)

    basis = sh_basis(dirs, degree=degree)                          # [N, k, D]
    if include_distance:
        # Normalize distances by their median so the feature is scene-scale invariant
        dists = dists / (dists.median() + 1e-8)
        basis = torch.cat([basis, dists[..., None]], dim=-1)
    return basis
