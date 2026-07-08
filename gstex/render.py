"""
gsplat rendering wrapper + random viewpoint sampling.

Cameras follow gsplat conventions: viewmats are world->camera with OpenCV axes
(x right, y down, z forward), intrinsics are pinhole K matrices.

Random viewpoints are sampled on a sphere around the scene: azimuth uniform,
elevation uniform in sin (area-uniform on the sphere).
"""

import math

import torch
from gsplat import rasterization


@torch.no_grad()
def sample_random_cameras(n_views, center, radius,
                          distance_factor=2.5, fov_deg=60.0,
                          width=224, height=224,
                          elevation_range_deg=(-75.0, 75.0),
                          up=(0.0, 1.0, 0.0),
                          device='cuda:0'):
    """
    Sample cameras on a sphere around the scene, looking at its center.

    :param center: [3] scene center
    :param radius: scalar scene radius (cameras placed at radius * distance_factor)
    :return: (viewmats [C, 4, 4], Ks [C, 3, 3])
    """
    center = torch.as_tensor(center, dtype=torch.float32, device=device)
    up = torch.as_tensor(up, dtype=torch.float32, device=device)
    distance = float(radius) * distance_factor

    azimuth = torch.rand(n_views, device=device) * 2.0 * math.pi
    sin_lo = math.sin(math.radians(elevation_range_deg[0]))
    sin_hi = math.sin(math.radians(elevation_range_deg[1]))
    elevation = torch.asin(torch.rand(n_views, device=device) * (sin_hi - sin_lo) + sin_lo)

    # Camera positions (y-up world)
    positions = center + distance * torch.stack([
        torch.cos(elevation) * torch.cos(azimuth),
        torch.sin(elevation),
        torch.cos(elevation) * torch.sin(azimuth),
    ], dim=1)                                                       # [C, 3]

    return look_at_cameras(positions, center, up, fov_deg, width, height)


@torch.no_grad()
def look_at_cameras(positions, target, up, fov_deg, width, height):
    """
    Build gsplat-convention (OpenCV) world->camera matrices and intrinsics.

    :param positions: [C, 3] camera centers
    :param target: [3] look-at point
    :return: (viewmats [C, 4, 4], Ks [C, 3, 3])
    """
    device = positions.device
    n = positions.shape[0]

    forward = target[None] - positions
    forward = forward / (forward.norm(dim=-1, keepdim=True) + 1e-8)       # cam z
    right = torch.cross(forward, up[None].expand_as(forward), dim=-1)     # cam x
    right = right / (right.norm(dim=-1, keepdim=True) + 1e-8)
    down = torch.cross(forward, right, dim=-1)                            # cam y

    rot = torch.stack([right, down, forward], dim=1)                      # [C, 3, 3] rows = axes
    trans = -torch.einsum('cij,cj->ci', rot, positions)                   # [C, 3]

    viewmats = torch.eye(4, device=device)[None].repeat(n, 1, 1)
    viewmats[:, :3, :3] = rot
    viewmats[:, :3, 3] = trans

    focal = 0.5 * height / math.tan(0.5 * math.radians(fov_deg))
    Ks = torch.tensor([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], device=device)[None].repeat(n, 1, 1)

    return viewmats, Ks


def render_gaussians(gaussians, colors, viewmats, Ks, width, height,
                     sh_degree, background=None):
    """
    Differentiable rendering of a 3DGS scene with (possibly NCA-produced) SH colors.

    :param gaussians: GaussianModel (frozen geometry)
    :param colors: [N, K, 3] SH coefficients (grad flows through this)
    :param viewmats: [C, 4, 4], Ks: [C, 3, 3]
    :param sh_degree: SH degree matching K = (sh_degree+1)^2
    :param background: None (black), "white", "random", or a [3] tensor
    :return: images [C, 3, H, W] clamped to [0, 1]
    """
    device = colors.device

    # Composite the background ourselves: gsplat 1.5.x's packed path rejects
    # [C, 3] backgrounds and does out-of-bounds per-camera reads on a [3] one.
    if background is None or background == 'black':
        bg = None
    elif background == 'white':
        bg = torch.ones(3, device=device)
    elif background == 'random':
        bg = torch.rand(3, device=device)
    else:
        bg = torch.as_tensor(background, dtype=torch.float32, device=device)

    renders, alphas, _ = rasterization(
        means=gaussians.means,
        quats=gaussians.normalized_quats,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode='RGB',
    )                                    # renders [C, H, W, 3], alphas [C, H, W, 1]

    if bg is not None:
        renders = renders + (1.0 - alphas) * bg

    images = renders.permute(0, 3, 1, 2)                             # [C, 3, H, W]
    return images.clamp(0.0, 1.0)
