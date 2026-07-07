"""
Loading / saving of 3D Gaussian Splatting scenes in the standard INRIA .ply format,
plus activation helpers to turn the raw (storage-space) parameters into the
rasterizer-ready quantities expected by gsplat.

Storage conventions of the .ply format:
    - x, y, z                  : splat centers
    - nx, ny, nz               : unused normals (kept for compatibility)
    - f_dc_0..2                : 0th-order SH coefficients (DC color term)
    - f_rest_0..3*(K-1)-1      : higher-order SH coefficients, channel-major
                                 (i.e. [3, K-1] flattened per splat)
    - opacity                  : opacity logit (sigmoid -> [0, 1])
    - scale_0..2               : log-scales (exp -> positive scales)
    - rot_0..3                 : unnormalized quaternion (wxyz)
"""

import numpy as np
import torch
from plyfile import PlyData, PlyElement

# 0th-order real SH basis constant: rgb = SH_C0 * f_dc + 0.5
SH_C0 = 0.28209479177387814


class GaussianModel:
    """
    Holds the raw (storage-space) parameters of a 3DGS scene as torch tensors.
    Geometry is frozen in this project; only the SH coefficients get rewritten
    by the NCA before saving.
    """

    def __init__(self, means, log_scales, quats, opacity_logits, f_dc, f_rest, device='cuda:0'):
        """
        :param means:           [N, 3] splat centers
        :param log_scales:      [N, 3] log-space scales
        :param quats:           [N, 4] unnormalized quaternions (wxyz)
        :param opacity_logits:  [N]    opacity logits
        :param f_dc:            [N, 3] DC SH coefficients
        :param f_rest:          [N, K-1, 3] higher-order SH coefficients
        """
        self.device = device
        self.means = means.to(device)
        self.log_scales = log_scales.to(device)
        self.quats = quats.to(device)
        self.opacity_logits = opacity_logits.to(device)
        self.f_dc = f_dc.to(device)
        self.f_rest = f_rest.to(device)

        num_rest = self.f_rest.shape[1]  # K - 1
        self.sh_degree = int(round(np.sqrt(num_rest + 1))) - 1
        assert (self.sh_degree + 1) ** 2 == num_rest + 1, \
            f"f_rest has {num_rest} coefficients per channel, which does not match any SH degree"

    @property
    def num_gaussians(self):
        return self.means.shape[0]

    # ------------------------------------------------------------------ #
    # Activations (storage space -> rasterizer space)
    # ------------------------------------------------------------------ #
    @property
    def scales(self):
        return torch.exp(self.log_scales)

    @property
    def opacities(self):
        return torch.sigmoid(self.opacity_logits)

    @property
    def normalized_quats(self):
        return self.quats / (self.quats.norm(dim=-1, keepdim=True) + 1e-8)

    def scene_center_and_radius(self):
        """Robust center / radius estimate used to place cameras and normalize positions."""
        with torch.no_grad():
            center = self.means.mean(dim=0)
            radius = (self.means - center).norm(dim=-1).quantile(0.97)
        return center, radius

    # ------------------------------------------------------------------ #
    # SH assembly
    # ------------------------------------------------------------------ #
    @staticmethod
    def assemble_sh(f_dc, f_rest):
        """
        :param f_dc:   [..., 3]
        :param f_rest: [..., K-1, 3]
        :return: colors [..., K, 3] as expected by gsplat's sh rendering
        """
        return torch.cat([f_dc.unsqueeze(-2), f_rest], dim=-2)

    @staticmethod
    def split_sh(colors):
        """Inverse of assemble_sh: [..., K, 3] -> ([..., 3], [..., K-1, 3])."""
        return colors[..., 0, :], colors[..., 1:, :]

    def get_colors(self):
        """Original SH colors of the scene, [N, K, 3]."""
        return self.assemble_sh(self.f_dc, self.f_rest)

    # ------------------------------------------------------------------ #
    # IO
    # ------------------------------------------------------------------ #
    @classmethod
    def load_ply(cls, path, device='cuda:0'):
        ply = PlyData.read(path)
        v = ply['vertex']

        means = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        log_scales = np.stack([v[f'scale_{i}'] for i in range(3)], axis=1).astype(np.float32)
        quats = np.stack([v[f'rot_{i}'] for i in range(4)], axis=1).astype(np.float32)
        opacity_logits = np.asarray(v['opacity']).astype(np.float32)
        f_dc = np.stack([v[f'f_dc_{i}'] for i in range(3)], axis=1).astype(np.float32)

        rest_names = sorted(
            [p.name for p in v.properties if p.name.startswith('f_rest_')],
            key=lambda name: int(name.split('_')[-1])
        )
        if len(rest_names) > 0:
            f_rest = np.stack([v[name] for name in rest_names], axis=1).astype(np.float32)
            num_rest = len(rest_names) // 3
            # stored channel-major: [N, 3, K-1] -> [N, K-1, 3]
            f_rest = f_rest.reshape(-1, 3, num_rest).transpose(0, 2, 1)
        else:
            f_rest = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

        return cls(
            means=torch.from_numpy(means),
            log_scales=torch.from_numpy(log_scales),
            quats=torch.from_numpy(quats),
            opacity_logits=torch.from_numpy(opacity_logits),
            f_dc=torch.from_numpy(f_dc),
            f_rest=torch.from_numpy(np.ascontiguousarray(f_rest)),
            device=device,
        )

    def save_ply(self, path, f_dc=None, f_rest=None):
        """
        Save the scene, optionally with updated SH coefficients (geometry unchanged).

        :param f_dc:   optional [N, 3] replacement DC coefficients
        :param f_rest: optional [N, K-1, 3] replacement higher-order coefficients
        """
        f_dc = self.f_dc if f_dc is None else f_dc
        f_rest = self.f_rest if f_rest is None else f_rest
        assert f_dc.shape == (self.num_gaussians, 3)
        assert f_rest.shape[0] == self.num_gaussians and f_rest.shape[2] == 3

        n = self.num_gaussians
        num_rest = f_rest.shape[1]

        names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        names += [f'f_dc_{i}' for i in range(3)]
        names += [f'f_rest_{i}' for i in range(3 * num_rest)]
        names += ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]

        dtype = [(name, 'f4') for name in names]
        elements = np.empty(n, dtype=dtype)

        def to_np(t):
            return t.detach().cpu().numpy().astype(np.float32)

        data = np.concatenate([
            to_np(self.means),
            np.zeros((n, 3), dtype=np.float32),                       # normals
            to_np(f_dc),
            to_np(f_rest).transpose(0, 2, 1).reshape(n, -1),          # back to channel-major
            to_np(self.opacity_logits)[:, None],
            to_np(self.log_scales),
            to_np(self.quats),
        ], axis=1)

        elements[:] = list(map(tuple, data))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
