"""
Generate a synthetic 3DGS scene: a sphere covered with disc-shaped gaussians.

Points are placed with a Fibonacci lattice for even coverage. Each gaussian is
flattened along the surface normal so the splats tile the sphere like scales.
Output is a standard INRIA-format .ply loadable by GaussianModel / smoke_test.

Usage:
    python scripts/make_sphere.py --out data/scenes/sphere.ply [--n 10000] [--radius 1.0]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gstex.gaussian_model import GaussianModel, SH_C0


def fibonacci_sphere(n):
    """[n, 3] roughly uniform unit directions."""
    i = np.arange(n, dtype=np.float64)
    golden = (1 + 5 ** 0.5) / 2
    phi = 2 * np.pi * i / golden
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(np.clip(1 - z ** 2, 0, 1))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1).astype(np.float32)


def quat_z_to(normals):
    """[n, 4] wxyz quaternions rotating +z onto each normal."""
    w = 1 + normals[:, 2]
    xyz = np.stack([-normals[:, 1], normals[:, 0], np.zeros(len(normals), np.float32)], axis=1)
    q = np.concatenate([w[:, None], xyz], axis=1)
    # degenerate case n = -z: any 180-degree flip about a tangent axis works
    q[w < 1e-6] = np.array([0, 1, 0, 0], dtype=np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='data/scenes/sphere.ply')
    p.add_argument('--n', type=int, default=10000, help='number of gaussians')
    p.add_argument('--radius', type=float, default=1.0)
    p.add_argument('--sh_degree', type=int, default=3,
                   help='SH degree of the output (bands above 0 are zero)')
    args = p.parse_args()

    normals = fibonacci_sphere(args.n)
    means = normals * args.radius

    # each disc covers ~ its share of the sphere's area, with some overlap
    disc_radius = 1.6 * args.radius * np.sqrt(4.0 / args.n)
    log_scales = np.log(np.array([disc_radius, disc_radius, 0.1 * disc_radius],
                                 dtype=np.float32))
    log_scales = np.tile(log_scales, (args.n, 1))

    quats = quat_z_to(normals)
    opacity_logits = np.full(args.n, 4.0, dtype=np.float32)  # sigmoid(4) ~ 0.98

    # color by normal direction so orientation is visible in renders
    rgb = 0.5 + 0.5 * normals
    f_dc = (rgb - 0.5) / SH_C0
    f_rest = np.zeros((args.n, (args.sh_degree + 1) ** 2 - 1, 3), dtype=np.float32)

    model = GaussianModel(
        means=torch.from_numpy(means),
        log_scales=torch.from_numpy(log_scales),
        quats=torch.from_numpy(quats),
        opacity_logits=torch.from_numpy(opacity_logits),
        f_dc=torch.from_numpy(f_dc.astype(np.float32)),
        f_rest=torch.from_numpy(f_rest),
        device='cpu',
    )

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    model.save_ply(args.out)
    print(f'wrote {args.n} gaussians (sh_degree={model.sh_degree}) to {args.out}')


if __name__ == '__main__':
    main()
