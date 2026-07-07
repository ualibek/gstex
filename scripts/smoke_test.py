"""
Milestone smoke tests (run before any training):

  1. IO + render:  load a .ply, render the ORIGINAL scene from a few views,
                   save PNGs, and round-trip save_ply. Catches camera-convention
                   and SH-layout bugs before any NCA is involved.
  2. Graph + perception: build the KNN graph, check degrees, run the perception
                   forward and check shapes.
  3. NCA forward:  run T steps, assert geometry channels unchanged;
                   render the (untrained) NCA output.

Usage:
    python scripts/smoke_test.py --scene data/scenes/<scene>.ply [--out_dir out/smoke]
"""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gstex import GaussianModel, GaussianNCA
from gstex.nca import N_GEOMETRY
from gstex.render import sample_random_cameras, render_gaussians


def save_image_grid(images, path):
    """images: [C, 3, H, W] in [0, 1] -> horizontal strip PNG."""
    grid = torch.cat(list(images.permute(0, 2, 3, 1)), dim=1)  # [H, C*W, 3]
    Image.fromarray((grid.clamp(0, 1).cpu().numpy() * 255).astype('uint8')).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='out/smoke')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--n_views', type=int, default=4)
    p.add_argument('--image_size', type=int, default=512)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- Milestone 1: IO + render ---------------- #
    print("== Milestone 1: IO + render ==")
    g = GaussianModel.load_ply(args.scene, device=args.device)
    print(f"  {g.num_gaussians} gaussians, SH degree {g.sh_degree}")
    center, radius = g.scene_center_and_radius()
    print(f"  center={center.tolist()}, radius={radius.item():.3f}")

    viewmats, Ks = sample_random_cameras(
        args.n_views, center, radius, width=args.image_size, height=args.image_size,
        device=args.device)
    with torch.no_grad():
        images = render_gaussians(g, g.get_colors(), viewmats, Ks,
                                  args.image_size, args.image_size,
                                  sh_degree=g.sh_degree, background='white')
    save_image_grid(images, os.path.join(args.out_dir, 'original_renders.png'))
    print(f"  saved {args.out_dir}/original_renders.png  <- INSPECT: scene must look correct")

    roundtrip_path = os.path.join(args.out_dir, 'roundtrip.ply')
    g.save_ply(roundtrip_path)
    g2 = GaussianModel.load_ply(roundtrip_path, device=args.device)
    for name in ['means', 'log_scales', 'quats', 'opacity_logits', 'f_dc', 'f_rest']:
        a, b = getattr(g, name), getattr(g2, name)
        assert torch.allclose(a, b, atol=1e-6), f"round-trip mismatch in {name}"
    print("  save_ply round-trip OK")

    # ---------------- Milestone 2 + 3: graph, perception, NCA ---------------- #
    print("== Milestone 2/3: graph + perception + NCA forward ==")
    nca = GaussianNCA(g, sh_mode='dc', n_hidden=8, knn_k=8).to(args.device)
    nb = nca.perception.neighbors
    assert nb.shape == (g.num_gaussians, 8), "unexpected neighbor matrix shape"
    self_idx = torch.arange(g.num_gaussians, device=nb.device)[:, None]
    assert (nb != self_idx).all(), "self-loops in KNN neighborhood"
    print(f"  KNN neighborhood: {tuple(nb.shape)}, basis dim {nca.perception.neighbor_basis.shape[-1]}")

    state = nca.seed(batch_size=1)
    perception = nca.perception(state)
    assert perception.shape == (1, g.num_gaussians, nca.perception.out_multiplier * nca.n_channels)
    print(f"  perception output: {tuple(perception.shape)} OK")

    with torch.no_grad():
        nca.eval()
        state_t = nca(state.clone(), n_steps=16)

    assert torch.equal(state_t[..., :N_GEOMETRY], state[..., :N_GEOMETRY]), \
        "geometry channels changed!"
    # Note: with the zero-initialized update MLP the delta is exactly 0 before
    # training, so SH channels are expected to be unchanged here.
    print("  geometry frozen OK")

    with torch.no_grad():
        colors = nca.decode_colors(state_t)[0]
        images = render_gaussians(g, colors, viewmats, Ks,
                                  args.image_size, args.image_size,
                                  sh_degree=nca.render_sh_degree, background='white')
    save_image_grid(images, os.path.join(args.out_dir, 'nca_seed_renders.png'))
    print(f"  saved {args.out_dir}/nca_seed_renders.png  <- should be uniform gray silhouettes")

    # Gradient flow check: loss on rendered pixels must reach NCA params
    nca.train()
    state = nca.seed(batch_size=1)
    state_t = nca(state, n_steps=4)
    colors = nca.decode_colors(state_t)[0]
    grad_viewmats, grad_Ks = sample_random_cameras(
        1, center, radius, width=64, height=64, device=args.device)
    images = render_gaussians(g, colors, grad_viewmats, grad_Ks,
                              64, 64, sh_degree=nca.render_sh_degree)
    images.mean().backward()
    grads = [p.grad.abs().sum().item() for p in nca.parameters() if p.grad is not None]
    assert len(grads) > 0 and sum(grads) > 0, "no gradient reached the NCA parameters"
    print("  gradient flow renderer -> NCA params OK")

    print("\nAll smoke tests passed.")


if __name__ == '__main__':
    main()
