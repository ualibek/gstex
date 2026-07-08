"""
Apply a trained GaussianNCA checkpoint to a (possibly unseen) 3DGS scene.

The NCA's learnable weights (perception filter bank + update MLP) are
scene-agnostic; the scene-specific buffers (geometry context, KNN graph,
edge-direction basis) are rebuilt for the target scene. Outputs a textured
.ply and renders from random viewpoints.

Example:
    python scripts/apply_nca.py --checkpoint out/run/checkpoints/nca_000500.pt \
        --scene data/scenes/Tacua/scene.ply --out_dir out/apply_Tacua
"""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gstex import GaussianModel, load_nca_for_scene
from gstex.render import sample_random_cameras, render_gaussians


def parse_args():
    p = argparse.ArgumentParser(description='Apply a trained NCA to a new scene')
    p.add_argument('--checkpoint', type=str, required=True, help='trained nca_*.pt')
    p.add_argument('--scene', type=str, required=True, help='target 3DGS .ply')
    p.add_argument('--out_dir', type=str, default=None,
                   help='default: out/apply_<scene dir name>')
    p.add_argument('--n_steps', type=int, default=None,
                   help='NCA rollout steps (default: max_steps from the checkpoint)')
    p.add_argument('--n_views', type=int, default=8)
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--background', type=str, default='black',
                   choices=['black', 'white', 'random'])
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def save_image(tensor, path):
    """[3, H, W] float in [0, 1] -> PNG."""
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    Image.fromarray(array).save(path)


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    scene_name = os.path.basename(os.path.dirname(os.path.abspath(args.scene)))
    out_dir = args.out_dir or os.path.join('out', f'apply_{scene_name}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading scene {args.scene} ...")
    gaussians = GaussianModel.load_ply(args.scene, device=device)
    print(f"  {gaussians.num_gaussians} gaussians, SH degree {gaussians.sh_degree}")

    nca, cfg = load_nca_for_scene(checkpoint, gaussians, device)
    nca.eval()
    print(f"  loaded NCA from {args.checkpoint} "
          f"(trained {checkpoint['iteration']} iters on {cfg['scene']})")

    n_steps = args.n_steps or cfg['max_steps']
    print(f"Growing texture from seed for {n_steps} steps ...")
    state = nca.seed(batch_size=1, device=device)
    state = nca(state, n_steps)

    # Textured .ply (same convention as Trainer.export_textured_ply)
    f_dc, f_rest = nca.decode_dc_rest(state)
    if nca.sh_mode == 'dc':
        f_rest = torch.zeros_like(gaussians.f_rest)
    ply_path = os.path.join(out_dir, 'textured.ply')
    gaussians.save_ply(ply_path, f_dc=f_dc, f_rest=f_rest)
    print(f"Saved {ply_path}")

    # Renders from random viewpoints
    center, radius = gaussians.scene_center_and_radius()
    viewmats, Ks = sample_random_cameras(
        args.n_views, center, radius,
        distance_factor=cfg['camera_distance'], fov_deg=cfg['fov'],
        width=args.image_size, height=args.image_size, device=device,
    )
    colors = nca.decode_colors(state)[0]
    images = render_gaussians(
        gaussians, colors, viewmats, Ks, args.image_size, args.image_size,
        sh_degree=nca.render_sh_degree, background=args.background,
    )

    save_image(torch.cat(list(images), dim=2), os.path.join(out_dir, 'grid.png'))
    print(f"Saved {images.shape[0]} views + grid to {os.path.join(out_dir, 'renders')}")


if __name__ == '__main__':
    main()
