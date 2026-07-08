"""
Entry point: texturize a pretrained 3DGS scene with a Neural Cellular Automaton.

Example:
    python train.py --scene data/scenes/chair.ply --texture data/textures/bubbly_0101.jpg \
        --sh_mode dc --out_dir out/chair_bubbly
"""

import argparse

import torch

from gstex import AppearanceLoss, GaussianModel, GaussianNCA, Trainer


def parse_args():
    p = argparse.ArgumentParser(description='GSTex: texturize 3DGS with NCA')
    # data
    p.add_argument('--scene', type=str, required=True, help='path to the pretrained 3DGS .ply')
    p.add_argument('--texture', type=str, required=True, help='path to the target texture image')
    p.add_argument('--out_dir', type=str, default='out/run')
    # NCA
    p.add_argument('--sh_mode', type=str, default='dc', choices=['dc', 'full'],
                   help='dc: view-independent color only; full: all SH bands')
    p.add_argument('--n_hidden', type=int, default=16, help='hidden state channels')
    p.add_argument('--knn_k', type=int, default=8, help='KNN neighbors per splat')
    p.add_argument('--basis_degree', type=int, default=2, help='SH degree of the edge-direction basis')
    p.add_argument('--n_filters', type=int, default=8, help='learnable perception filters')
    p.add_argument('--hidden_dim', type=int, default=128, help='update MLP width')
    p.add_argument('--fire_rate', type=float, default=0.5)
    # training
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--pool_size', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--min_steps', type=int, default=32)
    p.add_argument('--max_steps', type=int, default=96)
    p.add_argument('--reseed_interval', type=int, default=4,
                   help='every k iterations, restart one batch entry from seed')
    p.add_argument('--overflow_weight', type=float, default=1e2)
    p.add_argument('--checkpoint_steps', action=argparse.BooleanOptionalAction, default=True,
                   help='gradient-checkpoint each NCA step to save memory '
                        '(disable with --no-checkpoint_steps for small scenes)')
    # rendering
    p.add_argument('--n_views', type=int, default=4, help='random views per batch state')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--camera_distance', type=float, default=2.5,
                   help='camera distance as a multiple of the scene radius')
    p.add_argument('--fov', type=float, default=60.0)
    p.add_argument('--background', type=str, default='black',
                   choices=['black', 'white', 'random'])
    # misc
    p.add_argument('--render_interval', type=int, default=50,
                   help='iterations between summary render snapshots')
    p.add_argument('--eval_interval', type=int, default=10,
                   help='iterations between deterministic eval-loss probes')
    p.add_argument('--checkpoint_interval', type=int, default=500)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading scene {args.scene} ...")
    gaussians = GaussianModel.load_ply(args.scene, device=args.device)
    print(f"  {gaussians.num_gaussians} gaussians, SH degree {gaussians.sh_degree}")

    nca = GaussianNCA(
        gaussians,
        sh_mode=args.sh_mode,
        n_hidden=args.n_hidden,
        knn_k=args.knn_k,
        basis_degree=args.basis_degree,
        n_filters=args.n_filters,
        hidden_dim=args.hidden_dim,
        fire_rate=args.fire_rate,
    ).to(args.device)
    n_params = sum(p.numel() for p in nca.parameters() if p.requires_grad)
    print(f"  NCA: {nca.n_channels} state channels "
          f"({nca.n_sh} SH + {args.n_hidden} hidden), {n_params} trainable params")

    loss_fn = AppearanceLoss(
        image_size=(args.image_size, args.image_size),
        target_images_path=args.texture,
        device=args.device,
    )

    config = vars(args)
    trainer = Trainer(gaussians, nca, loss_fn, config)
    trainer.train()


if __name__ == '__main__':
    main()
