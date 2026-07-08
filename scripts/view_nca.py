"""
Watch a trained NCA grow its texture on a 3DGS scene, live.

Interactive mode (default) serves a viser web viewer: orbit the scene in the
browser while a background thread steps the automaton. Over SSH, forward the
port first: `ssh -L 8080:localhost:8080 <host>`.

Video mode (--video) skips the server and writes an orbiting-camera mp4 of
the evolution instead.

Examples:
    python scripts/view_nca.py --checkpoint out/run/checkpoints/nca_000500.pt \
        --scene data/scenes/Tacua/scene.ply
    python scripts/view_nca.py --checkpoint out/run/checkpoints/nca_000500.pt \
        --scene data/scenes/Tacua/scene.ply --video --n_steps 500
"""

import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gstex import GaussianModel, load_nca_for_scene
from gstex.render import look_at_cameras, render_gaussians


def parse_args():
    p = argparse.ArgumentParser(description='Live viewer / video of NCA texture evolution')
    p.add_argument('--checkpoint', type=str, required=True, help='trained nca_*.pt')
    p.add_argument('--scene', type=str, required=True, help='target 3DGS .ply')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--background', type=str, default='black', choices=['black', 'white'])
    # interactive mode
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--max_steps', type=int, default=0,
                   help='stop evolving after this many steps (0 = run forever)')
    # video mode
    p.add_argument('--video', nargs='?', const='__default__', default=None, metavar='PATH',
                   help='write an evolution mp4 instead of serving the viewer '
                        '(default path: out/<scene>_evolution.mp4)')
    p.add_argument('--n_steps', type=int, default=500, help='video: NCA rollout steps')
    p.add_argument('--render_every', type=int, default=2, help='video: steps per frame')
    p.add_argument("--updates_per_sec", type=int, default=10, help="update how many times per sec")
    p.add_argument('--n_orbits', type=float, default=2.0, help='video: camera orbits over the rollout')
    p.add_argument('--elevation', type=float, default=20.0, help='video: camera elevation (deg)')
    p.add_argument('--image_size', type=int, default=512, help='video: frame resolution')
    p.add_argument('--fps', type=int, default=30)
    return p.parse_args()


def setup(args):
    """Load scene + NCA and return everything both modes need."""
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"Loading scene {args.scene} ...")
    gaussians = GaussianModel.load_ply(args.scene, device=args.device)
    print(f"  {gaussians.num_gaussians} gaussians, SH degree {gaussians.sh_degree}")
    nca, cfg = load_nca_for_scene(checkpoint, gaussians, args.device)
    nca.eval()
    print(f"  loaded NCA from {args.checkpoint} "
          f"(trained {checkpoint['iteration']} iters on {cfg['scene']})")
    torch.manual_seed(args.seed)
    return gaussians, nca, cfg


def orbit_camera(center, radius, distance_factor, azimuth_deg, elevation_deg,
                 fov_deg, size, device):
    """Single orbiting camera looking at the scene center. :return: (viewmat, K)"""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    distance = float(radius) * distance_factor
    position = center + distance * torch.tensor(
        [math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)],
        device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    return look_at_cameras(position[None], center, up, fov_deg, size, size)


# ---------------------------------------------------------------------- #
@torch.no_grad()
def run_video(args, gaussians, nca, cfg):
    import imageio.v2 as imageio

    if args.video == '__default__':
        scene_name = os.path.basename(os.path.dirname(os.path.abspath(args.scene)))
        path = os.path.join('out', f'{scene_name}_evolution.mp4')
    else:
        path = args.video
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    center, radius = gaussians.scene_center_and_radius()

    frames_per_update = int(args.fps / args.updates_per_sec)
    n_frames = args.n_steps * frames_per_update
    state = nca.seed(batch_size=1, device=args.device)
    writer = imageio.get_writer(path, fps=args.fps)

    for frame in range(n_frames):
        azimuth = 360.0 * args.n_orbits * frame / max(n_frames - 1, 1)
        viewmat, K = orbit_camera(center, radius, cfg['camera_distance'],
                                  azimuth, args.elevation, cfg['fov'],
                                  args.image_size, args.device)
        image = render_gaussians(
            gaussians, nca.decode_colors(state)[0], viewmat, K,
            args.image_size, args.image_size,
            sh_degree=nca.render_sh_degree, background=args.background,
        )[0]
        writer.append_data(
            (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        if frame % frames_per_update == 0 and frame < n_frames - 1:
            state = nca(state, args.render_every)
    writer.close()
    print(f"Saved {n_frames} frames ({args.n_steps} NCA steps, "
          f"{n_frames / args.fps:.1f}s) to {path}")


# ---------------------------------------------------------------------- #
def run_viewer(args, gaussians, nca):
    import nerfview
    import viser

    device = args.device
    lock = threading.Lock()
    shared = {
        'state': nca.seed(batch_size=1, device=device),
        'step': 0,
    }

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(control_layout='collapsible')
    playing = server.gui.add_checkbox('evolve', initial_value=True)
    sps = server.gui.add_slider('steps / sec', min=1, max=60, step=1, initial_value=10)
    step_counter = server.gui.add_number('step', initial_value=0, disabled=True)
    reset_button = server.gui.add_button('reset to seed')

    @reset_button.on_click
    def _(_):
        with lock:
            shared['state'] = nca.seed(batch_size=1, device=device)
            shared['step'] = 0

    @torch.no_grad()
    def render_fn(camera_state, img_wh):
        width, height = img_wh
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        viewmat = torch.linalg.inv(c2w)[None]
        K = torch.from_numpy(camera_state.get_K(img_wh)).float().to(device)[None]
        with lock:
            colors = nca.decode_colors(shared['state'])[0]
        image = render_gaussians(
            gaussians, colors, viewmat, K, width, height,
            sh_degree=nca.render_sh_degree, background=args.background,
        )[0]
        return image.permute(1, 2, 0).cpu().numpy()

    viewer = nerfview.Viewer(server=server, render_fn=render_fn, mode='rendering')

    print(f"Viewer running at http://localhost:{args.port} "
          f"(over SSH: ssh -L {args.port}:localhost:{args.port} <host>)")

    @torch.no_grad()
    def evolve():
        while True:
            if playing.value and (args.max_steps <= 0 or shared['step'] < args.max_steps):
                with lock:
                    shared['state'] = nca.step(shared['state'])
                    shared['step'] += 1
                step_counter.value = shared['step']
                viewer.rerender(None)
            time.sleep(1.0 / sps.value)

    threading.Thread(target=evolve, daemon=True).start()
    while True:
        time.sleep(3600)


def main():
    args = parse_args()
    gaussians, nca, cfg = setup(args)
    if args.video is not None:
        run_video(args, gaussians, nca, cfg)
    else:
        run_viewer(args, gaussians, nca)


if __name__ == '__main__':
    main()
