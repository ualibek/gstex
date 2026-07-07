"""
Sample-pool NCA training loop (persistence training, as in the NCA literature):
a pool of M states is maintained; each iteration a batch is sampled, evolved
for a random number of steps, rendered from random viewpoints, scored against
the target texture with the MeshNCA appearance loss, and written back.

Only the NCA weights are optimized. Splat geometry is frozen and the SH colors
are *produced* by the automaton, never optimized directly.
"""

import os
import random

import torch
from tqdm import tqdm

from .nca import GaussianNCA
from .render import sample_random_cameras, render_gaussians


class Trainer:
    def __init__(self, gaussians, nca: GaussianNCA, appearance_loss, config):
        """
        :param gaussians: GaussianModel (frozen)
        :param nca: GaussianNCA
        :param appearance_loss: AppearanceLoss (expects rendered_images [b, 3, H, W])
        :param config: dict, see train.py for keys
        """
        self.gaussians = gaussians
        self.nca = nca
        self.loss_fn = appearance_loss
        self.cfg = config
        self.device = config['device']

        self.optimizer = torch.optim.Adam(nca.parameters(), lr=config['lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(config['iterations'] * m) for m in (0.5, 0.8)],
            gamma=0.3,
        )

        # State pool lives on CPU; batches are moved to the device on demand.
        with torch.no_grad():
            self.pool = nca.seed(batch_size=config['pool_size']).cpu()

        self.center, self.radius = gaussians.scene_center_and_radius()

        self.out_dir = config['out_dir']
        os.makedirs(os.path.join(self.out_dir, 'renders'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)

        self.loss_csv = os.path.join(self.out_dir, 'loss_log.csv')
        with open(self.loss_csv, 'w') as f:
            f.write('iteration,loss,style_loss,overflow,n_steps,eval_loss\n')

    # ------------------------------------------------------------------ #
    def _render_batch(self, states, n_views, width, height, background):
        """Render each batch state from its own random viewpoints.
        :return: images [B * n_views, 3, H, W]
        """
        images = []
        for b in range(states.shape[0]):
            viewmats, Ks = sample_random_cameras(
                n_views, self.center, self.radius,
                distance_factor=self.cfg['camera_distance'],
                fov_deg=self.cfg['fov'],
                width=width, height=height,
                device=self.device,
            )
            colors = self.nca.decode_colors(states[b:b + 1])[0]         # [N, K, 3]
            images.append(render_gaussians(
                self.gaussians, colors, viewmats, Ks, width, height,
                sh_degree=self.nca.render_sh_degree, background=background,
            ))
        return torch.cat(images, dim=0)

    @staticmethod
    def _overflow_loss(states, evolving_slice, bound=3.0):
        """Soft penalty keeping the evolving channels bounded (NCA stability)."""
        x = states[..., evolving_slice]
        return ((x.abs() - bound).clamp(min=0.0) ** 2).mean()

    # ------------------------------------------------------------------ #
    def train_step(self, iteration):
        cfg = self.cfg
        self.nca.train()

        batch_idx = random.sample(range(cfg['pool_size']), cfg['batch_size'])
        states = self.pool[batch_idx].to(self.device)

        # Persistence training: periodically restart one batch entry from seed
        # so the automaton keeps learning to grow the texture from scratch.
        if iteration % cfg['reseed_interval'] == 0:
            states[0] = self.nca.seed(batch_size=1, device=self.device)[0]

        n_steps = random.randint(cfg['min_steps'], cfg['max_steps'])
        states = self.nca(states, n_steps, checkpoint=cfg['checkpoint_steps'])

        images = self._render_batch(
            states, cfg['n_views'], cfg['image_size'], cfg['image_size'], cfg['background'])

        style_loss, loss_log, summary = self.loss_fn(
            {'rendered_images': images},
            return_summary=(iteration % cfg['log_interval'] == 0),
        )
        overflow = self._overflow_loss(states, self.nca.evolving_slice)
        loss = style_loss + cfg['overflow_weight'] * overflow

        self.optimizer.zero_grad()
        loss.backward()
        # NCA-standard per-parameter gradient normalization + clipping
        for p in self.nca.parameters():
            if p.grad is not None:
                p.grad /= (p.grad.norm() + 1e-8)
        torch.nn.utils.clip_grad_norm_(self.nca.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        # Write evolved states back to the pool
        with torch.no_grad():
            self.pool[batch_idx] = states.detach().cpu()

        log = {
            'loss': loss.item(),
            'style_loss': style_loss.item(),
            'overflow': overflow.item(),
            'n_steps': n_steps,
        }
        return log, summary

    @torch.no_grad()
    def eval_probe(self):
        """
        Deterministic loss probe for tracking progress: fixed RNG for seed state,
        cameras, fire masks, background and the OT loss' feature subsampling, and
        a fixed rollout length. Comparable across iterations, unlike the (very
        noisy) per-iteration training loss.
        """
        cfg = self.cfg
        self.nca.eval()
        devices = [self.device] if 'cuda' in str(self.device) else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            state = self.nca.seed(batch_size=1, device=self.device)
            state = self.nca(state, cfg['max_steps'])
            images = self._render_batch(
                state, cfg['n_views'], cfg['image_size'], cfg['image_size'], cfg['background'])
            loss, _, _ = self.loss_fn({'rendered_images': images}, return_summary=False)
        self.nca.train()
        return loss.item()

    def train(self):
        cfg = self.cfg
        progress = tqdm(range(1, cfg['iterations'] + 1), desc='train', dynamic_ncols=True)
        for iteration in progress:
            log, summary = self.train_step(iteration)

            eval_loss = None
            if iteration % cfg['eval_interval'] == 0:
                eval_loss = self.eval_probe()
                tqdm.write(f"[eval @ {iteration:6d}] deterministic style loss: {eval_loss:.4f}")

            progress.set_postfix(
                loss=f"{log['loss']:.4f}",
                style=f"{log['style_loss']:.4f}",
                overflow=f"{log['overflow']:.4f}",
                steps=log['n_steps'],
            )
            with open(self.loss_csv, 'a') as f:
                f.write(f"{iteration},{log['loss']:.6f},{log['style_loss']:.6f},"
                        f"{log['overflow']:.6f},{log['n_steps']},"
                        f"{'' if eval_loss is None else f'{eval_loss:.6f}'}\n")

            if iteration % cfg['log_interval'] == 0 and summary is not None:
                summary['images'].save(
                    os.path.join(self.out_dir, 'renders', f'iter_{iteration:06d}.png'))

            if iteration % cfg['checkpoint_interval'] == 0 or iteration == cfg['iterations']:
                self.save_checkpoint(iteration)

        self.export_textured_ply(os.path.join(self.out_dir, 'textured.ply'))

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, iteration):
        path = os.path.join(self.out_dir, 'checkpoints', f'nca_{iteration:06d}.pt')
        torch.save({
            'iteration': iteration,
            'nca_state_dict': self.nca.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg,
        }, path)

    @torch.no_grad()
    def export_textured_ply(self, path, n_steps=None):
        """Grow the texture from seed and bake the resulting SH into a .ply."""
        self.nca.eval()
        n_steps = n_steps or self.cfg['max_steps']
        state = self.nca.seed(batch_size=1, device=self.device)
        state = self.nca(state, n_steps)
        f_dc, f_rest = self.nca.decode_dc_rest(state)

        if self.nca.sh_mode == 'dc':
            # DC-only automaton: zero out the higher bands of the original scene
            f_rest = torch.zeros_like(self.gaussians.f_rest)

        self.gaussians.save_ply(path, f_dc=f_dc, f_rest=f_rest)
        tqdm.write(f"Saved textured scene to {path}")
