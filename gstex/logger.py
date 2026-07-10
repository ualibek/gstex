"""
Training telemetry, collected in one place: the tqdm progress bar, the
per-iteration CSV log, eval-probe messages, and periodic render snapshots.
"""

import os

from tqdm import tqdm


class TrainLogger:
    def __init__(self, out_dir, n_iterations, render_interval, eval_interval):
        self.out_dir = out_dir
        self.n_iterations = n_iterations
        self.render_interval = render_interval
        self.eval_interval = eval_interval
        self.progress = None
        self.eval_history = []                  # (iteration, eval_loss) pairs

        self.renders_dir = os.path.join(out_dir, 'renders')
        os.makedirs(self.renders_dir, exist_ok=True)

        self.loss_csv = os.path.join(out_dir, 'loss_log.csv')
        with open(self.loss_csv, 'w') as f:
            f.write('iteration,loss,style_loss,overflow,n_steps,eval_loss\n')

    def iterations(self):
        """Iterate 1..n_iterations while driving the progress bar."""
        self.progress = tqdm(range(1, self.n_iterations + 1),
                             desc='train', dynamic_ncols=True)
        return self.progress

    def render_due(self, iteration):
        """Whether a summary render should be produced/saved this iteration."""
        return iteration % self.render_interval == 0

    def eval_due(self, iteration):
        """Whether the deterministic eval probe should run this iteration."""
        return iteration % self.eval_interval == 0

    def log_iteration(self, iteration, log, summary=None, eval_loss=None):
        """
        Log one training iteration: progress-bar postfix and a CSV row always;
        an eval message when eval_loss is given, a render snapshot when a
        summary is given.
        """
        if eval_loss is not None:
            self.eval_history.append((iteration, eval_loss))
            self.message(f"[eval @ {iteration:6d}] deterministic style loss: {eval_loss:.4f}")

        self.progress.set_postfix(
            loss=f"{log['loss']:.4f}",
            style=f"{log['style_loss']:.4f}",
            overflow=f"{log['overflow']:.4f}",
            steps=log['n_steps'],
        )
        with open(self.loss_csv, 'a') as f:
            f.write(f"{iteration},{log['loss']:.6f},{log['style_loss']:.6f},"
                    f"{log['overflow']:.6f},{log['n_steps']},"
                    f"{'' if eval_loss is None else f'{eval_loss:.6f}'}\n")

        if summary is not None:
            self.save_render(iteration, summary['images'])

    def save_render(self, iteration, image):
        """Save a PIL summary image for this iteration."""
        image.save(os.path.join(self.renders_dir, f'iter_{iteration:06d}.png'))

    def plot_eval_loss(self):
        """Plot the deterministic eval-loss curve to <out_dir>/eval_loss.png."""
        if not self.eval_history:
            self.message("No eval probes were recorded; skipping eval-loss plot.")
            return None

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        iterations, losses = zip(*self.eval_history)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(iterations, losses, color='#2a78d6', linewidth=1.8,
                marker='o' if len(iterations) <= 60 else None, markersize=3.5)
        ax.set_xlabel('iteration')
        ax.set_ylabel('deterministic style loss')
        ax.set_title('Eval loss (fixed seed, cameras and rollout)')
        ax.set_xlim(left=0)
        ax.grid(color='#e3e3e3', linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        path = os.path.join(self.out_dir, 'eval_loss.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.message(f"Saved eval-loss plot to {path}")
        return path

    @staticmethod
    def message(text):
        """Print a line without corrupting the progress bar."""
        tqdm.write(text)
