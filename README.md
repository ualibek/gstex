# GSTex — Texturizing 3D Gaussian Splatting with Neural Cellular Automata

Port of the MeshNCA idea to 3DGS: cells are Gaussian splats, the neighborhood is a
KNN graph over splat centers, and an NCA repeatedly perceives each splat's neighbors
and emits a state delta that **only edits the SH color coefficients** — geometry
(position, scale, rotation, opacity) is frozen. Training is end-to-end: render the
splats with gsplat from random viewpoints and match a target texture image with the
MeshNCA VGG / optimal-transport style loss.

## Architecture

| Module | Role |
|---|---|
| `gstex/gaussian_model.py` | 3DGS `.ply` load/save (INRIA format), storage→rasterizer activations, SH assembly |
| `gstex/graph.py` | KNN graph + fixed per-edge real-SH direction basis (computed once, geometry frozen) |
| `gstex/perception.py` | Learnable graph-conv perception: filter bank over the edge basis + identity |
| `gstex/nca.py` | `GaussianNCA` — state `[geometry(11, read-only) \| SH \| hidden]`, zero-init update MLP, stochastic per-splat update mask (fire rate) |
| `gstex/render.py` | gsplat rasterization wrapper + random spherical viewpoint sampling |
| `gstex/trainer.py` | Sample-pool persistence training; only NCA weights are optimized |
| `gstex/logger.py` | `TrainLogger` — all training telemetry: progress bar, per-iteration CSV log, eval messages, render snapshots (owns `--render_interval` / `--eval_interval` schedule) |
| `gstex/appearance_loss.py` | Reused from MeshNCA (VGG-16 + relaxed-EMD optimal transport style loss) |

State SH scope is configurable: `--sh_mode dc` (view-independent color, render SH degree 0)
or `--sh_mode full` (all SH bands of the scene, view-dependent texture).

## Setup

```bash
pip install torch torchvision   # matching your CUDA version
pip install -r requirements.txt
```

## Usage

```bash
# 0. Smoke tests (do this first — verifies IO, camera conventions, graph, NCA invariants)
python scripts/smoke_test.py --scene data/scenes/<scene>.ply

# 1. Train (start with dc mode)
python train.py --scene data/scenes/<scene>.ply --texture data/textures/<tex>.jpg \
    --sh_mode dc --out_dir out/<run>

# Outputs: out/<run>/renders/  (progress grids: target vs. rendered, every --render_interval)
#          out/<run>/loss_log.csv  (per-iteration losses + periodic eval-probe loss)
#          out/<run>/checkpoints/
#          out/<run>/textured.ply  (open in any 3DGS viewer)
```

```bash
# 2. Apply the trained NCA to another (unseen) scene — outputs textured.ply + renders
python scripts/apply_nca.py --checkpoint out/<run>/checkpoints/nca_<iter>.pt \
    --scene data/scenes/<other>/scene.ply

# 3. Watch the texture grow: live web viewer (orbit while the NCA evolves) ...
python scripts/view_nca.py --checkpoint out/<run>/checkpoints/nca_<iter>.pt \
    --scene data/scenes/<other>/scene.ply          # http://localhost:8080

# ... or render the evolution as an orbiting-camera video
python scripts/view_nca.py --checkpoint ... --scene ... --video --n_steps 500
```

Gradient checkpointing of NCA steps is **on by default** (required at real scene scales;
`--no-checkpoint_steps` to disable for small scenes). Further memory knobs: smaller
`--batch_size` / `--max_steps` / `--n_filters` / `--knn_k`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` against fragmentation.
