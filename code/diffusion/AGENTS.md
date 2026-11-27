# Agent Notes for Diffusion Learning

Project-specific working context for coding agents (Codex, Claude Code).

## Project Goal

Learn DDPM-style diffusion from scratch, step by step. The goal is clarity and understanding, not maximum image quality.

The current implementation focuses on:

1. dataset loading and visualization
2. noise schedule
3. forward noising process
4. timestep embedding
5. small U-Net noise predictor
6. DDPM training loss
7. sampling / reverse process

## Location and Environment

This code lives in the blog repo `chaowang15/deep-learning` under `code/diffusion/` (imported with full git history from the old `deep_learning_lab` repo on 2026-11-27; develop here from now on). Jekyll excludes `code/`.

Main machine: Mac (Apple Silicon). Use `requirements.txt` (PyPI torch, MPS backend). The Windows desktop (RTX 3080) uses `requirements-cuda121.txt`.

```bash
cd code/diffusion
conda activate diffusion
python scripts/check_env.py   # cuda / mps / cpu
python -m pytest -q
```

## Data

Datasets live at `code/data/` (configs use `../data` relative to `code/diffusion/`). `data/` is ignored by Git. Only small sample runs happen here; large-scale training goes to a separate repo.

## Stage numbering

Each stage = one notebook + its `src/` modules; the first docstring line of every `src/*.py` names its stage. Keep the table in README.md in sync when a stage is finished.

## Current Progress

The project has completed the setup and forward-process learning phase.

Implemented:

- `src/datasets.py`
  - Supports MNIST, FashionMNIST, and CIFAR-10.
  - Normalizes images from `[0, 1]` to `[-1, 1]`.
- `src/schedule.py`
  - Builds linear beta schedule.
  - Computes `alphas`, `alpha_bars`, `sqrt_alpha_bars`, `sqrt_one_minus_alpha_bars`, `sqrt_recip_alphas`, and `posterior_variance`.
- `src/forward_process.py`
  - Implements `q_sample(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars, noise=None)`.
  - Handles `[B, C, H, W]` images and batch timesteps `[B]`.
- `notebooks/00_check_dataset.ipynb`
  - Checks MNIST and CIFAR-10 shapes, ranges, and image grids.
- `notebooks/01_forward_process.ipynb`
  - Visualizes schedule curves.
  - Visualizes forward noising for MNIST and CIFAR-10 with three example images each.

Validated:

- MNIST dataset loading
- CIFAR-10 dataset loading
- schedule visualization
- `q_sample` / forward noising visualization for MNIST and CIFAR-10

Important current concepts already covered:

- Image tensors are `[B, C, H, W]`; grayscale MNIST still has `C=1`.
- `repeat_interleave(..., dim=0)` was used in the notebook to show every image at every selected timestep.
- Notebook outputs should be cleared before commit, but the user has run and visually checked the outputs locally.

Next recommended step:

```text
src/time_embedding.py
```

After that:

```text
src/unet.py
notebooks/02_noise_prediction.ipynb
src/diffusion.py
```

Recommended continuation order:

1. Implement sinusoidal timestep embeddings in `src/time_embedding.py`.
2. Add a small visualization or sanity check for embedding shape and values.
3. Implement a minimal U-Net/noise-prediction network in `src/unet.py`.
4. Use `notebooks/02_noise_prediction.ipynb` to train on a tiny MNIST/FashionMNIST batch and verify loss decreases.
5. Move the loss wiring into `src/diffusion.py`.
6. Only after noise prediction works, implement reverse sampling / `p_sample`.

Do not jump directly to `p_sample`: it depends on a trained noise predictor.

## Handoff Checklist for a New Machine

See README.md "环境". Then run `python -m pytest -q` and re-run notebooks 00 and 01.

## Coding Style

This is a learning project. Prefer simple, explicit code over clever abstractions.

For core diffusion math:

- Keep formulas close to the implementation.
- Add comments explaining why each term is needed.
- If the user adds Chinese comments, keep the Chinese meaning. It is OK to shorten them and add a concise English explanation.
- Existing English comments do not need to be translated unless it improves learning clarity.

For notebooks:

- It is OK for Codex to add visualization and explanation code.
- Clear notebook outputs before committing.
- Keep notebooks as walkthroughs; reusable implementation should live in `src/`.

For source modules:

- The user prefers to implement core modules manually.
- Codex can review, lightly refactor, and add robustness, but should avoid jumping ahead with full implementations unless asked.

## Git Hygiene

Before committing:

```bash
git status --short
```

Do not commit:

- `data/`
- Conda/venv folders
- `__pycache__/`
- notebook execution outputs
- model checkpoints or generated samples

Useful checks:

```bash
python -m pytest -q
```

Notebook JSON can be checked with:

```bash
python -m json.tool notebooks/01_forward_process.ipynb > /dev/null
```
