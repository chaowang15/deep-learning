# Flow matching experiments

Experiments for the blog post `flow-matching`. CPU only. Convention: t=0 data, t=1 noise,
x_t = (1-t) x0 + t eps, target velocity eps - x0 (same as SD3 and the DDPM/DDIM posts).

```bash
cd code/flow
python3 toy_exp.py         # 2-D 8 Gaussians: 1-Rectified Flow, Reflow -> 2-RF, straightness and few-step quality (~1 min)
python3 toy_figs.py        # toy_paths.png, toy_steps.png
python3 fmnist_fm.py 2500  # class-conditional flow matching on FashionMNIST, DDPM post's Tiny U-Net (~19 min)
python3 fmnist_figs.py     # fmnist_steps.png, fmnist_solvers.png, fmnist_curve.png, Euler == DDIM check
```

Uses `code/diffusion/src/unet.py` (ClassConditionedTinyUNet) and `code/vae/common.py` (data loader).
Checkpoints go to `/tmp`; figures and JSON results go to `assets/flow-matching/`.
