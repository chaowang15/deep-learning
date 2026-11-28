# DiT experiments

Experiments for the blog post `dit`. CPU only. Same flow matching loss as the `flow-matching` post
(t=0 data, t=1 noise, target eps - x0; t is multiplied by 1000 before the timestep embedding).

```bash
cd code/dit
python3 check_paper.py                                  # params + GMACs of all 12 paper configs (DiT-S/B/L/XL x p=2/4/8)
python3 train_dit.py adaln_zero 2500 main               # p=4, D=192, L=6, 3 heads (4.2M), ~23 min
for m in adaln in_context cross_attn; do python3 train_dit.py $m 400 abl_$m; done   # block ablation, ~3 min each
python3 train_dit.py adaln_zero 2500 p2 2 128 6 4       # p=2, D=128, L=6, 4 heads (1.9M), ~60 min
RESUME=/tmp/dit_main.pt python3 train_dit.py adaln_zero 5000 main_cont   # continue p=4 to 7500 steps, ~41 min
python3 classifier.py                                   # small FashionMNIST CNN used for scoring (test acc 89.3%)
python3 eval_figs.py                                    # U-Net vs DiT p=4 vs p=2: class accuracy, feature FD; figures
python3 eval_longer.py                                  # per-t velocity error, DiT at 7500 steps, dit_longer.png
python3 train_unet_cont.py 5000                         # continue the flow post's U-Net to 7500 steps, same protocol, ~39 min
python3 eval_7500.py                                    # same-step comparison: dit_vs_unet_7500.png, dit_curves_7500.png
```

`dit.py` is the model (with a `cond_mode` switch for the four block designs). The U-Net baseline is
`/tmp/flow_fmnist.pt` from `code/flow/fmnist_fm.py`. Checkpoints go to `/tmp`; figures and JSON go to `assets/dit/`.
