# ViT + CLIP experiments

Code for the post [ViT 与 CLIP](https://chaowang15.github.io/deep-learning/vit-clip/). Pure PyTorch, CPU, FashionMNIST (loaded via `code/vae/common.py`).

| File | What it does |
|---|---|
| `vit.py` | ViT (patch embed, [CLS], learned pos-embed, Pre-LN blocks), torchvision weight loader, `interpolate_pos`, configs |
| `check_vit.py` | copies torchvision `vit_b_16` weights into `vit.py`, checks max output diff (0.0), prints config param counts |
| `clip.py` | CLIP image tower (ViT, ln_pre/ln_post/proj, QuickGELU), text tower (causal mask, [EOT] feature), learned logit scale, symmetric InfoNCE, OpenAI configs |
| `check_clip.py` | param counts of ViT-B/32, B/16, L/14, L/14@336 (151.3M / 149.6M / 427.6M / 427.9M); loss ≈ log B at init |
| `exp_data_size.py` | tiny ViT vs small CNN on 1k / 6k / 60k images, 2000 steps each (~10 min total) |
| `exp_clip.py` | tiny CLIP on synthetic captions; zero-shot, prompt ensembling, linear probe, dark/light test (~4 min) |
| `figs.py` | all figures in `assets/vit-clip/` |

```bash
python3 check_vit.py && python3 check_clip.py
python3 exp_data_size.py 2000          # -> assets/vit-clip/data_size.json, /tmp/vit_60000.pt
python3 exp_clip.py 1500 256           # -> assets/vit-clip/clip_results.json, /tmp/clip_tiny.pt
python3 exp_clip.py 1500 256 0.05 decay_all   # ablation: weight decay on logit_scale too
python3 figs.py
```
