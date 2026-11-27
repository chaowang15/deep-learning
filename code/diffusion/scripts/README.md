# scripts/

Command-line entry points. Notebooks are for learning; scripts are for running the same code end to end.

| Script | Purpose |
|---|---|
| `check_env.py` | Check torch install and device (CUDA / MPS / CPU) |
| `train.py` (later, Stage 03) | Train DDPM from a config: `python scripts/train.py --config configs/fashion_mnist_ddpm.yaml` |
| `sample.py` (later, Stage 04) | Generate samples from a checkpoint |
