# VAE / VQ-VAE experiments

Experiments for the blog post `generative-models-1-vae`. FashionMNIST from `code/data/FashionMNIST/raw`, CPU only.

```bash
cd code/vae
python3 vae_exp.py      # MLP VAE: BCE(sum)+KL vs MSE(mean)+KL, plus d=2 VAE   (~2 min)
python3 vae_figs.py     # vae_recon_samples.png, vae_latent2d.png
python3 vae_sigma.py    # Gaussian decoder, sigma^2 sweep                       (~4 min)
python3 vqvae_exp.py    # VQ-VAE (K=64, D=16, 7x7) + tiny Transformer prior      (~16 min)
python3 vq_figs.py      # vq_recon.png, vq_samples.png
```

Checkpoints go to `/tmp`; figures and JSON results go to `assets/generative-models-1-vae/`.
