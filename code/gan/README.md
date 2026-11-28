# GAN experiments

Experiments for the blog post `generative-models-2-gan`. CPU only.

```bash
cd code/gan
python3 toy_exp.py      # 2-D ring of 8 Gaussians: saturating vs non-saturating vs WGAN-GP, 5 D steps each (~3 min)
python3 mnist_gan.py    # FashionMNIST MLP GAN and DCGAN, 10 epochs each (~9 min)
python3 gan_figs.py     # fmnist_samples.png, fmnist_curves.png
python3 vqgan.py 40     # run check of the VQGAN training step (needs /tmp/vqvae.pt from code/vae/vqvae_exp.py)
```

Figures and logs go to `assets/generative-models-2-gan/`.
