"""Stage 03 (notebooks/03_ddpm_training.ipynb): training loop over epochs and batches."""

from src.diffusion import ddpm_training_step


def train_ddpm(model, optimizer, dataloader, sched, num_epochs, device,
               conditional=False, log_every=200):
    """Loop: for each epoch, for each batch -> one ddpm_training_step. Returns per-step losses."""
    losses, step = [], 0
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            x0 = images.to(device)                            # [B, C, H, W] in [-1, 1]
            y = labels.to(device) if conditional else None
            loss = ddpm_training_step(model, optimizer, x0, sched, labels=y)
            losses.append(loss.item())
            if step % log_every == 0:
                print(f"epoch {epoch + 1}/{num_epochs}  step {step}  loss {loss.item():.4f}")
            step += 1
    return losses
