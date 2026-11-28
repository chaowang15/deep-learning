"""VQ-VAE on FashionMNIST (7x7 tokens) + a tiny Transformer prior over the tokens."""
import json, time, math, torch, torch.nn as nn, torch.nn.functional as F
from common import load_fmnist, OUT

torch.manual_seed(0)
xtr, ytr = load_fmnist("train"); xte, yte = load_fmnist("test")


class VectorQuantizer(nn.Module):
    def __init__(self, K=64, D=16, beta=0.25):
        super().__init__()
        self.K, self.D, self.beta = K, D, beta
        self.codebook = nn.Embedding(K, D)
        self.codebook.weight.data.uniform_(-1 / K, 1 / K)

    def forward(self, z_e):                                   # z_e: [B, D, H, W]
        B, D, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).reshape(-1, D)            # [BHW, D]
        e = self.codebook.weight                              # [K, D]
        dist = z.pow(2).sum(1, keepdim=True) - 2 * z @ e.T + e.pow(2).sum(1)   # ||z - e_k||^2, [BHW, K]
        idx = dist.argmin(1)                                  # k = argmin_k ||z_e - e_k||
        z_q = e[idx].view(B, H, W, D).permute(0, 3, 1, 2)     # [B, D, H, W]
        codebook_loss = F.mse_loss(z_q, z_e.detach())         # ||sg[z_e] - e||^2
        commit_loss = F.mse_loss(z_e, z_q.detach())           # ||z_e - sg[e]||^2
        z_q_st = z_e + (z_q - z_e).detach()                   # straight-through: forward z_q, backward identity
        return z_q_st, codebook_loss + self.beta * commit_loss, idx.view(B, H, W)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x + self.net(x)


class VQVAE(nn.Module):
    def __init__(self, K=64, D=16, ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.ReLU(),             # 28 -> 14
            nn.Conv2d(ch, ch, 4, 2, 1),                       # 14 -> 7
            ResBlock(ch), ResBlock(ch), nn.ReLU(),
            nn.Conv2d(ch, D, 1))                              # [B, D, 7, 7]
        self.vq = VectorQuantizer(K, D)
        self.decoder = nn.Sequential(
            nn.Conv2d(D, ch, 3, padding=1),
            ResBlock(ch), ResBlock(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.ReLU(),   # 7 -> 14
            nn.ConvTranspose2d(ch, 1, 4, 2, 1))               # 14 -> 28, logits

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, idx = self.vq(z_e)
        return self.decoder(z_q), vq_loss, idx

    def decode_indices(self, idx):                            # idx: [B, 7, 7]
        z_q = self.vq.codebook(idx).permute(0, 3, 1, 2)
        return torch.sigmoid(self.decoder(z_q))


class TinyGPT(nn.Module):
    """Autoregressive prior p(k_1..k_49) over flattened 7x7 token maps."""
    def __init__(self, K=64, L=49, d=128, n_layer=3, n_head=4):
        super().__init__()
        self.K, self.L = K, L
        self.tok = nn.Embedding(K + 1, d)                     # id K = <BOS>
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        layer = nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d, K)
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, seq):                                   # seq: [B, L] tokens, predict seq from shifted input
        inp = torch.cat([torch.full_like(seq[:, :1], self.K), seq[:, :-1]], 1)
        h = self.tok(inp) + self.pos
        return self.head(self.blocks(h, mask=self.mask))      # [B, L, K]

    @torch.no_grad()
    def sample(self, n, temperature=1.0):
        seq = torch.zeros(n, self.L, dtype=torch.long)
        for i in range(self.L):
            logits = self(seq)[:, i] / temperature
            seq[:, i] = torch.multinomial(logits.softmax(-1), 1).squeeze(1)
        return seq


def perplexity(idx, K):
    p = torch.bincount(idx.flatten(), minlength=K).float(); p /= p.sum()
    return math.exp(-(p * (p + 1e-10).log()).sum().item())


if __name__ == "__main__":
    t0 = time.time(); res = {"vq": [], "prior": []}
    K = 64
    m = VQVAE(K=K); opt = torch.optim.Adam(m.parameters(), 2e-4 * 5)
    bs = 128
    for ep in range(8):
        m.train(); perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            x = xtr[perm[i:i + bs]]
            logits, vq_loss, _ = m(x)
            rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
            loss = rec + 784 * vq_loss                        # vq_loss is a per-element mean; put it on a per-image scale
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            logits, vq_loss, idx = m(xte)
            bce = F.binary_cross_entropy_with_logits(logits, xte, reduction="sum").item() / len(xte)
            mse = F.mse_loss(torch.sigmoid(logits), xte).item()
        r = dict(epoch=ep + 1, bce=round(bce, 2), mse=round(mse, 5), used=int(idx.unique().numel()), perplexity=round(perplexity(idx, K), 1))
        res["vq"].append(r); print("vq", r, flush=True)
    torch.save(m.state_dict(), "/tmp/vqvae.pt")

    with torch.no_grad():
        tok_tr = torch.cat([m(xtr[i:i + 1000])[2] for i in range(0, len(xtr), 1000)]).view(len(xtr), -1)
        tok_te = m(xte)[2].view(len(xte), -1)
    torch.save(tok_tr, "/tmp/vq_tokens_train.pt")
    prior = TinyGPT(K=K); popt = torch.optim.AdamW(prior.parameters(), 1e-3, weight_decay=0.01)
    for ep in range(6):
        prior.train(); perm = torch.randperm(len(tok_tr))
        for i in range(0, len(tok_tr), 256):
            seq = tok_tr[perm[i:i + 256]]
            loss = F.cross_entropy(prior(seq).reshape(-1, K), seq.reshape(-1))
            popt.zero_grad(); loss.backward(); popt.step()
        prior.eval()
        with torch.no_grad():
            nll = F.cross_entropy(prior(tok_te).reshape(-1, K), tok_te.reshape(-1)).item()
        r = dict(epoch=ep + 1, nll_per_token=round(nll, 3), bits_per_image=round(nll * 49 / math.log(2), 1))
        res["prior"].append(r); print("prior", r, flush=True)
    torch.save(prior.state_dict(), "/tmp/vq_prior.pt")
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(OUT / "vqvae_results.json", "w"), indent=1)
    print("done", res["seconds"])
