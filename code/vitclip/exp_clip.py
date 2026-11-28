"""Experiment 3: a tiny CLIP on FashionMNIST with synthetic captions.

Each training image gets a random caption: template x class synonym, plus with
prob. 0.5 a "dark"/"light" adjective from the garment's brightness (above or
below its class median). Two towers (tiny ViT, tiny causal text Transformer)
are trained only with the symmetric contrastive loss: no class labels are used.
Then: zero-shot classification via prompts, prompt ensembling, a linear probe
on the frozen image features, and text-to-image retrieval.

usage: python3 exp_clip.py [steps] [batch] [wd] [decay_all]     (~4 min on 4 CPU cores)
writes /tmp/clip_tiny.pt and assets/vit-clip/clip_results.json
"""
import json, math, random, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "vae"))
from common import load_fmnist  # noqa: E402
from clip import CLIP, ImageEncoder, TextEncoder, clip_loss  # noqa: E402

CLASSES = ["t-shirt", "trousers", "pullover", "dress", "coat",
           "sandal", "shirt", "sneaker", "bag", "ankle boot"]
SYNONYMS = [["t-shirt", "tee", "top"], ["trousers", "pants", "jeans"], ["pullover", "sweater", "jumper"],
            ["dress", "gown"], ["coat", "jacket"], ["sandal", "flip flop"], ["shirt", "blouse"],
            ["sneaker", "trainer", "running shoe"], ["bag", "handbag", "purse"], ["ankle boot", "boot"]]
TRAIN_TPL = ["a photo of a {}", "a picture of a {}", "an image of a {}", "a {}",
             "this is a {}", "a product photo of a {}"]
CTX = 12


def build_vocab():
    words = set()
    for t in TRAIN_TPL:
        words |= set(t.replace("{}", "").split())
    for syn in SYNONYMS:
        for s in syn:
            words |= set(s.split())
    words |= {"dark", "light", "the", "pair", "of"}
    vocab = ["<pad>", "<unk>"] + sorted(words) + ["<sot>", "<eot>"]   # <eot> gets the largest id
    return {w: i for i, w in enumerate(vocab)}


VOCAB = build_vocab()


def tokenize(texts):
    ids = torch.zeros(len(texts), CTX, dtype=torch.long)
    for i, t in enumerate(texts):
        toks = [VOCAB["<sot>"]] + [VOCAB.get(w, VOCAB["<unk>"]) for w in t.split()][:CTX - 2] + [VOCAB["<eot>"]]
        ids[i, :len(toks)] = torch.tensor(toks)
    return ids


def brightness(x):   # mean of garment pixels (x in [0,1])
    m = (x > 0.05).float()
    return (x * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1)


def make_dark_flags(x, y):
    b = brightness(x)
    med = torch.stack([b[y == c].median() for c in range(10)])
    return b < med[y]


def caption(label, dark):
    noun = random.choice(SYNONYMS[label])
    if random.random() < 0.5:
        noun = ("dark " if dark else "light ") + noun
    tpl = random.choice(TRAIN_TPL)
    if noun[0] in "aeiou" and tpl.endswith("a {}"):
        tpl = tpl[:-4] + "an {}"
    return tpl.format(noun)


def tiny_clip(init_temp=0.07):
    img = ImageEncoder(img=28, patch=4, in_ch=1, dim=64, depth=4, heads=4, embed=64)
    txt = TextEncoder(vocab=len(VOCAB), ctx=CTX, dim=64, depth=2, heads=4, embed=64)
    return CLIP(img, txt, init_temp)


@torch.no_grad()
def encode_images(m, x):
    m.eval()
    return torch.cat([F.normalize(m.image(x[i:i + 1000]), dim=-1) for i in range(0, len(x), 1000)])


@torch.no_grad()
def encode_texts(m, texts):
    m.eval()
    return F.normalize(m.text(tokenize(texts)), dim=-1)


def an(t):
    return t.replace(" a a", " an a").replace(" a e", " an e").replace(" a i", " an i").replace(" a o", " an o")


@torch.no_grad()
def zero_shot(m, xt, yt, prompts_per_class):
    """prompts_per_class: list (10) of lists of strings. Class embedding = mean of
    its normalised prompt embeddings, renormalised (prompt ensembling)."""
    u = encode_images(m, xt)
    W = torch.stack([F.normalize(encode_texts(m, ps).mean(0), dim=-1) for ps in prompts_per_class])
    return ((u @ W.t()).argmax(1) == yt).float().mean().item()


def linear_probe(ftr, ytr, fte, yte, steps=500):
    clf = nn.Linear(ftr.shape[1], 10)
    opt = torch.optim.Adam(clf.parameters(), 1e-2)
    for _ in range(steps):
        loss = F.cross_entropy(clf(ftr), ytr); opt.zero_grad(); loss.backward(); opt.step()
    return (clf(fte).argmax(1) == yte).float().mean().item()


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    random.seed(0); torch.manual_seed(0)
    x01, y = load_fmnist("train"); xt01, yt = load_fmnist("test")
    dark, dark_t = make_dark_flags(x01, y), make_dark_flags(xt01, yt)
    x, xt = x01 * 2 - 1, xt01 * 2 - 1
    m = tiny_clip()
    # no weight decay on logit_scale, biases and norm gains (as in OpenCLIP); decaying
    # logit_scale pulls 1/tau back toward 1 and the temperature barely moves
    decay = [p for n, p in m.named_parameters() if p.ndim >= 2 and "pos" not in n]
    no_decay = [p for n, p in m.named_parameters() if not (p.ndim >= 2 and "pos" not in n)]
    wd = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    if len(sys.argv) > 4 and sys.argv[4] == "decay_all":       # ablation: the naive setting
        groups = [{"params": list(m.parameters()), "weight_decay": wd}]
    opt = torch.optim.AdamW(groups, 1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) *
                                              0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    log, t0 = [], time.time()
    for s in range(steps):
        m.train()
        j = torch.randint(0, len(x), (bs,))
        ids = tokenize([caption(int(y[k]), bool(dark[k])) for k in j])
        loss = clip_loss(m(x[j], ids))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        # in-batch duplicates: other images whose class equals this one's (false negatives)
        dup = (y[j][:, None] == y[j][None]).float().sum(1).mean().item()
        if s % 25 == 0 or s == steps - 1:
            log.append({"step": s, "loss": loss.item(), "scale": m.logit_scale.exp().item(), "same_class": dup})
        if s % 250 == 0:
            print(s, round(loss.item(), 3), "1/tau", round(m.logit_scale.exp().item(), 2), round(time.time() - t0), "s", flush=True)
    if "decay_all" not in sys.argv:
        torch.save(m.state_dict(), "/tmp/clip_tiny.pt")

    # zero-shot classification on the 10k test set, no labels were used in training
    single = [[an(f"a photo of a {c}")] for c in CLASSES]
    ens = [[an(t.format(s)) for t in TRAIN_TPL for s in syn] for syn in SYNONYMS]
    unseen = [[an(f"a photo of the {c}")] for c in CLASSES]       # template never seen in training
    res = {"steps": steps, "batch": bs, "sec": round(time.time() - t0), "log": log,
           "zs_single": zero_shot(m, xt, yt, single), "zs_ensemble": zero_shot(m, xt, yt, ens),
           "zs_unseen_template": zero_shot(m, xt, yt, unseen)}
    # linear probe on frozen image features (uses labels; an upper reference)
    ftr, fte = encode_images(m, x[:10000]), encode_images(m, xt)
    res["linear_probe"] = linear_probe(ftr, y[:10000], fte, yt)
    # dark/light: among test images of each class, does "dark X" vs "light X" pick the right half?
    hits = []
    for c in range(10):
        idx = (yt == c).nonzero().squeeze(1)
        u = encode_images(m, xt[idx])
        w = encode_texts(m, [an(f"a photo of a dark {CLASSES[c]}"), an(f"a photo of a light {CLASSES[c]}")])
        pred_dark = (u @ w.t()).argmax(1) == 0
        hits.append((pred_dark == dark_t[idx]).float().mean().item())
    res["dark_light_acc_per_class"] = hits
    res["dark_light_acc"] = sum(hits) / 10
    res["final_scale"] = m.logit_scale.exp().item()
    res["avg_same_class_in_batch"] = sum(r["same_class"] for r in log) / len(log)
    print({k: v for k, v in res.items() if k != "log"})
    out = ROOT / "assets" / "vit-clip"; out.mkdir(parents=True, exist_ok=True)
    (out / ("clip_results_decay_all.json" if "decay_all" in sys.argv else "clip_results.json")).write_text(json.dumps(res, indent=1))
