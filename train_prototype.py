"""Stage 2 - train a purely visual classifier on cleaned contest prototypes.

Does NOT use class names / all_class_predictions.json.  It loads the pre-computed
visual features, the clean mask from class_prototype.py, and trains a cosine
classifier whose class embeddings are initialised with the discovered prototypes.

Training tricks:
  * GCE loss (q=0.7) for robustness
  * CE loss on the highest-confidence subset for stronger gradients
  * EMA teacher model for stability
  * Class-balanced sampling so small clean classes are not ignored
"""

import os
import json
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def gce_loss(logits, targets, q=0.7):
    """Generalized Cross Entropy (Zhang & Sabuncu, 2018)."""
    probs = F.softmax(logits, dim=1)
    probs = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
    probs = torch.clamp(probs, min=1e-7)
    if q == 0.0:
        loss = -torch.log(probs)
    else:
        loss = (1.0 - probs ** q) / q
    return loss.mean()


class VisualClassifier(nn.Module):
    """Cosine classifier on top of frozen CLIP features."""

    def __init__(self, n_classes=500, dim=512, temp=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, dim))
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.temp = temp
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.weight)
        self.weight.data = F.normalize(self.weight.data, dim=1)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        logits = x @ w.T
        logits = logits / self.temp + self.bias
        return logits


class FeatureDataset(Dataset):
    def __init__(self, features, labels, confidences=None):
        self.features = features
        self.labels = labels
        self.confidences = confidences if confidences is not None else torch.ones(len(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.features[i], self.labels[i], self.confidences[i]


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        # no-op: caller is expected to reload from checkpoint if needed
        pass


def build_balanced_sampler(labels):
    """Sample each class with inverse frequency."""
    counts = torch.bincount(labels)
    weights = 1.0 / (counts[labels].float() + 1e-8)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def train_epoch(model, loader, optimizer, device, q=0.7, ce_frac=0.3, ce_conf_thr=0.8):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for feats, labs, confs in loader:
        feats = feats.to(device)
        labs = labs.to(device)
        confs = confs.to(device)

        logits = model(feats)

        # GCE on all clean samples
        loss = gce_loss(logits, labs, q=q)

        # extra CE on high-confidence subset for stronger gradient
        if ce_frac > 0:
            high_conf = confs >= ce_conf_thr
            if high_conf.sum() > 0:
                ce = F.cross_entropy(logits[high_conf], labs[high_conf])
                loss = loss + ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def main():
    p = argparse.ArgumentParser(description="Train visual classifier on cleaned prototypes")
    p.add_argument("--prototype-dir", default="output/contest_prototype")
    p.add_argument("--output-dir", default="output/contest_train")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--q", type=float, default=0.7, help="GCE q parameter")
    p.add_argument("--ce-frac", type=float, default=0.3, help="CE loss weight on high-conf samples")
    p.add_argument("--ce-conf-thr", type=float, default=0.85, help="confidence threshold for CE loss")
    p.add_argument("--temp", type=float, default=0.01, help="cosine classifier temperature")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # ---- load stage 1 artifacts ----
    features = torch.load(os.path.join(args.prototype_dir, "features.pt"), weights_only=True)
    labels = torch.load(os.path.join(args.prototype_dir, "labels.pt"), weights_only=True)
    clean_mask = torch.load(os.path.join(args.prototype_dir, "clean_mask.pt"), weights_only=True)
    confidences = torch.load(os.path.join(args.prototype_dir, "sample_confidence.pt"), weights_only=True)
    prototypes = torch.load(os.path.join(args.prototype_dir, "prototypes.pt"), weights_only=True)

    n_classes = int(labels.max().item()) + 1
    dim = features.size(1)
    print(f"[train] loaded features {features.shape}, classes {n_classes}")
    print(f"[train] clean samples: {int(clean_mask.sum().item())}/{len(clean_mask)}")

    # ---- keep only clean samples for stage 2 ----
    train_features = features[clean_mask]
    train_labels = labels[clean_mask]
    train_conf = confidences[clean_mask]

    ds = FeatureDataset(train_features, train_labels, train_conf)
    sampler = build_balanced_sampler(train_labels)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)

    # ---- model ----
    model = VisualClassifier(n_classes=n_classes, dim=dim, temp=args.temp)
    # initialise with discovered prototypes
    if prototypes.shape == model.weight.shape:
        model.weight.data = F.normalize(prototypes.float(), dim=1)
    model = model.to(device)

    ema = EMA(model, decay=args.ema_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- training loop ----
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, loader, optimizer, device, q=args.q,
                           ce_frac=args.ce_frac, ce_conf_thr=args.ce_conf_thr)
        ema.update(model)
        scheduler.step()

        print(f"[train] epoch {epoch}/{args.epochs}  loss {loss:.4f}  "
              f"lr {scheduler.get_last_lr()[0]:.6f}  time {time.time()-t0:.1f}s", flush=True)

        if loss < best_loss:
            best_loss = loss
            ckpt = {
                "weight": F.normalize(model.weight.data, dim=1).cpu(),
                "bias": model.bias.data.cpu(),
                "temp": args.temp,
                "epoch": epoch,
                "loss": loss,
                "args": vars(args),
                "ema": {k: v.cpu() for k, v in ema.shadow.items()},
            }
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))

        if epoch % args.save_every == 0:
            ckpt = {
                "weight": F.normalize(model.weight.data, dim=1).cpu(),
                "bias": model.bias.data.cpu(),
                "temp": args.temp,
                "epoch": epoch,
                "loss": loss,
                "args": vars(args),
                "ema": {k: v.cpu() for k, v in ema.shadow.items()},
            }
            torch.save(ckpt, os.path.join(args.output_dir, f"epoch_{epoch}.pt"))

    # ---- save final EMA checkpoint ----
    ema.apply_shadow(model)
    ckpt = {
        "weight": F.normalize(model.weight.data, dim=1).cpu(),
        "bias": model.bias.data.cpu(),
        "temp": args.temp,
        "epoch": args.epochs,
        "loss": loss,
        "args": vars(args),
        "ema": {k: v.cpu() for k, v in ema.shadow.items()},
        "is_ema": True,
    }
    torch.save(ckpt, os.path.join(args.output_dir, "ema_final.pt"))
    print(f"[train] done. best loss {best_loss:.4f}")


if __name__ == "__main__":
    main()
