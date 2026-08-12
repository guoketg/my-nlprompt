"""Direction 4 — Triple-signal iterative self-training (Stage D).

Builds on self_train_v2.py (Stage C/C2, online 67.86 %) by combining THREE
partially-independent denoising signals instead of relying on pseudo-label
consistency alone.

Why (HANDOVER.md S6.4):
  C2 drove `mismatch` from 5157 -> 2837 and then plateaued: the consistency
  gate cannot see noise that the model *confidently agrees with* (confirmation
  bias).  We therefore add two signals that are only weakly correlated with the
  model's own argmax decision:

    1. Consistency  C : folder_label == argmax  &  conf >= thresh      (Stage C)
    2. Small-loss   L : per-class CE-loss rank, keep lowest keep_ratio (Stage B)
    3. Prototype    P : cosine sim to the class prototype, per-class rank

  Crucially P is RECOMPUTED every round from the *current fine-tuned* backbone.
  Reusing output/contest_prototype/ would only cover the 56,565/103,218 samples
  that survived the old K-means pass (rest = NaN -> whole classes skipped), which
  would reintroduce the Stage-A information bottleneck.

Voting -> sample weight:
    3 signals pass -> 1.00  (core clean)
    2 signals pass -> 0.50  (probable clean)
    1 signal  pass -> 0.05  (weak supervision)
    0 pass & C says high-conf mismatch -> DROPPED

Run:
  ./.venv/bin/python -u self_train_v3.py \
      --seed-ckpt output/contest_ft_lora_c2/best.pt \
      --output-dir output/contest_ft_lora_d \
      --rounds 5
"""

import os
import json
import math
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets.contest import load_contest_classnames, build_train_list
import contest_clip as cc
import train_clip_lora as tl
from self_train_v2 import (
    NORM,
    _PseudoPredDS,
    _collate_pseudo,
    consistency_clean,
    set_seed,
    make_transforms,
    _init_model_and_optim,
)


# --------------------------------------------------------------------------- #
# One fused pass: logits -> pred/conf, per-sample CE loss, and L2 features
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_all_with_feats(model, data_root, paths_all, labels_all, resolution,
                           batch_size, num_workers, amp):
    """Single forward sweep over the full candidate pool.

    Returns (preds, confs, losses, feats):
        preds  int32   [N]      argmax class
        confs  float32 [N]      max softmax prob
        losses float32 [N]      CE loss w.r.t. the *folder* label
        feats  float32 [N, D]   L2-normalised image embeddings
    Invalid / unreadable images keep NaN so they are excluded downstream.
    """
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"
    model.eval()

    eval_t = T.Compose([
        T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(resolution),
        T.ToTensor(),
        NORM,
    ])
    ds = _PseudoPredDS(paths_all, labels_all, eval_t, data_root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate_pseudo,
                        pin_memory=True)

    n = len(paths_all)
    preds = np.full(n, -1, dtype=np.int32)
    confs = np.full(n, np.nan, dtype=np.float32)
    losses = np.full(n, np.nan, dtype=np.float32)
    feats = None

    for batch in loader:
        if batch is None:
            continue
        imgs, idxs, labels = batch
        imgs = imgs.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and is_cuda):
            f = model.visual(imgs.type(next(model.visual.parameters()).dtype))
            if isinstance(f, tuple):
                f = f[0]
            f = f.float()
            logits = model.head(f)
        probs = logits.softmax(dim=1)
        c, p = probs.max(dim=1)
        lv = F.cross_entropy(logits, labels_d, reduction="none")
        fn = F.normalize(f, dim=1)

        if feats is None:
            feats = np.full((n, fn.shape[1]), np.nan, dtype=np.float32)
        ii = idxs.numpy().astype(np.int64)
        preds[ii] = p.cpu().numpy().astype(np.int32)
        confs[ii] = c.cpu().numpy().astype(np.float32)
        losses[ii] = lv.float().cpu().numpy().astype(np.float32)
        feats[ii] = fn.cpu().numpy().astype(np.float32)

    return preds, confs, losses, feats


# --------------------------------------------------------------------------- #
# Signal 2: per-class small-loss selection
# --------------------------------------------------------------------------- #
def small_loss_mask(labels_all, losses, keep_ratio):
    """Per class, keep the `keep_ratio` fraction with the LOWEST CE loss."""
    n = len(labels_all)
    mask = np.zeros(n, dtype=bool)
    for c in np.unique(labels_all):
        idx = np.where(labels_all == c)[0]
        valid = idx[~np.isnan(losses[idx])]
        if valid.size == 0:
            continue
        k = max(1, int(round(keep_ratio * valid.size)))
        order = valid[np.argsort(losses[valid], kind="stable")]
        mask[order[:k]] = True
    return mask


# --------------------------------------------------------------------------- #
# Signal 3: prototype similarity, recomputed from current features
# --------------------------------------------------------------------------- #
def prototype_mask(labels_all, feats, keep_ratio, trim_ratio=0.5):
    """Per class, keep the `keep_ratio` fraction most similar to its prototype.

    The prototype is built robustly: a first mean over all class samples, then
    re-averaged over the top `trim_ratio` closest samples, so a minority of
    outlier/irrelevant crawled images cannot drag the centroid away.
    Returns (mask, sims) where sims is NaN for invalid samples.
    """
    n = len(labels_all)
    mask = np.zeros(n, dtype=bool)
    sims = np.full(n, np.nan, dtype=np.float32)
    if feats is None:
        return mask, sims

    for c in np.unique(labels_all):
        idx = np.where(labels_all == c)[0]
        valid = idx[~np.isnan(feats[idx, 0])]
        if valid.size == 0:
            continue
        fv = feats[valid]

        proto = fv.mean(axis=0)
        nrm = np.linalg.norm(proto)
        if nrm < 1e-8:
            continue
        proto /= nrm

        # robust re-estimation on the closest trim_ratio subset
        s0 = fv @ proto
        m = max(1, int(round(trim_ratio * valid.size)))
        core = np.argsort(-s0, kind="stable")[:m]
        proto = fv[core].mean(axis=0)
        nrm = np.linalg.norm(proto)
        if nrm < 1e-8:
            continue
        proto /= nrm

        s = fv @ proto
        sims[valid] = s.astype(np.float32)
        k = max(1, int(round(keep_ratio * valid.size)))
        keep = valid[np.argsort(-s, kind="stable")[:k]]
        mask[keep] = True

    return mask, sims


# --------------------------------------------------------------------------- #
# Fuse the three votes into per-sample weights
# --------------------------------------------------------------------------- #
def fuse_signals(clean_mask, mismatch_mask, loss_mask, proto_mask,
                 w_all3=1.0, w_two=0.5, w_one=0.05):
    """Majority-vote fusion -> (train_mask, sample_weight, stats)."""
    votes = (clean_mask.astype(np.int8)
             + loss_mask.astype(np.int8)
             + proto_mask.astype(np.int8))

    w = np.zeros(len(votes), dtype=np.float32)
    w[votes == 3] = w_all3
    w[votes == 2] = w_two
    w[votes == 1] = w_one

    train_mask = votes >= 1
    # a confidently-wrong prediction with no other supporting vote is real noise
    train_mask &= ~(mismatch_mask & (votes <= 1))
    w[~train_mask] = 0.0

    stats = {
        "votes3": int((votes == 3).sum()),
        "votes2": int((votes == 2).sum()),
        "votes1": int((votes == 1).sum()),
        "votes0": int((votes == 0).sum()),
        "dropped": int((~train_mask).sum()),
        "signal_consistency": int(clean_mask.sum()),
        "signal_loss": int(loss_mask.sum()),
        "signal_proto": int(proto_mask.sum()),
    }
    return train_mask, w, stats


def parse_args():
    p = argparse.ArgumentParser(
        description="Direction 4: triple-signal iterative self-training")
    p.add_argument("--seed-ckpt", required=True)
    p.add_argument("--data-root", default="/root/datasets/contest")
    p.add_argument("--json", default="all_class_predictions.json")
    p.add_argument("--clip-weights", default="/root/weights/ViT-B-32.pt")
    p.add_argument("--output-dir", default="output/contest_ft_lora_d")
    # self-training
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--consistent-conf-threshold", type=float, default=0.7)
    p.add_argument("--mismatch-conf-threshold", type=float, default=0.8)
    # triple-signal knobs
    p.add_argument("--loss-keep-ratio", type=float, default=0.75,
                   help="per-class fraction kept by the small-loss signal")
    p.add_argument("--proto-keep-ratio", type=float, default=0.75,
                   help="per-class fraction kept by the prototype signal")
    p.add_argument("--proto-trim-ratio", type=float, default=0.5,
                   help="fraction used to robustly re-estimate each prototype")
    p.add_argument("--weight-all3", type=float, default=1.0)
    p.add_argument("--weight-two", type=float, default=0.5)
    p.add_argument("--weight-one", type=float, default=0.05)
    # LoRA
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="out_proj,c_fc,c_proj")
    # train per round
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


class _TrainDS(torch.utils.data.Dataset):
    def __init__(self, idxs, paths, labels, transform, data_root, weights):
        self.idxs = idxs
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.data_root = data_root
        self.weights = weights

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        try:
            img = Image.open(os.path.join(
                self.data_root, self.paths[self.idxs[i]])).convert("RGB")
            return (self.transform(img), int(self.labels[self.idxs[i]]),
                    float(self.weights[i]))
        except Exception:
            return None


def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    weights = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return imgs, labels, weights


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]

    classnames, _, _ = load_contest_classnames(args.json)
    n_classes = len(classnames)

    entries = build_train_list(args.data_root)
    paths_all = np.array([e[0] for e in entries], dtype=object)
    labels_all = np.array([int(e[1]) for e in entries], dtype=np.int64)
    n_total = len(entries)
    assert n_classes == int(labels_all.max()) + 1
    print(f"[data] n_classes={n_classes}, n_total={n_total}", flush=True)

    train_t, eval_t = make_transforms(args.resolution)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    ckpt = args.seed_ckpt
    summary_log = []

    for round_i in range(1, args.rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"[round {round_i}/{args.rounds}]", flush=True)

        total_steps = args.epochs * max(1, math.ceil(n_total / args.batch_size))
        model, optimizer, sched = _init_model_and_optim(
            ckpt, args.clip_weights, device, n_classes, targets,
            args.lora_rank, args.lora_alpha, args.lora_dropout,
            args.lora_lr, args.head_lr, args.wd,
            total_steps, args.warmup_ratio,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=args.amp and device.startswith("cuda"))

        # ---- Step 1: one fused sweep -> preds / conf / loss / features ----
        preds, confs, losses, feats = predict_all_with_feats(
            model, args.data_root, paths_all, labels_all, args.resolution,
            args.batch_size, args.num_workers, args.amp)
        print(f"[predict] valid: {int((~np.isnan(confs)).sum())}/{n_total}",
              flush=True)

        # ---- Step 2: three signals ---------------------------------------
        clean_mask, mismatch_mask, uncertain_mask = consistency_clean(
            labels_all, preds, confs,
            args.consistent_conf_threshold, args.mismatch_conf_threshold)
        loss_mask = small_loss_mask(labels_all, losses, args.loss_keep_ratio)
        proto_msk, sims = prototype_mask(labels_all, feats,
                                         args.proto_keep_ratio,
                                         args.proto_trim_ratio)
        print(f"[signals] C={int(clean_mask.sum())} "
              f"L={int(loss_mask.sum())} P={int(proto_msk.sum())} "
              f"mismatch={int(mismatch_mask.sum())}", flush=True)

        # ---- Step 3: fuse votes ------------------------------------------
        train_mask, sw, stats = fuse_signals(
            clean_mask, mismatch_mask, loss_mask, proto_msk,
            args.weight_all3, args.weight_two, args.weight_one)
        train_idx = np.where(train_mask)[0]
        print(f"[fuse] votes3={stats['votes3']} votes2={stats['votes2']} "
              f"votes1={stats['votes1']} dropped={stats['dropped']} "
              f"-> train_n={len(train_idx)}", flush=True)

        if len(train_idx) < args.batch_size:
            print(f"[WARN] too few samples ({len(train_idx)}), stopping")
            break

        sample_weights = sw[train_idx]

        # ---- Step 4: retrain on the weighted, purified set ---------------
        # Hold out 5 % of the highest-trust (3-vote) samples for monitoring only.
        g = torch.Generator().manual_seed(args.seed + round_i * 100)
        n_val = max(1, int(0.05 * len(train_idx)))
        perm = torch.randperm(len(train_idx), generator=g).numpy()
        val_sel = np.zeros(len(train_idx), dtype=bool)
        val_sel[perm[:n_val]] = True

        val_idx = train_idx[val_sel]
        tr_idx = train_idx[~val_sel]
        tr_w = sample_weights[~val_sel]

        train_ds = _TrainDS(tr_idx, paths_all, labels_all, train_t,
                            args.data_root, tr_w)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=_collate, pin_memory=True,
            prefetch_factor=(4 if args.num_workers else None),
            persistent_workers=bool(args.num_workers))

        val_ds = _TrainDS(val_idx, paths_all, labels_all, eval_t,
                          args.data_root, np.ones(len(val_idx), dtype=np.float32))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                collate_fn=_collate, pin_memory=True)

        best_state, best_val = None, -1.0
        round_log = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, seen = 0.0, 0
            for batch in train_loader:
                if batch is None:
                    continue
                imgs, labels, weights = batch
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits = model(imgs)
                    loss_vec = criterion(logits, labels)
                    loss = (loss_vec * weights).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                sched.step()
                b = imgs.shape[0]
                loss_sum += float(loss.detach().cpu()) * b
                seen += b

            val_acc = tl._evaluate(model, val_loader, device, args.amp)
            round_log.append({"epoch": epoch, "loss": loss_sum / max(seen, 1),
                              "val_acc": val_acc})
            print(f"[round {round_i} epoch {epoch}/{args.epochs}] "
                  f"loss={loss_sum / max(seen, 1):.4f} val_acc={val_acc:.4f}",
                  flush=True)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {n: v.detach().cpu()
                              for n, v in model.state_dict().items()
                              if ("lora_" in n) or n.startswith("head.")}

        if best_state is None:
            best_state = {n: v.detach().cpu()
                          for n, v in model.state_dict().items()
                          if ("lora_" in n) or n.startswith("head.")}
        round_ckpt = os.path.join(args.output_dir, f"round{round_i}.pt")
        torch.save(best_state, round_ckpt)
        torch.save(best_state, os.path.join(args.output_dir, "best.pt"))
        ckpt = round_ckpt

        summary_log.append({
            "round": round_i,
            "train_n": int(len(train_idx)),
            "clean": int(clean_mask.sum()),
            "mismatch": int(mismatch_mask.sum()),
            "uncertain": int(uncertain_mask.sum()),
            "signals": stats,
            "mean_proto_sim": float(np.nanmean(sims)) if np.isfinite(sims).any() else None,
            "best_val_acc": best_val,
            "epoch_log": round_log,
        })
        with open(os.path.join(args.output_dir, "self_train_log.json"), "w") as f:
            json.dump({"args": vars(args), "rounds": summary_log}, f, indent=2)
        print(f"[round {round_i}] best_val={best_val:.4f} -> {round_ckpt}",
              flush=True)

    print(f"\n[done] {args.rounds} rounds -> {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
