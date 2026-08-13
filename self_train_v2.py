"""Stage C — Iterative self-training with pseudo-label consistency cleaning.

Motivation (HANDOVER.md S9.3):
  * No class names are provided by the contest — zero-shot text-side cleaning is
    impossible.  All denoising must come from *image-space self-consistency*.
  * Dynamic small-loss selection (Stage B) only ensures "easy to fit", not
    "label is correct".  This script adds a **pseudo-label consistency** gate:
      - Use the trained model M to predict all 103k training images.
      - folder_label == predicted_label  &  high confidence  →  clean (strong sup).
      - folder_label != predicted_label  &  high confidence  →  likely mislabeled /
        wrong folder → down-weight or drop.
      - low confidence → hold out (keep for future rounds).
  * Re-train on the cleaned set → better model → cleaner set → iterate.
    This is the single largest lever remaining for pushing 55 % → ~63 % under
    the ViT-B/32 / single-model / no-external-data constraints.

Run (after Stage B produces a seed model):
  python self_train_v2.py \
      --seed-ckpt output/contest_ft_lora_b/best.pt \
      --data-root /root/datasets/contest \
      --json all_class_predictions.json \
      --output-dir output/contest_ft_lora_c \
      --rounds 3 \
      --consistent-conf-threshold 0.8 \
      --mismatch-conf-threshold 0.9
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
import train_clip_lora as tl  # LoRALinear / inject_lora / CLIPLoRAClassifier / _evaluate

NORM = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                   std=(0.26862954, 0.26130258, 0.27577711))


# --------------------------------------------------------------------------- #
# Pseudo-label prediction (full training set, no GT needed for forward)
# --------------------------------------------------------------------------- #
class _PseudoPredDS(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform, data_root):
        self.paths = paths
        self.labels = labels  # kept for reference, not used in forward
        self.transform = transform
        self.data_root = data_root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(os.path.join(self.data_root, self.paths[i])).convert("RGB")
            return self.transform(img), i, self.labels[i]
        except Exception:
            return None


def _collate_pseudo(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs = torch.stack([b[0] for b in batch])
    idxs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, idxs, labels


@torch.no_grad()
def predict_all(model, data_root, paths_all, labels_all, resolution, batch_size,
                num_workers, amp):
    """Return (pred_classes, confidences, softmax_probs) for every image.

    softmax_probs has shape (n_total, n_classes) and serves as the *soft*
    pseudo-label target for soft-label self-training (distillation).
    """
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"
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
    probs_all = np.full((n, model.head.out_features), np.nan, dtype=np.float32)
    for batch in loader:
        if batch is None:
            continue
        imgs, idxs, labels = batch
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and is_cuda):
            logits = model(imgs)
            probs = logits.softmax(dim=1)
        c, p = probs.max(dim=1)
        ii = idxs.numpy().astype(np.int64)
        preds[ii] = p.cpu().numpy().astype(np.int32)
        confs[ii] = c.cpu().numpy().astype(np.float32)
        probs_all[ii] = probs.cpu().numpy().astype(np.float32)
    return preds, confs, probs_all


# --------------------------------------------------------------------------- #
# Consistency-based cleaning
# --------------------------------------------------------------------------- #
def consistency_clean(folder_labels, pred_labels, confs,
                      consistent_conf_thresh, mismatch_conf_thresh):
    """Partition the training set into three groups based on model self-consistency.

    Returns:
        clean_mask: folder == pred AND conf >= consistent_conf_thresh
        mismatch_mask: folder != pred AND conf >= mismatch_conf_thresh (likely wrong label)
        uncertain_mask: everything else (low confidence or uncertain)
    """
    consistent = (folder_labels == pred_labels)
    # match: consistent & high confidence
    clean_mask = consistent & (confs >= consistent_conf_thresh)
    # mismatch: inconsistent & high confidence -> likely noise
    mismatch_mask = (~consistent) & (confs >= mismatch_conf_thresh)
    # uncertain: everything left (low confidence in either direction)
    uncertain_mask = ~(clean_mask | mismatch_mask)
    return clean_mask, mismatch_mask, uncertain_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(size):
    train_t = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        NORM,
    ])
    eval_t = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        NORM,
    ])
    return train_t, eval_t


# --------------------------------------------------------------------------- #
# Main loop: predict -> clean -> retrain -> iterate
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Stage C: iterative self-training")
    p.add_argument("--seed-ckpt", required=True,
                   help="LoRA checkpoint to start from (e.g. Stage B best.pt)")
    p.add_argument("--data-root", default="/root/datasets/contest")
    p.add_argument("--json", default="all_class_predictions.json")
    p.add_argument("--clip-weights", default="/root/weights/ViT-B-32.pt")
    p.add_argument("--output-dir", default="output/contest_ft_lora_c")
    # self-training
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--consistent-conf-threshold", type=float, default=0.8,
                   help="min confidence for folder==pred to be 'clean'")
    p.add_argument("--mismatch-conf-threshold", type=float, default=0.9,
                   help="min confidence for folder!=pred to be 'likely noise'")
    p.add_argument("--drop-mismatch", dest="drop_mismatch",
                   action="store_true", default=False,
                   help="fully DROP high-conf mismatch samples (likely mislabeled) "
                   "instead of down-weighting them. More aggressive denoising.")
    p.add_argument("--use-uncertain", dest="use_uncertain", action="store_true",
                   default=False,
                   help="include uncertain (low-conf) samples with reduced weight")
    p.add_argument("--soft-labels", dest="soft_labels", action="store_true",
                   default=False,
                   help="use the model's full softmax prediction as a SOFT target "
                   "(KL distillation) instead of a hard pseudo-label. Mitigates "
                   "confirmation bias from wrong hard labels (HANDOVER S11.6).")
    p.add_argument("--soft-temp", type=float, default=1.0,
                   help="temperature for softening the soft target (T>1 = smoother)")
    # LoRA
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="out_proj,c_fc,c_proj")
    # train per round
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)  # shorter per round
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def _init_model_and_optim(ckpt, clip_weights, device, n_classes, targets,
                          rank, alpha, dropout, lora_lr, head_lr, wd,
                          steps, warmup_ratio):
    clip_model = cc.load_clip_to_cpu(clip_weights, device="cpu").float()
    model = tl.CLIPLoRAClassifier(clip_model, n_classes).to(device)
    tl.freeze_all(model.visual)
    tl.inject_lora(model.visual, targets, rank, alpha, dropout)
    model.to(device)
    for p in model.head.parameters():
        p.requires_grad_(True)

    if ckpt is not None and os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=False)
        print(f"[init] loaded ckpt {ckpt}")
    else:
        print("[init] no ckpt — training from scratch (round 0 seed)")

    lora_ps = [p for n, p in model.named_parameters()
               if p.requires_grad and ("lora_" in n) and "head." not in n]
    optimizer = torch.optim.AdamW([
        {"params": lora_ps, "lr": lora_lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ], weight_decay=wd)
    warmup_steps = max(1, int(steps * warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup_steps) if s < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, steps - warmup_steps))),
    )
    return model, optimizer, sched


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
    print(f"[data] n_classes={n_classes}, n_total={n_total}")

    train_t, eval_t = make_transforms(args.resolution)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    ckpt = args.seed_ckpt
    summary_log = []

    for round_i in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"[round {round_i}/{args.rounds}]", flush=True)

        # ---- Step 1: predict all images with current model ---------------
        total_steps = args.epochs * max(1, math.ceil(n_total / args.batch_size))
        model, optimizer, sched = _init_model_and_optim(
            ckpt, args.clip_weights, device, n_classes, targets,
            args.lora_rank, args.lora_alpha, args.lora_dropout,
            args.lora_lr, args.head_lr, args.wd,
            total_steps, args.warmup_ratio,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.startswith("cuda"))

        pred_preds = predict_all(model, args.data_root, paths_all, labels_all,
                                  args.resolution, args.batch_size,
                                  args.num_workers, args.amp)
        preds, confs, probs_all = pred_preds
        soft_targets_all = None
        if args.soft_labels:
            # soften + normalize the model's prediction distribution
            if args.soft_temp != 1.0:
                probs_all = probs_all ** (1.0 / args.soft_temp)
            soft_targets_all = probs_all / probs_all.sum(axis=1, keepdims=True)
        n_valid = int((~np.isnan(confs)).sum())
        print(f"[predict] valid predictions: {n_valid}/{n_total}"
              + ("  [SOFT labels]" if args.soft_labels else ""))

        # ---- Step 2: consistency cleaning --------------------------------
        clean_mask, mismatch_mask, uncertain_mask = consistency_clean(
            labels_all, preds, confs,
            args.consistent_conf_threshold, args.mismatch_conf_threshold,
        )
        print(f"[clean] clean={int(clean_mask.sum())} "
              f"mismatch={int(mismatch_mask.sum())} "
              f"uncertain={int(uncertain_mask.sum())}")

        # Build training set for this round
        if args.drop_mismatch:
            train_mask = clean_mask.copy()
            if args.use_uncertain:
                train_mask |= uncertain_mask
            sw = np.ones(n_total, dtype=np.float32)
            sw[uncertain_mask] = 0.05
        else:
            train_mask = clean_mask | mismatch_mask
            if args.use_uncertain:
                train_mask |= uncertain_mask
            sw = np.ones(n_total, dtype=np.float32)
            sw[mismatch_mask] = 0.1
            sw[uncertain_mask] = 0.05
        train_idx = np.where(train_mask)[0]
        sample_weights = torch.tensor(sw[train_idx], dtype=torch.float32)
        print(f"[train] training on {len(train_idx)} samples "
              f"({'DROP mismatch' if args.drop_mismatch else 'w/ mismatch downweight'})")

        # ---- Step 3: train on cleaned set --------------------------------
        if len(train_idx) < args.batch_size:
            print(f"[WARN] too few training samples ({len(train_idx)}), skipping round")
            break

        class _TrainDS(torch.utils.data.Dataset):
            def __init__(self, idxs, paths, labels, transform, data_root, weights,
                         soft_targets=None):
                self.idxs = idxs
                self.paths = paths
                self.labels = labels
                self.transform = transform
                self.data_root = data_root
                self.weights = weights
                self.soft_targets = soft_targets  # (n_total, n_classes) or None

            def __len__(self):
                return len(self.idxs)

            def __getitem__(self, i):
                try:
                    img = Image.open(os.path.join(self.data_root,
                                                  self.paths[self.idxs[i]])).convert("RGB")
                    if self.soft_targets is not None:
                        return (self.transform(img),
                                self.soft_targets[self.idxs[i]].astype(np.float32),
                                self.weights[i])
                    return (self.transform(img), int(self.labels[self.idxs[i]]),
                            self.weights[i])
                except Exception:
                    return None

        def _collate(batch):
            batch = [b for b in batch if b is not None]
            if not batch:
                return None
            imgs = torch.stack([b[0] for b in batch])
            if isinstance(batch[0][1], np.ndarray):
                labels = torch.tensor(np.stack([b[1] for b in batch]),
                                      dtype=torch.float32)  # soft targets
            else:
                labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
            weights = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            return imgs, labels, weights

        train_ds = _TrainDS(train_idx, paths_all, labels_all, train_t,
                            args.data_root, sample_weights,
                            soft_targets=soft_targets_all)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=_collate, pin_memory=True,
                                  prefetch_factor=(4 if args.num_workers else None),
                                  persistent_workers=bool(args.num_workers))

        # Simple val on a held-out fraction of clean set (noisy monitoring only)
        g = torch.Generator().manual_seed(args.seed + round_i * 100)
        n_val = max(1, int(0.05 * len(train_idx)))
        perm = torch.randperm(len(train_idx), generator=g)
        val_mask = np.zeros(len(train_idx), dtype=bool)
        val_mask[perm[:n_val]] = True
        val_idx = train_idx[val_mask]
        train_one = train_idx[~val_mask]

        class _EvalDS(torch.utils.data.Dataset):
            def __init__(self, idxs, paths, labels, transform, data_root):
                self.idxs = idxs
                self.paths = paths
                self.labels = labels
                self.transform = transform
                self.data_root = data_root

            def __len__(self):
                return len(self.idxs)

            def __getitem__(self, i):
                try:
                    img = Image.open(os.path.join(self.data_root,
                                                  self.paths[self.idxs[i]])).convert("RGB")
                    return (self.transform(img), int(self.labels[self.idxs[i]]),
                            self.idxs[i])
                except Exception:
                    return None

        val_ds = _EvalDS(val_idx, paths_all, labels_all, eval_t, args.data_root)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=_collate,
                                pin_memory=True)

        best_state = None
        best_val = -1.0
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
                    if labels.dim() == 2:  # soft targets: KL distillation
                        logp = torch.log_softmax(logits, dim=1)
                        loss_vec = -(labels * logp).sum(dim=1)
                    else:  # hard pseudo-labels
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
            r = {"epoch": epoch, "loss": loss_sum / max(seen, 1),
                 "val_acc": val_acc}
            round_log.append(r)
            print(f"[round {round_i} epoch {epoch}/{args.epochs}] loss={r['loss']:.4f} "
                  f"val_acc={val_acc:.4f}", flush=True)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {n: v.detach().cpu() for n, v in model.state_dict().items()
                              if ("lora_" in n) or n.startswith("head.")}

        # Save round checkpoint
        round_ckpt = os.path.join(args.output_dir, f"round{round_i}.pt")
        if best_state is None:
            best_state = {n: v.detach().cpu() for n, v in model.state_dict().items()
                          if ("lora_" in n) or n.startswith("head.")}
        torch.save(best_state, round_ckpt)
        # Also update best.pt as the latest round's best
        torch.save(best_state, os.path.join(args.output_dir, "best.pt"))
        ckpt = round_ckpt  # next round starts from this

        summary = {
            "round": round_i,
            "train_n": int(len(train_idx)),
            "clean": int(clean_mask.sum()),
            "mismatch": int(mismatch_mask.sum()),
            "uncertain": int(uncertain_mask.sum()),
            "best_val_acc": best_val,
            "epoch_log": round_log,
        }
        summary_log.append(summary)
        with open(os.path.join(args.output_dir, "self_train_log.json"), "w") as f:
            json.dump({"args": vars(args), "rounds": summary_log}, f, indent=2)
        print(f"[round {round_i}] best_val={best_val:.4f} -> saved {round_ckpt}")

    print(f"\n[done] {args.rounds} rounds complete → {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
