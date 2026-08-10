"""Step 2b - LoRA fine-tuning of the CLIP ViT-B/32 visual backbone (robust to label noise).

Why this script exists:
  * The previous stage-2/3 pipeline FREEZES the CLIP visual encoder and only trains a
    linear cosine head (or prompt tokens). On this web-scraped, 500-class, heavy-noise
    benchmark that caps accuracy (~48%).
  * The sibling PGDF pipeline reaches ~59% on the SAME CLIP ViT-B/32 by
      (1) LoRA-fine-tuning the visual backbone (attention + mlp projections),
      (2) periodically re-selecting the small-loss + high-prototype samples and
          re-training on them, and
      (3) using a larger input resolution (336 here).
  * This script ports that exact recipe onto the NLPrompt contest data, reusing the
    clean_mask / features / labels produced by class_prototype.py (stage 1).

Constraints satisfied (requirements.md):
  * backbone stays CLIP ViT-B/32 (weights fixed, only LoRA adapters are added/learned)
  * no external data, single model (no ensemble), fully deterministic (fixed seed)

Run:
  python train_clip_lora.py --data-root /root/datasets/contest \
      --json all_class_predictions.json \
      --proto-dir output/contest_prototype \
      --output-dir output/contest_ft_lora
"""

import os
import json
import math
import time
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

torch.backends.cudnn.benchmark = False
NORM = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                   std=(0.26862954, 0.26130258, 0.27577711))


# --------------------------------------------------------------------------- #
# LoRA utilities (ported from PGDF gcdd.lora_training.LoRALinear / inject_lora)
# --------------------------------------------------------------------------- #
class _LoRALinear(torch.nn.Linear):
    """nn.Linear subclass with a LoRA branch.

    Subclassing (not wrapping) is essential here: CLIP's nn.MultiheadAttention
    accesses ``out_proj.weight`` / ``.bias`` directly, so the module MUST keep
    the standard Linear attributes while adding a trainable LoRA residual.
    """

    def __init__(self, base_layer, rank, alpha, dropout):
        super().__init__(base_layer.in_features, base_layer.out_features,
                         bias=base_layer.bias is not None)
        with torch.no_grad():
            self.weight.copy_(base_layer.weight)
            if base_layer.bias is not None:
                self.bias.copy_(base_layer.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_a = torch.nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_b = torch.nn.Linear(rank, base_layer.out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.scaling = float(alpha) / float(rank)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return super().forward(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def inject_lora(model, target_substrings, rank, alpha, dropout):
    replaced = []
    for name, child in list(model.named_modules()):
        if not isinstance(child, torch.nn.Linear):
            continue
        if not any(t in name for t in target_substrings):
            continue
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], _LoRALinear(child, rank, alpha, dropout))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No Linear matched LoRA targets {target_substrings}")
    return replaced


def freeze_all(module):
    for p in module.parameters():
        p.requires_grad_(False)


# --------------------------------------------------------------------------- #
# Dynamic sample selection (ported from PGDF gcdd.lora_dynamic)
# --------------------------------------------------------------------------- #
def select_small_loss_classwise(losses, labels, candidate_mask, keep_ratio):
    selected = np.zeros(len(labels), dtype=bool)
    for lab in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == lab))[0]
        if len(idx) == 0:
            continue
        if np.any(np.isnan(losses[idx])):
            # Lost loss for this class (corrupt images dropped at load time):
            # treat as untrusted -> do not select via loss for this class.
            continue
        keep = len(idx) if keep_ratio >= 1.0 else max(1, int(math.floor(len(idx) * keep_ratio)))
        order = np.argsort(losses[idx], kind="mergesort")
        selected[idx[order[:keep]]] = True
    return selected


def select_top_proto_classwise(proto_scores, labels, candidate_mask, keep_ratio):
    selected = np.zeros(len(labels), dtype=bool)
    for lab in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == lab))[0]
        if len(idx) == 0:
            continue
        if np.any(np.isnan(proto_scores[idx])):
            continue
        keep = len(idx) if keep_ratio >= 1.0 else max(1, int(math.floor(len(idx) * keep_ratio)))
        order = np.argsort(-proto_scores[idx], kind="mergesort")
        selected[idx[order[:keep]]] = True
    return selected


def combine_loss_proto(loss_sel, proto_pass, losses, labels, candidate_mask):
    selected = loss_sel & proto_pass
    for lab in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == lab))[0]
        if len(idx) == 0 or bool(selected[idx].any()):
            continue
        fb = idx[proto_pass[idx]]
        if len(fb) == 0:
            fb = idx
        fb_ok = fb[~np.isnan(losses[fb])]
        if len(fb_ok) == 0:
            fb_ok = fb
        best = fb_ok[np.argmin(losses[fb_ok])]
        selected[int(best)] = True
    return selected


@torch.no_grad()
def compute_losses(model, loader, device, total_n, amp):
    losses = np.full(total_n, np.nan, dtype=np.float32)
    loaded = []
    model.eval()
    for batch in loader:
        if batch is None:
            continue
        imgs, labels, idxs = batch
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.startswith("cuda")):
            logits = model(imgs)
            ce = F.cross_entropy(logits, labels, reduction="none")
        ii = idxs.numpy().astype(np.int64)
        losses[ii] = ce.detach().cpu().numpy().astype(np.float32)
        loaded.extend(ii.tolist())
    return losses, np.array(loaded, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform, data_root):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.data_root = data_root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(os.path.join(self.data_root, self.paths[i])).convert("RGB")
            return self.transform(img), int(self.labels[i]), i
        except Exception:
            return None


def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    idxs = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, labels, idxs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(size):
    train = T.Compose([
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
    return train, eval_t


# --------------------------------------------------------------------------- #
# Model: frozen CLIP visual + LoRA visual + linear head
# --------------------------------------------------------------------------- #
class CLIPLoRAClassifier(torch.nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.visual = clip_model.visual
        self.head = torch.nn.Linear(int(self.visual.output_dim), num_classes)

    def forward(self, images):
        feats = self.visual(images.type(next(self.visual.parameters()).dtype))
        if isinstance(feats, tuple):
            feats = feats[0]
        return self.head(feats.float())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune CLIP ViT-B/32 for contest")
    p.add_argument("--data-root", default="/root/datasets/contest")
    p.add_argument("--json", default="all_class_predictions.json")
    p.add_argument("--proto-dir", default="output/contest_prototype")
    p.add_argument("--clip-weights", default="/root/weights/ViT-B-32.pt")
    p.add_argument("--output-dir", default="output/contest_ft_lora")
    # lora
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="out_proj,c_fc,c_proj")
    # train
    p.add_argument("--resolution", type=int, default=336)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    # dynamic selection
    p.add_argument("--retention-ratio", type=float, default=0.9)
    p.add_argument("--proto-keep-ratio", type=float, default=0.7)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--update-interval", type=int, default=5)
    p.add_argument("--use-dynamic", dest="use_dynamic", action="store_true", default=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    # Stage-A: save the final-epoch model instead of the warmup-era best_val model,
    # because val_acc is a noisy (in-distribution) proxy and the aggressive dynamic
    # selection pushed the early "best" checkpoint to epoch 5, freezing the submission
    # model at the warmup stage. The last epoch carries the full LoRA fine-tune.
    p.add_argument("--keep-final", dest="keep_final", action="store_true", default=True,
                   help="save the last-epoch state as best.pt (recommended)")
    p.add_argument("--keep-best-val", dest="keep_final", action="store_false",
                   help="legacy: save max-val_acc epoch as best.pt")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]

    # n_classes derived from the data labels (folder ids), which is authoritative.
    classnames, folder_to_idx, keys = load_contest_classnames(args.json)
    print(f"[data] classname json has {len(classnames)} entries")

    # Stage-1 artifacts (torch .pt). features/labels/clean_mask are for the KEPT
    # subset; kept_idx.json maps them back to build_train_list() positions.
    proto_dir = args.proto_dir
    feats_k = torch.load(os.path.join(proto_dir, "features.pt"), weights_only=False).numpy().astype(np.float32)
    labels_k = torch.load(os.path.join(proto_dir, "labels.pt"), weights_only=False).numpy().astype(np.int64)
    clean_k = torch.load(os.path.join(proto_dir, "clean_mask.pt"), weights_only=False).numpy().astype(bool)
    with open(os.path.join(proto_dir, "kept_idx.json")) as f:
        kept_idx = np.array(json.load(f), dtype=np.int64)
    assert len(feats_k) == len(labels_k) == len(clean_k) == len(kept_idx)
    print(f"[data] kept={len(kept_idx)}; clean(kept)={int(clean_k.sum())}")

    # full-length alignment
    entries = build_train_list(args.data_root)
    n_total = len(entries)
    paths_all = np.array([e[0] for e in entries], dtype=object)
    labels_all = np.array([int(e[1]) for e in entries], dtype=np.int64)
    n_classes = int(labels_all.max()) + 1
    assert n_classes == len(classnames), f"n_classes mismatch {n_classes} vs {len(classnames)}"
    print(f"[data] n_classes={n_classes}, n_total={n_total}")

    clean_full = np.zeros(n_total, dtype=bool)
    clean_full[kept_idx[clean_k]] = True

    # prototype scores: cosine to per-class mean of clean kept features
    feats_k = feats_k / (np.linalg.norm(feats_k, axis=1, keepdims=True) + 1e-8)
    class_means = np.zeros((n_classes, feats_k.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = feats_k[clean_k & (labels_k == c)]
        if len(m):
            class_means[c] = m.mean(axis=0)
    class_means = class_means / (np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-8)
    proto_scores_full = np.full(n_total, np.nan, dtype=np.float32)
    proto_scores_full[kept_idx] = (feats_k * class_means[labels_k]).sum(axis=1)

    # candidate set = stage-1 clean (full positions)
    candidate_mask = clean_full.copy()
    candidate_idx = np.where(candidate_mask)[0]
    print(f"[data] candidates={len(candidate_idx)}")

    # val split (noisy; monitoring only)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(candidate_idx), generator=g)
    n_val = int(round(args.val_ratio * len(candidate_idx)))
    val_idx = candidate_idx[perm[:n_val]]
    train_idx0 = candidate_idx[perm[n_val:]]

    train_t, eval_t = make_transforms(args.resolution)
    print(f"[model] loading CLIP ViT-B/32 from {args.clip_weights}")
    clip_model = cc.load_clip_to_cpu(args.clip_weights, device="cpu").float()
    model = CLIPLoRAClassifier(clip_model, n_classes).to(device)
    freeze_all(model.visual)
    replaced = inject_lora(model.visual, targets, args.lora_rank, args.lora_alpha, args.lora_dropout)
    model.to(device)
    for p in model.head.parameters():
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] LoRA injected into {len(replaced)} modules; trainable={n_trainable}")

    lora_ps = [p for n, p in model.named_parameters()
               if p.requires_grad and ("lora_" in n) and "head." not in n]
    optimizer = torch.optim.AdamW([
        {"params": lora_ps, "lr": args.lora_lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.wd)

    def est_steps(idxs):
        return max(1, math.ceil(len(idxs) / args.batch_size))

    total_steps = (args.warmup_epochs + (args.epochs - args.warmup_epochs)) * est_steps(train_idx0)
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup_steps) if s < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps))),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.startswith("cuda"))
    criterion = torch.nn.CrossEntropyLoss()

    def make_loader(idxs, shuffle):
        ds = _ImageDataset(paths_all[idxs], labels_all[idxs],
                           (train_t if shuffle else eval_t), args.data_root)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, collate_fn=_collate,
                          pin_memory=True, prefetch_factor=(4 if args.num_workers else None),
                          persistent_workers=bool(args.num_workers))

    loss_ds = _ImageDataset(paths_all[candidate_idx], labels_all[candidate_idx], eval_t, args.data_root)
    loss_loader = DataLoader(loss_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=_collate, pin_memory=True)

    selected_mask = candidate_mask.copy()
    best_val, best_state, best_val_epoch = -1.0, None, 0
    log_f = open(os.path.join(args.output_dir, "train_log.jsonl"), "w")

    for epoch in range(1, args.epochs + 1):
        epoch_idx = np.where(selected_mask)[0]
        loader = make_loader(epoch_idx, shuffle=True)
        model.train()
        loss_sum, seen = 0.0, 0
        for batch in loader:
            if batch is None:
                continue
            imgs, labels, _ = batch
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            b = imgs.shape[0]
            loss_sum += float(loss.detach().cpu()) * b
            seen += b

        val_acc = _evaluate(model, make_loader(val_idx, False), device, args.amp)
        row = {"epoch": epoch, "loss": loss_sum / max(seen, 1), "val_acc": val_acc,
               "selected": int(selected_mask.sum())}
        log_f.write(json.dumps(row) + "\n"); log_f.flush()
        print(f"[epoch {epoch}/{args.epochs}] loss={row['loss']:.4f} val_acc={val_acc:.4f} "
              f"selected={row['selected']}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_val_epoch = epoch
            best_state = {n: v.detach().cpu() for n, v in model.state_dict().items()
                          if ("lora_" in n) or n.startswith("head.")}
            if not args.keep_final:
                torch.save(best_state, os.path.join(args.output_dir, "best.pt"))
            print(f"  -> new best val_acc={best_val:.4f} (epoch {epoch})", flush=True)

        if args.use_dynamic and epoch < args.epochs and \
           epoch >= args.warmup_epochs and (epoch - args.warmup_epochs) % args.update_interval == 0:
            losses, loaded_idx = compute_losses(model, loss_loader, device, n_total, args.amp)
            # permanently drop corrupt/unloaded images from the candidate pool
            if len(loaded_idx) < candidate_mask.sum():
                dropped = candidate_mask.sum() - len(loaded_idx)
                candidate_mask &= np.isin(np.arange(n_total), loaded_idx)
                print(f"[select] dropped {int(dropped)} unloadable imgs from candidates", flush=True)
            loss_sel = select_small_loss_classwise(losses, labels_all, candidate_mask, args.retention_ratio)
            proto_pass = select_top_proto_classwise(proto_scores_full, labels_all, candidate_mask, args.proto_keep_ratio)
            selected_mask = combine_loss_proto(loss_sel, proto_pass, losses, labels_all, candidate_mask)
            print(f"[select] epoch={epoch} selected={int(selected_mask.sum())} "
                  f"loss_sel={int(loss_sel.sum())} proto_pass={int(proto_pass.sum())}", flush=True)

    # Stage-A: the submission model is the LAST epoch (full fine-tune) by default,
    # because val_acc is a noisy in-distribution proxy and early stopping froze the
    # previous submission at the warmup stage (epoch 5). When --keep-best-val is given,
    # fall back to the max-val_acc epoch.
    if args.keep_final:
        best_state = {n: v.detach().cpu() for n, v in model.state_dict().items()
                      if ("lora_" in n) or n.startswith("head.")}
        saved_epoch = args.epochs
    else:
        saved_epoch = best_val_epoch
    torch.save(best_state, os.path.join(args.output_dir, "best.pt"))
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(dict(args=vars(args), best_val_acc=best_val, best_val_epoch=int(best_val_epoch),
                       submission_epoch=int(saved_epoch), n_classes=n_classes,
                       replaced_modules=replaced, resolution=args.resolution),
                  f, indent=2)
    print(f"[done] best_val_acc={best_val:.4f} (epoch {best_val_epoch}) "
          f"-> submission model = epoch {saved_epoch} (keep_final={args.keep_final})")


@torch.no_grad()
def _evaluate(model, loader, device, amp):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        if batch is None:
            continue
        imgs, labels, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.startswith("cuda")):
            logits = model(imgs)
        correct += int((logits.argmax(1) == labels).sum().item())
        total += int(labels.shape[0])
    return correct / max(total, 1)


if __name__ == "__main__":
    main()
