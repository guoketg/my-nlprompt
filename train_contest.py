"""Step 2 - robust prompt tuning on the CLEANED contest dataset (no feature-cache files).

Pipeline (see also clean_contest.py and the contest design notes):
  * Load the CLEAN manifest produced by clean_contest.py (corrupted / truncated /
    degenerate images already quarantined there).
  * We do NOT read pre-computed feature files. Instead we ENCODE the clean images
    ONCE into memory at the start of training (a runtime data load, nothing is
    written to disk) and then train 200 epochs on those in-memory features -- this
    keeps training fast without "feature caching" files on disk.
  * Optimise ONLY the NLPrompt context vectors; all CLIP weights stay frozen.
  * Base loss = Generalized Cross Entropy (GCE, Zhang & Sabuncu 2018), which is
    robust to label noise and never re-labels a sample.
  * Open-set noise handling (request.md 四.(三).1: "papers / other species mixed
    into a class"): after a warm-up we compute per-sample confidence and
        - EXCLUDE the lowest-confidence samples (the off-distribution outliers)
          from the loss entirely -- we do NOT re-label them to another class,
          because their true class is outside the 500-way label space;
        - train the high-confidence ("clean-looking") samples with plain CE for a
          stronger gradient.
  * Default 200 epochs (standard prompt-tuning recipe, e.g. CoOp).

Run:
  source .venv/bin/activate
  python clean_contest.py --data-root /root/datasets/contest --output-dir output/contest_clean
  python train_contest.py --data-root /root/datasets/contest \
      --json all_class_predictions.json \
      --clean-manifest output/contest_clean/clean_train_manifest.json \
      --output-dir output/contest
"""

import os
import json
import time
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets.contest import load_contest_classnames, _default_transform
import contest_clip as cc

torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Robust NLPrompt tuning on cleaned contest images")
    p.add_argument("--data-root", default="/root/datasets/contest")
    p.add_argument("--json", default="all_class_predictions.json")
    p.add_argument("--clean-manifest", default="output/contest_clean/clean_train_manifest.json")
    p.add_argument("--clip-weights", default="/root/weights/ViT-B-32.pt",
                   help="local fast-disk ViT-B/32 (loads in seconds; the ViT-L on the "
                        "slow network disk took ~3min to load and 4x longer to run)")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--output-dir", default="output/contest")
    # prompt learner
    p.add_argument("--n-ctx", type=int, default=16)
    p.add_argument("--ctx-init", default="a photo of a")
    p.add_argument("--csc", action="store_true")
    p.add_argument("--class-token-position", default="end", choices=["begin", "end"])
    # optimisation
    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--wd", type=float, default=0.001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    # initial in-memory encode
    p.add_argument("--encode-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8,
                   help="parallel image-decoding workers for the one-off in-memory "
                        "encode (cgroup CPU quota is 4; more workers just thrash)")
    # robustness
    p.add_argument("--gce-q", type=float, default=0.7)
    p.add_argument("--no-selection", action="store_true",
                   help="disable open-set outlier exclusion (pure GCE on all)")
    p.add_argument("--warmup", type=int, default=1,
                   help="epochs of GCE-only before enabling outlier exclusion")
    p.add_argument("--outlier-frac", type=float, default=0.15,
                   help="fraction (by confidence) of lowest-conf samples excluded as outliers")
    p.add_argument("--clean-thr", type=float, default=0.5,
                   help="samples with confidence above this are trained with CE")
    p.add_argument("--max-train", type=int, default=0,
                   help="if >0, only use this many training samples (dry-run)")
    return p.parse_args()


def _drop_none_collate(batch):
    """Module-level (picklable) collate: drop items whose image failed to decode."""
    return [x for x in batch if x is not None]


def _worker_init(_):
    # cgroup quota is only 4 CPUs; keep each worker single-threaded.
    torch.set_num_threads(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_manifest(manifest, data_root, preprocess, encoder, device, batch_size, num_workers):
    """Encode all images listed in manifest into an in-memory (N, D) float32 tensor."""
    paths = [os.path.join(data_root, rel) for rel, _ in manifest]
    labels = [int(l) for _, l in manifest]
    ds = torch.utils.data.Dataset()
    # simple indexable dataset
    class _Ds(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            try:
                img = Image.open(paths[i]).convert("RGB")
                return preprocess(img), labels[i], i
            except Exception:
                return None
    ds = _Ds()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_drop_none_collate,
                        pin_memory=False,
                        prefetch_factor=(8 if num_workers > 0 else None),
                        persistent_workers=(num_workers > 0),
                        worker_init_fn=(_worker_init if num_workers > 0 else None))
    encoder.eval()
    dev = next(encoder.parameters()).device
    print(f"[encode] encoder device = {dev} (expect cuda)", flush=True)
    feats, kept_labels, kept_idx = [], [], []
    n = len(paths)
    done = 0
    print(f"[encode] in-memory encoding {n} clean images ...", flush=True)
    t0 = time.time()
    for batch in loader:
        if not batch:
            continue
        imgs = torch.stack([b[0] for b in batch]).to(device, non_blocking=True)
        imgs = imgs.to(next(encoder.parameters()).dtype)
        lbs = [b[1] for b in batch]
        idxs = [b[2] for b in batch]
        raw = encoder(imgs)
        torch.cuda.synchronize()
        for j in range(len(batch)):
            f = raw[j]
            if torch.isnan(f).any() or torch.isinf(f).any() or f.norm() < 1e-3:
                continue  # should not happen (already cleaned), skip defensively
            feats.append(f / f.norm())
            kept_labels.append(lbs[j])
            kept_idx.append(idxs[j])
        done += len(batch)
        if done % (batch_size * 20) < batch_size:
            el = time.time() - t0
            rate = done / max(1e-6, el)
            print(f"  [encode] {done}/{n}  {el:.0f}s elapsed  "
                  f"{rate:.1f} img/s  eta {(n - done) / max(1e-6, rate) / 60:.1f}min",
                  flush=True)
    F_tensor = torch.stack(feats).float().cpu()
    L_tensor = torch.tensor(kept_labels, dtype=torch.long)
    print(f"[encode] done: {F_tensor.shape[0]} features in {time.time() - t0:.1f}s")
    return F_tensor, L_tensor


@torch.no_grad()
def evaluate(feats, labels, prompt_learner, text_encoder, tokenized, logit_scale,
             device, batch_size=4096):
    prompt_learner.eval()
    text_features = text_encoder(prompt_learner(), tokenized).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    correct, total = 0, 0
    for i in range(0, feats.shape[0], batch_size):
        x = feats[i:i + batch_size].to(device)
        y = labels[i:i + batch_size].to(device)
        logits = logit_scale * x @ text_features.t()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    classnames, folder_to_idx, keys = load_contest_classnames(args.json)
    with open(args.clean_manifest) as f:
        manifest = json.load(f)
    print(f"[data] clean manifest: {len(manifest)} images")

    # in-memory encode (no disk cache)
    preprocess = _default_transform(args.resolution)
    print(f"[model] loading CLIP image encoder from {args.clip_weights}")
    clip_model = cc.load_clip_to_cpu(args.clip_weights, device="cpu")
    clip_model = clip_model.to(device).float()
    feats, labels = encode_manifest(
        manifest, args.data_root, preprocess, clip_model.visual, device,
        args.encode_batch_size, args.num_workers,
    )
    dim = feats.shape[1]

    n_total = feats.shape[0]
    if args.max_train and args.max_train < n_total:
        n_total = args.max_train
        feats = feats[:n_total]
        labels = labels[:n_total]
    N = feats.shape[0]

    # val split (noisy labels, used only as a training signal / monitoring)
    n_val = int(round(args.val_ratio * N))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_feats, tr_labels = feats[tr_idx], labels[tr_idx]
    val_feats, val_labels = feats[val_idx], labels[val_idx]

    # ----- model (full CLIP, only prompt learner trained) -----
    print(f"[model] building CustomCLIP from {args.clip_weights}")
    model = cc.CustomCLIP(classnames, clip_model, args.n_ctx, args.ctx_init,
                          args.csc, args.class_token_position)
    model = model.to(device).float()           # fp32 to avoid fp16 gradient NaN
    model.dtype = torch.float32
    model.text_encoder.dtype = torch.float32
    model.prompt_learner.dtype = torch.float32
    for pname, p in model.named_parameters():
        p.requires_grad = False
    for p in model.prompt_learner.parameters():
        p.requires_grad = True

    tokenized = model.prompt_learner.tokenized_prompts.to(device)
    logit_scale = model.logit_scale.exp().float()
    text_encoder = model.text_encoder
    prompt_learner = model.prompt_learner

    gce = cc.GeneralizedCrossEntropy(q=args.gce_q)
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(
        prompt_learner.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.wd,
    )

    train_ds = TensorDataset(tr_feats, tr_labels, torch.arange(tr_feats.shape[0]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)

    # persistent masks (refreshed each epoch when selection is on)
    outlier_mask = torch.zeros(tr_feats.shape[0], dtype=torch.bool)
    clean_mask = torch.zeros(tr_feats.shape[0], dtype=torch.bool)

    def refresh_masks():
        nonlocal outlier_mask, clean_mask
        with torch.no_grad():
            prompt_learner.eval()
            tf = text_encoder(prompt_learner(), tokenized).float()
            tf = tf / tf.norm(dim=-1, keepdim=True)
            logits = logit_scale * tr_feats.to(device) @ tf.t()
            probs = F.softmax(logits.float(), dim=1)
            conf = probs.max(dim=1).values
            pred = probs.argmax(dim=1)
        q = torch.quantile(conf, args.outlier_frac)
        outlier_mask = (conf < q)
        clean_mask = (conf > args.clean_thr)
        prompt_learner.train()
        return float(conf.mean()), int(outlier_mask.sum()), int(clean_mask.sum())

    best_val_acc = -1.0
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    log_f = open(log_path, "w")

    for epoch in range(args.epochs):
        use_sel = (not args.no_selection) and epoch >= args.warmup
        extra = ""
        if use_sel:
            cmean, n_out, n_clean = refresh_masks()
            extra = f" conf_mean={cmean:.3f} outliers={n_out} clean={n_clean}"

        prompt_learner.train()
        running_loss, n_batches = 0.0, 0
        for x_b, y_b, idx_b in train_loader:
            x_b = x_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            tf = text_encoder(prompt_learner(), tokenized).float()
            tf = tf / tf.norm(dim=-1, keepdim=True)
            logits = logit_scale * x_b @ tf.t()
            loss_per = gce(logits, y_b)
            if use_sel:
                om = outlier_mask[idx_b].to(device)
                cm = clean_mask[idx_b].to(device)
                ce_b = ce(logits, y_b)
                w = (~om).float()
                loss = ((loss_per * (~cm) + ce_b * cm) * w).sum() / w.sum().clamp(min=1)
            else:
                loss = loss_per.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        val_acc = evaluate(val_feats, val_labels, prompt_learner, text_encoder,
                           tokenized, logit_scale, device)
        print(f"[epoch {epoch + 1}/{args.epochs}] "
              f"loss={running_loss / max(n_batches, 1):.4f} val_acc={val_acc:.4f}{extra}")
        log_f.write(json.dumps({
            "epoch": epoch + 1, "loss": running_loss / max(n_batches, 1),
            "val_acc": val_acc, "use_selection": use_sel,
            "n_outliers": int(outlier_mask.sum()) if use_sel else 0,
            "n_clean": int(clean_mask.sum()) if use_sel else 0,
        }) + "\n")
        log_f.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(prompt_learner.state_dict(),
                       os.path.join(args.output_dir, "prompt_learner_best.pt"))
            print(f"  -> new best val_acc={val_acc:.4f}, saved best prompt learner")

    torch.save(prompt_learner.state_dict(),
               os.path.join(args.output_dir, "prompt_learner_last.pt"))
    log_f.close()

    meta = dict(
        classnames=classnames, folder_keys=keys, n_ctx=args.n_ctx,
        ctx_init=args.ctx_init, csc=args.csc,
        class_token_position=args.class_token_position,
        clip_weights=args.clip_weights, resolution=args.resolution,
        feature_dim=int(feats.shape[1]), epochs=args.epochs, lr=args.lr,
        gce_q=args.gce_q, outlier_frac=args.outlier_frac, clean_thr=args.clean_thr,
        use_selection=not args.no_selection, best_val_acc=best_val_acc,
        args=vars(args),
    )
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] best_val_acc={best_val_acc:.4f} -> {args.output_dir}")


if __name__ == "__main__":
    main()
