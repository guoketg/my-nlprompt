"""Stage 1 - per-class visual prototype discovery for the contest dataset.

This script does NOT use class names (no all_class_predictions.json). It:
  1. loads the frozen CLIP ViT-B/32 visual encoder,
  2. encodes every training image into L2-normalised visual features,
  3. runs k-means (k=3) inside every class folder,
  4. treats the largest cluster as the "dominant species" and computes its mean
     as the class prototype,
  5. marks every image in the dominant cluster as a "clean" sample.

Outputs (under --output-dir):
  prototypes.pt            (n_classes, D)  L2-normalised class prototypes
  clean_mask.pt            (N,) bool       True for images in dominant clusters
  sample_confidence.pt     (N,) float      cosine similarity to class prototype
  features.pt              (N, D)          all visual features (for reuse)
  labels.pt                (N,) long       folder ids (0..499) for each feature
  prototype_info.json      per-class stats (n_total, n_clean, purity, ...)
"""

import os
import json
import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets.contest import build_train_list, _default_transform
import contest_clip as cc


torch.backends.cudnn.benchmark = False


class _FeatureDataset(Dataset):
    """Returns (tensor, global_idx) or None on decode failure."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.transform(img), i
        except Exception:
            return None


def _collate(batch):
    batch = [b for b in batch if b is not None]
    return batch if batch else None


def _worker_init(_):
    torch.set_num_threads(1)


def kmeans_torch(X, k=3, max_iter=100, seed=0):
    """Lightweight cosine k-means on GPU/CPU tensors, k-means++ init.

    Args:
        X: (N, D) float tensor, already L2-normalised.
    Returns:
        labels: (N,) long tensor
        centers: (k, D) float tensor, L2-normalised
    """
    N, D = X.shape
    if N <= k:
        labels = torch.arange(N, device=X.device) % k
        centers = torch.zeros(k, D, device=X.device)
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = F.normalize(X[mask].mean(0, keepdim=True), dim=1).squeeze(0)
            else:
                centers[j] = X[torch.randint(N, (1,), device=X.device).item()]
        return labels, centers

    torch.manual_seed(seed)
    # k-means++ on cosine distance
    first_idx = torch.randint(N, (1,), device=X.device).item()
    centers = [X[first_idx]]
    for _ in range(1, k):
        cstack = torch.stack(centers)  # (m, D)
        sim = X @ cstack.T  # (N, m)
        min_dist = 1.0 - sim.max(dim=1)[0]  # (N,)
        s = min_dist.sum().item()
        if s < 1e-8:
            # all remaining points coincide with chosen centers
            next_idx = torch.randint(N, (1,), device=X.device).item()
        else:
            probs = min_dist / s
            # numerical guard
            probs = torch.clamp(probs, min=0.0)
            if not torch.isfinite(probs).all():
                next_idx = torch.randint(N, (1,), device=X.device).item()
            else:
                next_idx = torch.multinomial(probs, 1).item()
        centers.append(X[next_idx])
    centers = F.normalize(torch.stack(centers), dim=1)

    labels = None
    for _ in range(max_iter):
        sim = X @ centers.T  # (N, k) cosine similarity
        labels_new = sim.argmax(dim=1)
        if labels is not None and (labels == labels_new).all():
            break
        labels = labels_new
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = F.normalize(X[mask].mean(0, keepdim=True), dim=1).squeeze(0)
            else:
                centers[j] = X[torch.randint(N, (1,), device=X.device).item()]
    return labels, centers


@torch.no_grad()
def extract_features(paths, preprocess, encoder, device, batch_size=128, num_workers=8):
    ds = _FeatureDataset(paths, preprocess)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate,
        pin_memory=(device == "cuda"),
        prefetch_factor=(8 if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
        worker_init_fn=(_worker_init if num_workers > 0 else None),
    )
    encoder.eval()

    all_feats = []
    all_idx = []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        if batch is None:
            continue
        imgs = torch.stack([b[0] for b in batch]).to(device)
        imgs = imgs.to(next(encoder.parameters()).dtype)
        idxs = [b[1] for b in batch]

        f = encoder(imgs)
        # normalise and filter degenerate features on the fly
        norm = f.norm(dim=-1, keepdim=True)
        bad = torch.isnan(f).any(1) | torch.isinf(f).any(1) | (norm.squeeze(1) < 1e-3)
        f = f / (norm + 1e-8)
        f[bad] = 0.0

        all_feats.append(f.float().cpu())
        all_idx.extend(idxs)
        if bi % 20 == 0 or bi == len(loader) - 1:
            print(f"  [encode] batch {bi}/{len(loader)}  "
                  f"kept {len(all_idx)}  elapsed {time.time()-t0:.1f}s", flush=True)

    feats = torch.cat(all_feats, dim=0)
    return feats, all_idx


def discover_prototypes(features, labels, n_classes, k=3, min_cluster_ratio=0.15,
                        outlier_std=2.0, seed=0):
    """Return clean mask, prototypes, confidences and per-class info.

    For each class:
      - run k-means(k)
      - pick the largest cluster as the dominant species
      - within that cluster, drop points whose cosine distance is > mean + outlier_std*std
      - the mean of remaining points is the prototype
    """
    device = features.device
    labels = labels.to(device)
    clean_mask = torch.zeros(features.size(0), dtype=torch.bool, device=device)
    confidences = torch.zeros(features.size(0), dtype=torch.float, device=device)
    prototypes = torch.zeros(n_classes, features.size(1), device=device)
    info = []

    for c in range(n_classes):
        mask_c = labels == c
        idx_c = mask_c.nonzero(as_tuple=True)[0]
        Xc = features[idx_c]
        n_total = Xc.size(0)

        if n_total == 0:
            prototypes[c] = torch.randn(features.size(1), device=device)
            prototypes[c] = F.normalize(prototypes[c].unsqueeze(0), dim=1).squeeze(0)
            info.append({
                "class": c, "folder_id": f"{c:04d}",
                "n_total": 0, "n_clean": 0, "purity": 0.0,
                "k_used": 0, "note": "empty class",
            })
            continue

        # adaptive k
        k_used = min(k, max(2, n_total // 5))
        km_labels, centers = kmeans_torch(Xc, k=k_used, seed=seed + c)

        # dominant cluster = largest
        sizes = [(km_labels == j).sum().item() for j in range(k_used)]
        dom_j = int(max(range(k_used), key=lambda j: sizes[j]))
        dom_mask = (km_labels == dom_j)

        # within dominant cluster, drop far outliers
        dom_feats = Xc[dom_mask]
        dom_idx = idx_c[dom_mask]
        proto = dom_feats.mean(0)
        proto = F.normalize(proto.unsqueeze(0), dim=1).squeeze(0)
        sim = dom_feats @ proto  # cosine similarity
        mean_sim = sim.mean().item()
        std_sim = sim.std().item()
        thr = mean_sim - outlier_std * std_sim
        keep = sim >= thr

        clean_idx = dom_idx[keep]
        clean_mask[clean_idx] = True
        confidences[clean_idx] = sim[keep]

        # recompute prototype from kept samples
        kept_feats = features[clean_idx]
        proto = F.normalize(kept_feats.mean(0).unsqueeze(0), dim=1).squeeze(0)
        prototypes[c] = proto

        # also give non-clean samples a confidence score
        sim_all = Xc @ proto
        confidences[idx_c] = sim_all

        info.append({
            "class": c, "folder_id": f"{c:04d}",
            "n_total": n_total,
            "n_clean": int(keep.sum().item()),
            "purity": float(keep.sum().item() / max(1, n_total)),
            "k_used": k_used,
            "dominant_cluster_size": sizes[dom_j],
            "mean_sim": mean_sim,
            "std_sim": std_sim,
        })

        if c % 50 == 0 or c == n_classes - 1:
            n_clean = int(clean_mask.sum().item())
            print(f"  [proto] class {c}/{n_classes}  "
                  f"clean {n_clean}/{features.size(0)}", flush=True)

    return clean_mask, prototypes, confidences, info


def main():
    p = argparse.ArgumentParser(description="Discover per-class visual prototypes")
    p.add_argument("--data-root", default="/root/datasets/contest")
    p.add_argument("--clean-manifest", default=None,
                   help="optional clean manifest from clean_contest.py; "
                        "if None, scan data-root directly")
    p.add_argument("--clip-weights", default="/root/weights/ViT-B-32.pt")
    p.add_argument("--output-dir", default="output/contest_prototype")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--k", type=int, default=3, help="k-means clusters per class")
    p.add_argument("--outlier-std", type=float, default=2.0,
                   help="drop dominant-cluster samples below mean_sim - outlier_std*std_sim")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # ---- build sample list ----
    if args.clean_manifest:
        with open(args.clean_manifest) as f:
            manifest = json.load(f)
        data_root = args.data_root
        paths = [os.path.join(data_root, rel) for rel, _ in manifest]
        labels = [int(lab) for _, lab in manifest]
        print(f"[proto] loaded clean manifest: {len(paths)} samples")
    else:
        entries = build_train_list(args.data_root)
        paths = [e[0] for e in entries]
        labels = [e[1] for e in entries]
        print(f"[proto] scanned data-root: {len(paths)} samples")

    labels_t = torch.tensor(labels, dtype=torch.long)
    n_classes = int(labels_t.max().item()) + 1
    print(f"[proto] classes: {n_classes}")

    # ---- load CLIP ----
    print(f"[proto] loading CLIP from {args.clip_weights}")
    clip_model = cc.load_clip_to_cpu(args.clip_weights, device="cpu")
    clip_model = clip_model.to(device).float()
    encoder = clip_model.visual
    encoder.eval()

    # ---- extract features ----
    preprocess = _default_transform(args.resolution)
    print(f"[proto] encoding {len(paths)} images ...")
    t0 = time.time()
    feats, kept_idx = extract_features(
        paths, preprocess, encoder, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"[proto] encoded {feats.size(0)} images in {time.time()-t0:.1f}s")

    # align labels with kept features
    labels_kept = labels_t[kept_idx]

    # ---- discover prototypes ----
    print(f"[proto] discovering prototypes (k={args.k}) ...")
    t0 = time.time()
    clean_mask, prototypes, confidences, info = discover_prototypes(
        feats.to(device), labels_kept.to(device), n_classes,
        k=args.k, outlier_std=args.outlier_std, seed=args.seed,
    )
    print(f"[proto] done in {time.time()-t0:.1f}s")

    total = clean_mask.size(0)
    n_clean = int(clean_mask.sum().item())
    print(f"[proto] clean samples: {n_clean}/{total} "
          f"({100.0*n_clean/total:.1f}%)")

    # ---- save artifacts ----
    torch.save(prototypes.cpu(), os.path.join(args.output_dir, "prototypes.pt"))
    torch.save(clean_mask.cpu(), os.path.join(args.output_dir, "clean_mask.pt"))
    torch.save(confidences.cpu(), os.path.join(args.output_dir, "sample_confidence.pt"))
    torch.save(feats.cpu(), os.path.join(args.output_dir, "features.pt"))
    torch.save(labels_kept.cpu(), os.path.join(args.output_dir, "labels.pt"))

    # build a per-sample index map from kept_idx so downstream can map back
    with open(os.path.join(args.output_dir, "kept_idx.json"), "w") as f:
        json.dump(kept_idx, f)

    with open(os.path.join(args.output_dir, "prototype_info.json"), "w") as f:
        json.dump({
            "n_classes": n_classes,
            "feature_dim": feats.size(1),
            "n_total": total,
            "n_clean": n_clean,
            "clean_ratio": n_clean / total,
            "classes": info,
        }, f, indent=2)

    print(f"[proto] artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
