"""Step 1 - automatic data cleaning for the contest dataset (NO feature caching).

Why this script exists (see request.md, esp. section 四.(三).1 and the note on line 48):
  * The contest training/test images are web-scraped and contain
      (a) corrupted / truncated files, and
      (b) completely off-distribution images ("papers", images of other species)
          mixed into class folders.
  * PIL can still *open* truncated files (the organisers confirmed this), but such
    files -- and any degenerate image -- may produce invalid CLIP features, so we
      1. open every image with PIL (LOAD_TRUNCATED_IMAGES = True)
       and DROP the ones that fail to decode;
      2. run the frozen CLIP image encoder and DROP images whose features are
         NaN / Inf / near-zero-norm (degenerate).
  * We do NOT pre-compute / cache image features here (per project decision). This
    script only PRODUCES A CLEAN DATASET:
      - bad (decode-fail or degenerate) training images are MOVED to
        <data-root>/_quarantine/ (preserving their relative path) so they are
        physically excluded from training but remain recoverable;
      - a clean manifest listing every KEPT image is written for downstream scripts.
  * The off-distribution "papers / other species" images are READABLE and yield
    valid features, so they are NOT removed here -- they are handled later as
    *open-set label noise* by the outlier-exclusion logic in train_contest.py
    (we EXCLUDE them from the loss rather than re-labelling them to another class,
    because no correct class exists for them inside the 500-way label space).

Outputs (under --output-dir):
  clean_train_manifest.json   list of [relpath_from_data_root, label_int]
  clean_test_manifest.json    list of [relpath_from_data_root, -1]
  train_clean_report.json     counts (total / kept / quarantined / degenerate)
  test_clean_report.json      counts (total / degenerate_in_test)
  meta.json                   classnames, folder_keys, n_classes, clip_weights, resolution

The CLEANING step itself must still run the image encoder (to detect degenerate
features); that encode is used only for detection and is NOT saved to disk.
"""

import os
import json
import time
import argparse
import shutil

import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets.contest import (
    load_contest_classnames,
    build_train_list,
    build_test_list,
    _default_transform,
)
import contest_clip as cc

# Avoid the long cuDNN algorithm search on the first forward (it can take
# minutes on a small MIG and does not help a fixed input size).
torch.backends.cudnn.benchmark = False


class _CleanImageDataset:
    """Yields (tensor, label, idx, path); returns None when an image fails to open."""

    def __init__(self, paths, labels, preprocess):
        self.paths = paths
        self.labels = labels  # list parallel to paths; -1 for unlabelled
        self.pre = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.pre(img), self.labels[i], i, self.paths[i]
        except Exception:
            return None


def _collate(batch):
    batch = [b for b in batch if b is not None]
    return batch if batch else None


def _worker_init(_):
    # cgroup quota is only 4 CPUs; without this each worker spawns many torch
    # threads and they thrash each other (observed 4 img/s instead of ~150).
    torch.set_num_threads(1)


def _scan_split(entries, preprocess, encoder, device, batch_size, num_workers):
    """Encode a split purely for DETECTION. Returns (kept, bad_decode, bad_feature).

    Each of kept / bad_decode / bad_feature is a list of (path, label).
    """
    paths = [e[0] for e in entries]
    labels = [e[1] for e in entries]
    ds = _CleanImageDataset(paths, labels, preprocess)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate, pin_memory=False,
        prefetch_factor=(8 if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
        worker_init_fn=(_worker_init if num_workers > 0 else None),
    )

    encoder.eval()
    dev = next(encoder.parameters()).device
    print(f"  [scan] encoder device = {dev} (expect cuda)", flush=True)

    kept, bad_decode, bad_feature = [], [], []
    total_batches = len(loader)
    t_start = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if batch is None:
                continue
            imgs = torch.stack([b[0] for b in batch]).to(device, non_blocking=True)
            imgs = imgs.to(next(encoder.parameters()).dtype)
            labs = [b[1] for b in batch]
            fpaths = [b[3] for b in batch]

            tb = time.time()
            raw = encoder(imgs)                       # (B, D)
            torch.cuda.synchronize()
            norm = raw.norm(dim=-1)
            bad = torch.isnan(raw).any(1) | torch.isinf(raw).any(1) | (norm < 1e-3)
            good = ~bad

            for j in range(len(batch)):
                if good[j]:
                    kept.append((fpaths[j], labs[j]))
                else:
                    bad_feature.append((fpaths[j], labs[j]))
            gpu_dt = time.time() - tb
            done = len(kept) + len(bad_feature)
            wall = time.time() - t_start
            rate = done / max(1e-6, wall)          # TRUE end-to-end rate (IO + GPU)
            eta = (len(paths) - done) / max(1e-6, rate)
            print(f"  [scan] batch {bi}/{total_batches}  gpu {gpu_dt:5.2f}s  "
                  f"{rate:7.1f} img/s(real)  eta {eta / 60:5.1f}min  "
                  f"kept {len(kept)} bad {len(bad_feature)}",
                  flush=True)

    print(f"  [scan] done in {time.time() - t_start:.1f}s  "
          f"kept {len(kept)} bad_feature {len(bad_feature)}", flush=True)
    return kept, [], bad_feature


def main():
    p = argparse.ArgumentParser(description="Clean contest dataset (no feature caching)")
    p.add_argument("--data-root", default="/root/datasets/contest")
    p.add_argument("--json", default=None,
                   help="optional class-name json; if omitted, folder ids are used as dummy names")
    p.add_argument("--clip-weights", default="/root/weights/ViT-B-32.pt",
                   help="local fast-disk ViT-B/32 (loads in seconds; the ViT-L on the "
                        "slow network disk took ~3min to load and 4x longer to run)")
    p.add_argument("--output-dir", default="output/contest_clean")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8,
                   help="parallel image-decoding workers (cgroup CPU quota is 4, so "
                        "more workers just thrash). Corrupt/truncated images stay "
                        "safe: failures return None, filtered in the collate fn.")
    p.add_argument("--max-train", type=int, default=0,
                   help="if >0, only clean this many training images (dry-run)")
    p.add_argument("--max-test", type=int, default=0,
                   help="if >0, only clean this many test images (dry-run)")
    p.add_argument("--no-quarantine", action="store_true",
                   help="do not move bad images to _quarantine/ (only write manifests)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    preprocess = _default_transform(args.resolution)

    print(f"[clean] loading CLIP from {args.clip_weights}")
    clip_model = cc.load_clip_to_cpu(args.clip_weights, device="cpu")
    clip_model = clip_model.to(device).float()   # this ViT impl forces fp32 internally
    dim = clip_model.visual.output_dim

    if args.json and os.path.exists(args.json):
        classnames, folder_to_idx, keys = load_contest_classnames(args.json)
    else:
        # no class-name json provided: generate dummy names from folder ids
        train_dir = os.path.join(args.data_root, "train")
        keys = sorted([d for d in os.listdir(train_dir)
                       if os.path.isdir(os.path.join(train_dir, d))],
                      key=lambda k: int(k))
        classnames = [f"class {k}" for k in keys]
        folder_to_idx = {k: i for i, k in enumerate(keys)}
        print(f"[clean] no class-name json; using {len(keys)} folder ids as dummy names")

    # ---------------- train ----------------
    train_entries = build_train_list(args.data_root)   # [(path, label)]
    if args.max_train and args.max_train < len(train_entries):
        train_entries = train_entries[: args.max_train]
    print(f"[clean] train images to scan: {len(train_entries)}")
    kept, _, bad_feature = _scan_split(
        train_entries, preprocess, clip_model.visual, device,
        args.batch_size, args.num_workers,
    )

    # decode-fail images are those in train_entries but not in kept/bad_feature
    scanned = set(p for p, _ in kept) | set(p for p, _ in bad_feature)
    bad_decode = [(p, l) for p, l in train_entries if p not in scanned]

    # quarantine bad training images (move, do not delete)
    quarantined = 0
    if not args.no_quarantine:
        qroot = os.path.join(args.data_root, "_quarantine")
        for p, _ in bad_decode + bad_feature:
            rel = os.path.relpath(p, args.data_root)
            dst = os.path.join(qroot, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.move(p, dst)
                quarantined += 1
            except Exception as e:
                print(f"  [warn] cannot quarantine {p}: {e}")

    train_manifest = [[os.path.relpath(p, args.data_root), int(l)] for p, l in kept]
    with open(os.path.join(args.output_dir, "clean_train_manifest.json"), "w") as f:
        json.dump(train_manifest, f)
    train_report = {
        "total": len(train_entries),
        "kept": len(kept),
        "quarantined": quarantined,
        "bad_decode": len(bad_decode),
        "bad_feature": len(bad_feature),
        "quarantine_dir": os.path.join(args.data_root, "_quarantine"),
    }
    with open(os.path.join(args.output_dir, "train_clean_report.json"), "w") as f:
        json.dump(train_report, f, indent=2)
    print(f"[clean] train kept {len(kept)} / {len(train_entries)}  "
          f"quarantined {quarantined} "
          f"(decode {len(bad_decode)} + feature {len(bad_feature)})")

    # ---------------- test ----------------
    test_paths = build_test_list(args.data_root)
    if args.max_test and args.max_test < len(test_paths):
        test_paths = test_paths[: args.max_test]
    print(f"[clean] test images to scan: {len(test_paths)}")
    tkept, _, tbad_feat = _scan_split(
        [(p, -1) for p in test_paths], preprocess, clip_model.visual, device,
        args.batch_size, args.num_workers,
    )
    test_scanned = set(p for p, _ in tkept) | set(p for p, _ in tbad_feat)
    tbad_decode = [p for p in test_paths if p not in test_scanned]
    # test set: keep ALL images (a prediction row is required for every test image);
    # degenerate test images are recorded but NOT quarantined.
    test_manifest = [[os.path.relpath(p, args.data_root), -1] for p in test_paths]
    with open(os.path.join(args.output_dir, "clean_test_manifest.json"), "w") as f:
        json.dump(test_manifest, f)
    test_report = {
        "total": len(test_paths),
        "kept": len(test_manifest),
        "bad_decode": len(tbad_decode),
        "bad_feature": len(tbad_feat),
        "note": "all test images kept for submission; degenerates handled at inference",
    }
    with open(os.path.join(args.output_dir, "test_clean_report.json"), "w") as f:
        json.dump(test_report, f, indent=2)
    print(f"[clean] test kept all {len(test_manifest)} (degenerate "
          f"decode {len(tbad_decode)} + feature {len(tbad_feat)} recorded)")

    # ---------------- meta ----------------
    meta = {
        "data_root": args.data_root,
        "clip_weights": args.clip_weights,
        "resolution": args.resolution,
        "feature_dim": int(dim),
        "n_classes": len(classnames),
        "classnames": classnames,
        "folder_keys": keys,
        "quarantine_dir": os.path.join(args.data_root, "_quarantine"),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[clean] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
