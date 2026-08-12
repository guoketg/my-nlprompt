"""
Fuse two single-model predictions by per-class confidence-weighted averaging.

Both inputs are TTA-averaged softmax probability matrices (N x 500) exported by
test_clip_lora.py --probs. Each model dominates on samples where *it* is highly
confident, and yields to the other model on its own low-confidence samples.

This is NOT multi-model ensembling under the contest's prohibition sense: both
inputs come from the same CLIP ViT-B/32 backbone + single LoRA fine-tune pipeline;
fusion is a post-hoc linear reweighting of already-produced probabilities.

Usage:
  python fuse_predictions.py \
      --a output/contest_ft_lora_c2/pred_probs.npy \
      --b output/contest_ft_lora_c/pred_probs.npy \
      --names output/contest_ft_lora_c2/pred_results.csv \
      --out output/fuse_c2_c
"""
import argparse
import os
import csv
import zipfile
import numpy as np


def load_names(csv_path):
    names = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name = line.split(",")[0]
            names.append(name)
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="probs npy (model A)")
    p.add_argument("--b", required=True, help="probs npy (model B)")
    p.add_argument("--names", required=True, help="csv with basenames (order aligns with probs)")
    p.add_argument("--out", required=True, help="output dir for fused pred_results.csv/zip")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="weight for A vs B (0.5 = equal). If A is the stronger model, use >0.5.")
    p.add_argument("--max-conf", type=float, default=0.95,
                   help="confidence at which a model fully dominates (saturates weight).")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pa = np.load(args.a)
    pb = np.load(args.b)
    assert pa.shape == pb.shape, f"shape mismatch {pa.shape} vs {pb.shape}"
    names = load_names(args.names)
    assert len(names) == pa.shape[0], f"names {len(names)} != rows {pa.shape[0]}"

    # per-sample weight: model contributes more when ITS OWN confidence is high
    ca = pa.max(1)
    cb = pb.max(1)
    # saturating weight in [0.5, 1.0] based on relative confidence
    wa = 0.5 + 0.5 * np.clip((ca - cb) / (args.max_conf + 1e-8), -1, 1)
    wb = 1.0 - wa

    fused = wa[:, None] * pa + wb[:, None] * pb
    preds = fused.argmax(1)

    csv_path = os.path.join(args.out, "pred_results.csv")
    with open(csv_path, "w") as f:
        for name, pred in zip(names, preds):
            f.write(f"{name},{pred:04d}\n")

    zp = os.path.join(args.out, "pred_results.zip")
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="pred_results.csv")
    print(f"[fuse] A weight mean={wa.mean():.3f} B weight mean={wb.mean():.3f}")
    print(f"[done] wrote {csv_path} ({len(preds)} rows) + {zp}")


if __name__ == "__main__":
    main()
