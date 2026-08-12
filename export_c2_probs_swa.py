"""Export TTA probs for all C2-line checkpoints and build SWA fusions.

Step 1: for each ckpt in output/contest_ft_lora_c2/{round1..5,best}.pt, run the
        same 11-view TTA inference and dump pred_probs.npy (reuses test_clip_lora.predict).
Step 2: average a chosen subset of those probs (equal or alpha-weighted) and write
        a submission package.

This is single-model / single-pipeline checkpoint averaging -> compliant with the
contest prohibition on multi-model ensembling (see HANDOVER §11.5/§11.7).

Run:
  python export_c2_probs_swa.py --export          # export all probs (GPU, ~45min each)
  python export_c2_probs_swa.py --fuse all        # equal-weight average of all 6
  python export_c2_probs_swa.py --fuse best5 --alpha 0.4   # best weighted more
"""
import os
import argparse
import zipfile
import csv
import numpy as np

import test_clip_lora as tt

C2_DIR = "output/contest_ft_lora_c2"
CKPTS = ["round1.pt", "round2.pt", "round3.pt", "round4.pt", "round5.pt", "best.pt"]


def export_all(data_root, resolution, batch_size, num_workers):
    names = None
    for ck in CKPTS:
        ckpt = os.path.join(C2_DIR, ck)
        if not os.path.isfile(ckpt):
            print(f"[skip] missing {ckpt}")
            continue
        out_dir = os.path.join(C2_DIR, "probs_" + ck.replace(".pt", ""))
        prob_path = os.path.join(out_dir, "pred_probs.npy")
        if os.path.isfile(prob_path):
            print(f"[skip] {prob_path} exists")
            continue
        os.makedirs(out_dir, exist_ok=True)
        nms, preds, logits = tt.predict(
            ckpt, data_root, resolution, batch_size, num_workers, True, tta_enhanced=True)
        probs = tt.softmax(logits, axis=1)
        np.save(prob_path, probs)
        names = nms
        print(f"[export] {ck} -> {prob_path} shape={probs.shape}")
    # save names once (aligned across all ckpts)
    if names is not None:
        with open(os.path.join(C2_DIR, "test_names.txt"), "w") as f:
            for n in names:
                f.write(n + "\n")
        print(f"[names] saved {len(names)} test names")


def _load_names():
    p = os.path.join(C2_DIR, "test_names.txt")
    if not os.path.isfile(p):
        # fallback: reuse any existing pred_results.csv order
        for ck in CKPTS:
            cand = os.path.join(C2_DIR, "probs_" + ck.replace(".pt", ""), "pred_probs.npy")
            if os.path.isfile(cand):
                pass
        raise SystemExit("test_names.txt missing; run --export first")
    with open(p) as f:
        return [l.strip() for l in f if l.strip()]


def fuse(mode, alpha):
    names = _load_names()
    n = len(names)
    mats = []
    used = []
    for ck in CKPTS:
        pp = os.path.join(C2_DIR, "probs_" + ck.replace(".pt", ""), "pred_probs.npy")
        if not os.path.isfile(pp):
            print(f"[skip] no probs for {ck}")
            continue
        m = np.load(pp)
        assert m.shape[0] == n, f"row mismatch {ck}: {m.shape[0]} vs {n}"
        mats.append(m)
        used.append(ck)
    print(f"[fuse] using {used}")
    if mode == "all":
        fused = np.mean(mats, axis=0)
        tag = "swa_all6"
    elif mode == "best5":
        # best gets alpha, the other 5 share (1-alpha)
        w = np.ones(len(mats)) * ((1 - alpha) / (len(mats) - 1))
        w[used.index("best.pt")] = alpha
        fused = np.sum(wi * mi for wi, mi in zip(w, mats))
        tag = f"swa_best{int(alpha*100)}"
    else:
        raise SystemExit(f"unknown mode {mode}")
    preds = fused.argmax(1)
    out_dir = os.path.join("output", "fuse_" + tag)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pred_results.csv")
    with open(csv_path, "w") as f:
        for name, pred in zip(names, preds):
            f.write(f"{name},{pred:04d}\n")
    zp = os.path.join(out_dir, "pred_results.zip")
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="pred_results.csv")
    print(f"[done] wrote {csv_path} ({len(preds)} rows) + {zp}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true", help="export TTA probs for all ckpts")
    p.add_argument("--fuse", choices=["all", "best5"], default=None,
                   help="build SWA fusion package")
    p.add_argument("--alpha", type=float, default=0.4,
                   help="weight for best.pt in best5 mode")
    p.add_argument("--data-root", default="/root/datasets/contest")
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args()
    if a.export:
        export_all(a.data_root, a.resolution, a.batch_size, a.num_workers)
    if a.fuse:
        fuse(a.fuse, a.alpha)
    if not a.export and not a.fuse:
        print("specify --export and/or --fuse")


if __name__ == "__main__":
    main()
