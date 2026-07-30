"""Step 3 - inference on the unlabelled contest test set (no feature-cache files).

Loads the CLEAN test manifest (output/contest_clean/clean_test_manifest.json) which
lists every test image, encodes them ONCE into memory (nothing written to disk), then
runs a prompt learner trained by train_contest.py and writes the submission files:

  <output-dir>/pred_results.csv      filename,label   (label = 4-digit class id)
  <output-dir>/pred_results.zip      zipped pred_results.csv (contest submission)
  <output-dir>/contest_predictions.json / .name.csv   (for inspection)

Run:
  source .venv/bin/activate
  python test_contest.py --checkpoint output/contest/prompt_learner_best.pt \
      --data-root /root/datasets/contest \
      --json all_class_predictions.json \
      --clean-test-manifest output/contest_clean/clean_test_manifest.json \
      --output-dir output/contest
"""

import os
import json
import time
import argparse
import zipfile
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets.contest import load_contest_classnames, _default_transform
import contest_clip as cc

torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Infer on the unlabelled contest test set")
    p.add_argument("--data-root", default="/root/datasets/contest")
    p.add_argument("--json", default="all_class_predictions.json")
    p.add_argument("--clean-test-manifest",
                   default="output/contest_clean/clean_test_manifest.json")
    p.add_argument("--checkpoint", required=True,
                   help="prompt_learner*.pt produced by train_contest.py")
    p.add_argument("--meta", default=None,
                   help="train meta.json (defaults to <checkpoint-dir>/meta.json)")
    p.add_argument("--clip-weights", default=None,
                   help="CLIP weights path; defaults to meta value")
    p.add_argument("--output-dir", default=None,
                   help="defaults to the checkpoint's directory")
    p.add_argument("--encode-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


@torch.no_grad()
def encode_test(manifest, data_root, preprocess, encoder, device, batch_size, num_workers):
    paths = [os.path.join(data_root, rel) for rel, _ in manifest]

    class _Ds(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            try:
                img = Image.open(paths[i]).convert("RGB")
                return preprocess(img), i
            except Exception:
                return None
    ds = _Ds()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=lambda b: [x for x in b if x],
                        pin_memory=False)
    encoder.eval()
    dev = next(encoder.parameters()).device
    print(f"[encode] encoder device = {dev} (expect cuda)", flush=True)
    feats, kept_idx = [], []
    mean_feat = None
    print(f"[encode] in-memory encoding {len(paths)} test images ...", flush=True)
    t0 = time.time()
    for batch in loader:
        if not batch:
            continue
        imgs = torch.stack([b[0] for b in batch]).to(device, non_blocking=True)
        imgs = imgs.to(next(encoder.parameters()).dtype)
        idxs = [b[1] for b in batch]
        raw = encoder(imgs)
        torch.cuda.synchronize()
        for j in range(len(batch)):
            f = raw[j]
            if torch.isnan(f).any() or torch.isinf(f).any() or f.norm() < 1e-3:
                continue  # degenerate -> fill with mean later
            feats.append(f / f.norm())
            kept_idx.append(idxs[j])
    F_tensor = torch.stack(feats).float().cpu() if feats else torch.zeros(1, encoder.output_dim)
    mean_feat = F_tensor.mean(dim=0)
    # rebuild full ordered tensor (fill missing with mean feature)
    out = mean_feat.unsqueeze(0).repeat(len(paths), 1).clone()
    for k, i in enumerate(kept_idx):
        out[i] = F_tensor[k]
    print(f"[encode] done: {len(paths)} test images in {time.time() - t0:.1f}s "
          f"({len(kept_idx)} valid, {len(paths) - len(kept_idx)} filled with mean)")
    return out


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    meta_path = args.meta or os.path.join(ckpt_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    classnames = meta["classnames"]
    folder_keys = meta["folder_keys"]
    n_ctx = meta.get("n_ctx", 16)
    ctx_init = meta.get("ctx_init", "a photo of a")
    csc = meta.get("csc", False)
    ctp = meta.get("class_token_position", "end")
    clip_weights = args.clip_weights or meta.get("clip_weights")
    assert clip_weights, "CLIP weights path missing (pass --clip-weights)"

    output_dir = args.output_dir or ckpt_dir
    os.makedirs(output_dir, exist_ok=True)

    # ----- test features (encoded in memory, all images kept) -----
    with open(args.clean_test_manifest) as f:
        test_manifest = json.load(f)
    preprocess = _default_transform(meta.get("resolution", 336))
    print(f"[model] loading CLIP from {clip_weights}")
    clip_model = cc.load_clip_to_cpu(clip_weights, device="cpu")
    clip_model = clip_model.to(device).float()
    feats = encode_test(
        test_manifest, args.data_root, preprocess, clip_model.visual, device,
        args.encode_batch_size, args.num_workers,
    )
    names = [entry[0] for entry in test_manifest]   # basename / relpath
    print(f"[data] test features {tuple(feats.shape)}, files {len(names)}")

    # ----- model -----
    model = cc.CustomCLIP(classnames, clip_model, n_ctx, ctx_init, csc, ctp)
    model = model.to(device).float()
    model.dtype = torch.float32
    model.text_encoder.dtype = torch.float32
    model.prompt_learner.dtype = torch.float32
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.prompt_learner.load_state_dict(sd)
    model.eval()

    tokenized = model.prompt_learner.tokenized_prompts.to(device)
    logit_scale = model.logit_scale.exp().float()

    # ----- inference -----
    all_probs = []
    for i in range(0, feats.shape[0], args.batch_size):
        x = feats[i:i + args.batch_size].to(device)
        tf = model.text_encoder(model.prompt_learner(), tokenized).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
        logits = logit_scale * x @ tf.t()
        probs = F.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu())
    all_probs = torch.cat(all_probs, dim=0)
    topk_vals, topk_idx = all_probs.topk(args.topk, dim=1)

    preds = {}
    for j, name in enumerate(names):
        pred_idx = int(topk_idx[j][0].item())
        preds[name] = {
            "pred": pred_idx,
            "label": folder_keys[pred_idx],
            "class_name": classnames[pred_idx],
            "top5": [classnames[int(i)] for i in topk_idx[j].tolist()],
        }

    # ----- write outputs -----
    json_path = os.path.join(output_dir, "contest_predictions.json")
    with open(json_path, "w") as f:
        json.dump(preds, f, indent=2)

    name_csv_path = os.path.join(output_dir, "contest_predictions_name.csv")
    with open(name_csv_path, "w") as fn:
        fn.write("filename,label,class_name\n")
        for name, d in preds.items():
            fn.write(f"{name},{d['label']},{d['class_name']}\n")

    csv_path = os.path.join(output_dir, "pred_results.csv")
    with open(csv_path, "w") as fc:
        fc.write("filename,label\n")
        for name, d in preds.items():
            fc.write(f"{name},{d['label']}\n")

    zip_path = os.path.join(output_dir, "pred_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="pred_results.csv")

    print(f"[done] {len(preds)} test predictions written to {output_dir}")
    print(f"  - {csv_path}")
    print(f"  - {zip_path}  (submit this)")
    print(f"  - {json_path}")
    dist = Counter(d["pred"] for d in preds.values())
    print(f"[sanity] predicted classes used: {len(dist)}/{len(classnames)}")


if __name__ == "__main__":
    main()
