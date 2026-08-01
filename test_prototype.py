"""Stage 4 - inference on the unlabelled contest test set using the visual classifier.

Loads:
  * frozen CLIP ViT-B/32 visual encoder
  * the trained VisualClassifier checkpoint (best.pt / ema_final.pt)
  * the clean test manifest from clean_contest.py

Outputs:
  <output-dir>/pred_results.csv
  <output-dir>/pred_results.zip   (contest submission)
  <output-dir>/contest_predictions.json

Optional TTA: --tta crops.
"""

import os
import json
import argparse
import zipfile
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets.contest import _default_transform
import contest_clip as cc


torch.backends.cudnn.benchmark = False


class _TestDataset(Dataset):
    def __init__(self, paths, transform, tta=False):
        self.paths = paths
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            if self.tta:
                # return a list of tensors for TTA
                views = [self.transform(img)]
                views.append(self.transform(img.transpose(Image.FLIP_LEFT_RIGHT)))
                return views, i
            return self.transform(img), i
        except Exception:
            return None


def _collate(batch):
    batch = [b for b in batch if b is not None]
    return batch if batch else None


def _worker_init(_):
    torch.set_num_threads(1)


class VisualClassifier(torch.nn.Module):
    def __init__(self, n_classes=500, dim=512, temp=0.01):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n_classes, dim))
        self.bias = torch.nn.Parameter(torch.zeros(n_classes))
        self.temp = temp

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return (x @ w.T) / self.temp + self.bias


@torch.no_grad()
def encode_test(paths, preprocess, encoder, device, batch_size, num_workers, tta=False):
    ds = _TestDataset(paths, preprocess, tta=tta)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate,
        pin_memory=False,
        prefetch_factor=(8 if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
        worker_init_fn=(_worker_init if num_workers > 0 else None),
    )
    encoder.eval()
    dtype = next(encoder.parameters()).dtype

    all_feats, kept_idx = [], []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        if batch is None:
            continue
        if tta:
            # batch is list of (views, idx)
            flat_imgs, flat_idx = [], []
            for views, idx in batch:
                flat_imgs.extend(views)
                flat_idx.extend([idx] * len(views))
            imgs = torch.stack(flat_imgs).to(device).to(dtype)
            raw = encoder(imgs)
            norm = raw.norm(dim=-1, keepdim=True)
            bad = torch.isnan(raw).any(1) | torch.isinf(raw).any(1) | (norm.squeeze(1) < 1e-3)
            raw = raw / (norm + 1e-8)
            raw[bad] = 0.0
            # average views per image
            B = len(batch)
            views_per = len(flat_imgs) // B
            raw = raw.view(B, views_per, -1).mean(1)
            idxs = [b[1] for b in batch]
        else:
            imgs = torch.stack([b[0] for b in batch]).to(device).to(dtype)
            raw = encoder(imgs)
            norm = raw.norm(dim=-1, keepdim=True)
            bad = torch.isnan(raw).any(1) | torch.isinf(raw).any(1) | (norm.squeeze(1) < 1e-3)
            raw = raw / (norm + 1e-8)
            raw[bad] = 0.0
            idxs = [b[1] for b in batch]

        all_feats.append(raw.float().cpu())
        kept_idx.extend(idxs)
        if bi % 20 == 0 or bi == len(loader) - 1:
            print(f"  [encode] batch {bi}/{len(loader)}  kept {len(kept_idx)}  "
                  f"elapsed {time.time()-t0:.1f}s", flush=True)

    feats = torch.cat(all_feats, dim=0)
    # rebuild full ordered tensor, filling missing with mean
    out = feats.mean(0).unsqueeze(0).repeat(len(paths), 1).clone()
    for k, i in enumerate(kept_idx):
        out[i] = feats[k]
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Infer on contest test set with visual classifier")
    p.add_argument("--checkpoint", required=True, help="best.pt or ema_final.pt from train_prototype.py")
    p.add_argument("--data-root", default="/root/datasets/contest")
    p.add_argument("--clean-test-manifest", default="output/contest_clean/clean_test_manifest.json")
    p.add_argument("--clip-weights", default="/root/weights/ViT-B-32.pt")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--encode-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tta", action="store_true", help="enable test-time augmentation")
    p.add_argument("--resolution", type=int, default=224)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    output_dir = args.output_dir or ckpt_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(args.clean_test_manifest) as f:
        test_manifest = json.load(f)
    paths = [os.path.join(args.data_root, rel) for rel, _ in test_manifest]
    names = [rel for rel, _ in test_manifest]

    # ---- load CLIP ----
    print(f"[test] loading CLIP from {args.clip_weights}")
    clip_model = cc.load_clip_to_cpu(args.clip_weights, device="cpu")
    clip_model = clip_model.to(device).float()
    preprocess = _default_transform(args.resolution)

    print(f"[test] encoding {len(paths)} test images (tta={args.tta}) ...")
    feats = encode_test(paths, preprocess, clip_model.visual, device,
                        args.encode_batch_size, args.num_workers, tta=args.tta)
    print(f"[test] test features {tuple(feats.shape)}")

    # ---- load classifier ----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    n_classes = ckpt["weight"].size(0)
    dim = ckpt["weight"].size(1)
    model = VisualClassifier(n_classes=n_classes, dim=dim, temp=ckpt.get("temp", 0.01))
    model.weight.data = F.normalize(ckpt["weight"].float(), dim=1)
    if "bias" in ckpt:
        model.bias.data = ckpt["bias"].float()
    model = model.to(device).float()
    model.eval()

    # ---- inference ----
    all_probs = []
    for i in range(0, feats.size(0), args.batch_size):
        x = feats[i:i + args.batch_size].to(device)
        logits = model(x)
        probs = F.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu())
    all_probs = torch.cat(all_probs, dim=0)
    preds_idx = all_probs.argmax(dim=1).tolist()

    # folder ids are just 4-digit class indices
    with open(os.path.join(output_dir, "pred_results.csv"), "w") as fc:
        fc.write("filename,label\n")
        for name, pred in zip(names, preds_idx):
            fc.write(f"{name},{pred:04d}\n")

    with zipfile.ZipFile(os.path.join(output_dir, "pred_results.zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(output_dir, "pred_results.csv"), arcname="pred_results.csv")

    pred_dict = {name: {"pred": pred, "label": f"{pred:04d}"} for name, pred in zip(names, preds_idx)}
    with open(os.path.join(output_dir, "contest_predictions.json"), "w") as f:
        json.dump(pred_dict, f, indent=2)

    dist = Counter(preds_idx)
    print(f"[test] wrote {len(preds_idx)} predictions to {output_dir}")
    print(f"[test] predicted classes used: {len(dist)}/{n_classes}")


if __name__ == "__main__":
    main()
