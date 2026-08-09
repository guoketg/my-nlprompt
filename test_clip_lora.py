"""Inference with the LoRA-fine-tuned CLIP ViT-B/32 (stage 2b).

Loads output/contest_ft_lora/best.pt and produces the submission file:
  output/contest_ft_lora/pred_results.csv  (+ .zip)
Format (per format_request.md): NO header, ``<filename>,<4-digit label>``.

Run:
  python test_clip_lora.py --data-root /root/datasets/contest \
      --ckpt output/contest_ft_lora/best.pt
"""

import os
import argparse
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets.contest import build_test_list
import contest_clip as cc
import train_clip_lora as tl  # reuse LoRALinear / inject_lora / CLIPLoRAClassifier

NORM = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                   std=(0.26862954, 0.26130258, 0.27577711))


def tta_transforms(size):
    return [
        T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                   T.CenterCrop(size), T.ToTensor(), NORM]),
        T.Compose([T.Resize(int(size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
                   T.RandomCrop(size), T.ToTensor(), NORM]),
        T.Compose([T.Resize(int(size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
                   T.RandomCrop(size), T.RandomHorizontalFlip(), T.ToTensor(), NORM]),
        T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                   T.RandomCrop(size), T.ToTensor(), NORM]),
        T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                   T.RandomCrop(size), T.RandomHorizontalFlip(), T.ToTensor(), NORM]),
    ]


class _TestTTA(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def predict(ckpt, data_root, resolution, batch_size, num_workers, amp):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = cc.load_clip_to_cpu("/root/weights/ViT-B-32.pt", device="cpu").float()
    model = tl.CLIPLoRAClassifier(clip_model, 500).to(device)
    tl.freeze_all(model.visual)
    tl.inject_lora(model.visual, ["out_proj", "c_fc", "c_proj"], 8, 16.0, 0.05)
    model.to(device)  # move freshly-created LoRA params to device
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    test_paths = build_test_list(data_root)
    basenames = [os.path.basename(p) for p in test_paths]
    tfms = tta_transforms(resolution)
    all_logits = np.zeros((len(test_paths), 500), dtype=np.float32)
    for t in tfms:
        ds = _TestTTA(test_paths, t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        start = 0
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.startswith("cuda")):
                logits = model(imgs)
            all_logits[start:start + len(imgs)] += logits.detach().cpu().numpy().astype(np.float32)
            start += len(imgs)
    preds = all_logits.argmax(1)
    return basenames, preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/root/datasets/contest")
    p.add_argument("--ckpt", default="output/contest_ft_lora/best.pt")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    args = p.parse_args()
    out_dir = args.output_dir or os.path.dirname(args.ckpt)
    os.makedirs(out_dir, exist_ok=True)

    names, preds = predict(args.ckpt, args.data_root, args.resolution,
                           args.batch_size, args.num_workers, args.amp)

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


if __name__ == "__main__":
    main()
