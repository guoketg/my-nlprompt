"""Stage 3 - iterative self-training to recover more clean samples.

Loads a trained VisualClassifier and the full feature bank, then expands the
clean mask by adding samples where:
  * the model's prediction matches the given folder label, AND
  * the model's confidence is above a threshold.

The expanded clean mask can be fed back into train_prototype.py for another
round of training (warm-start from the previous checkpoint).
"""

import os
import json
import argparse

import torch
import torch.nn.functional as F


def load_classifier(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    n_classes = ckpt["weight"].size(0)
    dim = ckpt["weight"].size(1)
    temp = ckpt.get("temp", 0.01)

    class VisualClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(n_classes, dim))
            self.bias = torch.nn.Parameter(torch.zeros(n_classes))
            self.temp = temp

        def forward(self, x):
            x = F.normalize(x, dim=1)
            w = F.normalize(self.weight, dim=1)
            return (x @ w.T) / self.temp + self.bias

    model = VisualClassifier()
    model.weight.data = F.normalize(ckpt["weight"].float(), dim=1)
    if "bias" in ckpt:
        model.bias.data = ckpt["bias"].float()
    return model


def main():
    p = argparse.ArgumentParser(description="Expand clean mask via self-training")
    p.add_argument("--prototype-dir", default="output/contest_prototype")
    p.add_argument("--checkpoint", required=True, help="trained best.pt or ema_final.pt")
    p.add_argument("--output-dir", default="output/contest_prototype_selftrain")
    p.add_argument("--conf-thr", type=float, default=0.9,
                   help="minimum softmax confidence to accept pseudo-label")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- load stage 1 artifacts ----
    features = torch.load(os.path.join(args.prototype_dir, "features.pt"), weights_only=True)
    labels = torch.load(os.path.join(args.prototype_dir, "labels.pt"), weights_only=True)
    clean_mask = torch.load(os.path.join(args.prototype_dir, "clean_mask.pt"), weights_only=True)
    confidences = torch.load(os.path.join(args.prototype_dir, "sample_confidence.pt"), weights_only=True)
    prototypes = torch.load(os.path.join(args.prototype_dir, "prototypes.pt"), weights_only=True)

    # ---- load model and score all samples ----
    model = load_classifier(args.checkpoint).to(device).float().eval()

    all_probs = []
    batch_size = 8192
    with torch.no_grad():
        for i in range(0, features.size(0), batch_size):
            x = features[i:i + batch_size].to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
    all_probs = torch.cat(all_probs, dim=0)
    pred_labels = all_probs.argmax(dim=1)
    pred_conf = all_probs.max(dim=1)[0]

    # ---- expand clean mask ----
    correct = (pred_labels == labels)
    high_conf = pred_conf >= args.conf_thr
    recovered = correct & high_conf & (~clean_mask)

    new_clean_mask = clean_mask.clone()
    new_clean_mask[recovered] = True

    # update confidences with model confidence for recovered samples
    new_confidences = confidences.clone()
    new_confidences[recovered] = pred_conf[recovered]

    # recompute prototypes with expanded set
    n_classes = int(labels.max().item()) + 1
    new_prototypes = torch.zeros_like(prototypes)
    for c in range(n_classes):
        mask = (labels == c) & new_clean_mask
        if mask.any():
            new_prototypes[c] = F.normalize(features[mask].mean(0, keepdim=True), dim=1).squeeze(0)
        else:
            new_prototypes[c] = prototypes[c]

    n_orig = int(clean_mask.sum().item())
    n_new = int(new_clean_mask.sum().item())
    n_recovered = int(recovered.sum().item())

    print(f"[selftrain] original clean: {n_orig}")
    print(f"[selftrain] recovered:      {n_recovered}")
    print(f"[selftrain] new clean:      {n_new}/{len(labels)} ({100.0*n_new/len(labels):.1f}%)")

    # ---- save ----
    torch.save(new_clean_mask, os.path.join(args.output_dir, "clean_mask.pt"))
    torch.save(new_confidences, os.path.join(args.output_dir, "sample_confidence.pt"))
    torch.save(new_prototypes, os.path.join(args.output_dir, "prototypes.pt"))
    # symlink/copy the unchanged features/labels for convenience
    torch.save(features, os.path.join(args.output_dir, "features.pt"))
    torch.save(labels, os.path.join(args.output_dir, "labels.pt"))

    with open(os.path.join(args.output_dir, "selftrain_info.json"), "w") as f:
        json.dump({
            "prototype_dir": args.prototype_dir,
            "checkpoint": args.checkpoint,
            "conf_thr": args.conf_thr,
            "n_total": len(labels),
            "n_original_clean": n_orig,
            "n_recovered": n_recovered,
            "n_new_clean": n_new,
            "new_clean_ratio": n_new / len(labels),
        }, f, indent=2)

    print(f"[selftrain] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
