"""Quick inspection of prototype discovery results."""

import os
import json
import argparse

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prototype-dir", default="output/contest_prototype")
    args = p.parse_args()

    with open(os.path.join(args.prototype_dir, "prototype_info.json")) as f:
        info = json.load(f)

    purities = [c["purity"] for c in info["classes"]]
    n_cleans = [c["n_clean"] for c in info["classes"]]
    n_totals = [c["n_total"] for c in info["classes"]]

    print(f"Total classes: {info['n_classes']}")
    print(f"Total samples: {info['n_total']}")
    print(f"Clean samples: {info['n_clean']} ({100*info['clean_ratio']:.1f}%)")
    print()
    print(f"Purity stats:")
    print(f"  mean: {sum(purities)/len(purities):.3f}")
    print(f"  median: {sorted(purities)[len(purities)//2]:.3f}")
    print(f"  min: {min(purities):.3f} (class {purities.index(min(purities)):04d})")
    print(f"  max: {max(purities):.3f} (class {purities.index(max(purities)):04d})")
    print()
    print("Top 10 cleanest classes:")
    for idx in sorted(range(len(purities)), key=lambda i: -purities[i])[:10]:
        c = info["classes"][idx]
        print(f"  {c['folder_id']}: purity={c['purity']:.3f}  "
              f"clean={c['n_clean']}/{c['n_total']}")
    print()
    print("Top 10 noisiest classes:")
    for idx in sorted(range(len(purities)), key=lambda i: purities[i])[:10]:
        c = info["classes"][idx]
        print(f"  {c['folder_id']}: purity={c['purity']:.3f}  "
              f"clean={c['n_clean']}/{c['n_total']}")
    print()
    print("Classes with < 30 clean samples (risk of prototype collapse):")
    low = [i for i, n in enumerate(n_cleans) if n < 30]
    for idx in low[:20]:
        c = info["classes"][idx]
        print(f"  {c['folder_id']}: clean={c['n_clean']}/{c['n_total']}")
    if len(low) > 20:
        print(f"  ... and {len(low)-20} more")


if __name__ == "__main__":
    main()
