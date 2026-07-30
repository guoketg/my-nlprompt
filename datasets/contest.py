"""Standalone dataset definition for the "contest" dataset.

This dataset is a REAL noisy dataset:
  * The training images live in ``/root/datasets/contest/train/<class_id>/images``
    where ``<class_id>`` is a 4-digit folder id (0000..0499) and is also the
    ground-truth *class index* (0..499).
  * The human/auto generated class NAMES (used as CLIP text prompts) are stored in
    ``all_class_predictions.json`` and come from heterogeneous, partly noisy sources
    (``ground_truth``, ``kimi_vision``, ``qwen_api``, ``clip_high_conf``,
    ``clip_analysis``).  This is the main source of label noise.
  * The test set ``/root/datasets/contest/test/*.jpg`` is UNLABELLED (random hex
    filenames), so it can only be used for inference / submission.

The module is intentionally dependency-free (no Dassl) so it can be imported by the
new standalone train/test scripts.
"""

import os
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_contest_classnames(json_path):
    """Return (classnames, folder_to_idx, keys).

    ``classnames[i]`` is the class name for class index ``i`` (i.e. folder id
    ``keys[i]``).  The canonical class index is the numeric order of the folder id.
    """
    with open(json_path) as f:
        J = json.load(f)
    keys = sorted(J.keys(), key=lambda k: int(k))
    classnames = [J[k]["prediction"] for k in keys]
    folder_to_idx = {k: i for i, k in enumerate(keys)}
    return classnames, folder_to_idx, keys


def _default_transform(resolution):
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=Image.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def build_train_list(data_root):
    """Scan ``train/<folder>/images`` and return a list of (path, label).

    ``label`` is the integer folder id (== canonical class index)."""
    train_dir = os.path.join(data_root, "train")
    entries = []
    for folder in sorted(os.listdir(train_dir), key=lambda x: int(x)):
        folder_path = os.path.join(train_dir, folder, "images")
        if not os.path.isdir(folder_path):
            folder_path = os.path.join(train_dir, folder)
        label = int(folder)
        for fn in os.listdir(folder_path):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                entries.append((os.path.join(folder_path, fn), label))
    return entries


def build_test_list(data_root):
    """Return list of absolute image paths in the (unlabelled) test set."""
    test_dir = os.path.join(data_root, "test")
    paths = []
    for fn in os.listdir(test_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            paths.append(os.path.join(test_dir, fn))
    paths.sort()
    return paths


class ContestTrainDataset(Dataset):
    """Returns ``(image, label, index)``; the trailing ``index`` is the global
    sample index and is used by the confidence-based sample-selection routine."""

    def __init__(self, data_root, transform=None, resolution=336):
        self.entries = build_train_list(data_root)
        self.transform = transform or _default_transform(resolution)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, idx


class ContestTestDataset(Dataset):
    """Returns ``(image, basename)`` for the unlabelled test set."""

    def __init__(self, data_root, transform=None, resolution=336):
        self.paths = build_test_list(data_root)
        self.transform = transform or _default_transform(resolution)
        self.basename = [os.path.basename(p) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.basename[idx]
