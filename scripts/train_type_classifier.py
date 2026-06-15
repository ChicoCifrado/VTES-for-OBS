#!/usr/bin/env python3
"""
Train a card TYPE classifier (14 VTES types).

MobileNetV3-large → classifier head → 14 type logits.
This replaces the YOLO-based type classifier for more accurate filtering.

Usage:
    python3 scripts/train_type_classifier.py \
        --data /mnt/d/Hermes/vtes/dataset/by_type \
        --vtes-json /mnt/d/Hermes/vtes/vtes.json \
        --output-dir /mnt/c/Users/JackSuicide/VTES/vtes_obs_detect/data \
        --epochs 100 --batch-size 64
"""

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

# ---------------------------------------------------------------------------
# VTES types (14 classes from vtes.json)
# ---------------------------------------------------------------------------

VTES_TYPES = [
    "Action", "Action Modifier", "Ally", "Combat", "Conviction",
    "Equipment", "Event", "Imbued", "Master", "Political Action",
    "Power", "Reaction", "Retainer", "Vampire",
]


def build_card_to_type_map(vtes_json_path: str) -> dict:
    with open(vtes_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def norm(name: str) -> str:
        return re.sub(r'[^a-z0-9]', '', name.lower())

    mapping = {}
    for card in data:
        types = card.get('types', [])
        primary = types[0] if types else "Unknown"
        names = set()
        for field in ['printed_name', 'name', '_name']:
            if field in card and card[field]:
                names.add(card[field])
        for variant in card.get('name_variants', []):
            names.add(variant)
        for name in names:
            n = norm(name)
            if n and n not in mapping:
                mapping[n] = primary
    return mapping


def lookup_type(stem: str, type_map: dict) -> str:
    n = re.sub(r'[^a-z0-9]', '', stem.lower())
    if n in type_map:
        return type_map[n]
    if stem in type_map:
        return type_map[stem]
    return None

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def build_transform(train: bool):
    if not train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomAffine(degrees=20, translate=(0.08, 0.08),
                                scale=(0.80, 1.20), fill=128),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.5, fill=128),
        transforms.ColorJitter(brightness=0.4, contrast=0.3,
                               saturation=0.25, hue=0.1),
        transforms.RandomApply(
            [transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TypeDataset(Dataset):
    def __init__(self, data_root: str, vtes_json: str, train: bool = True):
        self.transform = build_transform(train)
        self.type_names = VTES_TYPES
        self.type_to_idx = {t: i for i, t in enumerate(self.type_names)}

        type_map = build_card_to_type_map(vtes_json)
        self.samples = []  # (img_path, type_idx)

        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        unmatched = 0
        for type_dir in sorted(Path(data_root).iterdir()):
            if not type_dir.is_dir():
                continue
            for img_path in sorted(type_dir.iterdir()):
                if img_path.suffix.lower() not in exts:
                    continue
                card_type = lookup_type(img_path.stem, type_map)
                if card_type is None or card_type not in self.type_to_idx:
                    unmatched += 1
                    continue
                self.samples.append((str(img_path),
                                     self.type_to_idx[card_type]))

        self.num_classes = len(self.type_names)
        counts = {}
        for _, t in self.samples:
            n = self.type_names[t]
            counts[n] = counts.get(n, 0) + 1
        print(f"[Dataset] {len(self.samples)} images, {unmatched} unmatched")
        for t, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TypeClassifier(nn.Module):
    def __init__(self, num_types: int = 14, pretrained: str = None):
        super().__init__()
        self.backbone = mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[0].in_features  # 960
        self.backbone.classifier = nn.Identity()

        if pretrained and os.path.exists(pretrained):
            state = torch.load(pretrained, map_location='cpu')
            if 'model_state_dict' in state:
                s = state['model_state_dict']
            else:
                s = state
            bb_keys = {}
            for k, v in s.items():
                if k.startswith('backbone.'):
                    bb_keys[k.replace('backbone.', '')] = v
                elif k.startswith('features.') or k.startswith('classifier.'):
                    continue
                elif k.startswith('model_state_dict'):
                    continue
                else:
                    bb_keys[k] = v
            missing, _ = self.backbone.load_state_dict(bb_keys, strict=False)
            print(f"[Model] Loaded backbone: {len(missing)} missing")

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_types),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--vtes-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = torch.cuda.is_available() and not args.no_amp
    print(f"[Main] Device: {device} AMP: {amp}")
    os.makedirs(args.output_dir, exist_ok=True)

    ds = TypeDataset(args.data, args.vtes_json, train=True)
    dl = DataLoader(ds, args.batch_size, shuffle=True,
                    num_workers=min(4, os.cpu_count() or 1) if os.name != 'nt' else 0,
                    pin_memory=True, drop_last=True)

    model = TypeClassifier(num_types=ds.num_classes,
                           pretrained=args.checkpoint).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    print(f"[Main] {ds.num_classes} types, {len(ds)} images, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Type weights for class imbalance
    counts = [0] * ds.num_classes
    for _, t in ds.samples:
        counts[t] += 1
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights = weights / weights.sum() * ds.num_classes
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = tot = cor = 0
        t0 = time.time()
        for images, labels in dl:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            _, pred = logits.max(1)
            tot += labels.size(0)
            cor += pred.eq(labels).sum().item()

        scheduler.step()
        acc = 100.0 * cor / tot
        elapsed = time.time() - t0
        eta = time.strftime('%H:%M:%S',
                            time.gmtime((args.epochs - epoch - 1) * elapsed))
        print(f"[Epoch {epoch:3d}/{args.epochs}] loss={total_loss/len(dl):.4f} "
              f"acc={acc:.1f}% lr={optimizer.param_groups[0]['lr']:.2e} "
              f"({elapsed:.0f}s, ETA {eta})")

        ckpt = {'epoch': epoch, 'acc': acc, 'model': model.state_dict()}
        torch.save(ckpt, os.path.join(args.output_dir,
                                      'type_classifier_last.pth'))
        if acc > best_acc:
            best_acc = acc
            torch.save(ckpt, os.path.join(args.output_dir,
                                          'type_classifier_best.pth'))
            print(f"  ** Best: {acc:.1f}%")

    print(f"[Main] Best acc: {best_acc:.1f}%")

    # Export ONNX
    onnx_path = os.path.join(args.output_dir, 'vtes_type_classifier.onnx')
    model.eval().cpu()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=['input'], output_names=['type_logits'],
                      dynamic_axes={'input': {0: 'batch'},
                                    'type_logits': {0: 'batch'}},
                      opset_version=18)
    print(f"[Export] ONNX: {onnx_path}")

    # Save type label mapping
    with open(os.path.join(args.output_dir, 'type_labels.json'), 'w') as f:
        json.dump(VTES_TYPES, f, indent=2)
    print("[Main] Done!")


if __name__ == '__main__':
    train()
