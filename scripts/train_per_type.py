#!/usr/bin/env python3
"""
Train per-type ArcFace embedding models.

For each VTES card type, train a separate MobileNetV3-small ArcFace model.
Each model only needs to discriminate ~200-1700 cards instead of 4156.

Usage:
    python3 scripts/train_per_type.py \
        --data /mnt/d/Hermes/vtes/dataset/by_type \
        --vtes-json /mnt/d/Hermes/vtes/vtes.json \
        --output-dir /mnt/c/Users/JackSuicide/VTES/vtes_obs_detect/data/per_type \
        --epochs 200 --batch-size 64
"""

import argparse
import json
import os
import sys
import time
import re
import subprocess
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

VTES_TYPES = [
    "Action", "Action Modifier", "Ally", "Combat", "Conviction",
    "Equipment", "Event", "Imbued", "Master", "Political Action",
    "Power", "Reaction", "Retainer", "Vampire",
]


def build_card_to_type_map(vtes_json_path):
    with open(vtes_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def norm(name):
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


def lookup_type(stem, type_map):
    n = re.sub(r'[^a-z0-9]', '', stem.lower())
    if n in type_map:
        return type_map[n]
    if stem in type_map:
        return type_map[stem]
    return None

# ---------------------------------------------------------------------------
# Augmentation (webcam-realistic)
# ---------------------------------------------------------------------------

def build_transform(train=True):
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
        transforms.RandomResizedCrop(224, scale=(0.80, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomAffine(degrees=25, translate=(0.10, 0.10),
                                scale=(0.75, 1.25), fill=128),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5, fill=128),
        transforms.ColorJitter(brightness=0.5, contrast=0.4,
                               saturation=0.3, hue=0.15),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ---------------------------------------------------------------------------
# Dataset per type
# ---------------------------------------------------------------------------

class PerTypeDataset(Dataset):
    def __init__(self, data_root, vtes_json, card_type, train=True):
        self.transform = build_transform(train)
        type_map = build_card_to_type_map(vtes_json)

        self.samples = []  # (img_path, class_label)
        self.card_names = []

        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        label = 0
        for type_dir in sorted(Path(data_root).iterdir()):
            if not type_dir.is_dir():
                continue
            for img_path in sorted(type_dir.iterdir()):
                if img_path.suffix.lower() not in exts:
                    continue
                t = lookup_type(img_path.stem, type_map)
                if t is None or t != card_type:
                    continue
                self.samples.append((str(img_path), label))
                self.card_names.append(img_path.stem)
                label += 1

        self.num_classes = label
        print(f"[{card_type}] {self.num_classes} images")

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
# ArcFace
# ---------------------------------------------------------------------------

class ArcFace(nn.Module):
    def __init__(self, embedding_dim, num_classes, m=0.5, s=64.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_dim, num_classes))
        nn.init.xavier_normal_(self.W)
        self.m = m
        self.s = s
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, embeddings, labels):
        W = F.normalize(self.W, dim=0)
        embeddings = F.normalize(embeddings, dim=1)
        cos_theta = embeddings @ W
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=1e-9))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = torch.where(one_hot.bool(), phi, cos_theta)
        output *= self.s
        return output

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes, embedding_dim=1024):
    backbone = mobilenet_v3_small(weights=None)
    in_features = backbone.classifier[0].in_features
    backbone.classifier = nn.Identity()

    embedding = nn.Sequential(
        nn.Linear(in_features, embedding_dim),
        nn.BatchNorm1d(embedding_dim),
    )
    arcface = ArcFace(embedding_dim, num_classes, m=0.5)
    return backbone, embedding, arcface


def export_onnx(backbone, embedding, output_path, device='cpu'):
    class Embedder(nn.Module):
        def __init__(self, backbone, embedding):
            super().__init__()
            self.backbone = backbone
            self.embedding = embedding
        def forward(self, x):
            x = self.backbone(x)
            x = self.embedding(x)
            x = F.normalize(x, dim=1)
            return x

    model = Embedder(backbone, embedding).eval().to(device)
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=['input'], output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'embedding': {0: 'batch_size'}},
        opset_version=18,
    )
    print(f"    ONNX: {output_path}")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--vtes-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embedding-dim', type=int, default=1024)
    parser.add_argument('--types', nargs='*', default=None,
                        help='Specific types to train (default: all)')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = torch.cuda.is_available() and not args.no_amp
    print(f"[Main] Device: {device} AMP: {amp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Type labels to train
    types_to_train = args.types if args.types else VTES_TYPES
    output_meta = {}

    for card_type in types_to_train:
        print(f"\n{'='*60}")
        print(f"Training: {card_type}")
        print(f"{'='*60}")

        # Dataset for this type
        ds = PerTypeDataset(args.data, args.vtes_json, card_type, train=True)
        if ds.num_classes < 2:
            print(f"  Skipping {card_type}: only {ds.num_classes} cards")
            continue

        dl = DataLoader(ds, args.batch_size, shuffle=True,
                        num_workers=min(4, os.cpu_count() or 1) if os.name != 'nt' else 0,
                        pin_memory=True, drop_last=ds.num_classes >= args.batch_size)

        # Model
        backbone, embedding, arcface = build_model(ds.num_classes, args.embedding_dim)
        backbone.to(device); embedding.to(device); arcface.to(device)

        optimizer = optim.AdamW(
            list(backbone.parameters()) + list(embedding.parameters()) + list(arcface.parameters()),
            lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        warmup = min(10, args.epochs)
        scaler = torch.amp.GradScaler('cuda', enabled=amp)

        total_params = sum(p.numel() for p in backbone.parameters()) + \
                       sum(p.numel() for p in embedding.parameters()) + \
                       sum(p.numel() for p in arcface.parameters())
        print(f"  Params: {total_params/1e6:.1f}M, Classes: {ds.num_classes}")

        best_acc = 0.0
        for epoch in range(args.epochs):
            backbone.train(); embedding.train(); arcface.train()
            total_loss = cor = tot = 0
            t0 = time.time()

            for images, labels in dl:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=amp):
                    features = backbone(images)
                    embeds = embedding(features)
                    embeds = F.normalize(embeds, dim=1)
                    output = arcface(embeds, labels)
                    loss = F.cross_entropy(output, labels)

                if amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                _, pred = output.max(1)
                tot += labels.size(0)
                cor += pred.eq(labels).sum().item()

            if epoch < warmup:
                pass  # LR stays at base during warmup (implicit via cosine start)
            scheduler.step()

            acc = 100.0 * cor / tot
            elapsed = time.time() - t0
            remaining = (args.epochs - epoch - 1) * elapsed
            eta = time.strftime('%H:%M:%S', time.gmtime(remaining))
            print(f"  [{card_type[:8]:>8} {epoch:3d}/{args.epochs}] "
                  f"loss={total_loss/len(dl):.3f} acc={acc:.1f}% "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"({elapsed:.0f}s ETA {eta})")

            if acc > best_acc:
                best_acc = acc

        print(f"  [{card_type}] Best acc: {best_acc:.1f}%")

        # Export ONNX
        safe_name = card_type.lower().replace(' ', '_')
        onnx_name = f"vtes_embedder_{safe_name}.onnx"
        onnx_path = os.path.join(args.output_dir, onnx_name)
        export_onnx(backbone.cpu(), embedding.cpu(), onnx_path, device='cpu')

        # Build embedding index for this type
        index_path = os.path.join(args.output_dir, f"embeddings_{safe_name}.bin")
        meta_path = os.path.join(args.output_dir, f"embeddings_{safe_name}_meta.json")

        # Compute embeddings for all cards of this type
        embedder = None
        try:
            # Create embedder from trained backbone+embedding
            class Embedder(nn.Module):
                def __init__(self, bb, emb):
                    super().__init__()
                    self.backbone = bb
                    self.embedding = emb
                def forward(self, x):
                    x = self.backbone(x)
                    x = self.embedding(x)
                    return F.normalize(x, dim=1)

            embedder = Embedder(backbone.cpu(), embedding.cpu()).eval()
            all_embs = []
            all_names = []
            loader = DataLoader(ds_eval, args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=False)
            with torch.no_grad():
                for images, _ in loader:
                    emb = embedder(images).numpy()
                    all_embs.append(emb)

            # Build stem → (card_id, printed_name) lookup from vtes.json
            def _norm(name):
                return re.sub(r'[^a-z0-9]', '', name.lower())
            card_lookup = {}
            with open(args.vtes_json) as _f:
                _cards = json.load(_f)
            for _c in _cards:
                _cid = str(_c['id'])
                _names = set()
                for _field in ['printed_name', 'name', '_name']:
                    if _field in _c and _c[_field]:
                        _names.add(_c[_field])
                for _v in _c.get('name_variants', []):
                    _names.add(_v)
                for _name in _names:
                    _n = _norm(_name)
                    if _n and _n not in card_lookup:
                        card_lookup[_n] = (_cid, _c.get('printed_name', _c.get('name', _name)))

            # Build meta entries with real card IDs and printed names
            meta_entries = []
            for type_dir in sorted(Path(args.data).iterdir()):
                if not type_dir.is_dir():
                    continue
                for img_path in sorted(type_dir.iterdir()):
                    if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                        continue
                    t = lookup_type(img_path.stem, build_card_to_type_map(args.vtes_json))
                    if t != card_type:
                        continue
                    _stem = img_path.stem
                    _cid, _cname = card_lookup.get(_stem, (str(len(meta_entries)), _stem))
                    meta_entries.append({
                        "id": _cid,
                        "name": _cname,
                        "file": str(img_path),
                    })

            if all_embs:
                all_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
                # L2 normalize
                norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
                norms[norms == 0] = 1
                all_embs /= norms

                all_embs.tofile(index_path)
                with open(meta_path, 'w') as f:
                    json.dump(meta_entries, f, indent=2)
                print(f"    Index: {index_path} ({all_embs.nbytes/1024/1024:.1f}MB, "
                      f"{len(meta_entries)} cards)")
        except Exception as e:
            print(f"    ERROR building index: {e}")

        output_meta[card_type] = {
            "onnx": onnx_name,
            "index": f"embeddings_{safe_name}.bin",
            "meta": f"embeddings_{safe_name}_meta.json",
            "classes": ds.num_classes,
        }

        # Save checkpoint
        ckpt = {
            'card_type': card_type, 'best_acc': best_acc,
            'backbone': backbone.state_dict(),
            'embedding': embedding.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.output_dir,
                                      f"checkpoint_{safe_name}.pth"))

    # Save type manifest
    manifest_path = os.path.join(args.output_dir, 'per_type_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(output_meta, f, indent=2)
    print(f"\n[Main] Manifest: {manifest_path}")
    print("[Main] Done!")


if __name__ == '__main__':
    train()
