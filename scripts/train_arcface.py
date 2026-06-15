#!/usr/bin/env python3
"""
Train an ArcFace embedding model for VTES card identification.

Key improvements over v1:
- Stronger data augmentation (perspective, blur, no flip)
- More epochs (default 200) with warmup + cosine schedule
- Label smoothing (0.1)
- Mixed precision training (AMP)
- MobileNetV3-Large backbone option (--backbone large)

Usage:
    python3 scripts/train_arcface.py \
        --data /mnt/d/Hermes/vtes/dataset/by_type \
        --output-dir /mnt/c/Users/JackSuicide/VTES/vtes_obs_detect/data \
        --vtes-json /mnt/d/Hermes/vtes/vtes.json \
        --epochs 200 --batch-size 128 --backbone large

    # To resume from previous checkpoint:
    python3 scripts/train_arcface.py \
        --data /mnt/d/Hermes/vtes/dataset/by_type \
        --output-dir /mnt/c/Users/JackSuicide/VTES/vtes_obs_detect/data \
        --vtes-json /mnt/d/Hermes/vtes/vtes.json \
        --resume /mnt/c/Users/JackSuicide/VTES/vtes_obs_detect/data/arcface_checkpoint_last.pth \
        --epochs 500 --backbone large

    # To continue the existing 30-epoch training with improved aug:
    python3 scripts/train_arcface.py \
        --data /mnt/d/Hermes/vtes/dataset/by_type \
        --output-dir /mnt/c/Users/JackSuicide/VTES/vtes_obs_detect/data \
        --vtes-json /mnt/d/Hermes/vtes/vtes.json \
        --resume /mnt/c/Users/JackSuicide/VTES/vtes_obs_detect/data/arcface_checkpoint_last.pth \
        --epochs 200
"""

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

# ---------------------------------------------------------------------------
# ArcFace head
# ---------------------------------------------------------------------------

class ArcFace(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, m: float = 0.5, s: float = 64.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_dim, num_classes))
        nn.init.xavier_normal_(self.W)
        self.m = m
        self.s = s
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
# Dataset with strong augmentation
# ---------------------------------------------------------------------------

def build_augmentation(is_train: bool = True):
    """Stronger augmentation to bridge training->webcam domain gap.

    Key changes from v1:
    - RandomResizedCrop instead of Resize (simulates framing variation)
    - RandomPerspective (simulates viewing angle)
    - GaussianBlur (simulates slight defocus / motion blur)
    - Stronger ColorJitter (bigger lighting variation)
    - No RandomHorizontalFlip (cards aren't symmetric; mirrored text is unrealistic)
    """
    if not is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.90, 1.10)),
        transforms.RandomAffine(degrees=20, translate=(0.08, 0.08),
                                scale=(0.80, 1.20), fill=128),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.5, fill=128),
        transforms.ColorJitter(brightness=0.4, contrast=0.3,
                               saturation=0.25, hue=0.1),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class VTESCardDataset(Dataset):
    def __init__(self, data_root: str, transform=None, is_train: bool = True):
        self.samples = []
        self.class_names = []

        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        label_idx = 0
        root = Path(data_root)
        for type_dir in sorted(root.iterdir()):
            if not type_dir.is_dir():
                continue
            for img_path in sorted(type_dir.iterdir()):
                if img_path.suffix.lower() not in exts:
                    continue
                self.samples.append((str(img_path), label_idx))
                self.class_names.append(img_path.stem)
                label_idx += 1

        self.num_classes = label_idx
        self.transform = transform or build_augmentation(is_train=is_train)
        print(f"[Dataset] {len(self.samples)} images, {self.num_classes} classes")

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
# Model builder
# ---------------------------------------------------------------------------

BACKBONES = {
    'small': (mobilenet_v3_small, 576),
    'large': (mobilenet_v3_large, 960),
}


def build_model(backbone_name: str = 'small', pretrained_path: str = None,
                num_classes: int = 4156, embedding_dim: int = 1024,
                arcface_m: float = 0.5):
    backbone_fn, in_features = BACKBONES[backbone_name]
    backbone = backbone_fn(weights=None)
    backbone.classifier = nn.Identity()

    if pretrained_path and os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location='cpu')
        if 'model_state_dict' in state:
            bb_state = {k.replace('backbone.', ''): v
                       for k, v in state['model_state_dict'].items()
                       if k.startswith('backbone.')}
            missing, _ = backbone.load_state_dict(bb_state, strict=False)
        else:
            bb_keys = {}
            for k, v in state.items():
                if k.startswith('backbone.'):
                    bb_keys[k.replace('backbone.', '')] = v
                elif any(k.startswith(p) for p in ('features.', 'classifier.')):
                    continue
                else:
                    bb_keys[k] = v
            missing, _ = backbone.load_state_dict(bb_keys, strict=False)
        print(f"[Model] Loaded backbone from {pretrained_path}: {len(missing)} missing")

    embedding = nn.Sequential(
        nn.Linear(in_features, embedding_dim),
        nn.BatchNorm1d(embedding_dim),
    )
    arcface = ArcFace(embedding_dim, num_classes, m=arcface_m)
    return backbone, embedding, arcface


def export_onnx(backbone, embedding, output_path: str, input_size: int = 224,
                device=torch.device('cpu')):
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
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=['input'], output_names=['embedding'],
        dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
        opset_version=18,
    )
    print(f"[Export] ONNX saved: {output_path}")

# ---------------------------------------------------------------------------
# Label smoothing cross-entropy
# ---------------------------------------------------------------------------

class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=1)
        n_classes = pred.size(1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(true_dist * log_probs).sum(dim=1).mean()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(backbone, embedding, arcface, loader, optimizer, criterion,
                scaler, device, epoch, use_amp):
    backbone.train()
    embedding.train()
    arcface.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            features = backbone(images)
            embeds = embedding(features)
            embeds = F.normalize(embeds, dim=1)
            output = arcface(embeds, labels)
            loss = criterion(output, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train ArcFace VTES card embedder')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset root (by_type/ subdirs)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output dir for ONNX, checkpoints, embeddings')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to classifier_best.pth (pretrained backbone)')
    parser.add_argument('--vtes-json', type=str, default=None,
                        help='Path to vtes.json (for build_embedding_index)')
    parser.add_argument('--build-script', type=str, default=None,
                        help='Path to build_embedding_index.py')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (reduce if OOM)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Peak learning rate')
    parser.add_argument('--embedding-dim', type=int, default=1024,
                        help='Output embedding dimension')
    parser.add_argument('--arcface-m', type=float, default=0.5,
                        help='ArcFace angular margin')
    parser.add_argument('--backbone', type=str, default='small',
                        choices=['small', 'large'],
                        help='Backbone: small (2.5M params, 576 feat) or '
                             'large (5.4M params, 960 feat)')
    parser.add_argument('--label-smooth', type=float, default=0.1,
                        help='Label smoothing epsilon')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of linear warmup epochs')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from arcface_checkpoint_last.pth')
    parser.add_argument('--no-build-index', action='store_true',
                        help='Skip embedding index rebuild after training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available() and not args.no_amp
    print(f"[Main] Device: {device}  AMP: {use_amp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset (training transform with augmentation)
    dataset = VTESCardDataset(args.data, is_train=True)
    num_classes = dataset.num_classes

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1) if os.name != 'nt' else 0,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    backbone, embedding, arcface = build_model(
        backbone_name=args.backbone,
        pretrained_path=args.checkpoint,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        arcface_m=args.arcface_m,
    )
    backbone.to(device)
    embedding.to(device)
    arcface.to(device)

    optimizer = optim.AdamW(
        list(backbone.parameters()) + list(embedding.parameters()) + list(arcface.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )

    # Warmup + Cosine scheduler
    warmup_epochs = min(args.warmup, args.epochs)

    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs)
    )

    # Loss with label smoothing
    criterion = LabelSmoothCrossEntropy(smoothing=args.label_smooth)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch = 0
    best_acc = 0.0

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location='cpu')
        backbone.load_state_dict(state['backbone_state_dict'])
        embedding.load_state_dict(state['embedding_state_dict'])
        arcface.load_state_dict(state['arcface_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1
        best_acc = state.get('best_acc', 0.0)
        print(f"[Main] Resumed from epoch {start_epoch} (best acc: {best_acc:.2f}%)")
        # Manually advance LR schedulers to the correct state
        for _ in range(start_epoch):
            if _ < warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
    else:
        # If no --resume but there's an existing last checkpoint, load it
        last_path = os.path.join(args.output_dir, 'arcface_checkpoint_last.pth')
        if os.path.exists(last_path):
            print(f"[Main] Found existing checkpoint at {last_path}, use --resume to resume")

    # Train
    total_params = (sum(p.numel() for p in backbone.parameters()) +
                    sum(p.numel() for p in embedding.parameters()) +
                    sum(p.numel() for p in arcface.parameters()))
    print(f"[Main] Training {args.epochs} epochs "
          f"(backbone={args.backbone}, params={total_params/1e6:.1f}M)")
    print(f"[Main] Augmentation: RandomResizedCrop+Affine+Perspective+ColorJitter+Blur")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        loss, acc = train_epoch(
            backbone, embedding, arcface, loader, optimizer, criterion,
            scaler, device, epoch, use_amp,
        )

        # LR scheduler step
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        remaining = (args.epochs - epoch - 1) * elapsed
        eta = time.strftime('%H:%M:%S', time.gmtime(remaining))
        print(f"[Epoch {epoch:3d}/{args.epochs}] loss={loss:.4f}  "
              f"acc={acc:.2f}%  lr={current_lr:.2e}  "
              f"({elapsed:.0f}s, ETA {eta})")

        ckpt = {
            'epoch': epoch, 'loss': loss, 'acc': acc,
            'best_acc': max(best_acc, acc),
            'backbone_state_dict': backbone.state_dict(),
            'embedding_state_dict': embedding.state_dict(),
            'arcface_state_dict': arcface.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        last_path = os.path.join(args.output_dir, 'arcface_checkpoint_last.pth')
        torch.save(ckpt, last_path)

        if acc > best_acc:
            best_acc = acc
            best_path = os.path.join(args.output_dir, 'arcface_checkpoint_best.pth')
            torch.save(ckpt, best_path)
            print(f"  ** Best model (acc={acc:.2f}%)")

    print(f"[Main] Training done. Best acc: {best_acc:.2f}%")

    # Export ONNX
    onnx_name = 'vtes_embedder_1024d.onnx'
    onnx_path = os.path.join(args.output_dir, onnx_name)
    print(f"[Main] Exporting ONNX...")
    export_onnx(backbone.cpu(), embedding.cpu(), onnx_path, device=torch.device('cpu'))

    # Rebuild embedding index
    if not args.no_build_index:
        build_script = args.build_script
        if not build_script:
            # Try common locations
            candidates = [
                os.path.join(os.path.dirname(__file__), 'build_embedding_index.py'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'scripts', 'build_embedding_index.py'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'build_embedding_index.py'),
            ]
            for c in candidates:
                if os.path.exists(c):
                    build_script = c
                    break

        if build_script and os.path.exists(build_script):
            vtes_json_arg = []
            if args.vtes_json:
                vtes_json_arg = ['--vtes-json', args.vtes_json]
            cmd = [
                sys.executable, build_script,
                '--onnx', onnx_path,
                '--data', args.data,
                '--output', args.output_dir,
            ] + vtes_json_arg
            print(f"[Main] Rebuilding index: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print("[Main] Index rebuilt.")
        else:
            print(f"[Main] build_embedding_index.py not found. Rebuild manually:")
            print(f"  python build_embedding_index.py --onnx {onnx_path} "
                  f"--data {args.data} --output {args.output_dir}")

    # Copy files to OBS plugin data directory
    plugin_data = r'C:\Users\JackSuicide\AppData\Roaming\obs-studio\plugin_config\vtes-card-scanner\data'
    for fname in ['vtes_embedder_1024d.onnx', 'embeddings_1024d.bin',
                  'embeddings_1024d_meta.json']:
        src = os.path.join(args.output_dir, fname)
        dst = os.path.join(plugin_data, fname)
        if os.path.exists(src):
            try:
                import shutil
                shutil.copy2(src, dst)
                print(f"[Main] Copied {fname} to plugin data dir")
            except Exception as e:
                print(f"[Main] Failed to copy {fname}: {e}")

    print("[Main] All done!")
    print(f"  ONNX: {onnx_path}")
    print(f"  Checkpoint: {last_path}")
    print(f"  Best checkpoint: {os.path.join(args.output_dir, 'arcface_checkpoint_best.pth')}")


if __name__ == '__main__':
    main()
