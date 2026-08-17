"""
train_cifarnet.py — ~1.6M-parameter CNN trained on CIFAR-10, from scratch.

Usage:
    python3 train_cifarnet.py                       # defaults: 20 epochs, bs=128
    python3 train_cifarnet.py --epochs 40 --lr 0.001
    python3 train_cifarnet.py --device cpu           # force CPU

First run downloads CIFAR-10 (~170MB) into ./data via torchvision.
Saves the best (by val accuracy) checkpoint to ./checkpoints/best.pt
and the final-epoch checkpoint to ./checkpoints/last.pt.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


class CIFARNet(nn.Module):
    """
    Plain CNN: 5 conv layers (with BatchNorm) + 2 FC layers.
    ~1.59M parameters total (32/64/128/128/256 channel progression).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 32x32 -> 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 16x16 -> 8x8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_dataloaders(batch_size: int, data_dir: str, num_workers: int = 2):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)  # CIFAR-10 stats
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--ckpt-dir", default="./checkpoints")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders(args.batch_size, args.data_dir, args.num_workers)

    model = CIFARNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
        scheduler.step()

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.2%}  ({dt:.1f}s)")

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "val_acc": val_acc, "args": vars(args)}, ckpt_dir / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc, "args": vars(args)}, ckpt_dir / "best.pt")

    print(f"\ndone. best val_acc={best_acc:.2%}. checkpoints in {ckpt_dir}/")


if __name__ == "__main__":
    main()
