"""
eval_confusion.py — confusion matrix and per-class breakdown for a trained
CIFARNet checkpoint (from train_cifarnet.py).

Usage:
    python3 eval_confusion.py                          # uses checkpoints/best.pt
    python3 eval_confusion.py --ckpt checkpoints/last.pt
"""
from __future__ import annotations
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from train_cifarnet import CIFARNet  # reuse the exact same architecture

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"loaded {args.ckpt}  (trained {ckpt['epoch']} epochs, "
          f"recorded val_acc={ckpt['val_acc']:.2%})")

    model = CIFARNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                             download=True, transform=test_tf)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    n = len(CLASSES)
    confusion = torch.zeros(n, n, dtype=torch.long)  # rows=true, cols=predicted

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion[t.item(), p.item()] += 1

    total_correct = confusion.diag().sum().item()
    total = confusion.sum().item()
    print(f"\noverall accuracy: {total_correct/total:.2%}  ({total_correct}/{total})\n")

    print("per-class accuracy:")
    for i, name in enumerate(CLASSES):
        class_total = confusion[i].sum().item()
        class_correct = confusion[i, i].item()
        print(f"  {name:<12} {class_correct/class_total:.2%}  ({class_correct}/{class_total})")

    print("\nconfusion matrix (rows=true label, cols=predicted):")
    header = "".join(f"{c[:4]:>6}" for c in CLASSES)
    print(" " * 12 + header)
    for i, name in enumerate(CLASSES):
        row = "".join(f"{confusion[i, j].item():>6}" for j in range(n))
        print(f"  {name:<10}{row}")

    print("\ntop confused pairs (true -> predicted, excluding correct predictions):")
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((confusion[i, j].item(), CLASSES[i], CLASSES[j]))
    pairs.sort(reverse=True)
    for count, true_name, pred_name in pairs[:10]:
        print(f"  {true_name:<12} -> {pred_name:<12}  {count} times")


if __name__ == "__main__":
    main()
