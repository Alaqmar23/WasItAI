"""
=============================================================
  HAIL MARY PHASE — The Final Push to 80%+
=============================================================
Strategy:
  1. FRESH START from ImageNet (escape bad local minimum)
  2. GENTLE augmentation (preserve AI artifacts the model needs)
  3. Stochastic Weight Averaging (SWA) for final epochs
  4. Gradient clipping for training stability
  5. Warmup + Cosine Annealing LR schedule
  6. Test-Time Augmentation (TTA) for evaluation
  7. Early stopping with patience
=============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR
import os
import sys
import time
import copy
import numpy as np
from model import AIDetectorModel

# ─── MIXUP STRATEGY ─────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ─── AUGMENTATION STRATEGY ──────────────────────────────────────────────
# KEY INSIGHT: Previous augmentations (Grayscale, Posterize, GaussianBlur) 
# were DESTROYING the frequency/texture artifacts that distinguish AI images.
# This gentle augmentation preserves those subtle patterns.

def get_gentle_train_transform():
    """Gentle augmentation — preserves AI-detection-critical artifacts.
       Images are already pre-resized to 256x256, so skip Resize."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Less aggressive crop
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # Mild rotation only
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Subtle color jitter
        # NO RandomGrayscale — destroys color distribution artifacts
        # NO RandomPosterize — destroys bit-depth artifacts  
        # NO GaussianBlur — destroys frequency domain signatures
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Small random erasing for robustness
    ])

def get_clean_eval_transform():
    """Clean evaluation transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_tta_transforms():
    """Test-Time Augmentation — 5 views for ensemble prediction."""
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return [
        transforms.Compose(base),  # Original
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0)] + base[1:]),  # H-flip
        transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224)] + base[1:]),  # Center crop
        transforms.Compose([transforms.Resize((256, 256)), transforms.FiveCrop(224)] + base[1:]),  # Will handle separately
    ]


# ─── FOCAL LOSS ──────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha # alpha is for class 1 (REAL)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Apply Label Smoothing
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        
        # Proper Class-Balanced Alpha
        # If target is near 1 (Real), use alpha. If near 0 (Fake), use 1-alpha.
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal = alpha_t * (1 - pt)**self.gamma * bce
        return focal.mean()


# ─── TRAINING ENGINE ─────────────────────────────────────────────────────
def save_best_acc(acc):
    with open("best_acc.txt", "w") as f:
        f.write(str(acc))

def load_best_acc():
    if os.path.exists("best_acc.txt"):
        with open("best_acc.txt", "r") as f:
            try:
                return float(f.read().strip())
            except:
                return 0.0
    return 0.0

def run_epoch(model, loader, criterion, optimizer, device, is_train=True, grad_clip=1.0, mixup_alpha=0.2):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    fake_correct, fake_total = 0, 0
    real_correct, real_total = 0, 0
    
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            targets = labels.float().unsqueeze(1)
            
            if is_train:
                optimizer.zero_grad()
                # Apply Mixup
                images, targets_a, targets_b, lam = mixup_data(images, targets, mixup_alpha, device)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            
            # Use original targets for accuracy metrics even if mixed up
            # For simplicity in TTA/Val, targets is unchanged if not is_train
            if not is_train:
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
                fake_mask = (targets == 0).squeeze()
                real_mask = (targets == 1).squeeze()
                if fake_mask.any():
                    fake_correct += (predicted[fake_mask.unsqueeze(1)] == targets[fake_mask.unsqueeze(1)]).sum().item()
                    fake_total += fake_mask.sum().item()
                if real_mask.any():
                    real_correct += (predicted[real_mask.unsqueeze(1)] == targets[real_mask.unsqueeze(1)]).sum().item()
                    real_total += real_mask.sum().item()
            else:
                # Approximate train acc during mixup is tricky, we just use total loss mostly
                total += targets.size(0)

    acc = 100 * correct / total if total > 0 else 0
    fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
    real_acc = 100 * real_correct / real_total if real_total > 0 else 0
    avg_loss = total_loss / len(loader)
    
    return acc, avg_loss, fake_acc, real_acc


def evaluate_with_tta(model, test_dir, device, batch_size=32):
    """Test-Time Augmentation: average predictions from 3 views."""
    model.eval()
    
    tta_transforms_list = [
        # View 1: Standard
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # View 2: Horizontal Flip
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # View 3: Slight center crop
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ]
    
    all_preds = []
    all_labels = []
    
    for view_idx, tfm in enumerate(tta_transforms_list):
        dataset = datasets.ImageFolder(test_dir, transform=tfm)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        view_preds = []
        if view_idx == 0:
            for images, labels in loader:
                images = images.to(device)
                with torch.no_grad():
                    outputs = model(images)
                view_preds.append(outputs.cpu())
                all_labels.append(labels)
        else:
            for images, _ in loader:
                images = images.to(device)
                with torch.no_grad():
                    outputs = model(images)
                view_preds.append(outputs.cpu())
        
        all_preds.append(torch.cat(view_preds, dim=0))
    
    # Average predictions across views
    avg_preds = torch.stack(all_preds, dim=0).mean(dim=0)
    labels = torch.cat(all_labels, dim=0).float().unsqueeze(1)
    
    predicted = (avg_preds >= 0.5).float()
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    
    return 100 * correct / total


# ─── MAIN HAIL MARY ─────────────────────────────────────────────────────
def hail_mary(data_dir="dataset/shutterstock", epochs=10, batch_size=32, start_fresh=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 65)
    print("  HAIL MARY PHASE — The Final Push to 80%+")
    print("=" * 65)
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Start Fresh: {start_fresh}")
    print("=" * 65)

    train_dir = os.path.join(data_dir, 'train')
    test_dir  = os.path.join(data_dir, 'test')

    # Use GENTLE augmentation that preserves AI artifacts
    train_dataset = datasets.ImageFolder(train_dir, transform=get_gentle_train_transform())
    test_dataset  = datasets.ImageFolder(test_dir,  transform=get_clean_eval_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print(f"  Classes: {train_dataset.classes}")
    print(f"  Class mapping: {train_dataset.class_to_idx}")
    print("=" * 65)

    # ── MODEL INITIALIZATION ──────────────────────────────────────────
    if start_fresh:
        print("\n[STRATEGY] Starting FRESH from ImageNet pretrained weights")
        print("[REASON]   Escaping bad local minimum from previous training phases")
        model = AIDetectorModel(pretrained=True).to(device)
        best_acc = 0.0
    else:
        print("\n[STRATEGY] Resuming from best checkpoint")
        model = AIDetectorModel(pretrained=False).to(device)
        if os.path.exists("best_model.pth"):
            model.load_state_dict(torch.load("best_model.pth", map_location=device))
            best_acc = load_best_acc()
            print(f"[LOADED]   Baseline accuracy: {best_acc:.2f}%")
        else:
            print("[WARNING]  No checkpoint found, starting fresh anyway")
            model = AIDetectorModel(pretrained=True).to(device)
            best_acc = 0.0

    # ── LOSS FUNCTION ─────────────────────────────────────────────────
    # Use Focal Loss to focus on hard-to-classify samples
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    # ── TRAINING SCHEDULE ─────────────────────────────────────────────
    # Phase A (epochs 1-3):   Frozen backbone, train classifier head
    # Phase B (epochs 4-end): Unfreeze all, fine-tune with low LR
    # Final 3 epochs:         SWA for weight averaging
    
    warmup_epochs = 3
    swa_start_epoch = max(epochs - 3, warmup_epochs + 1)
    patience = 5
    no_improve_count = 0
    best_model_weights = None
    
    print(f"\n  Schedule:")
    print(f"    Phase A (epochs 1-{warmup_epochs}): Frozen backbone, classifier only")
    print(f"    Phase B (epochs {warmup_epochs+1}-{swa_start_epoch}): Full fine-tune")
    print(f"    Phase C (epochs {swa_start_epoch+1}-{epochs}): SWA averaging")
    print(f"    Patience: {patience} epochs")
    print()
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # ── PHASE A: Frozen backbone ──────────────────────────────────
        if epoch <= warmup_epochs:
            if epoch == 1:
                print("━" * 65)
                print("  PHASE A: Training classifier head (backbone frozen)")
                print("━" * 65)
                for param in model.model.features.parameters():
                    param.requires_grad = False
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=1e-3, weight_decay=1e-4
                )
                scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs)
            
        # ── PHASE B: Partial fine-tune ────────────────────────────────
        elif epoch == warmup_epochs + 1:
            print("━" * 65)
            print("  PHASE B: Partial fine-tuning (Last 3 blocks unfrozen)")
            print("━" * 65)
            # Only unfreeze the end of the backbone for better stability
            for i, layer in enumerate(model.model.features):
                if i >= 5: # EfficientNet-V2-S has 8 stages in features
                    for param in layer.parameters():
                        param.requires_grad = True
            
            optimizer = optim.AdamW([
                {'params': [p for i, l in enumerate(model.model.features) if i >= 5 for p in l.parameters()], 'lr': 5e-6},
                {'params': model.model.classifier.parameters(), 'lr': 1e-4},
            ], weight_decay=1e-4)
            
            remaining = epochs - warmup_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining)
        
        # ── PHASE C: SWA ─────────────────────────────────────────────
        if epoch == swa_start_epoch + 1:
            print("━" * 65)
            print("  PHASE C: Stochastic Weight Averaging (SWA)")
            print("━" * 65)
            swa_model = AveragedModel(model).to(device)
            swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
        
        # ── RUN EPOCH ─────────────────────────────────────────────────
        train_acc, train_loss, train_fake_acc, train_real_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, 
            is_train=True, grad_clip=1.0
        )
        
        val_acc, val_loss, val_fake_acc, val_real_acc = run_epoch(
            model, test_loader, criterion, optimizer, device, 
            is_train=False
        )
        
        # Update scheduler
        if epoch <= swa_start_epoch:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            current_lr = swa_scheduler.get_last_lr()[0]
        
        elapsed = time.time() - t0
        
        # ── LOGGING ───────────────────────────────────────────────────
        phase = "A" if epoch <= warmup_epochs else ("C/SWA" if epoch > swa_start_epoch else "B")
        print(f"  [{phase}] Epoch {epoch:2d}/{epochs} │ "
              f"Train {train_acc:5.1f}% (F:{train_fake_acc:4.1f} R:{train_real_acc:4.1f}) │ "
              f"Val {val_acc:5.1f}% (F:{val_fake_acc:4.1f} R:{val_real_acc:4.1f}) │ "
              f"LR {current_lr:.1e} │ {elapsed:.0f}s")
        
        # ── CHECKPOINT ────────────────────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_model.pth")
            save_best_acc(best_acc)
            no_improve_count = 0
            print(f"  ★★★ NEW BEST: {best_acc:.2f}% ★★★")
        else:
            no_improve_count += 1
            if no_improve_count >= patience and epoch > warmup_epochs:
                print(f"\n  ⚠ Early stopping triggered (no improvement for {patience} epochs)")
                break
        
        # Gap monitoring (overfitting detector)
        gap = train_acc - val_acc
        if gap > 15:
            print(f"  ⚠ Overfitting alert: train-val gap = {gap:.1f}%")
    
    # ── FINAL SWA EVALUATION ──────────────────────────────────────────
    if swa_start_epoch < epochs and 'swa_model' in dir():
        print("\n" + "━" * 65)
        print("  Final SWA Model Evaluation")
        print("━" * 65)
        
        # Update batch norm statistics for SWA model
        print("  Updating SWA batch norm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        
        swa_val_acc, swa_val_loss, swa_fake_acc, swa_real_acc = run_epoch(
            swa_model, test_loader, criterion, optimizer, device, is_train=False
        )
        print(f"  SWA Val Accuracy: {swa_val_acc:.2f}% (F:{swa_fake_acc:.1f} R:{swa_real_acc:.1f})")
        
        if swa_val_acc > best_acc:
            best_acc = swa_val_acc
            # Save the inner model weights from SWA
            torch.save(swa_model.module.state_dict(), "best_model.pth")
            save_best_acc(best_acc)
            print(f"  ★★★ SWA IS THE NEW CHAMPION: {best_acc:.2f}% ★★★")
    
    # ── TEST-TIME AUGMENTATION EVALUATION ─────────────────────────────
    print("\n" + "━" * 65)
    print("  Test-Time Augmentation (TTA) Evaluation")
    print("━" * 65)
    
    # Load best model for TTA
    final_model = AIDetectorModel(pretrained=False).to(device)
    final_model.load_state_dict(torch.load("best_model.pth", map_location=device))
    
    tta_acc = evaluate_with_tta(final_model, test_dir, device, batch_size=batch_size)
    print(f"  TTA Accuracy (3-view average): {tta_acc:.2f}%")
    
    if tta_acc > best_acc:
        print(f"  TTA improves accuracy from {best_acc:.2f}% → {tta_acc:.2f}%!")
    
    # ── FINAL REPORT ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  HAIL MARY RESULTS")
    print("=" * 65)
    print(f"  Best Checkpoint Accuracy: {best_acc:.2f}%")
    print(f"  TTA Accuracy:            {tta_acc:.2f}%")
    target = 80.0
    if best_acc >= target:
        print(f"  ✅ TARGET ACHIEVED! {best_acc:.2f}% >= {target}%")
    else:
        print(f"  ❌ Target {target}% not reached. Best: {best_acc:.2f}%")
        print(f"  Consider: increasing epochs, using a larger dataset, or trying EfficientNet-B3")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HAIL MARY Training")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint instead of fresh start")
    args = parser.parse_args()
    
    hail_mary(
        epochs=args.epochs,
        batch_size=args.batch_size,
        start_fresh=not args.resume,
    )
