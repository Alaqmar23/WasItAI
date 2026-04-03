import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import os
import time

# ─── PHASE 10: HIGH-RES FINAL CONFIG ─────────────────────────────────────────
TRAIN_DIR = "dataset/shutterstock_resized/train"
TEST_DIR = "dataset/shutterstock_resized/test"
BATCH_SIZE = 16   # Optimized for 4GB VRAM @ 300px
ACCUMULATION_STEPS = 4 # Total effective batch = 64
RESOLUTION = 300  # High-Precision Resolution
EPOCHS = 8        
LEARNING_RATE = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIXUP_ALPHA = 1.0 
SMOOTHING = 0.1

# ─── MIXUP ENGINE ───────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ─── TRANSFORMS (300px High-Res) ───────────────────────────────────────────
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ─── TRAINING ENGINE ────────────────────────────────────────────────────────
def train_model():
    print(f"🚀 PHASE 10: HIGH-RES FINAL (300px | Backbone Frozen | Gradient Accum)")
    print(f"📊 Training on {DEVICE}")
    
    train_dataset = datasets.ImageFolder(TRAIN_DIR, get_train_transform())
    test_dataset = datasets.ImageFolder(TEST_DIR, get_test_transform())
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)
    
    # 1. NUCLEAR FREEZE: Block all memorization
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.classifier[1].in_features
    # Only training the top-level classifier
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(1024, 1)
    )
    
    # Enable gradients for the classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-1)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader) // ACCUMULATION_STEPS, epochs=EPOCHS,
        pct_start=0.3
    )
    
    scaler = torch.amp.GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')
    best_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        model.train()
        train_loss, corrects, total = 0.0, 0.0, 0
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE, dtype=torch.float32).unsqueeze(1)
            
            labels_smoothed = labels * (1 - SMOOTHING) + 0.5 * SMOOTHING
            inputs, y_a, y_b, lam = mixup_data(inputs, labels_smoothed, MIXUP_ALPHA)
            
            with torch.amp.autocast(device_type='cuda' if DEVICE.type == 'cuda' else 'cpu'):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                loss = loss / ACCUMULATION_STEPS # Scale loss for accumulation
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss += (loss.item() * ACCUMULATION_STEPS) * inputs.size(0)
            preds = (outputs > 0.0).float()
            y_a_hard = (y_a > 0.5).float()
            y_b_hard = (y_b > 0.5).float()
            c = (lam * torch.sum(preds == y_a_hard).item() + (1 - lam) * torch.sum(preds == y_b_hard).item())
            corrects += c
            total += labels.size(0)
            
        epoch_acc = corrects / total
        
        model.eval()
        val_corrects, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE, dtype=torch.float32).unsqueeze(1)
                outputs = model(inputs)
                preds = (outputs > 0.0).float()
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)

        val_acc = val_corrects.double() / val_total
        elapsed = time.time() - epoch_start
        
        print(f"Acc: {epoch_acc*100:.2f}% │ Val: {val_acc*100:.2f}% │ Time: {elapsed:.0f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            with open("best_acc.txt", "w") as f: f.write(str(val_acc.item() * 100))
            print(f"★★★ NEW BEST: {val_acc*100:.2f}% ★★★")

    print(f"\n🏁 Finished Phase 10. Best Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    train_model()
