import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms, models
from PIL import Image
import os
import random
from pathlib import Path
import shutil
from model import AIDetectorModel

# ─── CONFIG ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_model_global.pth"
ORIGINAL_DATA_DIR = "dataset/shutterstock/train"
FEEDBACK_DIR = "dataset/feedback"
PROCESSED_DIR = os.path.join(FEEDBACK_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Memory Replay Settings
MEMORY_SAMPLE_SIZE = 1200  # Per class (2400 total memory images)
LR = 5e-5
EPOCHS = 3
BATCH_SIZE = 16

# ─── DATASET CLASS ────────────────────────────────────────────────────────
class SimpleImageDataset(Dataset):
    """Simple dataset for loading specific lists of images with labels."""
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_train_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ─── MAIN RETRAINING ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FEEDBACK RETRAINING — Memory Replay Pipeline")
    print("=" * 60)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Error: Could not find {CHECKPOINT_PATH}. Cannot fine-tune.")
        return

    # 1. Gather original samples (Memory)
    all_paths = []
    all_labels = []

    print(f"📦 Sampling {MEMORY_SAMPLE_SIZE} images from each original class...")
    for class_name, label in [("FAKE", 0), ("REAL", 1)]:
        class_path = Path(ORIGINAL_DATA_DIR) / class_name
        if class_path.exists():
            files = list(class_path.glob("*"))
            sample = random.sample(files, min(len(files), MEMORY_SAMPLE_SIZE))
            all_paths.extend([str(f) for f in sample])
            all_labels.extend([label] * len(sample))
        else:
            print(f"⚠️ Warning: Could not find original directory {class_path}")

    # 2. Gather new feedback data
    feedback_count = 0
    print(f"📝 Gathering new verified feedback data...")
    for folder_name, label in [("verified_ai", 0), ("verified_authentic", 1)]:
        folder_path = Path(FEEDBACK_DIR) / folder_name
        if folder_path.exists():
            files = list(folder_path.glob("*"))
            all_paths.extend([str(f) for f in files])
            all_labels.extend([label] * len(files))
            feedback_count += len(files)
    
    if feedback_count == 0:
        print("⚠️ No new feedback images found. Training on memory only.")
    else:
        print(f"✅ Found {feedback_count} new images to learn from!")

    # 3. Create DataLoader
    train_dataset = SimpleImageDataset(all_paths, all_labels, get_train_transform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 4. Load Model
    print(f"🛠️ Loading model from {CHECKPOINT_PATH}...")
    model = AIDetectorModel(pretrained=False).to(DEVICE)
    # The AIDetectorModel's .model is the actual backbone
    model.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.train()

    # 5. Fine-tune (Strictly surgical)
    # Only unfreeze the head and the very last block of the backbone
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable classifier
    for param in model.model.classifier.parameters():
        param.requires_grad = True
    
    # Enable last block of features (block 7)
    for param in model.model.features[-1].parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print(f"🚀 Starting fine-tuning for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE, dtype=torch.float32).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        acc = 100 * correct / total
        print(f"  Epoch {epoch}/{EPOCHS} │ Loss: {avg_loss:.4f} │ Acc: {acc:.2f}%")

    # 6. Save Updated Weights
    torch.save(model.model.state_dict(), CHECKPOINT_PATH)
    print(f"\n✅ Retraining Complete! Updated weights saved to {CHECKPOINT_PATH}")

    # 7. Move feedback images to 'processed' so they aren't retrained twice
    if feedback_count > 0:
        print(f"📦 Moving {feedback_count} feedback images to processed folder...")
        for folder_name in ["verified_ai", "verified_authentic"]:
            folder_path = Path(FEEDBACK_DIR) / folder_name
            for f in folder_path.glob("*"):
                shutil.move(str(f), os.path.join(PROCESSED_DIR, f.name))
        print("💡 Images moved. They will be auto-deleted by the backend after 30 days.")

    print("=" * 60)

if __name__ == "__main__":
    main()
