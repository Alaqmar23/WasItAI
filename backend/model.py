import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

class AIDetectorModel(nn.Module):
    def __init__(self, pretrained=True):
        super(AIDetectorModel, self).__init__()
        # Use EfficientNet-V2-S
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_v2_s(weights=weights)
        
        # Match Phase 11 Clean architecture: Simple Head
        # Original: Sequential(Dropout(0.2), Linear(1280, 1000))
        # Ours:     Sequential(Dropout(0.3), Linear(1280, 1))
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        # Result is a single logit
        return self.model(x)

def get_transform():
    """Clean evaluation transform (Matched to clean_train.py)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def predict_image(image_bytes, model, device):
    """Predicts if an image is AI-generated (label 0 = FAKE)."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        return {"error": str(e)}

    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        
        # 1. Aggressive Temperature Scaling (8.0 instead of 3.0)
        # This squashes the raw outputs (logits) more forcefully towards 0.
        # This ensures the model feels "humble" and shows doubt in its verdict.
        TEMPERATURE = 8.0
        scaled_logits = logits / TEMPERATURE
        
        prob_real = torch.sigmoid(scaled_logits).item()
        prob_ai = 1 - prob_real
        
        # 2. Standard 50% Threshold 
        prediction = "AUTHENTIC" if prob_real >= 0.50 else "AI"
        
        # 3. Calculate Raw Confidence relative to the predicted class
        raw_confidence = (prob_real if prediction == "AUTHENTIC" else prob_ai) * 100

        # 4. Confidence Mapping (The "Humble Clip")
        # We cap the confidence at 96.50% so the model never claims to be "100% sure".
        # We also slightly "shave" the lower bound to keep it looking clean.
        # This makes the "doubt" visible even on very certain images.
        if raw_confidence > 96.50:
            confidence = 96.50
        else:
            confidence = raw_confidence

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "probability_ai": prob_ai,
        "probability_real": prob_real
    }
