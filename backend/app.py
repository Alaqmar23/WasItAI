import torch
import os
import hashlib
import shutil
import time
import requests
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from model import AIDetectorModel, predict_image
import uvicorn

# Discord Webhook configuration (Optional/Cloud-only)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

def send_to_discord(prediction, image_id, file_contents, filename):
    """Sends a notification to Discord with the image for incorrect feedback."""
    if not DISCORD_WEBHOOK_URL:
        return
    
    label = "AI" if prediction == "AUTHENTIC" else "AUTHENTIC"
    payload = {
        "content": f"🚨 **Feedback Received (Correction Required)**\n**Scan Result:** {prediction}\n**Ground Truth:** {label}\n**Image ID:** {image_id}"
    }
    files = {"file": (filename, file_contents)}
    
    try:
        requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files, timeout=10)
    except Exception as e:
        print(f"⚠️ Discord Webhook failed: {e}")

# Directory Setup - Keeping verified and processed folders
BASE_FEEDBACK_DIR = Path("dataset/feedback")
VERIFIED_AI_DIR = BASE_FEEDBACK_DIR / "verified_ai"
VERIFIED_AUTH_DIR = BASE_FEEDBACK_DIR / "verified_authentic"
PROCESSED_DIR = BASE_FEEDBACK_DIR / "processed"

for d in [VERIFIED_AI_DIR, VERIFIED_AUTH_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def cleanup_old_feedback():
    """Delete files older than 30 days from all feedback folders to save disk space."""
    now = time.time()
    cutoff = now - (30 * 24 * 60 * 60) # 30 days in seconds
    files_deleted = 0
    
    # We clean up everything in feedback to prevent infinite growth
    for folder in [VERIFIED_AI_DIR, VERIFIED_AUTH_DIR, PROCESSED_DIR]:
        if not folder.exists(): continue
        for file_path in folder.glob("*"):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff:
                    try:
                        file_path.unlink()
                        files_deleted += 1
                    except Exception as e:
                        print(f"⚠️ Error deleting {file_path}: {e}")
    
    if files_deleted > 0:
        print(f"🧹 Cleaned up {files_deleted} feedback images older than 30 days.")

# Run cleanup on startup to manage disk space automatically
cleanup_old_feedback()

app = FastAPI(title="AI Image Detector API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Backend running on device: {device}")

# Initialize and load model
model = AIDetectorModel(pretrained=True).to(device)
checkpoint_path = "best_model_global.pth"

if os.path.exists(checkpoint_path):
    print(f"✅ Loading New Godzilla Model from {checkpoint_path}")
    model.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print(f"⚠️ Model {checkpoint_path} not found. Searching for fallback...")
    for fallback in ["best_model_68.pth", "best_model.pth"]:
        if os.path.exists(fallback):
            model.model.load_state_dict(torch.load(fallback, map_location=device))
            break

model.eval()

@app.get("/")
def read_root():
    return {"status": "AI Image Detector backend is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Generate unique ID for matching if client gives feedback later
    file_hash = hashlib.md5(contents).hexdigest()
    image_id = f"{file_hash}_{int(time.time())}"
    
    # We NO LONGER save to a pending folder here. 
    # The image is processed in memory and then the request ends.
    result = predict_image(contents, model, device)
    
    # Return basic info plus the ID for the filename if they save it later
    result["image_id"] = image_id
    return result

@app.post("/feedback")
async def feedback(
    is_correct: bool = Form(...),
    prediction: str = Form(...),
    image_id: str = Form(...),
    file: UploadFile = File(None) # Optional: only sent if is_correct is False
):
    # Determine ground truth label
    if is_correct:
        # Prediction was right - nothing to save
        return {"status": "success", "message": "Correct prediction verified by user."}
    else:
        # Prediction was wrong - we need the file for the verified database
        if not file:
            return {"status": "error", "message": "File is required for incorrect feedback"}
        
        contents = await file.read()
        file_ext = Path(file.filename).suffix or ".jpg"
        
        # 1. Send to Discord Bridge (Cloud -> Local notify)
        send_to_discord(prediction, image_id, contents, f"{image_id}{file_ext}")
        
        # 2. Local save for retraining lab
        label = "AI" if prediction == "AUTHENTIC" else "AUTHENTIC"
        target_dir = VERIFIED_AI_DIR if label == "AI" else VERIFIED_AUTH_DIR
        
        save_path = target_dir / f"{image_id}{file_ext}"
        with open(save_path, "wb") as f:
            f.write(contents)
            
        return {"status": "success", "label": label, "saved_to": str(save_path)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
