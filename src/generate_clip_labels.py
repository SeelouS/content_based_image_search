import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------
IMAGES_DIR = "./images"
OUTPUT_FILE = "clip_labels_16.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load CLIP
# ----------------------------
model, preprocess = clip.load("ViT-B/16", device=DEVICE)
model.eval()

results = []

# ----------------------------
# Process images
# ----------------------------
image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

with torch.no_grad():
    for img_name in tqdm(image_files):
        img_path = os.path.join(IMAGES_DIR, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(DEVICE)

        # Get CLIP embedding
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Convert to list (for JSON)
        embedding = image_features.squeeze(0).cpu().tolist()

        # Original label (dataset category id)
        category_id = int(img_name.split("_")[0])

        results.append({
            "image": img_name,
            "category_id": category_id,
            "clip_embedding": embedding
        })

# ----------------------------
# Save results
# ----------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f)

print(f"✔ CLIP labels saved to {OUTPUT_FILE}")