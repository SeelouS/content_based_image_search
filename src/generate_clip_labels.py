import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm

# ----------------------------
# Configuración
# ----------------------------
IMAGES_DIR = "../GPR1200/images"
OUTPUT_FILE = "clip_labels.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Cargar CLIP
# ----------------------------
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

results = []

# ----------------------------
# Procesar imágenes
# ----------------------------
image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

with torch.no_grad():
    for img_name in tqdm(image_files):
        img_path = os.path.join(IMAGES_DIR, img_name)

        # Cargar imagen
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(DEVICE)

        # Obtener embedding CLIP
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Convertir a lista (para JSON)
        embedding = image_features.squeeze(0).cpu().tolist()

        # Label original (id de categoría del dataset)
        category_id = int(img_name.split("_")[0])

        results.append({
            "image": img_name,
            "category_id": category_id,
            "clip_embedding": embedding
        })

# ----------------------------
# Guardar resultados
# ----------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f)

print(f"✔ CLIP labels guardados en {OUTPUT_FILE}")