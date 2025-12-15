import os
import json
import argparse
from typing import List, Dict, Any
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from binaryFeatureExtractor import BinaryFeatureExtractor

class ImageFolderDataset(Dataset):
    def __init__(self, root: str, exts: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]):
        self.paths: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if any(f.lower().endswith(e) for e in exts):
                    self.paths.append(os.path.join(dirpath, f))
        if not self.paths:
            raise ValueError(f"No se encontraron imágenes en: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return img


class EmbeddingJsonDataset(Dataset):
    """
    Carga embeddings CLIP desde un archivo JSON con este formato:
      [
        {"image": "0123_abc.jpg", "category_id": 123, "clip_embedding": [ ... ]},
        ...
      ]
    También acepta JSONL con un objeto por línea en el mismo formato.
    """
    def __init__(self, json_path: str, normalize: bool = True):
        self.vectors: List[torch.Tensor] = []

        def to_tensor(vec: List[float]) -> torch.Tensor:
            t = torch.tensor(vec, dtype=torch.float32)
            if normalize:
                t = F.normalize(t, dim=0)
            return t

        # Intentar JSON estándar; si falla, intentar JSONL
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
            with open(json_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not isinstance(data, list):
            raise ValueError("Se esperaba una lista de objetos con 'clip_embedding'.")

        raw: List[List[float]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            vec = item.get("clip_embedding")
            if isinstance(vec, list) and vec and all(isinstance(x, (int, float)) for x in vec):
                raw.append([float(x) for x in vec])

        if not raw:
            raise ValueError(f"No se encontraron entradas válidas con 'clip_embedding' en: {json_path}")

        # Validar dimensión consistente
        dim_count: Dict[int, int] = {}
        for v in raw:
            dim_count[len(v)] = dim_count.get(len(v), 0) + 1
        target_dim = max(dim_count.items(), key=lambda kv: kv[1])[0]

        for v in raw:
            if len(v) == target_dim:
                self.vectors.append(to_tensor(v))

        if not self.vectors:
            raise ValueError("No se encontraron vectores con dimensión consistente en el JSON.")

        self.dim = int(self.vectors[0].numel())

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.vectors[idx]


def train(model: BinaryFeatureExtractor,
          dataloader: DataLoader,
          device: str,
          epochs: int,
          lr: float,
          weight_decay: float,
          reg_weight: float) -> None:
    # Entrenar las capas ocultas (el bloque de signo no tiene parámetros y no se usa)
    for p in model.hidden_layers.parameters():
        p.requires_grad_(True)
    model.train()

    optimizer = torch.optim.Adam(model.hidden_layers.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_sim = 0.0
        total_reg = 0.0
        n_batches = 0

        for clip_emb in dataloader:
            clip_emb = clip_emb.to(device)

            # Forward por las capas ocultas con tanh (sin signo)
            x = clip_emb
            for layer in model.hidden_layers:
                x = layer(x)
                x = model.tanh(x)

            # Pérdida de similitud: 1 - cosine_similarity(myEmbedding, clipEmbedding)
            sim = F.cosine_similarity(x, clip_emb, dim=-1)
            sim_loss = 1.0 - sim.mean()

            # Regularización: penaliza fuertemente valores cercanos a 0, casi nula cerca de +/-1
            # -log(|x| + eps): grande cerca de 0, ~0 cuando |x|->1
            eps = 1e-6
            reg = (-torch.log(torch.clamp(torch.abs(x), min=eps))).mean()

            loss = sim_loss + reg_weight * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sim += (1.0 - sim_loss.item())  # valor medio de similitud
            total_reg += reg.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_sim = total_sim / max(1, n_batches)
        avg_reg = total_reg / max(1, n_batches)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f} | cos_sim: {avg_sim:.6f} | reg: {avg_reg:.6f}")

    # Volver a modo evaluación y congelar de nuevo
    for p in model.hidden_layers.parameters():
        p.requires_grad_(False)
    model.eval()


def main():
    parser = argparse.ArgumentParser(description="Entrenar BinaryFeatureExtractor con embeddings CLIP desde JSON")
    parser.add_argument("--clip-json", required=True, help="Ruta al JSON con embeddings CLIP")
    parser.add_argument("--model-name", default="clip-ViT-B-32", help="Nombre del modelo SentenceTransformer")
    parser.add_argument("--hidden-layers", type=int, default=2, help="Número de capas ocultas")
    parser.add_argument("--last-dim", type=int, default=512, help="Dimensión de la última capa")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--reg-weight", type=float, default=0.1, help="Peso de la regularización -log(|x|)")
    parser.add_argument("--out", default="./artifacts/binary_extractor.pt", help="Ruta para guardar pesos")
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Dataset: embeddings CLIP del JSON
    emb_dataset = EmbeddingJsonDataset(args.clip_json)
    dataloader = DataLoader(emb_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = BinaryFeatureExtractor(model_name=args.model_name,
                                   num_of_hidden_layers=args.hidden_layers,
                                   device=device,
                                   last_layer_dimension=args.last_dim)

    # Validar dimensión: myEmbedding y clipEmbedding deben tener misma dimensión para cos_sim
    if emb_dataset.dim != args.last_dim:
        raise ValueError(f"La dimensión del JSON ({emb_dataset.dim}) debe coincidir con --last-dim ({args.last_dim}) para calcular la similitud.")

    train(model, dataloader, device, args.epochs, args.lr, args.weight_decay, args.reg_weight)

    # Guardar solo las capas entrenadas para un reuso ligero
    torch.save({
        "hidden_layers_state": model.hidden_layers.state_dict(),
        "config": {
            "model_name": args.model_name,
            "num_hidden_layers": args.hidden_layers,
            "last_dim": args.last_dim,
        }
    }, args.out)
    print(f"Pesos guardados en: {args.out}")


if __name__ == "__main__":
    main()
