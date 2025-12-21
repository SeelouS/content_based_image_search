"""Query Image → Find most similar image (uses binary extractor and JSON embeddings)

Usage examples:
  python -m src.query_image --query path/to/img.jpg --dataset binaryEmbeddings/testBin/binary_results.json --weights artifacts/twelfthOutput/binaryExtractor.pt --k 1 --metric auto

If --dataset is omitted the script will try to find a JSON under `binaryEmbeddings/`.
"""
from __future__ import annotations
import os
import argparse
import logging
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

# Import internal modules (must run from project root with venv active)
try:
    from src.apply_flann import load_json_embeddings
    from src.binaryFeatureExtractor import BinaryFeatureExtractor
except Exception as exc:
    raise ImportError(
        "Could not import project modules. Ensure you run 'python -m src.query_image' from the project root with your virtualenv activated and that dependencies are installed. "
        "If the error mentions 'sentence_transformers', install it using: 'python -m pip install sentence-transformers'. "
        f"Full error: {exc}"
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("query_image")


def find_first_json(root_dir: str = "binaryEmbeddings") -> str | None:
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith('.json'):
                return os.path.join(root, f)
    return None


def load_model(weights_path: str, device: str = 'cpu') -> BinaryFeatureExtractor:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)
    cfg = ckpt.get('config', {})
    model = BinaryFeatureExtractor(
        model_name=cfg.get('model_name', 'clip-ViT-B-32'),
        num_of_hidden_layers=cfg.get('num_hidden_layers', 1),
        last_layer_dimension=cfg.get('last_dim', 512),
        device=device,
    )
    if 'hidden_layers_state' in ckpt:
        model.hidden_layers.load_state_dict(ckpt['hidden_layers_state'])
    model.eval()
    return model


def compute_query_embedding(model: BinaryFeatureExtractor, image_path: str, device: str = 'cpu'):
    img = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        emb = model.forward(img)
    return emb.squeeze().cpu().numpy().astype(np.uint8)


def topk_by_distance(query_vec: np.ndarray, mat: np.ndarray, ids: List[str], k: int = 3, metric: str = 'auto') -> List[Tuple[str, float, int]]:
    # metric: 'auto' | 'hamming' | 'euclidean'
    is_binary = set(np.unique(mat).tolist()).issubset({0, 1})
    metric_used = 'hamming' if (metric == 'auto' and is_binary) or metric == 'hamming' else 'euclidean'

    if metric_used == 'hamming':
        q = query_vec.reshape(1, -1).astype(np.uint8)
        mat_bool = (mat != 0).astype(np.uint8)
        dists = np.count_nonzero(mat_bool != q, axis=1) / mat_bool.shape[1]
    else:
        q = query_vec.astype(np.float32)
        dists = np.linalg.norm(mat - q, axis=1)

    k = min(k, len(dists))
    idxs = np.argpartition(dists, k)[:k]
    idxs = idxs[np.argsort(dists[idxs])]
    return [(ids[int(i)], float(dists[int(i)]), int(i)) for i in idxs]


def main():
    n_images_to_return = 3
    p = argparse.ArgumentParser(description="Find most similar image to a query using your binary extractor and JSON embeddings")
    p.add_argument("--query", required=True, help="Path to query image")
    p.add_argument("--dataset", help="Path to dataset JSON (if omitted, first JSON under binaryEmbeddings/ is used)")
    p.add_argument("--weights", default="artifacts/twelfthOutput/binaryExtractor.pt", help="Path to the extractor checkpoint")
    p.add_argument("--k", type=int, default=3, help="Top-k to return (default 3)")
    p.add_argument("--metric", choices=["auto", "hamming", "euclidean"], default="auto", help="Distance metric")
    p.add_argument("--device", default=None, help="Device to run model on (cuda/cpu). If omitted, auto-detects")
    args = p.parse_args()

    dataset_file = args.dataset or find_first_json()
    if not dataset_file:
        logger.error("No dataset JSON found. Provide --dataset or ensure binaryEmbeddings/ contains a JSON file.")
        return
    if not os.path.exists(dataset_file):
        logger.error(f"Dataset file not found: {dataset_file}")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading dataset from {dataset_file}")
    ids, mat, meta, is_binary = load_json_embeddings(dataset_file)

    logger.info(f"Loading model from {args.weights} on device {device}")
    model = load_model(args.weights, device=device)

    logger.info(f"Computing query embedding for {args.query}")
    q_vec = compute_query_embedding(model, args.query, device=device)

    # Enforce a minimum of 3 results (so user gets multiple suggestions)
    effective_k = max(n_images_to_return, int(args.k))
    if args.k < 3:
        logger.info(f"Requested k={args.k} less than 3 — using k={effective_k} instead to return more results")
    logger.info("Searching top-k")
    top = topk_by_distance(q_vec, mat, ids, k=effective_k, metric=args.metric)

    for rank, (iid, dist, idx) in enumerate(top, 1):
        img_name = meta.get(iid, {}).get('image', iid)
        cat = meta.get(iid, {}).get('category_id')
        print(f"#{rank}: image={img_name}, id={iid}, distance={dist}, category_id={cat}")


if __name__ == "__main__":
    main()
