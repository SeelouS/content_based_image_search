#!/usr/bin/env python3
"""Normalize an eval file (JSON/PKL) into a numeric NumPy matrix (.npy).

- Supports JSON (including various embedding schemas via apply_flann.load_json_embeddings)
- Supports Pickle containing dict/list/list-of-dicts
- Outputs .npy float32 matrix of shape (N, D)
- Optionally writes ids to a .txt and a combined .npz

Usage:
  python ./src/normalize_evalfile.py \
    --in ./clipLabels/clip_labels.json \
    --out ./clipLabels/clip_labels.npy \
    --ids-out ./clipLabels/clip_labels_ids.txt \
    --npz-out ./clipLabels/clip_labels.npz
"""
from __future__ import annotations
import os
import sys
import argparse
import json
import pickle
import numpy as np
from typing import Any, Tuple, List, Dict


def to_numeric_matrix(x: Any) -> np.ndarray:
    # If dict with known keys, extract embeddings
    if isinstance(x, dict):
        for k in ("embeddings", "features", "vecs", "vectors"):
            if k in x:
                x = x[k]
                break
    # If list of dicts, extract known vector fields
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict):
        out: List[Any] = []
        for item in x:
            vec = None
            for k in ("binary_embedding", "embedding", "vec", "features"):
                if k in item:
                    vec = item[k]
                    break
            if vec is not None:
                out.append(vec)
        x = out
    arr = np.array(x, dtype=object)
    # Try coercions into numeric 2D
    if arr.dtype == object:
        try:
            arr = np.array(list(arr), dtype=np.float32)
        except Exception:
            pass
    if arr.dtype == object:
        try:
            arr = np.stack(arr.astype(object).tolist()).astype(np.float32)
        except Exception:
            pass
    if arr.dtype == object:
        raise ValueError("Could not coerce data to a numeric matrix. Inspect the input structure.")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array (N,D), got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)

def load_json_embeddings_local(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def is_list_of_dicts(x):
        return isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict)

    ids: List[str] = []
    vectors: List[List[float]] = []

    if isinstance(data, dict):
        if "ids" in data and "embeddings" in data and isinstance(data["ids"], list) and isinstance(data["embeddings"], list):
            ids = [str(x) for x in data["ids"]]
            vectors = [list(v) for v in data["embeddings"]]
        elif all(isinstance(v, (list, tuple)) for v in data.values()) and len(data) > 0:
            for k, v in data.items():
                ids.append(str(k))
                vectors.append(list(v))
        elif "results" in data and (isinstance(data["results"], list) or isinstance(data["results"], dict)):
            data = data["results"]
        elif "embeddings" in data:
            data = data["embeddings"]

    if isinstance(data, list):
        if is_list_of_dicts(data):
            for i, item in enumerate(data):
                identifier = item.get("id") or item.get("image") or item.get("image_id") or str(i)
                vec = item.get("clip_embedding") or item.get("vec") or item.get("features") or item.get("binary_embedding") or item.get("embedding")
                if vec is None:
                    continue
                ids.append(str(identifier))
                vectors.append(list(vec))
        elif len(data) > 0 and isinstance(data[0], (list, tuple)):
            for i, vec in enumerate(data):
                ids.append(str(i))
                vectors.append(list(vec))

    if len(ids) == 0 or len(vectors) == 0:
        raise ValueError(f"Could not parse embeddings from {path}")

    mat = np.asarray(vectors, dtype=np.float32)
    return ids, mat


def load_any(path: str) -> Tuple[np.ndarray, List[str] | None]:
    ext = os.path.splitext(path)[1].lower()
    ids: List[str] | None = None
    if ext in (".json", ".jsonl"):
        # First try local robust JSON parser
        ids, mat = load_json_embeddings_local(path)
        return mat.astype(np.float32, copy=False), ids
    elif ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # Try to capture ids if present
        if isinstance(obj, dict):
            if "ids" in obj and isinstance(obj["ids"], list):
                ids = [str(x) for x in obj["ids"]]
        mat = to_numeric_matrix(obj)
        return mat, ids
    elif ext == ".npy":
        arr = np.load(path)
        arr = to_numeric_matrix(arr)
        return arr, None
    else:
        raise ValueError(f"Unsupported extension for input: {ext}")


def parse_args():
    ap = argparse.ArgumentParser(description="Normalize eval file to .npy (float32, shape N,D)")
    ap.add_argument("--in", dest="inp", required=True, help="Input file (.json/.jsonl/.pkl/.npy)")
    ap.add_argument("--out", dest="out", required=True, help="Output .npy file")
    ap.add_argument("--ids-out", dest="ids_out", default=None, help="Optional path to write ids .txt")
    ap.add_argument("--npz-out", dest="npz_out", default=None, help="Optional path to write combined .npz (embeddings+ids)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mat, ids = load_any(args.inp)
    # Save .npy
    np.save(args.out, mat)
    # Optionally ids
    if args.ids_out and ids is not None:
        with open(args.ids_out, "w", encoding="utf-8") as f:
            f.write("\n".join(ids))
    # Optionally .npz
    if args.npz_out:
        if ids is not None:
            np.savez_compressed(args.npz_out, embeddings=mat, ids=np.array(ids, dtype=object))
        else:
            np.savez_compressed(args.npz_out, embeddings=mat)
    print(f"Saved .npy: {args.out} shape={mat.shape}")
    if args.ids_out and ids is not None:
        print(f"Saved ids: {args.ids_out} count={len(ids)}")
    if args.npz_out:
        print(f"Saved .npz: {args.npz_out}")


if __name__ == "__main__":
    main()
