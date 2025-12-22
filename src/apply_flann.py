"""Apply FLANN retrieval to JSON embeddings in a directory tree.

Behavior:
- For each JSON file under an input directory (default: `binaryEmbeddings`),
  try to parse embeddings in a few common formats and build an index.
- For each embedding (treated as a query), find the top-k nearest neighbors
  among the same file's embeddings (excluding itself) and write a JSON
  file with rankings under the output directory (default: `binaryEmbeddings_flann`).

Usage:
    python -m src.apply_flann --input-dir binaryEmbeddings --output-dir binaryEmbeddings_flann --k 10

The script prefers `pyflann` (FLANN) if installed; otherwise it falls back to scikit-learn's NearestNeighbors.
"""

from __future__ import annotations
import os
import json
import argparse
import logging
from typing import Tuple, List, Dict, Any

import numpy as np

from tqdm import tqdm
HAS_TQDM = True

# Try to import FLANN first, then sklearn, otherwise fallback to brute force
USE_FLANN = False
USE_SKLEARN = False

import faiss
USE_FLANN = True

from sklearn.neighbors import NearestNeighbors
USE_SKLEARN = True



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_json_embeddings(path: str) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]], bool]:
    """Load embeddings from a JSON file and return (ids, matrix, meta, is_binary).

    Supported formats:
    - List of dicts: [{"id": "xxx", "embedding": [..]}, ...]
    - Dict of id->embedding: {"id1": [..], ...}
    - List of dicts with keys {"image", "category_id", "binary_embedding"}
    - Top-level key 'results' or 'embeddings' containing one of the above
    - List of lists (no ids): returns generated ids as str(index)

    Returns:
      ids: list of identifiers (strings)
      mat: numpy array of shape (n, d)
      meta: mapping id -> metadata dict (may include 'image' and 'category_id')
      is_binary: True if vectors appear to be binary (0/1)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def is_embedding_list_of_dicts(x):
        return isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict)

    ids: List[str] = []
    vectors: List[List[float]] = []
    meta: Dict[str, Dict[str, Any]] = {}

    # unwrap common containers
    if isinstance(data, dict):
        # Support our own output format with top-level 'ids' and 'embeddings'
        if "ids" in data and "embeddings" in data and isinstance(data["ids"], list) and isinstance(data["embeddings"], list):
            ids = [str(x) for x in data["ids"]]
            vectors = [list(v) for v in data["embeddings"]]
            meta = data.get("meta", {}) or {}
        # direct map of id->embedding (legacy)
        elif all(isinstance(v, (list, tuple)) for v in data.values()):
            for k, v in data.items():
                ids.append(str(k))
                vectors.append(list(v))
                meta[str(k)] = {}
        # results key may contain either embeddings or neighbor mappings; prefer embeddings
        elif "results" in data and (isinstance(data["results"], list) or isinstance(data["results"], dict)):
            data = data["results"]
        elif "embeddings" in data:
            data = data["embeddings"]

    if isinstance(data, list):
        if is_embedding_list_of_dicts(data):
            # keys may be 'id' and 'embedding' or variants, or image/category_id/binary_embedding
            for i, item in enumerate(data):
                identifier = item.get("id") or item.get("image") or item.get("image_id") or str(i)
                # Try known embedding fields
                vec = item.get("clip_embedding") or item.get("vec") or item.get("features") or item.get("binary_embedding")
                if vec is None:
                    continue
                ids.append(str(identifier))
                vectors.append(list(vec))
                # keep any useful metadata
                m: Dict[str, Any] = {}
                if "image" in item:
                    m["image"] = item.get("image")
                if "category_id" in item:
                    m["category_id"] = item.get("category_id")
                meta[str(identifier)] = m
        elif len(data) > 0 and isinstance(data[0], (list, tuple)):
            # list of vectors without ids
            for i, vec in enumerate(data):
                ids.append(str(i))
                vectors.append(list(vec))
                meta[str(i)] = {}

    if len(ids) == 0 or len(vectors) == 0:
        raise ValueError(f"Could not parse embeddings from {path}")

    mat = np.asarray(vectors)

    # Detect binary embeddings (0/1 only)
    unique_vals = np.unique(mat)
    is_binary = bool(set(unique_vals.tolist()).issubset({0, 1}))

    # Ensure float32 output for metric computations (except we keep binary as int->bool if needed)
    mat = mat.astype(np.float32)
    return ids, mat, meta, is_binary


def apply_flann_to_matrix(ids: List[str], mat: np.ndarray, k: int = 10, metric: str = "euclidean", is_binary: bool = False, batch_size: int = 256) -> Dict[str, List[Dict[str, Any]]]:
    n = mat.shape[0]
    if n == 0:
        return {}
    k_query = min(k + 1, n)  # +1 because we'll drop self if present

    logger.info(f"Indexing {n} vectors (dim={mat.shape[1]}) using metric={metric} (pyflann={USE_FLANN}, sklearn={USE_SKLEARN}), batch_size={batch_size}")

    # If metric is 'auto' and embeddings are binary, prefer hamming
    if metric == "auto":
        metric_used = "hamming" if is_binary else "euclidean"
    else:
        metric_used = metric

    results: Dict[str, List[Dict[str, Any]]] = {qid: [] for qid in ids}

    # Prefer ANN libraries when available
    if metric_used == "hamming":
        mat_bool = (mat != 0).astype(np.uint8)
        if USE_SKLEARN:
            nn = NearestNeighbors(n_neighbors=k_query, algorithm='auto', metric='hamming', n_jobs=-1)
            nn.fit(mat_bool)
            dists, indices = nn.kneighbors(mat_bool)
            for i, qid in enumerate(ids):
                neighs = []
                for idx, dist in zip(indices[i], dists[i]):
                    if int(idx) == i:
                        continue
                    neighs.append({"id": ids[int(idx)], "distance": float(dist)})
                    if len(neighs) >= k:
                        break
                results[qid] = neighs
            return results
        else:
            # Brute-force but in batches to avoid huge memory use
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                q = mat_bool[start:end]
                # compute hamming distance chunk: fraction of differing bits
                dists_chunk = np.count_nonzero(q[:, None, :] != mat_bool[None, :, :], axis=2).astype(np.float32) / mat_bool.shape[1]
                # for queries that overlap with database (same dataset), set self-distance to large value
                for i_rel, i_abs in enumerate(range(start, end)):
                    if i_abs < n:
                        dists_chunk[i_rel, i_abs] = np.inf
                # find top-k_query for each query in chunk
                idx_part = np.argpartition(dists_chunk, k_query, axis=1)[:, :k_query]
                dists_part = np.take_along_axis(dists_chunk, idx_part, axis=1)
                order = np.argsort(dists_part, axis=1)
                idx_sorted = np.take_along_axis(idx_part, order, axis=1)
                dists_sorted = np.take_along_axis(dists_part, order, axis=1)
                for i_rel, i_abs in enumerate(range(start, end)):
                    neighs = []
                    for idx, dist in zip(idx_sorted[i_rel], dists_sorted[i_rel]):
                        neighs.append({"id": ids[int(idx)], "distance": float(dist)})
                        if len(neighs) >= k:
                            break
                    results[ids[i_abs]] = neighs
            return results
    else:
        # Euclidean or other numeric distances
        if USE_FLANN and metric_used == "euclidean":

            # FAISS requiere float32
            mat32 = mat.astype(np.float32)

            # Dimensión de los vectores
            dim = mat32.shape[1]

            # IndexFlatL2 = distancia euclídea exacta
            index = faiss.IndexFlatL2(dim)

            # Añadir todos los vectores al índice
            index.add(mat32)

            # Búsqueda k-NN
            dists, indices = index.search(mat32, k_query)

            # Construcción del resultado igual que antes
            for i, qid in enumerate(ids):
                neighs = []
                for idx, dist in zip(indices[i], dists[i]):

                    # Evitar el propio punto
                    if int(idx) == i:
                        continue

                    neighs.append({
                        "id": ids[int(idx)],
                        "distance": float(dist)
                    })

                    if len(neighs) >= k:
                        break

                results[qid] = neighs

            return results

        elif USE_SKLEARN:
            nn = NearestNeighbors(n_neighbors=k_query, algorithm='auto', metric='euclidean', n_jobs=-1)
            nn.fit(mat)
            dists, indices = nn.kneighbors(mat)
            for i, qid in enumerate(ids):
                neighs = []
                for idx, dist in zip(indices[i], dists[i]):
                    if int(idx) == i:
                        continue
                    neighs.append({"id": ids[int(idx)], "distance": float(dist)})
                    if len(neighs) >= k:
                        break
                results[qid] = neighs
            return results
        else:
            # Brute-force with batching and argpartition
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                q = mat[start:end]
                # compute distances chunk: shape (batch, n)
                dists_chunk = np.linalg.norm(q[:, None, :] - mat[None, :, :], axis=2)
                # set self distances to inf
                for i_rel, i_abs in enumerate(range(start, end)):
                    dists_chunk[i_rel, i_abs] = np.inf
                # find top-k_query per row using argpartition
                idx_part = np.argpartition(dists_chunk, k_query, axis=1)[:, :k_query]
                dists_part = np.take_along_axis(dists_chunk, idx_part, axis=1)
                order = np.argsort(dists_part, axis=1)
                idx_sorted = np.take_along_axis(idx_part, order, axis=1)
                dists_sorted = np.take_along_axis(dists_part, order, axis=1)
                for i_rel, i_abs in enumerate(range(start, end)):
                    neighs = []
                    for idx, dist in zip(idx_sorted[i_rel], dists_sorted[i_rel]):
                        neighs.append({"id": ids[int(idx)], "distance": float(dist)})
                        if len(neighs) >= k:
                            break
                    results[ids[i_abs]] = neighs
            return results


def process_file(inpath: str, outpath: str, k: int = 10, metric: str = "auto", batch_size: int = 256, output_format: str = "mapping", overwrite: bool = True):
    logger.info(f"Processing {inpath} -> {outpath} (format={output_format})")
    try:
        ids, mat, meta, is_binary = load_json_embeddings(inpath)
    except Exception as e:
        logger.warning(f"Skipping {inpath}: {e}")
        return

    # Keep original embeddings in a JSON-serializable format
    if is_binary:
        embeddings_list = mat.astype(np.uint8).tolist()
    else:
        embeddings_list = mat.tolist()

    results = apply_flann_to_matrix(ids, mat, k=k, metric=metric, is_binary=is_binary, batch_size=batch_size)

    # If the user wants a simple annotated list with fields ordered, build it
    if output_format == "annotated_list":
        try:
            with open(inpath, "r", encoding="utf-8") as rf:
                raw = json.load(rf)
        except Exception:
            raw = None

        annotated = []
        # If original was a list of dicts, preserve original fields order where possible
        if isinstance(raw, list):
            # Build a mapping from known identifiers to original entries
            by_id = {}
            for i, item in enumerate(raw):
                identifier = item.get("id") or item.get("image") or item.get("image_id") or str(i)
                by_id[str(identifier)] = item

            for qid in ids:
                orig = by_id.get(qid, {})
                # ordered fields: image, category_id, binary_embedding
                entry = {}
                if "image" in orig:
                    entry["image"] = orig.get("image")
                if "category_id" in orig:
                    entry["category_id"] = orig.get("category_id")
                # prefer binary_embedding, else embedding/vec/features
                if "binary_embedding" in orig:
                    entry["binary_embedding"] = orig.get("binary_embedding")
                elif "embedding" in orig:
                    entry["binary_embedding"] = orig.get("embedding")
                elif "vec" in orig:
                    entry["binary_embedding"] = orig.get("vec")
                else:
                    # fallback to our embeddings_list
                    idx = ids.index(qid)
                    entry["binary_embedding"] = embeddings_list[idx]

                # add neighbors (only ids + distance)
                neigh = results.get(qid, [])
                entry["neighbors"] = neigh
                annotated.append(entry)
        else:
            # If raw is not a list, build entries from ids/meta/embeddings
            for i, qid in enumerate(ids):
                entry = {}
                m = meta.get(qid, {})
                if "image" in m:
                    entry["image"] = m.get("image")
                if "category_id" in m:
                    entry["category_id"] = m.get("category_id")
                entry["binary_embedding"] = embeddings_list[i]
                entry["neighbors"] = results.get(qid, [])
                annotated.append(entry)

        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(annotated, f, indent=2, ensure_ascii=False)
        return

    # Default: mapping style (backwards compatible)
    out_obj = {
        "ids": ids,
        "embeddings": embeddings_list,
        "meta": meta,
        "results": results,
        "meta_source_file": os.path.basename(inpath)
    }

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)


def walk_and_process(input_dir: str, output_dir: str, k: int = 10, metric: str = "auto", pattern: str = ".json", show_progress: bool = True, batch_size: int = 256, output_format: str = "mapping"):
    # Gather files first so we can display a global progress bar if desired
    files_to_process = []
    for root, dirs, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        for fname in files:
            if not fname.lower().endswith(pattern):
                continue
            inpath = os.path.join(root, fname)
            outdir = os.path.join(output_dir, rel) if rel != os.curdir else output_dir
            outpath = os.path.join(outdir, fname)
            files_to_process.append((inpath, outpath))

    iterator = files_to_process
    if show_progress and HAS_TQDM:
        iterator = tqdm(files_to_process, desc="Processing files", unit="file")

    for inpath, outpath in iterator:
        process_file(inpath, outpath, k=k, metric=metric, batch_size=batch_size, output_format=output_format)


def parse_args():
    p = argparse.ArgumentParser(description="Apply FLANN retrieval to JSON embedding files")
    p.add_argument("--input-dir", default="binaryEmbeddings", help="Input directory containing embedding JSONs")
    p.add_argument("--output-dir", default="binaryEmbeddings_flann", help="Output directory to write ranked JSONs")
    p.add_argument("--k", type=int, default=10, help="Number of neighbors to keep per query")
    p.add_argument("--metric", choices=["auto", "euclidean", "hamming"], default="auto", help="Distance metric to use (auto detects binary embeddings)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for brute-force computation (memory/speed tradeoff)")
    p.add_argument("--output-format", choices=["mapping", "annotated_list", "neighbor_ids"], default="mapping", help="Output format per file: mapping (default) or annotated_list (list of ordered image objects) or neighbor_ids (compact)")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar (useful if tqdm is not installed or when running in non-interactive environments)")
    return p.parse_args()


def main():
    args = parse_args()
    show_progress = not args.no_progress
    if show_progress and not HAS_TQDM:
        logger.info("tqdm is not installed; running without progress bars. Install 'tqdm' to enable progress bars.")
    walk_and_process(args.input_dir, args.output_dir, k=args.k, metric=args.metric, show_progress=show_progress, batch_size=args.batch_size, output_format=args.output_format)


if __name__ == "__main__":
    main()
