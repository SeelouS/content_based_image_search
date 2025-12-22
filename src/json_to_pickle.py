#!/usr/bin/env python3
"""Convert all JSON files in a directory into Pickle files.

- Stores each .pkl next to the original .json (same name, different extension).
- Supports recursive traversal by default.
- Overwrites existing .pkl by default (configurable).

Usage:
    python ./src/json_to_pickle.py --dir ./clipLabels

Options:
    --dir          Directory to scan for .json files (required)
    --no-recursive Disable recursive traversal (default: recursive)
    --pattern      Filename pattern/extension to match (default: .json)
    --no-overwrite Do not overwrite existing .pkl files

Security note:
    Only load JSON from trusted sources if later you plan to use the .pkl files,
    as Pickle loading executes code and is unsafe for untrusted inputs.
"""

from __future__ import annotations
import os
import argparse
import json
import pickle
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_json_files(root: str, recursive: bool, pattern: str = ".json") -> List[str]:
    files: List[str] = []
    pattern = pattern.lower()
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(pattern):
                    files.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(root):
            if f.lower().endswith(pattern):
                files.append(os.path.join(root, f))
    return files


def _load_json_forgiving(inpath: str):
    """Load JSON; if standard load fails, try JSON Lines (one JSON object per line)."""
    try:
        with open(inpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Try JSONL
        items = []
        try:
            with open(inpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        # Not JSONL; re-raise original
                        items = None
                        break
            if items is not None and len(items) > 0:
                return items
        except Exception:
            pass
        raise


def convert_json_to_pickle(inpath: str, overwrite: bool = True) -> Tuple[str, bool]:
    """Convert a single JSON file to Pickle alongside the original.

    Returns:
        (outpath, success)
    """
    base, _ = os.path.splitext(inpath)
    outpath = base + ".pkl"

    if (not overwrite) and os.path.exists(outpath):
        logger.info(f"Skip (exists): {outpath}")
        return outpath, False

    # Skip empty source files
    try:
        sz = os.stat(inpath).st_size
    except Exception:
        sz = -1
    if sz == 0:
        logger.warning(f"Skip empty JSON file: {inpath}")
        return outpath, False

    try:
        data = _load_json_forgiving(inpath)
    except Exception as e:
        logger.error(f"Failed to read JSON {inpath}: {e}")
        return outpath, False

    # Write atomically to avoid empty files on failure
    tmp_out = outpath + ".tmp"
    try:
        with open(tmp_out, "wb") as g:
            pickle.dump(data, g, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        # Clean up tmp
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
        logger.error(f"Failed to write Pickle {outpath}: {e}")
        return outpath, False

    # Replace existing file atomically
    try:
        os.replace(tmp_out, outpath)
    except Exception as e:
        # Fallback: copy contents
        try:
            with open(tmp_out, "rb") as src, open(outpath, "wb") as dst:
                dst.write(src.read())
            os.remove(tmp_out)
        except Exception as e2:
            logger.error(f"Failed to finalize Pickle {outpath}: {e2}")
            return outpath, False

    # Verify written file has content and is loadable
    try:
        out_sz = os.stat(outpath).st_size
        if out_sz == 0:
            logger.error(f"Written pickle is empty, removing: {outpath}")
            os.remove(outpath)
            return outpath, False
        with open(outpath, "rb") as f:
            obj = pickle.load(f)
        # Warn if object is empty structure
        if isinstance(obj, (list, dict)) and len(obj) == 0:
            logger.warning(f"Pickle contains empty {type(obj).__name__}: {outpath}")
    except Exception as e:
        logger.error(f"Verification failed for {outpath}: {e}")
        try:
            os.remove(outpath)
        except Exception:
            pass
        return outpath, False

    logger.info(f"Converted: {inpath} -> {outpath}")
    return outpath, True


def parse_args():
    p = argparse.ArgumentParser(description="Convert all JSON files in a directory to Pickle (.pkl) in place")
    p.add_argument("--dir", required=True, help="Directory containing JSON files")
    p.add_argument("--no-recursive", action="store_true", help="Disable recursive traversal")
    p.add_argument("--pattern", default=".json", help="File extension/pattern to match (default .json)")
    p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing .pkl files")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.dir
    recursive = not args.no_recursive
    overwrite = not args.no_overwrite

    if not os.path.isdir(root):
        raise SystemExit(f"Directory does not exist: {root}")

    json_files = find_json_files(root, recursive=recursive, pattern=args.pattern)
    if not json_files:
        logger.warning(f"No JSON files found in {root} (recursive={recursive}, pattern={args.pattern})")
        return

    converted, failed = 0, 0
    for inpath in json_files:
        _, ok = convert_json_to_pickle(inpath, overwrite=overwrite)
        converted += int(ok)
        failed += int(not ok)

    logger.info(f"Done. Converted={converted}, Failed={failed}")


if __name__ == "__main__":
    main()
