"""Mini GUI app to query images using existing embeddings and extractors.

Usage:
python -m src.miniapp --dataset binaryEmbeddings_faiss/your.json --weights artifacts/twelfthOutput/binaryExtractor.pt --k 5 --metric auto

Behavior:
- Loads dataset embeddings via `load_json_embeddings`.
- If `--weights` is a CLIP model name (starts with 'clip' and not an existing file), the app uses SentenceTransformer to compute float CLIP embeddings for queries.
- Otherwise it loads the project's BinaryFeatureExtractor checkpoint and uses it to compute binary embeddings.
- GUI lets you:
- Browse/select an image file to query
- Or select an image from the dataset list to use as query
- View top-k results and open result images with the OS default image viewer

"""
from __future__ import annotations
import argparse
import os
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("miniapp")

# Import project utilities
try:
    from src.apply_Indexer import load_json_embeddings
    from src.query_image import load_model, compute_query_embedding, topk_by_distance
except Exception as exc:
    raise ImportError("Could not import project modules. Run the app from project root with your venv active.") from exc


class MiniApp(tk.Tk):
    def __init__(self, dataset_file: str, weights: str, k: int, metric: str, device: str):
        super().__init__()
        self.title("Image Query MiniApp")
        self.geometry("1000x600")

        self.dataset_file = dataset_file
        self.weights = weights or ""
        self.k = k
        self.metric = metric
        self.device = device

        logger.info(f"Loading dataset {dataset_file}")
        try:
            self.ids, self.mat, self.meta, self.is_binary = load_json_embeddings(dataset_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
            self.destroy()
            return

        # Build list of display names
        self.display_names = [self.meta.get(i, {}).get('image', i) for i in self.ids]

        # UI frames
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        mid = ttk.Frame(self)
        mid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        # Left: dataset list
        ttk.Label(left, text="Dataset Images").pack()
        self.listbox = tk.Listbox(left, width=40, height=30)
        self.listbox.pack(side=tk.LEFT, fill=tk.Y)
        for name in self.display_names:
            self.listbox.insert(tk.END, name)
        self.listbox.bind('<Double-Button-1>', self.open_list_image)

        list_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=list_scroll.set)

        # Middle: query controls and results
        ttk.Label(mid, text="Query Image").pack()
        btn_frame = ttk.Frame(mid)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Browse image...", command=self.browse_image).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Use selected dataset image", command=self.use_selected_as_query).pack(side=tk.LEFT, padx=8)

        self.query_preview = ttk.Label(mid, text="No query selected", anchor="center")
        self.query_preview.pack(fill=tk.BOTH, expand=False, pady=8)

        ttk.Label(mid, text="Top-k Results").pack()
        self.results_list = tk.Listbox(mid, width=60, height=20)
        self.results_list.pack(fill=tk.BOTH, expand=True)
        self.results_list.bind('<Double-Button-1>', self.open_result_image)
        
        self.time_label = ttk.Label(right, text="Query time: 0.00 seconds")
        self.time_label.pack(pady=8)

        # Right: settings and actions
        ttk.Label(right, text="Settings").pack()
        ttk.Label(right, text=f"Weights: {self.weights}").pack()
        ttk.Label(right, text=f"k: {self.k}").pack()
        ttk.Label(right, text=f"metric: {self.metric}").pack()
        ttk.Button(right, text="Run query", command=self.run_query).pack(pady=8)

        # internal state
        self.query_path = None
        self.query_vec = None
        self.clip_mode = False

        # If weights argument indicates CLIP, prepare to compute CLIP embeddings
        if self.weights and (not os.path.exists(self.weights)) and self.weights.lower().startswith('clip'):
            self.clip_mode = True
            try:
                from sentence_transformers import SentenceTransformer
                self.clip_model = SentenceTransformer(self.weights)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CLIP model '{self.weights}': {e}")
                self.destroy()
                return
            if self.is_binary:
                messagebox.showerror("Error", "Dataset appears to contain binary embeddings. CLIP float embeddings cannot be compared to binary dataset.")
                self.destroy()
                return
        else:
            # load binary extractor model
            try:
                self.model = load_model(self.weights, device=self.device)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load extractor model: {e}")
                self.destroy()
                return

    def browse_image(self):
        p = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*")])
        if not p:
            return
        self.query_path = p
        self.show_query_preview(p)

    def use_selected_as_query(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("Info", "Select an image from the dataset list first")
            return
        idx = int(sel[0])
        iid = self.ids[idx]
        imgpath = self.meta.get(iid, {}).get('image') or iid
        imgpath = "..\\images\\" + imgpath
        if not os.path.exists(imgpath):
            messagebox.showerror("Error", f"Image file not found: {imgpath}")
            return
        self.query_path = imgpath
        self.show_query_preview(imgpath)

    def show_query_preview(self, path: str):
        try:
            im = Image.open(path).convert('RGB')
            im.thumbnail((256, 256))
            self._preview_im = ImageTk.PhotoImage(im)
            self.query_preview.config(image=self._preview_im, text='')
        except Exception as e:
            self.query_preview.config(text=f"Selected: {os.path.basename(path)}")

    def run_query(self):
        if not self.query_path:
            messagebox.showinfo("Info", "Select or browse a query image first")
            return
        start_time = time.time()
        
        try:
            if self.clip_mode:
                emb = self.clip_model.encode(Image.open(self.query_path).convert('RGB'), convert_to_tensor=False)
                q_vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            else:
                q_vec = compute_query_embedding(self.model, self.query_path, device=self.device)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute query embedding: {e}")
            return

        # compute top-k
        try:
            results = topk_by_distance(q_vec, self.mat, self.ids, k=self.k, metric=self.metric)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to search: {e}")
            return

        end_time = time.time()  # Captura el tiempo después de que se muestran los resultados
        elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido

        # Muestra el tiempo transcurrido en el label
        self.time_label.config(text=f"Query time: {elapsed_time:.2f} seconds")
        # display results
        self.results_list.delete(0, tk.END)
        self.current_results = results
        for rank, (iid, dist, idx) in enumerate(results, 1):
            name = self.meta.get(iid, {}).get('image', iid)
            self.results_list.insert(tk.END, f"#{rank}: {name} (id={iid}) dist={dist:.5f}")

    def open_list_image(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        iid = self.ids[idx]
        imgpath = self.meta.get(iid, {}).get('image') or iid
        imgpath = "..\\images\\" + imgpath
        if os.path.exists(imgpath):
            try:
                os.startfile(imgpath)
            except Exception:
                messagebox.showinfo("Open", f"Cannot open file: {imgpath}")
        else:
            messagebox.showinfo("Open", f"File not found: {imgpath}")

    def open_result_image(self, event=None):
        sel = self.results_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        iid = self.current_results[idx][0]
        imgpath = self.meta.get(iid, {}).get('image') or iid
        imgpath = "..\\images\\" + imgpath
        if os.path.exists(imgpath):
            try:
                os.startfile(imgpath)
            except Exception:
                messagebox.showinfo("Open", f"Cannot open file: {imgpath}")
        else:
            messagebox.showinfo("Open", f"File not found: {imgpath}")


def parse_args():
    p = argparse.ArgumentParser(description="Mini GUI for querying images")
    p.add_argument("--dataset", required=True, help="Path to dataset JSON (embeddings)")
    p.add_argument("--weights", default="artifacts/twelfthOutput/binaryExtractor.pt", help="Checkpoint path or CLIP model name (clip-*)")
    p.add_argument("--k", type=int, default=5, help="Number of neighbors to show")
    p.add_argument("--metric", choices=["auto", "hamming", "euclidean"], default="auto", help="Metric to use")
    p.add_argument("--device", default=None, help="Device to run model on (cuda/cpu)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    app = MiniApp(args.dataset, args.weights, args.k, args.metric, device)
    app.mainloop()
