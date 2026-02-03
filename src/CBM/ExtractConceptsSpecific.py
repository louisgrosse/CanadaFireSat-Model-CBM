import sys
import os
import glob
from typing import List, Optional
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy

# ----------------------------------------------------------------------
# MS-CLIP setup
# ----------------------------------------------------------------------

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model_, preprocess, tokenizer = build_model(
    model_name="Llama3-MS-CLIP-Base",  
    pretrained=True,
    ckpt_path=None,
    device=device,
    channels=10,                        
)
model = model_.to(device).eval()

# ----------------------------------------------------------------------
# Embedding + k-means on term embeddings
# ----------------------------------------------------------------------

def tokenize_phrases(
    phrases: List[str],
    batch_size: int,
) -> torch.Tensor:
    embs = []
    for i in tqdm(range(0, len(phrases), batch_size), desc="Encoding terms"):
        batch = phrases[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        z = model.inference_text(tokens)
        z = F.normalize(z, dim=-1)
        embs.append(z)
    return torch.cat(embs, dim=0)   # [N_terms, D]


def mini_batch_kmeans_with_early_stopping(
    data: torch.Tensor,
    k: int,
    batch_size: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> torch.Tensor:
    """
    data: [N, D] on GPU (L2-normalized).
    Returns centroids: [k, D] on GPU.
    """
    num_samples, _ = data.shape
    indices = torch.randperm(num_samples, device=data.device)[:k]
    centroids = data[indices]

    for _ in tqdm(range(max_iter), desc="Mini-batch k-means"):
        batch_indices = torch.randperm(num_samples, device=data.device)[:batch_size]
        batch = data[batch_indices]

        distances = torch.cdist(batch, centroids)            # [B, k]
        labels = torch.argmin(distances, dim=1)              # [B]

        new_centroids = centroids.clone()
        for i in range(k):
            cluster_points = batch[labels == i]
            if cluster_points.numel() > 0:
                new_centroids[i] = cluster_points.mean(dim=0)

        if torch.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    #Choose here the general dictionary to use
    captions_dir = "/home/louis/Code/CanadaFireSat-Model-CBM/results/DictionnaryWordsGeneral/spacy_msclip_general_f1k2n5_homogeneous.csv"  # <== adjust
    out_dir = "/home/louis/Code/CanadaFireSat-Model-CBM/results/DictionnaryWordsNoPropn"
    out_stub = "kmeans_k2n5_balanced"
    k = 10 #number in thousands

    terms = pd.read_csv(captions_dir)["concept"]

    N_terms = len(terms)
    print(f"[INFO] Using {N_terms} unique terms as k-means input.")

    # --------- 2) Embed terms in MS-CLIP space ----------
    term_embs = tokenize_phrases(terms, batch_size=512)
    # term_embs: [N_terms, D] on GPU

    # --------- 3) K-means on term embeddings ----------
    N_CLUSTERS = min(50_000, N_terms)  # cannot have more clusters than terms
    print(f"[INFO] Clustering {N_terms} terms into {N_CLUSTERS} clusters.")

    centroids = mini_batch_kmeans_with_early_stopping(
        term_embs,
        k=N_CLUSTERS,
        batch_size=4096,
        max_iter=100,
        tol=1e-4,
    )
    centroids = F.normalize(centroids, dim=1) 

    # --------- 4) Assign each term to a cluster ----------
    cluster_ids = torch.empty(N_terms, dtype=torch.long, device=device)
    assign_batch_size = 4096

    print("[INFO] Assigning terms to clusters...")
    for start in tqdm(range(0, N_terms, assign_batch_size), desc="Assigning clusters"):
        end = min(start + assign_batch_size, N_terms)
        batch = term_embs[start:end]       # [B, D]
        distances = torch.cdist(batch, centroids)  # [B, k]
        labels = torch.argmin(distances, dim=1)
        cluster_ids[start:end] = labels

    # --------- 5) For each cluster, pick the closest term as label ----------
    cluster_to_indices: defaultdict[int, List[int]] = defaultdict(list)
    for i in range(N_terms):
        cid = int(cluster_ids[i].item())
        cluster_to_indices[cid].append(i)

    cluster_label_term_idx: dict[int, int] = {}
    for cid, idxs in tqdm(cluster_to_indices.items(), desc="Selecting representatives"):
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
        sims = (term_embs[idxs_t] * centroids[cid]).sum(dim=1)  # cosine similarity
        best_pos = int(torch.argmax(sims).item())
        best_local = idxs[best_pos]
        cluster_label_term_idx[cid] = best_local

    # cluster_id -> term string
    cluster_label = {cid: terms[i] for cid, i in cluster_label_term_idx.items()}

    # --------- 6) Save dictionary ----------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_stub}_{str(k)}k.csv")

    df = pd.DataFrame({
        "cluster_id": list(cluster_label.keys()),
        "concept": list(cluster_label.values()),
    })


    df.to_csv(out_path, index=False)

    print(f"[INFO] Saved {len(cluster_label)} concepts to {out_path}")



if __name__ == "__main__":
    main()
