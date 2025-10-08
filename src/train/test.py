import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F

# Import MS-CLIP model builder
import sys
sys.path.append("MS-CLIP")
from msclip.inference.utils import build_model

# ---------------- CONFIG ----------------
parquet_path = "/home/louis/Code/wildfire-forecast/worldstrat/data/worldstrat_train_0001.parquet" 
num_samples = 50               # subset for speed
resize_to = 224
channels = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Llama3-MS-CLIP-Base"
ckpt_path = None  # or path to ckpt if local
# ----------------------------------------

print(f"Loading parquet: {parquet_path}")
df = pd.read_parquet(parquet_path)
if num_samples:
    df = df.sample(min(num_samples, len(df)), random_state=0)

print("Loading MS-CLIP...")
msclip_model, preprocess, tokenizer = build_model(
    model_name=model_name,
    pretrained=True,
    ckpt_path=ckpt_path,
    device=device,
    channels=channels,
)

encoder = msclip_model.image_encoder.eval().to(device)

# Utility
def preprocess_tensor(arr10, arr20):
    arr10 = torch.from_numpy(pickle.loads(arr10)).float()
    arr20 = torch.from_numpy(pickle.loads(arr20)).float()
    arr20 = F.interpolate(arr20.unsqueeze(0), size=arr10.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
    x = torch.cat([arr10, arr20], dim=0)
    x = F.interpolate(x.unsqueeze(0), size=(resize_to, resize_to), mode="bilinear", align_corners=False)
    return x

mse_vals = []
cos_vals = []

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing MS-CLIP Embedding MSE"):
        # Preprocess
        x_l1c = preprocess_tensor(row["10x_L1C"], row["20x_L1C"]).to(device)
        x_l2a = preprocess_tensor(row["10x_L2A"], row["20x_L2A"]).to(device)

        # Get patch embeddings
        feat_l1c = encoder.get_patch_embeddings(x_l1c)  # [1, P, D]
        feat_l2a = encoder.get_patch_embeddings(x_l2a)

        # Remove CLS if present
        if feat_l1c.shape[1] == feat_l2a.shape[1] + 1:
            feat_l1c, feat_l2a = feat_l1c[:, 1:], feat_l2a
        elif feat_l2a.shape[1] == feat_l1c.shape[1] + 1:
            feat_l2a, feat_l1c = feat_l2a[:, 1:], feat_l1c

        # Compute per-patch cosine + MSE
        mse = F.mse_loss(feat_l1c, feat_l2a).item()
        cos = F.cosine_similarity(feat_l1c, feat_l2a, dim=-1).mean().item()
        mse_vals.append(mse)
        cos_vals.append(cos)

mse_vals = np.array(mse_vals)
cos_vals = np.array(cos_vals)

print("\n✅ Embedding-level differences")
print(f"Mean embedding MSE : {mse_vals.mean():.6f} ± {mse_vals.std():.6f}")
print(f"Mean cosine similarity : {cos_vals.mean():.6f} ± {cos_vals.std():.6f}")
print(f"Min / Max MSE : {mse_vals.min():.6f} / {mse_vals.max():.6f}")
print(f"Min / Max Cosine : {cos_vals.min():.6f} / {cos_vals.max():.6f}")
