import os, math
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

# --- Dataset bits from the HF dataset card ---
from ssl4eos12_dataset import (
    SSL4EOS12Dataset, collate_fn,
    S2L1C_MEAN, S2L1C_STD
)

# --- Model / tokenizer ---
# We use IBM's helper to build the OpenCLIP-based model + tokenizer wired for MS-CLIP
from msclip.inference.utils import build_model  # provided in IBM/MS-CLIP repo
# (If you prefer raw open_clip, see comment at bottom.)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths (edit to match your layout)
DATA_DIR = "data/ssl4eo-s12"        # root containing train/val from SSL4EO-S12 v1.1
SPLIT_DIR = f"{DATA_DIR}/splits"    # contains ssl4eos12_train.txt etc.
SAVE_PATH = "msclip_l1c_finetune.pt"

# --- Data transforms (L1C only) ---
train_transform = transforms.Compose([
    transforms.RandomCrop(224),  # Input patches are 264x264 in SSL4EO-S12
    transforms.Normalize(mean=S2L1C_MEAN, std=S2L1C_STD),
])
val_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.Normalize(mean=S2L1C_MEAN, std=S2L1C_STD),
])

# We load just S2L1C + captions. single_timestamp=True increases diversity (IBM’s tip)
train_set = SSL4EOS12Dataset(
    data_dir=f"{DATA_DIR}/train",
    split_file=f"{SPLIT_DIR}/ssl4eos12_train.txt",
    modalities=["S2L1C", "captions"],
    transform=train_transform,
    single_timestamp=True,
    # Make sure we get 1 caption per sample (not [caption, q1, q2, q3]).
    caption_col="caption",
)

val_set = SSL4EOS12Dataset(
    data_dir=f"{DATA_DIR}/val",
    split_file=f"{SPLIT_DIR}/ssl4eos12_val.txt",
    modalities=["S2L1C", "captions"],
    transform=val_transform,
    single_timestamp=True,
    caption_col="caption",
)

# Each zarr chunk already holds many samples; batch_size=2 -> ~128 samples per step (64 per file)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

# --- Build model & tokenizer ---
# This downloads "Llama3-MS-CLIP-Base" weights from HF and sets up the CLIP text model + multispectral image tower.
# It returns (model, preprocess, tokenizer). We won't use 'preprocess' since we have our own transform for L1C.
model, _preprocess, tokenizer = build_model(model_name="Llama3-MS-CLIP-Base")
model = model.to(DEVICE)
model.train()

# --- Optimizer / AMP ---
optim = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.2)  # small LR for stable FT
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

# Standard CLIP loss: symmetric cross-entropy with learnable temperature
def clip_contrastive_loss(img_emb, txt_emb, logit_scale):
    # Normalize
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    # Similarities
    logits_per_image = logit_scale.exp() * img_emb @ txt_emb.t()
    logits_per_text  = logits_per_image.t()
    targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_i = nn.functional.cross_entropy(logits_per_image, targets)
    loss_t = nn.functional.cross_entropy(logits_per_text,  targets)
    return (loss_i + loss_t) * 0.5

@torch.no_grad()
def evaluate_one_epoch():
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        imgs = batch["S2L1C"].to(DEVICE, non_blocking=True)   # shape: [N, C, 224, 224]
        caps = batch["captions"]
        # Handle any empty/None captions gracefully
        caps = ["a satellite image" if (c is None or (isinstance(c, float) and math.isnan(c))) else c for c in caps]
        tokens = tokenizer(caps).to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            img_emb = model.encode_image(imgs)
            txt_emb = model.encode_text(tokens)
            total += clip_contrastive_loss(img_emb, txt_emb, model.logit_scale).item() * imgs.size(0)
            n += imgs.size(0)
    model.train()
    return total / max(n, 1)

EPOCHS = 1  # set higher for real training (e.g., 2–5)
log_every = 50
best_val = float("inf")

for epoch in range(EPOCHS):
    for step, batch in enumerate(train_loader, start=1):
        imgs = batch["S2L1C"].to(DEVICE, non_blocking=True)   # [N, C, 224, 224], channels = S2-L1C bands used by loader
        caps = batch["captions"]
        caps = ["a satellite image" if (c is None or (isinstance(c, float) and math.isnan(c))) else c for c in caps]
        tokens = tokenizer(caps).to(DEVICE)

        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            img_emb = model.encode_image(imgs)
            txt_emb = model.encode_text(tokens)
            loss = clip_contrastive_loss(img_emb, txt_emb, model.logit_scale)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        if step % log_every == 0:
            print(f"[epoch {epoch+1}] step {step} | loss {loss.item():.4f}")

    # quick val
    val_loss = evaluate_one_epoch()
    print(f"[epoch {epoch+1}] val_loss {val_loss:.4f}")

    # save (best or every epoch)
    ckpt = {
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optim.state_dict(),
        "val_loss": val_loss,
        "config": {
            "modality": "S2L1C",
            "image_size": 224,
            "single_timestamp": True,
        },
    }
    torch.save(ckpt, SAVE_PATH)
    if val_loss < best_val:
        best_val = val_loss
        torch.save(ckpt, SAVE_PATH.replace(".pt", "_best.pt"))

print("Saved:", SAVE_PATH)
