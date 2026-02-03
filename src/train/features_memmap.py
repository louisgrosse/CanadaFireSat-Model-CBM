"""
This python file saves list of all activations maps in a numpy mmap format in order to 
facilite the training of the downstream SAEs.
"""


import json
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional
import sys

import hydra
import numpy as np
import torch
from numpy.lib.format import open_memmap
from omegaconf import DictConfig, OmegaConf

from DeepSatModels.utils.torch_utils import get_device
from src.constants import CONFIG_PATH
from src.data import get_data
from src.models import get_model
from src.utils.torch_utils import load_from_checkpoint

@torch.no_grad()
def prehead_forward(model: torch.nn.Module, x: torch.Tensor, doy: Optional[torch.Tensor] = None,seq_len = None) -> torch.Tensor:
    """
    Recovers the activation maps from the forward of the msclipfactorize model
    """
    
    # [B, T, C, H, W]
    assert x.ndim == 5, f"inputs must be [B,T,C,H,W], got {x.ndim} dims"
    B, T, C, H, W = x.shape
    assert C == model.channels, f"channels mismatch: got {C}, expected {model.channels}"

    x = x.reshape(B * T, C, H, W)

    t_idx = torch.arange(T, device=x.device).unsqueeze(0)      # [1,T]
    valid_BT  = t_idx < seq_len.unsqueeze(1).to(x.device)                       # [B,T] True=valid
    P = model.num_patches
    valid_BPT = valid_BT.unsqueeze(1).expand(-1, P, -1)            # [B,P,T]
    valid_mask = valid_BPT.reshape(B * P, T)                       # [B*P,T]

    # Encoder
    pooled_feats, patch_feats = model.msclip_model.image_encoder(x)  # pooled_feats: [B*T, 512], patch_feats: [B*T, P, 768]
    patch_feats = patch_feats.view(B, T, model.num_patches, model.mix_dim) \
                            .permute(0, 2, 1, 3).contiguous() \
                            .view(B * model.num_patches, T, model.mix_dim)          # [B*P, T, 768]

    # DOY for mixer (768-d)
    doy_mix = None
    if model.use_doy and (doy is not None):
        if doy.ndim > 2:  
            doy = doy.view(B, T, -1)[:, :, 0]
        assert doy.shape == (B, T), f"DOY must be [B,T], got {tuple(doy.shape)}"
        d = model.doy_embed_mix(doy).unsqueeze(1).expand(-1, model.num_patches, -1, -1) \
                                .reshape(B * model.num_patches, T, model.mix_dim)
        doy_mix = d 

    # Temporal mixing in 768
    if model.use_mixer:
        if model.useCBM:
            with torch.no_grad():
                mix_out = model.temporal_mixer(patch_feats, doy_emb=doy_mix, mask = (~valid_mask))                    # [B*P, T, 768]
        else:
            mix_out = model.temporal_mixer(patch_feats, doy_emb=doy_mix, mask = (~valid_mask))                    # [B*P, T, 768]

    else:
        mix_out = patch_feats

    if model.ABMIL:
        proj_seq = model.vision.ln_post(mix_out)                                    # [B*P, T, 768]
        proj_seq = torch.einsum("btd,df->btf", proj_seq, model.vision.proj)         # [B*P, T, 512]

        doy_pool = None
        if model.use_doy and (doy is not None):
            d = model.doy_embed_pool(doy).unsqueeze(1).expand(-1, model.num_patches, -1, -1) \
                                    .reshape(B * model.num_patches, T, model.embed_dim)
            doy_pool = d

        patch_vec = model.temp_enc(proj_seq, doy_emb=doy_pool, mask = valid_mask)                      # [B*P, 512]
    else:
        last_idx  = (seq_len - 1).clamp_min(0)                      # [B]
        idx_bp    = last_idx.unsqueeze(1).expand(B, P).reshape(B*P) # [B*P]
        row_ids   = torch.arange(B*P, device=mix_out.device)
        last_768  = mix_out[row_ids, idx_bp, :]                     # [B*P,768]
        last_768  = model.vision.ln_post(last_768)
        patch_vec = last_768 @ model.vision.proj                     # [B*P,512]

    # [B*P, 512] -> [B, P, 512]
    patch_feats = patch_vec.view(B, model.num_patches, model.embed_dim)

    if model.use_cls_fusion and model.has_cls_token:
        cls_feats = pooled_feats.view(B, T, model.embed_dim)                         # [B, T, 512]
        cls_feats = model.cls_temp_enc(cls_feats,mask = valid_BT)                      # [B, 512]
        patch_feats = patch_feats + cls_feats.unsqueeze(1)                          

    # [B, P, 512] -> [B, 512, H_p, W_p]
    patch_feats = patch_feats.view(B, model.H_patch, model.W_patch, model.embed_dim) \
                            .permute(0, 3, 1, 2).contiguous()


    return patch_feats


def _first_batch_shapes(dataloader, model, device):
    b0 = next(iter(dataloader))
    if isinstance(b0, (list, tuple)):
        b0 = b0[0]
    x = b0["inputs"].to(device, non_blocking=True)
    doy = b0.get("doy", None)
    if doy is not None:
        if doy.ndim == 5:   # [B, T, H, W, C]
            if doy.shape[-1] == 1:
                doy = doy[..., 0] 
            doy = doy.float().mean(dim=(2,3))
        doy = doy.to(device, non_blocking=True)
    feats = prehead_forward(model.model, x, doy, seq_len = b0["seq_lengths"])  # [B,D,Hp,Wp]
    _, D, Hp, Wp = feats.shape
    return D, Hp, Wp


def _num_items(dataloader):
    try:
        return len(dataloader.dataset)
    except Exception:
        n = 0
        for b in dataloader:
            n += b["inputs"].shape[0]
        return n


def _split_loader(dm, name: str):
    try:
        if name == "train":
            return dm.train_dataloader()
        elif name == "validation":
            return dm.val_dataloader()
        elif name =="test":
            return dm.test_dataloader(split="test")
    except Exception:
        return None


def _open_mm(path: Path, shape, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


@hydra.main(version_base=None, config_path="/home/grosse/CanadaFireSat-Model-CBM/results/models/MS-CLIP/")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    out_root = Path(cfg["OUTPUT"]["out_root"])
    dtype = "float32"  # float16, float32
    splits = ["train", "validation","test"]

    device = 0
    dm = get_data(cfg)
    model = get_model(cfg, device)
    ckpt = cfg["CHECKPOINT"].get("load_from_checkpoint", "")
    if ckpt:
        load_from_checkpoint(model, ckpt)
    model.eval().to(device)

    manifest = {"out_dir": str(out_root), "checkpoint": ckpt, "dtype": dtype, "splits": []}

    for split in splits:
        dl = _split_loader(dm, split)
        if dl is None:
            print(f"[skip] split={split}")
            continue
        
        D, Hp, Wp = _first_batch_shapes(dl, model, device)
        N = _num_items(dl)

        f_path = out_root / split / "features.npy"
        feats_mm = _open_mm(f_path, shape=(N, D, Hp, Wp), dtype=dtype)
        labels_mm = np.lib.format.open_memmap(out_root / split / "labels.npy", mode="w+", dtype=np.uint8, shape=(N, Hp, Wp))


        total_batches = len(dl)

        pbar = tqdm(total=total_batches, desc=f"Dump {split}", unit="batch")
        cursor = 0
        for batch in dl:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            x = batch["inputs"].to(device, non_blocking=True)

            doy = batch.get("doy", None)
            if doy is not None:
                doy = doy.to(device, non_blocking=True) 
            feats = prehead_forward(model.model, x, doy,seq_len= batch["seq_lengths"])  # [B,D,Hp,Wp]
            B = feats.shape[0]
            out = feats.half().cpu().numpy() if dtype == "float16" else feats.float().cpu().numpy()
            feats_mm[cursor:cursor + B] = out
            lbl_small = torch.nn.functional.interpolate(batch["labels"].float(), size=feats.shape[-2:], mode="nearest").squeeze(1)
            labels_mm[cursor:cursor+B] = lbl_small.cpu().numpy().astype(np.uint8)


            cursor += B

            pbar.update(1)
            pbar.set_postfix(items=cursor)
        pbar.close()

        # close
        del feats_mm
        del labels_mm

        manifest["splits"].append({
            "split": split,
            "N": int(N),
            "D": int(D),
            "Hp": int(Hp),
            "Wp": int(Wp),
            "features_path": str(f_path),
        })

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()


