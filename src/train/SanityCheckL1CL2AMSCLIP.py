from src.data.hf_Canada.ssl4eos12_dataset import SSL4EOS12Dataset, collate_fn, S2L1C_MEAN, S2L1C_STD,S2L2A_MEAN, S2L2A_STD
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model

tf = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=S2L1C_MEAN + S2L2A_MEAN,
        std=S2L1C_STD + S2L2A_STD,
    ),
])

ds = SSL4EOS12Dataset(
    data_dir="/work/eceo/grosse/ssl4eo-s12/train",
    split_file="/work/eceo/grosse/ssl4eo-s12/splits/ssl4eos12_train.txt",
    modalities=["S2L1C","S2L2A"],
    transform=tf,
    single_timestamp=True,
)

dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

msclip_model, preprocess, tokenizer = build_model(
    model_name="Llama3-MS-CLIP-Base", pretrained=True, ckpt_path=None, device="cpu", channels=10
)

vision = msclip_model.clip_base_model.model.visual  
vision.output_tokens = True

def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return torch.clamp(x.norm(dim=dim), min=eps)

def _cosine(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    an = _safe_norm(a, dim=dim, eps=eps)
    bn = _safe_norm(b, dim=dim, eps=eps)
    return (a * b).sum(dim=dim) / (an * bn)

def _angle_deg_from_cos(cos: torch.Tensor) -> torch.Tensor:
    cos_clamped = torch.clamp(cos, -1.0, 1.0)
    return torch.acos(cos_clamped) * (180.0 / np.pi)
    
metrics = []
for i, batch in enumerate(dl):
    if i > 1000:
        break
    l1c = batch["S2L1C"] 
    l2a = batch["S2L2A"] 

    l1c = l1c[:, [1,2,3,4,5,6,7,8,11,12], :, :]  # keep B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 (drop 60m: B1,B9,B10)
    l2a = l2a[:, [1,2,3,4,5,6,7,8,10,11], :, :]  # keep same (L2A has no B10, so drop 60m: B1,B9)   

    poolL1C, ptoksL1C = msclip_model.image_encoder(l1c)
    poolL2A, ptoksL2A = msclip_model.image_encoder(l2a)

        # ---- compute per-pair metrics (aggregated over patches) ----
    # ptoks* expected [N, 1+P, D] with CLS at index 0
    cls_l1c   = ptoksL1C[:, 0, :]         # [N, D]
    cls_l2a   = ptoksL2A[:, 0, :]         # [N, D]
    patch_l1c = ptoksL1C[:, 1:, :]        # [N, P, D]
    patch_l2a = ptoksL2A[:, 1:, :]        # [N, P, D]

    # CLS metrics
    cls_cosine      = _cosine(cls_l1c, cls_l2a, dim=-1)                               # [N]
    cls_norm_l1c    = _safe_norm(cls_l1c, dim=-1)                                     # [N]
    cls_norm_l2a    = _safe_norm(cls_l2a, dim=-1)                                     # [N]
    cls_norm_ratio  = cls_norm_l1c / torch.clamp(cls_norm_l2a, min=1e-12)             # [N]
    cls_diff        = (cls_l1c - cls_l2a).abs()                                       # [N, D]
    cls_diff_mae    = cls_diff.mean(dim=-1)                                           # [N]
    cls_diff_max    = cls_diff.amax(dim=-1)                                           # [N]

    # Patch metrics (aggregate across patches)
    patch_cos       = _cosine(patch_l1c, patch_l2a, dim=-1)                            # [N, P]
    patch_cos_mean  = patch_cos.mean(dim=1)                                            # [N]
    patch_norm_l1c  = _safe_norm(patch_l1c, dim=-1)                                    # [N, P]
    patch_norm_l2a  = _safe_norm(patch_l2a, dim=-1)                                    # [N, P]
    patch_norm_ratio= (patch_norm_l1c / torch.clamp(patch_norm_l2a, min=1e-12)).mean(1)# [N]
    patch_norm_l1c_mean = patch_norm_l1c.mean(dim=1)                                   # [N]
    patch_norm_l2a_mean = patch_norm_l2a.mean(dim=1)                                   # [N]
    patch_diff      = (patch_l1c - patch_l2a).abs()                                    # [N, P, D]
    patch_diff_mae  = patch_diff.mean(dim=(1,2))                                       # [N]
    patch_diff_max  = patch_diff.amax(dim=(1,2))                                       # [N]

    # push one dict per pair into `metrics`
    for i in range(cls_cosine.size(0)):
        metrics.append({
            # CLS
            "cls_cosine":        cls_cosine[i].item(),
            "cls_norm_l1c":      cls_norm_l1c[i].item(),
            "cls_norm_l2a":      cls_norm_l2a[i].item(),
            "cls_norm_ratio":    cls_norm_ratio[i].item(),
            "cls_diff_mae":      cls_diff_mae[i].item(),
            "cls_diff_max":      cls_diff_max[i].item(),
            # PATCH (aggregated)
            "patch_cosine_mean": patch_cos_mean[i].item(),
            "patch_norm_l1c_mean": patch_norm_l1c_mean[i].item(),
            "patch_norm_l2a_mean": patch_norm_l2a_mean[i].item(),
            "patch_norm_ratio_mean": patch_norm_ratio[i].item(),
            "patch_diff_mae":    patch_diff_mae[i].item(),
            "patch_diff_max":    patch_diff_max[i].item(),
        })


df = pd.DataFrame(metrics)
df.to_csv("/home/grosse/CanadaFireSat-Model-CBM/results/stats/msclip_pairs_per_sample.csv", index=False)

mean_series = df.mean(numeric_only=True)
std_series  = df.std(numeric_only=True)

summary = pd.concat([mean_series, std_series], axis=1)
summary.columns = ["mean", "std"]
summary.to_csv("/home/grosse/CanadaFireSat-Model-CBM/results/stats/msclip_pairs_summary.csv")
print("Saved per-pair metrics -> msclip_pairs_per_sample.csv")
print("Saved dataset mean/std  -> msclip_pairs_summary.csv")

