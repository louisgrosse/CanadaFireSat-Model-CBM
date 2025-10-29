import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Any, Dict

from src.models.l1c2l2a_adapter import L1C2L2AAdapter
#from src.CBM.concepts_minimal import _load_sae_from_ckpt

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model
from msclip.inference.clearclip import maybe_patch_clearclip

import torch
import torch.nn as nn
import torch.nn.functional as F


class DOYEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(2, embed_dim)

    def forward(self, doy):
        # doy: [B, T] integers (0–1)
        theta = 2 * math.pi * doy
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        cyc = torch.stack([sin, cos], dim=-1)   # [B, T, 2]
        return self.fc(cyc)                     # [B, T, D]


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, learned_query=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        self.last_attn_weights = None 
        self.learned_query = learned_query

        if self.learned_query:
            self.pool_q = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, doy_emb=None):
        # x: [B*P, T, D]
        # doy_emb: [B*P, T, D] 
        if doy_emb is not None:
            q_in = x + doy_emb
            k_in = x + doy_emb
        else:
            q_in = k_in = x

        attn_out, _ = self.attn(q_in, k_in, x) 
        x = self.norm(attn_out + x)

        if self.learned_query:
            BPT, T, D = x.shape
            q = self.pool_q.expand(BPT, 1, D)
            pooled, w = self.attn(q, x, x)
            self.last_attn_weights = w.squeeze(1)  # [B*P, T]
            return pooled.squeeze(1)

        return x.mean(dim=1) 



class MSClipFactorizeModel(nn.Module):
    def __init__(
        self,
        model_name="Llama3-MS-CLIP-Base",
        ckpt_path=None,
        patch_size: int = 16,
        channels=10,
        num_classes=2,
        out_H=25,
        out_W=25,
        temp_enc_type="attention",
        freeze_msclip=True,
        use_doy=True,
        ds_labels=True,
        use_cls_fusion=False,
        image_size: int = 224,
        use_l1c2l2a_adapter: bool = False,
        l1c2l2a_dropout: int = 0,
        l1c2l2a_Adapter_loc:str = "",
        unfreeze_last_block:bool = False,
        learned_query: bool = False,
        pretrained: bool = True,
        model_config: Dict[str, Any] = None,
        sae_model_config: Dict[str, Any] = None,
        **kwargs,
        ):
        super().__init__()
        
        self.ds_labels = ds_labels
        self.out_H = out_H
        self.out_W = out_W
        self.channels = channels
        self.use_doy = use_doy
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_cls_fusion = use_cls_fusion

        msclip_model, preprocess, tokenizer = build_model(
            model_name=model_name, pretrained=pretrained, ckpt_path=ckpt_path, device="cpu", channels=channels
        )
        self.msclip_model   = msclip_model            
        self.image_encoder  = msclip_model.image_encoder 

        self.vision = self.msclip_model.clip_base_model.model.visual  
        self.vision.output_tokens = True
        
        if model_config is not None and "clearclip" in model_config and model_config["clearclip"]["enabled"]:
            num_patched = maybe_patch_clearclip(self.image_encoder, model_config["clearclip"])
            if num_patched > 0:
                print(f"[ClearCLIP] Patched last {num_patched} vision blocks "
                    f"(keep_ffn={model_config['clearclip'].get('keep_ffn', False)}, "
                    f"keep_residual={model_config['clearclip'].get('keep_residual', False)})")

        if freeze_msclip:
            for p in self.msclip_model.parameters():
                p.requires_grad = False
            
            if unfreeze_last_block:
                self._unfreeze_last_vit_blocks(n_blocks=1)
        else:
            for name, p in self.msclip_model.named_parameters():
                if not name.startswith("image_encoder."):
                    p.requires_grad = False

            for n, p in self.msclip_model.named_parameters():
                if any(n.startswith(prefix) for prefix in ["text_encoder.", "text_projection", "logit_scale", "text_transformer"]):
                    p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = True

            if hasattr(self.image_encoder.model.visual, "ln_post"):
                for p in self.image_encoder.model.visual.ln_post.parameters():
                    p.requires_grad = True

        self.embed_dim = 768
        self.H_patch = self.image_size // self.patch_size
        self.W_patch = self.image_size // self.patch_size
        self.num_patches = self.H_patch * self.W_patch
        self.has_cls_token = True

        self.use_l1c2l2a_adapter = use_l1c2l2a_adapter
        self.l1c2l2a_dropout = l1c2l2a_dropout

        if self.use_l1c2l2a_adapter:
            self.l1c2l2a = L1C2L2AAdapter(dim=self.embed_dim, dropout=self.l1c2l2a_dropout)
            adapter_weights = torch.load("/home/louis/Code/wildfire-forecast/worldstrat/l1c2l2a_linear.pt", map_location="cpu")
            self.l1c2l2a.load_state_dict(adapter_weights)
        

        if self.use_doy:
            self.doy_embed = DOYEmbed(self.embed_dim)

        if temp_enc_type == "attention":
            self.temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=8, dropout=0.1, learned_query=learned_query)
        else:
            raise ValueError(f"Unknown temp_enc_type: {temp_enc_type}")

        if self.use_cls_fusion and self.has_cls_token:
            self.cls_temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=4, dropout=0.1, learned_query=learned_query)
            self.cls_fuse_proj = nn.Linear(self.embed_dim, self.embed_dim)

        #self.sae = _load_sae_from_ckpt(sae_ckpt, device='cpu')
        #for p in self.sae.parameters():
          #  p.requires_grad = False

        self.head = nn.Conv2d(self.embed_dim, num_classes, 1)
    
    def _iter_vit_blocks(self, enc):
        """
        Return the list of Vision Transformer blocks inside MS-CLIP's image encoder.
        Specifically targets the path:
        enc.model.visual.transformer.resblocks
        """
        try:
            return list(enc.model.visual.transformer.resblocks)
        except AttributeError:
            raise RuntimeError(
                "Could not locate VisionTransformer.resblocks under image_encoder.model.visual.transformer."
            )

    def _unfreeze_last_vit_blocks(self, n_blocks: int = 1):
        blocks = self._iter_vit_blocks(self.image_encoder)
        n = max(1, min(n_blocks, len(blocks)))
        last_blocks = blocks[-n:]

        # then unfreeze the selected blocks + final LayerNorm if you wish
        for b in last_blocks:
            for p in b.parameters():
                p.requires_grad = True

        if hasattr(self.image_encoder.model.visual, "ln_post"):
            for p in self.image_encoder.model.visual.ln_post.parameters():
                p.requires_grad = True

    def forward(self, batch, doy=None):
        #[B, T, C, H, W] 
        assert batch.ndim == 5, f"inputs must be [B,T,C,H,W], got {batch.ndim} dims"
        B, T, C, H, W = batch.shape

        assert C == self.channels, f"channels mismatch: got {C}, expected {self.channels}"
        assert H == self.image_size and W == self.image_size, f"spatial mismatch: ({H},{W}) vs {self.image_size}"
        assert torch.isfinite(batch).all(), "inputs contain NaN/Inf"
        assert batch.dtype in (torch.float16, torch.float32, torch.bfloat16), f"bad dtype {batch.dtype}"

        x = batch.reshape(B * T, C, H, W)

        assert self.patch_size <= H and self.patch_size <= W, f"patch {self.patch_size} > ({H},{W})"

        requires_enc_grad = any(p.requires_grad for p in self.image_encoder.parameters())
        ctx = torch.enable_grad() if requires_enc_grad and self.training else torch.no_grad()
        with ctx:
            #pooled_feats, patch_feats = self.msclip_model.clip_base_model.model.visual(x)       # pooled: [B, D], patch_tokens: [B*T, P, D]  [120, 512] and [120, 196, 768]
            pooled_feats, patch_feats = self.msclip_model.image_encoder(x)       # pooled: [B, D], patch_tokens: [B*T, P, D]  [120, 512] and [120, 196, 768]

        # [B, T, P, D]
        patch_feats = patch_feats.view(B, T, self.num_patches, self.embed_dim)

        # (B*P, T, D)
        patch_feats = patch_feats.permute(0, 2, 1, 3).contiguous().view(B * self.num_patches, T, self.embed_dim)

        doy_emb = None
        if self.use_doy and doy is not None:
            assert doy.shape[0] == B and doy.shape[1] == T, f"DOY shape mismatch: {doy.shape} vs {(B,T)}"
            if doy.ndim > 2:  # [B,T,H,W,1] -> [B,T]
                doy = doy.view(B, T, -1)[:, :, 0]
            assert doy.shape == (B, T), f"DOY must be [B,T], got {tuple(doy.shape)}"
            assert torch.isfinite(doy).all(), "DOY has NaN/Inf"

            doy = doy.clamp(0, 1)
            doy_emb = self.doy_embed(doy)  # [B,T,D]
            doy_emb = doy_emb.unsqueeze(1).expand(-1, self.num_patches, -1, -1).reshape(
                B * self.num_patches, T, self.embed_dim
            )
        else:
            doy_emb = None


        patch_feats = self.temp_enc(patch_feats, doy_emb=doy_emb)  # [B*P, D]
        assert torch.isfinite(patch_feats).all(), "Temporal encoder produced NaN/Inf"

        # [B, P, D]
        patch_feats = patch_feats.view(B, self.num_patches, self.embed_dim)
        if self.use_cls_fusion and self.has_cls_token:
            cls_feats = pooled_feats.view(B, T, self.embed_dim)      
            cls_feats = self.cls_temp_enc(cls_feats)                   # [B, D]
            patch_feats = patch_feats + self.cls_fuse_proj(cls_feats).unsqueeze(1)

        # [B,D,H_p,W_p]
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        out = self.head(patch_feats)  # [B, num_classes, H_p, W_p]
        assert torch.isfinite(out).all(), "Head produced NaN/Inf"

        if not self.ds_labels:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        else:
            out = F.interpolate(out, size=(self.out_H, self.out_W), mode="bilinear", align_corners=False)

        return out

