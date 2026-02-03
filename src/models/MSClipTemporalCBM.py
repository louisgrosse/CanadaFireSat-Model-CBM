import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
from pathlib import Path

from src.models.l1c2l2a_adapter import L1C2L2AAdapter
#from src.CBM.concepts_minimal import _load_sae_from_ckpt

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model
from msclip.inference.clearclip import maybe_patch_clearclip
from msclip.inference.sclip import maybe_patch_sclip

from src.models.sae import plSAE
from overcomplete.sae.archetypal_dictionary import RelaxedArchetypalDictionary

import sys
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


class TemporalConvMixer(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio=2.0, dropout=0.1, kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

        #[B*P, D, T] -> [B*P, D, T] input size should be equal to ouptut size 
        self.temporal_conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim,
            bias=False,
        )
        self.temporal_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, doy_emb=None, mask=None):

        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()  # [B*P, T, 1]
            x = x * valid                          
            if doy_emb is not None:
                doy_emb = doy_emb * valid          

        y = x + (doy_emb if doy_emb is not None else 0.0)
        y = self.norm1(y)                 # [B*P, T, D]
        y = y.transpose(1, 2)             
        y = self.temporal_conv(y)
        y = self.temporal_dropout(y)
        y = y.transpose(1, 2)            

        x = x + y                         

        x = x + self.mlp(self.norm2(x))

        if mask is not None:
            x = x * valid                

        return x                          # [B*P, T, D]


class TemporalMixer(nn.Module):
    """
    Small, stable pre-projection temporal mixer (keeps sequence length T).
    Pre-norm MHA + MLP with residuals; DOY can nudge Q/K but leaves values on the identity path.
    """
    def __init__(self, embed_dim=768, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.last_attn = None  # [B*P, T, T]
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, doy_emb=None,mask=None):
        # x: [B*P, T, 768], doy_emb: [B*P, T, 768] or None
        qk = self.norm1(x + (doy_emb if doy_emb is not None else 0.0))
        y, attn = self.attn(qk, qk, x, need_weights=True, average_attn_weights=True,key_padding_mask= mask)
        self.last_attn = attn.detach()  # [B*P, T, T]
        x = x + y                              # residual
        x = x + self.mlp(self.norm2(x))        # FFN block
        return x                                # [B*P, T, 768]


class MSClipTemporalCBM(nn.Module):
    def __init__(
        self,
        model_name="Llama3-MS-CLIP-Base",
        ckpt_path=None,
        patch_size: int = 16,
        channels=10,
        num_classes=2,
        out_H=25,
        out_W=25,
        freeze_msclip=True,
        use_doy=True,
        ds_labels=True,
        use_cls_fusion=False,
        image_size: int = 224,
        use_l1c2l2a_adapter: bool = False,
        log_concepts: bool = False,
        l1c2l2a_dropout: int = 0,
        l1c2l2a_Adapter_loc:str = "",
        learned_query: bool = False,
        use_mixer: bool = True,
        use_CBM: bool = False,
        sae_config: str = None,
        pretrained: bool = True,
        clearclip: Dict[str, Any] = None,
        sclip: Dict[str, Any] = None,
        denseclip: Dict[str, Any] = None,
        sae_before_attention: bool = False,
        concept_attn_temperature: float = 1.0,
        sae_encode_chunk_size: int = 2048,
        **kwargs,
        ):
        super().__init__()

        print("### INITIALIZING MSCLIP MODEL ###")
        
        self.ds_labels = ds_labels
        self.out_H = out_H
        self.out_W = out_W
        self.channels = channels
        self.use_doy = use_doy
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_cls_fusion = use_cls_fusion
        self.use_mixer = use_mixer
        self.log_concepts = log_concepts
        self.sae_before_attention = bool(sae_before_attention)
        self.concept_attn_temperature = float(concept_attn_temperature)
        self.sae_encode_chunk_size = int(sae_encode_chunk_size)
        self.last_time_attn = None  # [B, P, T] when sae_before_attention=True
        self.concept_time_query = None  # nn.Parameter set after SAE is loaded
        self.editing_vector = None
        
        msclip_model, preprocess, tokenizer = build_model(
            model_name=model_name, pretrained=pretrained, ckpt_path=ckpt_path, device="cpu", channels=channels
        )
        self.msclip_model   = msclip_model            
        self.image_encoder  = msclip_model.image_encoder 
        self._tokenizer = tokenizer

        self.vision = self.msclip_model.clip_base_model.model.visual  
        self.vision.output_tokens = True
        
        # -- ClearCLIP
        if clearclip is not None and clearclip["enabled"]:
            num_patched = maybe_patch_clearclip(self.image_encoder, clearclip)
            print("Patched clearclip : ", num_patched)
            if num_patched > 0:
                print(f"[ClearCLIP] Patched last {num_patched} vision blocks "
                    f"(keep_ffn={clearclip.get('keep_ffn', False)}, "
                    f"keep_residual={clearclip.get('keep_residual', False)})")

        # --- SCLIP
        if sclip is not None and sclip["enabled"]:
            num_patched = maybe_patch_sclip(self.image_encoder, sclip)
            if num_patched > 0:
                print(f"[SCLIP] Patched last {num_patched} vision blocks "
                      f"(CSA attention)")


        self.embed_dim = 512
        self.mix_dim   = 768 
        self.H_patch = self.image_size // self.patch_size
        self.W_patch = self.image_size // self.patch_size
        self.num_patches = self.H_patch * self.W_patch
        self.has_cls_token = True

        self.use_l1c2l2a_adapter = use_l1c2l2a_adapter
        self.l1c2l2a_dropout = l1c2l2a_dropout

        if self.use_l1c2l2a_adapter:  #Test
            self.l1c2l2a = L1C2L2AAdapter(dim=self.embed_dim, dropout=self.l1c2l2a_dropout)
            adapter_weights = torch.load("/home/grosse/wildfire-forecast/worldstrat/l1c2l2a_linear.pt", map_location="cpu")
            self.l1c2l2a.load_state_dict(adapter_weights)
        

        if self.use_doy:
            self.doy_embed_mix  = DOYEmbed(self.mix_dim)
            self.doy_embed_pool = DOYEmbed(self.embed_dim)

        self.temporal_mixer = TemporalMixer(embed_dim=self.mix_dim, num_heads=4, mlp_ratio=2.0, dropout=0.1)

        self.useCBM = use_CBM
        self.last_concept_map = None
        self.last_concept_map_raw = None
        # When sae_before_attention=True and log_concepts=True, we store three concept maps
        self.last_concept_map_last = None
        self.last_concept_map_mean = None
        self.last_concept_map_delta = None
        self.last_concept_map_last_raw = None
        self.last_concept_map_mean_raw = None
        self.last_concept_map_delta_raw = None
        
        if self.useCBM:
            cfg_sae = OmegaConf.load(sae_config)
            sae_model_config = OmegaConf.to_container(cfg_sae["sae"], resolve=True)
            sae_model_config.pop("_target_", None)

            if cfg_sae["use_archetypal"]["enabled"]:
                points = torch.tensor(np.load(Path(cfg_sae["sae_ckpt_path"]).parent / "archetypalPoints.npy"))
            else:
                points = None

            self.sae = plSAE(points=points, **sae_model_config)

            ckpt = torch.load(cfg_sae["sae_ckpt_path"], map_location="cuda:0")
            state_dict = ckpt["state_dict"]

            if any(k.startswith("msclip_model.") for k in state_dict.keys()):
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("msclip_model.")}

            incompat = self.sae.load_state_dict(state_dict, strict=False)

            missing, unexpected = self.sae.load_state_dict(state_dict, strict=True)
            if missing or unexpected:
                print("SAE load_state_dict — missing:", missing, "unexpected:", unexpected)

            self.sae.eval().to("cuda:0")

            for p in self.sae.net.parameters():
                p.requires_grad = False

            concept_dim = int(cfg_sae["nb_concepts"])
            self.concept_dim = concept_dim

            # In sae_before_attention mode we append cyclic DOY as two channels (sin, cos) -> (C+2) = 8194
            # and use the Option-B temporal summary: [last, mean, delta] -> 3*(C+2) channels.
            if self.sae_before_attention:
                self.concept_dim_plus = concept_dim + 2
                self.head = nn.Conv2d(3 * self.concept_dim_plus, num_classes, 1)
            else:
                self.head = nn.Conv2d(concept_dim, num_classes, 1)


        else:
            self.head = nn.Conv2d(self.embed_dim, num_classes, 1)
            nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)


        for p in self.msclip_model.parameters():
            p.requires_grad = False


    def forward(self, batch, doy=None, seq_len = None):
        # [B, T, C, H, W]
        assert batch.ndim == 5, f"inputs must be [B,T,C,H,W], got {batch.ndim} dims"
        B, T, C, H, W = batch.shape

        if seq_len is None:
            # If caller doesn't provide true sequence lengths, assume all timesteps are valid.
            seq_len = torch.full((B,), T, device=batch.device, dtype=torch.long)

        assert C == self.channels, f"channels mismatch: got {C}, expected {self.channels}"

        x = batch.reshape(B * T, C, H, W)

        t_idx = torch.arange(T, device=batch.device).unsqueeze(0)      # [1,T]
        valid_BT  = t_idx < seq_len.unsqueeze(1)                       # [B,T] True=valid
        P = self.num_patches
        valid_BPT = valid_BT.unsqueeze(1).expand(-1, P, -1)            # [B,P,T]
        valid_mask = valid_BPT.reshape(B * P, T)                       # [B*P,T]

        # Encoder
        pooled_feats, patch_feats = self.msclip_model.image_encoder(x)  # pooled_feats: [B*T, 512], patch_feats: [B*T, P, 768]


        if self.useCBM and self.sae_before_attention:

            patch_512 = self.vision.ln_post(patch_feats)              # [B*T, P, 768]
            patch_512 = patch_512 @ self.vision.proj                 # [B*T, P, 512]
            patch_512 = patch_512.view(B, T, self.num_patches, self.embed_dim)  # [B, T, P, 512]

            # 2) Encode ALL (time,patch) tokens with the frozen SAE (no chunking)
            tokens = patch_512.reshape(-1, self.embed_dim)            # [B*T*P, 512]
            with torch.no_grad():
                z_pre, z0 = self.sae.net.encode(tokens.float())        # z0: [B*T*P, C]

            # Apply concept ablation gate AFTER logging raw concepts (pre-gate).
            if self.editing_vector is not None:
                gate = self.editing_vector.to(z0.device).view(1, -1)
                z = z0 * gate
            else:
                z = z0

            # 3) Reshape to concept maps per timestep: Z_raw/Z = [B, T, C, H_p, W_p]
            Z_raw = z0.view(B, T, self.num_patches, -1)
            Z_raw = Z_raw.view(B, T, self.H_patch, self.W_patch, -1).permute(0, 1, 4, 2, 3).contiguous()

            Z = z.view(B, T, self.num_patches, -1)
            Z = Z.view(B, T, self.H_patch, self.W_patch, -1).permute(0, 1, 4, 2, 3).contiguous()

            if doy is not None:
                if doy.ndim > 2:
                    doy_use = doy.view(B, T, -1)[:, :, 0]
                else:
                    doy_use = doy
                # doy is expected to already be in [0, 1] (fraction of the year)
                theta = 2.0 * math.pi * doy_use.to(Z.dtype)
                sin = torch.sin(theta)
                cos = torch.cos(theta)
                cyc = torch.stack([sin, cos], dim=2)  # [B, T, 2]
                doy_chan = cyc.unsqueeze(3).unsqueeze(4)  # [B, T, 2, 1, 1]
                doy_chan = doy_chan.expand(-1, -1, 2, self.H_patch, self.W_patch)  # [B, T, 2, H_p, W_p]
            else:
                doy_chan = torch.zeros((B, T, 2, self.H_patch, self.W_patch), device=Z.device, dtype=Z.dtype)

            # Append DOY channels to both raw and gated concept tensors
            Z_raw = torch.cat([Z_raw, doy_chan], dim=2)  # [B, T, C+2, H_p, W_p]
            Z     = torch.cat([Z,     doy_chan], dim=2)  # [B, T, C+2, H_p, W_p]

            # 5) Mask padded timesteps
            M = valid_BT.to(Z.dtype).view(B, T, 1, 1, 1)  # [B,T,1,1,1]
            Z_raw = Z_raw * M
            Z     = Z * M

            den = M.sum(dim=1).clamp_min(1.0)                              # [B,1,1,1]
            Z_mean     = Z.sum(dim=1) / den                                # [B,C+2,H_p,W_p]
            Z_mean_raw = Z_raw.sum(dim=1) / den                            # [B,C+2,H_p,W_p]

            last_idx = (seq_len - 1).clamp_min(0)                          # [B]
            prev_idx = (last_idx - 1).clamp_min(0)                         # [B]
            b_idx = torch.arange(B, device=Z.device)

            Z_last     = Z[b_idx, last_idx]                                # [B,C+2,H_p,W_p]
            Z_prev     = Z[b_idx, prev_idx]                                # [B,C+2,H_p,W_p]
            Z_delta    = Z_last - Z_prev                                   # [B,C+2,H_p,W_p]

            Z_last_raw  = Z_raw[b_idx, last_idx]                           # [B,C+2,H_p,W_p]
            Z_prev_raw  = Z_raw[b_idx, prev_idx]                           # [B,C+2,H_p,W_p]
            Z_delta_raw = Z_last_raw - Z_prev_raw                          # [B,C+2,H_p,W_p]

            # Final features: [B, 3*(C+2), H_p, W_p]
            feats = torch.cat([Z_last, Z_mean, Z_delta], dim=1)

            if self.log_concepts:
                # Store per-summary concept maps (exclude DOY channels) for evaluation / logging.
                n_extra = int(Z_last.shape[1] - getattr(self, 'concept_dim', Z_last.shape[1]))
                n_extra = max(n_extra, 0)

                if n_extra > 0:
                    z_last_log      = Z_last[:, :-n_extra].detach()
                    z_mean_log      = Z_mean[:, :-n_extra].detach()
                    z_delta_log     = Z_delta[:, :-n_extra].detach()

                    z_last_raw_log  = Z_last_raw[:, :-n_extra].detach()
                    z_mean_raw_log  = Z_mean_raw[:, :-n_extra].detach()
                    z_delta_raw_log = Z_delta_raw[:, :-n_extra].detach()
                else:
                    z_last_log      = Z_last.detach()
                    z_mean_log      = Z_mean.detach()
                    z_delta_log     = Z_delta.detach()

                    z_last_raw_log  = Z_last_raw.detach()
                    z_mean_raw_log  = Z_mean_raw.detach()
                    z_delta_raw_log = Z_delta_raw.detach()

                # RAW = pre-ablation (pre editing_vector)
                self.last_concept_map_last_raw  = z_last_raw_log.clone()
                self.last_concept_map_mean_raw  = z_mean_raw_log.clone()
                self.last_concept_map_delta_raw = z_delta_raw_log.clone()

                # (Optionally) also expose the gated versions actually used for prediction
                self.last_concept_map_last  = z_last_log
                self.last_concept_map_mean  = z_mean_log
                self.last_concept_map_delta = z_delta_log

                # Backward-compatible alias: last timestep summary
                self.last_concept_map_raw = self.last_concept_map_last_raw
                self.last_concept_map     = self.last_concept_map_last

            logits_map = self.head(feats)  # [B, num_classes, H_p, W_p]
            out = F.interpolate(
                logits_map, size=(H, W) if not self.ds_labels else (self.out_H, self.out_W),
                mode='bilinear', align_corners=False
            )
            return out

        patch_feats = patch_feats.view(B, T, self.num_patches, self.mix_dim) \
                                .permute(0, 2, 1, 3).contiguous() \
                                .view(B * self.num_patches, T, self.mix_dim)          # [B*P, T, 768]

        # DOY for mixer (768-d)
        doy_mix = None
        if self.use_doy and (doy is not None):
            if doy.ndim > 2:  
                doy = doy.view(B, T, -1)[:, :, 0]
            assert doy.shape == (B, T), f"DOY must be [B,T], got {tuple(doy.shape)}"
            d = self.doy_embed_mix(doy).unsqueeze(1).expand(-1, self.num_patches, -1, -1) \
                                    .reshape(B * self.num_patches, T, self.mix_dim)
            doy_mix = d 

        # Temporal mixing in 768
        if self.useCBM:
            with torch.no_grad():
                mix_out = self.temporal_mixer(patch_feats, doy_emb=doy_mix, mask = (~valid_mask))                    # [B*P, T, 768]
        else:
            mix_out = self.temporal_mixer(patch_feats, doy_emb=doy_mix, mask = (~valid_mask))                    # [B*P, T, 768]


        last_idx  = (seq_len - 1).clamp_min(0)                      # [B]
        idx_bp    = last_idx.unsqueeze(1).expand(B, P).reshape(B*P) # [B*P]
        row_ids   = torch.arange(B*P, device=mix_out.device)
        last_768  = mix_out[row_ids, idx_bp, :]                     # [B*P,768]
        last_768  = self.vision.ln_post(last_768)
        patch_vec = last_768 @ self.vision.proj                     # [B*P,512]

        # [B*P, 512] -> [B, P, 512]
        patch_feats = patch_vec.view(B, self.num_patches, self.embed_dim)                

        # [B, P, 512] -> [B, 512, H_p, W_p]
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim) \
                                .permute(0, 3, 1, 2).contiguous()

        if self.useCBM:
            patch_flat = patch_feats.permute(0,2,3,1).contiguous().view(B * self.num_patches, self.embed_dim)
            with torch.no_grad():
                z_pre, z = self.sae.net.encode(patch_flat)   # concepts
            patch_feats = z.view(B, self.H_patch, self.W_patch, -1).permute(0,3,1,2).contiguous()  # [B, C_concept, H_p, W_p]
            
            if self.log_concepts:
                self.last_concept_map_raw = patch_feats.detach().clone() 

            if self.editing_vector is not None:
                patch_feats *= self.editing_vector.view(1, -1, 1, 1)

            if self.log_concepts:
                self.last_concept_map = patch_feats.detach()


        out = self.head(patch_feats)
        out = F.interpolate(
            out, size=(H, W) if not self.ds_labels else (self.out_H, self.out_W),
            mode="bilinear", align_corners=False
        )
        return out

    @torch.no_grad()
    def encode_patches(self, batch):
        """
        Extract per-time, per-patch MS-CLIP embeddings *before* temporal aggregation.

        Args:
            batch: [B, T, C, H, W] tensor (same normalization as segmentation training)

        Returns:
            patch_feats: [B, T, P, D] where D=self.embed_dim (512),
                         P = H_patch * W_patch
        """
        assert batch.ndim == 5, f"Expected [B,T,C,H,W], got {batch.shape}"
        B, T, C, H, W = batch.shape

        x = batch.reshape(B * T, C, H, W)

        pooled_feats, patch_feats = self.msclip_model.image_encoder(x)  # patch_feats: [B*T, P, 768]

        patch_feats = self.vision.ln_post(patch_feats)                  # [B*T, P, 768]
        patch_feats = patch_feats @ self.vision.proj                    # [B*T, P, 512]

        patch_feats = patch_feats.view(B, T, self.num_patches, self.embed_dim)
        return patch_feats
