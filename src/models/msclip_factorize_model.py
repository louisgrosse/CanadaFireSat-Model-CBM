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


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, learned_query=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        self.last_attn_weights = None 
        self.learned_query = learned_query

        if self.learned_query:
            self.pool_q = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, doy_emb=None, mask=None):
        # x: [B*P or B, T, D], mask: [B*P or B, T]  True=valid, False=pad
        if doy_emb is not None:
            q_in = x + doy_emb
            k_in = x + doy_emb
        else:
            q_in = k_in = x

        # Self-attn mix (mask pads)
        pad_mask = (~mask) if mask is not None else None
        attn_out, _ = self.attn(q_in, k_in, x,
                                key_padding_mask=pad_mask,
                                need_weights=False, average_attn_weights=False)
        x = self.norm(attn_out + x)

        if self.learned_query:
            Bn, T, D = x.shape
            q = self.pool_q.expand(Bn, 1, D)
            pooled, w = self.attn(q, x, x,
                                  key_padding_mask=pad_mask,
                                  need_weights=True, average_attn_weights=True)
            self.last_attn_weights = w.squeeze(1).detach()  # [B*, T], sums to 1 over valid
            return pooled.squeeze(1)                        # [B*, D]

        # masked mean over time (ignore pads)
        if mask is None:
            return x.mean(dim=1)
        m = mask.float().unsqueeze(-1)                      # [B*,T,1]
        denom = m.sum(dim=1).clamp_min(1e-6)                # [B*,1,1]
        return (x * m).sum(dim=1) / denom                   # [B*,D]


class XAITemporalPool(nn.Module):
    """
    Weights-only temporal pooling:
      - x must already be in MS-CLIP space (512-d after ln_post+proj)
      - learns a query q and temperature tau to score timestamps
      - returns a convex combination of input tokens (values untouched)
    """
    def __init__(self, dim: int, init_temp: float = 2.0):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.q, std=1e-2)

        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(init_temp))))
        self.gate = nn.Parameter(torch.tensor(-2.0))  
        self._doy_raw = nn.Parameter(torch.tensor(0.1)) 
        self.value_scale = nn.Parameter(torch.tensor(1.0))
        self.last_attn_weights = None

    @property
    def doy_scale(self):
        # keeps seasonality modest and stable
        return 0.5 * torch.tanh(self._doy_raw)

    def forward(self, x: torch.Tensor, doy_emb: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B*P, T, D]; mask: [B*P, T] True=valid
        q   = F.normalize(self.q,  dim=0,  eps=1e-6)
        x_n = F.normalize(x,      dim=-1, eps=1e-6)

        scores = torch.matmul(x_n, q)                       # [B*P, T]
        if doy_emb is not None:
            doy_n  = F.normalize(doy_emb, dim=-1, eps=1e-6)
            scores = scores + self.doy_scale * torch.matmul(doy_n, q)

        tau = torch.exp(self.log_tau).clamp(0.5, 5.0)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)        # exclude pads
        w = torch.softmax(tau * scores, dim=1)              # [B*P, T]
        if mask is not None:
            w = w * mask.float()
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)   # renorm over valid only
        self.last_attn_weights = w.detach()

        if mask is not None:
            m = mask.float().unsqueeze(-1)
            z_mean = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
        else:
            z_mean = x.mean(dim=1)

        z_attn = torch.einsum("btd,bt->bd", x, w)           # convex combo
        z_attn = F.normalize(z_attn, dim=-1, eps=1e-6) * self.value_scale

        g = torch.sigmoid(self.gate)
        return (1 - g) * z_mean + g * z_attn


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
        use_mixer: bool = True,
        ABMIL: bool = False,
        use_CBM: bool = False,
        sae_config: str = None,
        pretrained: bool = True,
        clearclip: Dict[str, Any] = None,
        sclip: Dict[str, Any] = None,
        denseclip: Dict[str, Any] = None,
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
        self.temp_enc_type = temp_enc_type
        self.ABMIL = ABMIL
        self.use_mixer = use_mixer
        self.log_concepts = True
        self.editing_vector = None
        
        msclip_model, preprocess, tokenizer = build_model(
            model_name=model_name, pretrained=pretrained, ckpt_path=ckpt_path, device="cpu", channels=channels
        )
        self.msclip_model   = msclip_model            
        self.image_encoder  = msclip_model.image_encoder 

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

        if self.temp_enc_type == "attention":
            self.temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=8, dropout=0.1, learned_query=learned_query)
        else:
            self.temp_enc = XAITemporalPool(dim=self.embed_dim, init_temp=3.0) 

        if self.use_cls_fusion and self.has_cls_token:
            self.cls_temp_enc = TemporalAttention(embed_dim=self.embed_dim, num_heads=4, dropout=0.1, learned_query=learned_query)
        
        self.useCBM = use_CBM
        self.last_concept_map = None
        if self.useCBM:
            cfg_sae = OmegaConf.load(sae_config)
            sae_model_config = OmegaConf.to_container(cfg_sae["sae"], resolve=True)
            sae_model_config.pop("_target_", None)

            # 1) Build plSAE with the same config as during training
            self.sae = plSAE(**sae_model_config)

            # 2) If archetypal dictionary was used during training, recreate it
            if cfg_sae["use_archetypal"]["enabled"]:
                points = np.load(Path(cfg_sae["sae_ckpt_path"]).parent / "archetypalPoints.npy")

                archetypal_dict = RelaxedArchetypalDictionary(
                    in_dimensions=cfg_sae.sae.sae_kwargs["input_shape"],
                    nb_concepts=cfg_sae.sae.sae_kwargs["nb_concepts"],
                    points=torch.as_tensor(points),
                    delta=1.0,
                )
                # IMPORTANT: override dictionary *before* loading weights
                self.sae.net.dictionary = archetypal_dict

            # 3) Manually load the Lightning checkpoint
            ckpt = torch.load(cfg_sae["sae_ckpt_path"], map_location="cuda:0")
            state_dict = ckpt["state_dict"]

            missing, unexpected = self.sae.load_state_dict(state_dict, strict=True)
            if missing or unexpected:
                print("SAE load_state_dict — missing:", missing, "unexpected:", unexpected)

            self.sae.eval().to("cuda:0")

            for p in self.sae.net.parameters():
                p.requires_grad = False

            concept_dim = cfg_sae["nb_concepts"]
            self.head = nn.Conv2d(concept_dim, num_classes, 1)

        else:
            self.head = nn.Conv2d(self.embed_dim, num_classes, 1)
            nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

         # --- DenseCLIP
        self.use_denseclip = False
        self.denseclip_concat_scores = False
        self.denseclip_tau = 0.07
        self.last_denseclip_score_map = None
        self._tokenizer = tokenizer

        if denseclip is not None and denseclip["enabled"]:
            cfg_dense = denseclip

            # required
            class_names = cfg_dense["class_names"]   # list[str], must be length num_classes
            assert len(class_names) == num_classes, \
                f"denseclip.class_names must have len={num_classes}, got {len(class_names)}"

            prompt_template = cfg_dense["prompt_template"] if "prompt_template" in cfg_dense else "a photo of a {}."
            prompts = [prompt_template.format(n) for n in class_names]

            tokens = self._tokenizer(prompts)
            text_device = next(self.msclip_model.parameters()).device
            tokens = tokens.to(text_device)

            with torch.no_grad():
                # OpenCLIP-style (will crash if your base differs)
                text_feats = self.msclip_model.clip_base_model.encode_text(tokens)
                text_feats = F.normalize(text_feats, dim=-1)

            self.register_buffer("denseclip_text_features", text_feats)

            self.use_denseclip = True
            self.denseclip_tau = float(cfg_dense["tau"]) if "tau" in cfg_dense else 0.07
            self.denseclip_concat_scores = bool(cfg_dense["concat_scores"]) if "concat_scores" in cfg_dense else True

            if self.denseclip_concat_scores:
                # overwrite head to accept concatenated channels
                self.head = nn.Conv2d(self.embed_dim + num_classes, num_classes, 1)
                nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
                if self.head.bias is not None:
                    nn.init.zeros_(self.head.bias)

                print(f"[DenseCLIP] Enabled. concat_scores=True, tau={self.denseclip_tau}")
            else:
                print(f"[DenseCLIP] Enabled. concat_scores=False, tau={self.denseclip_tau}")

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

        


    def _denseclip_score_map(self, patch_feats_2d: torch.Tensor) -> torch.Tensor:
        """
        patch_feats_2d: [B, D, H_p, W_p] in CLIP projected space (D = embed_dim).
        returns score_map: [B, K, H_p, W_p]
        """
        B, D, H_p, W_p = patch_feats_2d.shape
        P = H_p * W_p

        z = patch_feats_2d.permute(0, 2, 3, 1).reshape(B, P, D)
        z = F.normalize(z, dim=-1)

        t = self.denseclip_text_features  # [K, D], already normalized
        scores = z @ t.t()                # [B, P, K]
        score_map = scores.reshape(B, H_p, W_p, self.num_classes).permute(0, 3, 1, 2).contiguous()
        return score_map


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

    def forward(self, batch, doy=None, seq_len = None):
        if self.temp_enc_type == "attention":
            return self.forwardAttention(batch,doy,seq_len)
        else:
            return self.forwardMixer(batch,doy,seq_len)

    def forwardAttention(self, batch, doy=None, seq_len = None):
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
            pooled_feats, patch_feats = self.msclip_model.image_encoder(x)       # pooled: [B, D], patch_tokens: [B*T, P, D]  [120, 512] and [120, 196, 768]
            patch_feats = self.vision.ln_post(patch_feats)      # [B*T, P, 768] -> LN
            patch_feats = patch_feats @ self.vision.proj

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
            doy_emb = self.doy_embed_pool(doy)  # [B,T,D]
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
            patch_feats = patch_feats + cls_feats.view(B,1,self.embed_dim)

        # [B,D,H_p,W_p]
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        if self.use_denseclip:
            score_map = self._denseclip_score_map(patch_feats)
            self.last_denseclip_score_map = score_map
            if self.denseclip_concat_scores:
                patch_feats = torch.cat([patch_feats, score_map], dim=1)

        out = self.head(patch_feats)  # [B, num_classes, H_p, W_p]
        assert torch.isfinite(out).all(), "Head produced NaN/Inf"

        if not self.ds_labels:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        else:
            out = F.interpolate(out, size=(self.out_H, self.out_W), mode="bilinear", align_corners=False)

        return out

    def forwardMixer(self, batch, doy=None, seq_len = None):
        # [B, T, C, H, W]
        assert batch.ndim == 5, f"inputs must be [B,T,C,H,W], got {batch.ndim} dims"
        B, T, C, H, W = batch.shape
        assert C == self.channels, f"channels mismatch: got {C}, expected {self.channels}"

        x = batch.reshape(B * T, C, H, W)

        t_idx = torch.arange(T, device=batch.device).unsqueeze(0)      # [1,T]
        valid_BT  = t_idx < seq_len.unsqueeze(1)                       # [B,T] True=valid
        P = self.num_patches
        valid_BPT = valid_BT.unsqueeze(1).expand(-1, P, -1)            # [B,P,T]
        valid_mask = valid_BPT.reshape(B * P, T)                       # [B*P,T]

        # Encoder
        pooled_feats, patch_feats = self.msclip_model.image_encoder(x)  # pooled_feats: [B*T, 512], patch_feats: [B*T, P, 768]
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
        if self.use_mixer:
            if self.useCBM:
                with torch.no_grad():
                    mix_out = self.temporal_mixer(patch_feats, doy_emb=doy_mix, mask = (~valid_mask))                    # [B*P, T, 768]
            else:
                mix_out = self.temporal_mixer(patch_feats, doy_emb=doy_mix, mask = (~valid_mask))                    # [B*P, T, 768]

        else:
            mix_out = patch_feats

        if self.ABMIL:
            proj_seq = self.vision.ln_post(mix_out)                                    # [B*P, T, 768]
            proj_seq = torch.einsum("btd,df->btf", proj_seq, self.vision.proj)         # [B*P, T, 512]

            doy_pool = None
            if self.use_doy and (doy is not None):
                d = self.doy_embed_pool(doy).unsqueeze(1).expand(-1, self.num_patches, -1, -1) \
                                        .reshape(B * self.num_patches, T, self.embed_dim)
                doy_pool = d

            patch_vec = self.temp_enc(proj_seq, doy_emb=doy_pool, mask = valid_mask)                      # [B*P, 512]
        else:
            last_idx  = (seq_len - 1).clamp_min(0)                      # [B]
            idx_bp    = last_idx.unsqueeze(1).expand(B, P).reshape(B*P) # [B*P]
            row_ids   = torch.arange(B*P, device=mix_out.device)
            last_768  = mix_out[row_ids, idx_bp, :]                     # [B*P,768]
            last_768  = self.vision.ln_post(last_768)
            patch_vec = last_768 @ self.vision.proj                     # [B*P,512]

        # [B*P, 512] -> [B, P, 512]
        patch_feats = patch_vec.view(B, self.num_patches, self.embed_dim)

        if self.use_cls_fusion and self.has_cls_token:
            cls_feats = pooled_feats.view(B, T, self.embed_dim)                         # [B, T, 512]
            cls_feats = self.cls_temp_enc(cls_feats,mask = valid_BT)                      # [B, 512]
            patch_feats = patch_feats + cls_feats.unsqueeze(1)                          

        # [B, P, 512] -> [B, 512, H_p, W_p]
        patch_feats = patch_feats.view(B, self.H_patch, self.W_patch, self.embed_dim) \
                                .permute(0, 3, 1, 2).contiguous()

        if self.use_denseclip:
            score_map = self._denseclip_score_map(patch_feats)
            self.last_denseclip_score_map = score_map
            if self.denseclip_concat_scores:
                patch_feats = torch.cat([patch_feats, score_map], dim=1)

        if self.useCBM:
            patch_flat = patch_feats.permute(0,2,3,1).contiguous().view(B * self.num_patches, self.embed_dim)
            with torch.no_grad():
                z_pre, z = self.sae.net.encode(patch_flat)   # concepts
            patch_feats = z.view(B, self.H_patch, self.W_patch, -1).permute(0,3,1,2).contiguous()  # [B, C_concept, H_p, W_p]

            if self.editing_vector is not None:
                patch_feats *= self.editing_vector.view(1, -1, 1, 1)

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

