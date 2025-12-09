import math
import random
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from overcomplete.sae import TopKSAE, JumpSAE, BatchTopKSAE, SAE
from overcomplete.sae.train import extract_input, _compute_reconstruction_error
from overcomplete.sae.trackers import DeadCodeTracker
from overcomplete.metrics import l0_eps, avg_l2_loss, hoyer
from overcomplete.sae.archetypal_dictionary import RelaxedArchetypalDictionary


from src.utils.sae_utils import criterion_factory, optimizer_factory, scheduler_factory, mse_criterion,\
    region_mse_bands_per_class, region_r2_bands_per_class,AuxKDeadTracker


class plSAE(pl.LightningModule):
    def __init__(
            self,
            lr: float = 0.001,
            weight_decay: float = 0.,
            sae_type: str = "topk",
            loss_type: str = "mse",
            optimizer_type: str = "adam",
            scheduler_type: Optional[str] = None,
            num_samples: int = 100000,
            # resample_steps: List[int] = [],
            resample_every_n_epochs: int = 1,
            resample_batch_size: int = 1,
            bind_init: bool = False,
            depth_scale_shift: int = 0,
            geo_embed_dim: int = 256,
            bias_fire: bool = False,
            decay_k: bool= False,
            align_start_epoch: int= 40,
            sae_kwargs: Dict[str, Any] = {},
            criterion_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = self.sae_factory(sae_type, **sae_kwargs)

        if loss_type == "mse_auxk_true":
            n_features = self.net.get_dictionary().shape[0]
            self.auxk_tracker = AuxKDeadTracker(n_features)
            criterion_kwargs = dict(criterion_kwargs)
            criterion_kwargs.setdefault("auxk_tracker", self.auxk_tracker)

        else:
            self.auxk_tracker = None

        self.criterion = criterion_factory(loss_type, **criterion_kwargs)
        
        self.geonet = None

        self.train_dead_tracker = None
        self.val_dead_tracker = None
        self.test_dead_tracker = None

        self.from_msclip = False
        self.msclip_model = None

        self.align_text_embs = None
        self.align_loss_coeff = self.hparams.criterion_kwargs.get("align_loss_coeff", 0.0)


        if bind_init:
            print("Binding the encoder and decoder weights")
            self._initialize_encoder_from_decoder()

        #self._training_outputs = []
        self._val_outputs = []
        self._test_outputs = []

    @staticmethod
    def sae_factory(sae_type: str,use_relaxed_dict: bool = False, **sae_kwargs) -> nn.Module:

        if sae_type == "topk":
            print("Using topk sae")            
            return TopKSAE(**sae_kwargs)
        elif sae_type == "jump":
            return JumpSAE(**sae_kwargs)
        elif sae_type == "batch_topk":
            return BatchTopKSAE(**sae_kwargs)
        elif sae_type == "vanilla":
            return SAE(**sae_kwargs)
        else:
            raise NotImplementedError
    
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def _spatial_embedding(self, lat: torch.Tensor, long: torch.Tensor) -> torch.Tensor:
        xpos = torch.concat(
            [torch.sin(lat * (math.pi / 180)), torch.cos(lat * (math.pi / 180)), torch.sin(long * (math.pi / 180)), torch.cos(long * (math.pi / 180))],
            dim=1,
        )
        return xpos

    def step(self, batch: Any, tracker: Any, flag_mse: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["inputs"]

        if self.from_msclip:
                if x.ndim == 4:
                    x = x.unsqueeze(1) 
                
                    
        if x.ndim == 5:
            assert self.from_msclip and (self.msclip_model is not None), \
                "plSAE.from_msclip=True and a msclip_model must be attached for 5D inputs"

            B, T, C, H, W = x.shape

            with torch.no_grad():
                patch_feats = self.msclip_model.encode_patches(x)   # [B, T, P, D]

            B, T, P, D = patch_feats.shape
            H_p = self.msclip_model.H_patch
            W_p = self.msclip_model.W_patch
            assert P == H_p * W_p, f"num_patches mismatch: P={P}, H_p*W_p={H_p*W_p}"

            seq_len = batch["seq_lengths"]
            t_idx = torch.arange(T, device=batch["inputs"].device).unsqueeze(0)      # [1,T]
            valid_BT  = t_idx < seq_len.unsqueeze(1)                       # [B,T] True=valid
            P = self.msclip_model.num_patches
            valid_BPT = valid_BT.unsqueeze(1).expand(-1, P, -1)            # [B,P,T]
            patch_feats = patch_feats[valid_BPT[:,0,:],:,:]
            B, _,_ = patch_feats.shape

            # [B*T, P, D] -> [B*T, D, H_p, W_p]
            x = patch_feats.view(B, H_p, W_p, D).permute(0, 3, 1, 2).contiguous()

        elif x.ndim == 4:
            # Old path: we already have [B, D, H, W] activations
            pass
        else:
            raise ValueError(f"Unexpected input shape {tuple(x.shape)}, expected 4D or 5D.")

        B, D, H, W = x.shape

        # Flatten spatially to tokens [B*H*W, D]
        x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, D)
        x = x.float()

        
        #x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, unbiased=False, keepdim=True) + 1e-6)

        x = x.to(device=self.device, dtype=self.net.encoder.final_block[0].weight.dtype, non_blocking=True)
        
        z_pre, z, x_hat = self.net(x)
        loss = self.criterion(x, x_hat, z_pre, z, self.net.get_dictionary())

        # --- optional concept/text alignment term ---
        if getattr(self, "align_text_embs", None) is not None and getattr(self, "align_loss_coeff", 0.0) > 0 and self.current_epoch >= self.hparams.align_start_epoch:
            D = self.net.get_dictionary()                          # [C, D]
            D = torch.nn.functional.normalize(D, dim=1)
            T = torch.nn.functional.normalize(self.align_text_embs.to(D.device).float(), dim=1)  # [N, D]
            sims = D @ T.t()                                       # [C, N]
            top1 = sims.max(dim=1).values                          # [C]
            loss = loss - self.align_loss_coeff * top1.mean()      


        if self.hparams.bias_fire:
            m = None
            lbl = batch["label"]
            lbl = lbl.squeeze(1) if lbl.ndim==4 else lbl
            m = (lbl.reshape(-1) > 0).to(x.device)

            pos_mse = (x[m] - x_hat[m]).square().mean()
            loss = 0.7 * pos_mse + 0.3 * loss            # weight toward positives
            
        if tracker is None:
            tracker = DeadCodeTracker(z.shape[1], self.device)
        tracker.update(z)

        if flag_mse:
            mse = mse_criterion(x, x_hat, z_pre, z, self.net.get_dictionary())
            lat   = batch.get("latitude") if isinstance(batch, dict) else None
            label = batch.get("label")    if isinstance(batch, dict) else None
            region_mse = region_mse_bands_per_class(
                x, x_hat, z_pre, z, self.net.get_dictionary(), lat, label
            )
            return loss, z_pre, z, x, x_hat, (mse, region_mse)

        return loss, z_pre, z, x, x_hat

    def on_train_epoch_start(self):
        if self.current_epoch > 0 and self.current_epoch % self.hparams.resample_every_n_epochs == 0:
            print(f"Epoch {self.current_epoch}: Resampling dead codes...")
            self.net.eval()
            self._resample_dead_codes()
            self.net.train()
            self.log("train/resampled_codes", 1, prog_bar=True)

        if self.hparams.decay_k:
            start_K, end_K, T = 16, 4, 35  # decay to 4 over 15 epochs
            K = max(end_K, start_K - (self.current_epoch * (start_K-end_K)//T))
            self.net.top_k = int(K)

        self.train_dead_tracker = None
        self.train_dead_tracker = DeadCodeTracker(self.net.get_dictionary().shape[0], self.device)

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, _, codes, inputs, rec_inputs = self.step(batch, tracker=self.train_dead_tracker)
        sparsity_error = l0_eps(codes, 0).sum().item()
        rec_error = _compute_reconstruction_error(inputs, rec_inputs)
        self.log("train/r2", rec_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/l0", sparsity_error, on_step=False, on_epoch=True, prog_bar=True)
        # self._training_outputs.append({"loss": loss, "inputs": inputs.detach().cpu(), "rec_inputs": rec_inputs.detach().cpu()})
        return {"loss": loss}

    def on_train_epoch_end(self):
        dead_ratio = self.train_dead_tracker.get_dead_ratio()
        print(f"Dead Train Ratio {dead_ratio}")
        self.log("train/dead_features", dead_ratio, prog_bar=True)
        #self.train_dead_tracker = None

    def on_validation_epoch_start(self):
        self.val_dead_tracker = DeadCodeTracker(self.net.get_dictionary().shape[0], self.device)

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        
        loss, _, codes, inputs, rec_inputs, (mse, region_mse) = self.step(batch, tracker=self.val_dead_tracker, flag_mse=True)
        if isinstance(batch, dict) and ("label" in batch):
            lbl = batch["label"]
            if lbl.ndim == 4: lbl = lbl.squeeze(1)
            m = (lbl.reshape(-1) > 0).to(inputs.device)
            if m.any(): self.log("val/r2_fire", _compute_reconstruction_error(inputs[m], rec_inputs[m]),
                                on_step=False, on_epoch=True, prog_bar=True)
        sparsity_error = l0_eps(codes, 0).sum().item()
        hoyer_error = hoyer(codes).mean().item()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        if region_mse is not None:
            for bands in region_mse.keys():
                if region_mse[bands] is not None:
                    self.log(f"val/region_mse_{bands}", region_mse[bands], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/l0", sparsity_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/l2", avg_l2_loss(inputs, rec_inputs), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/hoyer", hoyer_error, on_step=False, on_epoch=True, prog_bar=True)
        self._val_outputs.append({"loss": loss, "inputs": inputs.detach().cpu(), "rec_inputs": rec_inputs.detach().cpu()})
        return {"loss": loss}

    # Potentially Add Frechet & Wasserstein
    def on_validation_epoch_end(self):
        outputs = self._val_outputs
        inputs = torch.cat([x["inputs"] for x in outputs], dim=0)
        rec_inputs = torch.cat([x["rec_inputs"] for x in outputs], dim=0)
        self._val_outputs.clear()
        rec_error = _compute_reconstruction_error(inputs, rec_inputs)
        self.log("val/r2", rec_error, prog_bar=True)

        # Computing Val Dead Ratio
        dead_ratio = self.val_dead_tracker.get_dead_ratio()
        print(f"Dead Val Ratio {dead_ratio}")
        self.log("val/dead_features", dead_ratio, prog_bar=True)
        self.val_alive_mask = self.val_dead_tracker.alive_features.detach().clone()
        self.val_dead_tracker = None

    def on_test_epoch_start(self):
        self.test_dead_tracker = DeadCodeTracker(self.net.get_dictionary().shape[0], self.device)

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:        

        loss, _, codes, inputs, rec_inputs, (mse, region_mse) = self.step(batch, tracker=self.test_dead_tracker, flag_mse=True)
        if isinstance(batch, dict) and ("label" in batch):
            lbl = batch["label"]
            if lbl.ndim == 4: lbl = lbl.squeeze(1)
            m = (lbl.reshape(-1) > 0).to(inputs.device)
            if m.any(): self.log("test/r2_fire", _compute_reconstruction_error(inputs[m], rec_inputs[m]),
                                on_step=False, on_epoch=True, prog_bar=True)
        sparsity_error = l0_eps(codes, 0).sum().item()
        hoyer_error = hoyer(codes).mean().item()
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        if region_mse is not None:
            for bands in region_mse.keys():
                if region_mse[bands] is not None:
                    self.log(f"test/region_mse_{bands}", region_mse[bands], on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/l0", sparsity_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/l2", avg_l2_loss(inputs, rec_inputs), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/hoyer", hoyer_error, on_step=False, on_epoch=True, prog_bar=True)
        self._test_outputs.append({"loss": loss, "inputs": inputs.detach().cpu(), "rec_inputs": rec_inputs.detach().cpu()})
        return {"loss": loss}

    # Potentially Add Frechet & Wasserstein
    def on_test_epoch_end(self):
        outputs = self._test_outputs
        inputs = torch.cat([x["inputs"] for x in outputs], dim=0)
        rec_inputs = torch.cat([x["rec_inputs"] for x in outputs], dim=0)
        self._test_outputs.clear()
        rec_error = _compute_reconstruction_error(inputs, rec_inputs)
        self.log("test/r2", rec_error, prog_bar=True)

        # Computing Test Dead Ratio
        dead_ratio = self.test_dead_tracker.get_dead_ratio()
        print(f"Dead Test Ratio {dead_ratio}")
        self.log("test/dead_features", dead_ratio, prog_bar=True)
        #self.test_dead_tracker = None


    @torch.no_grad()
    def _resample_dead_codes(self):
        """Resample dead codes using high-reconstruction-error tokens.
        Based on Appendix E.3 of transformer-circuits SAE work, adapted to our tensor shapes.
        """

        if self.geonet is not None:
            raise NotImplementedError("Resampling not implemented for Geo-Conditioned SAE")

        dead_indices = torch.where(self.train_dead_tracker.alive_features == False)[0]
        if len(dead_indices) == 0:
            return

        dataset = self.trainer.train_dataloader.dataset

        num_samples = min(self.hparams.num_samples, len(dataset))

        sampled_indices = random.sample(range(len(dataset)), num_samples)
        subset = Subset(dataset, sampled_indices)

        subset_dataloader = DataLoader(
            subset,
            batch_size=self.hparams.resample_batch_size,
            shuffle=False,
        )

        mse_loss = criterion_factory(loss_type="mse", aggregate_batch=False)

        all_losses = []
        all_tokens = []

        for batch in subset_dataloader:
            x = batch["inputs"]

            B, D, H, W = x.shape

            x_flat = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, D)

            x_flat = x_flat.float()
            #x_flat = (x_flat - x_flat.mean(dim=0, keepdim=True)) / (
            #    x_flat.std(dim=0, unbiased=False, keepdim=True) + 1e-6
            #)
            x_flat = x_flat.to(
                device=self.device,
                dtype=self.net.encoder.final_block[0].weight.dtype,
                non_blocking=True,
            )

            z_pre, z, x_hat = self.net(x_flat)

            loss_vec = mse_loss(
                x_flat, x_hat, z_pre, z, self.net.get_dictionary()
            ) 

            all_losses.append(loss_vec.pow(2).detach().cpu())
            all_tokens.append(x_flat.detach().cpu())

        all_losses = torch.cat(all_losses, dim=0)          # [N_tokens]
        all_tokens = torch.cat(all_tokens, dim=0)          # [N_tokens, D]

        probs = all_losses / (all_losses.sum() + 1e-8)

        num_dead = len(dead_indices)
        need_replacement = num_dead > all_tokens.shape[0]

        chosen_token_ids = torch.multinomial(
            probs,
            num_samples=num_dead,
            replacement=need_replacement,
        )

        for dead_idx, chosen_id in zip(dead_indices, chosen_token_ids):
            sampled_input = all_tokens[chosen_id].to(self.device)

            sampled_input = sampled_input / (sampled_input.norm(p=2) + 1e-8)

            self.net.dictionary._weights[dead_idx, :] = sampled_input

            alive_norm = self.net.encoder.final_block[0].weight[
                self.train_dead_tracker.alive_features, :
            ].norm(dim=1)
            mean_alive_norm = alive_norm.mean()
            target_norm = mean_alive_norm * 0.2

            self.net.encoder.final_block[0].weight[dead_idx, :] = (
                sampled_input * target_norm
            )
            self.net.encoder.final_block[0].bias[dead_idx] = 0.0

            optimizer = self.optimizers()
            if isinstance(optimizer, torch.optim.Adam):
                enc_weight_param = self.net.encoder.final_block[0].weight
                enc_bias_param = self.net.encoder.final_block[0].bias
                dec_weight_param = self.net.dictionary._weights

                for param, index in [
                    (enc_weight_param, dead_idx),
                    (enc_bias_param, dead_idx),
                    (dec_weight_param, dead_idx),
                ]:
                    state = optimizer.state.get(param, {})
                    if state:
                        state = optimizer.state[param]
                        if "exp_avg" in state and "exp_avg_sq" in state:
                            state["exp_avg"][index].zero_()
                            state["exp_avg_sq"][index].zero_()
            else:
                raise NotImplementedError("Specify reset for other optimizer types")



    @torch.no_grad()
    def _initialize_encoder_from_decoder(self):
        """Set encoder final layer weights equal to decoder dictionary weights."""
        self.net.encoder.final_block[0].weight.copy_(self.net.dictionary._weights)
        self.net.encoder.final_block[0].bias.zero_()


    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        optimizer = optimizer_factory(
            optim_type=self.hparams.optimizer_type, params=self.parameters(),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = scheduler_factory(
            scheduler_type=self.hparams.scheduler_type, optimizer=optimizer
        )

        if lr_scheduler is None:
            return optimizer
        else:
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": "train/loss"}