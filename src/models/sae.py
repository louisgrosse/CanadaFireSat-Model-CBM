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


from src.utils.sae_utils import criterion_factory, optimizer_factory, scheduler_factory, mse_criterion,\
    region_mse_bands_per_class, region_r2_bands_per_class


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
            sae_kwargs: Dict[str, Any] = {},
            criterion_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = self.sae_factory(sae_type, **sae_kwargs)
        self.criterion = criterion_factory(loss_type, **criterion_kwargs)

        if depth_scale_shift > 0:
            self.geonet = nn.ModuleList()
            self.geonet.append(nn.Linear(4, geo_embed_dim)) # Apply Sin / Cos -> 4
            self.geonet.append(nn.GELU())
            self.geonet.append(nn.LayerNorm(geo_embed_dim))
            for _ in range(depth_scale_shift - 1):
                self.geonet.append(nn.Linear(geo_embed_dim, geo_embed_dim))
                self.geonet.append(nn.GELU())
                self.geonet.append(nn.LayerNorm(geo_embed_dim))
            
            out_layer = nn.Linear(geo_embed_dim, 2*self.net.dictionary.in_dimensions)
            nn.init.zeros_(out_layer.bias)
            nn.init.xavier_uniform_(out_layer.weight, gain=0.01)
            self.geonet.append(out_layer) # scale & shift
            self.geonet = nn.Sequential(*self.geonet)

        else:
            self.geonet = None

        self.train_dead_tracker = None
        self.val_dead_tracker = None
        self.test_dead_tracker = None

        if bind_init:
            print("Binding the encoder and decoder weights")
            self._initialize_encoder_from_decoder()

        #self._training_outputs = []
        self._val_outputs = []
        self._test_outputs = []

    @staticmethod
    def sae_factory(sae_type: str, **sae_kwargs) -> nn.Module:

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
        x = extract_input(batch)

        B, D, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, D)

        x = x.float()
        
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, unbiased=False, keepdim=True) + 1e-6)

        x = x.to(device=self.device, dtype=self.net.encoder.final_block[0].weight.dtype, non_blocking=True)
        
        z_pre, z, x_hat = self.net(x)
        loss = self.criterion(x, x_hat, z_pre, z, self.net.get_dictionary())

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
        # self.train_dead_tracker = None

    def on_validation_epoch_start(self):
        self.val_dead_tracker = DeadCodeTracker(self.net.get_dictionary().shape[0], self.device)

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, _, codes, inputs, rec_inputs, (mse, region_mse) = self.step(batch, tracker=self.val_dead_tracker, flag_mse=True)
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
        self.val_dead_tracker = None

    def on_test_epoch_start(self):
        self.test_dead_tracker = DeadCodeTracker(self.net.get_dictionary().shape[0], self.device)

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, _, codes, inputs, rec_inputs, (mse, region_mse) = self.step(batch, tracker=self.test_dead_tracker, flag_mse=True)
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
        self.test_dead_tracker = None


    @torch.no_grad()
    def _resample_dead_codes(self):
        """Implements resampling of dead codes from
        https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-resampling"""

        if self.geonet is not None:
            raise NotImplementedError("Resampling not implemented for Geo-Conditioned SAE")

        dead_indices = torch.where(self.train_dead_tracker.alive_features == False)[0]
        if len(dead_indices) == 0:
            return

        dataset = self.trainer.train_dataloader.dataset

        sampled_indices = random.sample(range(len(dataset)), self.hparams.num_samples)
        subset = Subset(dataset, sampled_indices)
        subset_dataloader = DataLoader(subset, batch_size=self.hparams.resample_batch_size, shuffle=False)

        mse_loss = criterion_factory(loss_type="mse", aggregate_batch=False)

        tot_indices = []
        tot_loss = []
        for i, batch in enumerate(subset_dataloader):
            x = extract_input(batch)
            x = x.to(self.device)
            z_pre, z, x_hat = self.net(x)
            loss = mse_loss(x, x_hat, z_pre, z, self.net.get_dictionary())

            start = i * self.hparams.resample_batch_size
            end = start + loss.shape[0]

            tot_indices.extend(sampled_indices[start:end])
            tot_loss.append(loss.pow(2).detach().cpu())

        tot_loss = torch.cat(tot_loss, dim=0)
        tot_probs = tot_loss / tot_loss.sum()

        chosen_indices= torch.multinomial(tot_probs, num_samples=len(dead_indices), replacement=False)
        chosen_dataset_indices = [tot_indices[i] for i in chosen_indices]
        for dead_idx, chosen_idx in zip(dead_indices, chosen_dataset_indices):
            raw_input = dataset[chosen_idx]
            sampled_input = extract_input(raw_input).to(self.device)
            sampled_input = sampled_input / sampled_input.norm(p=2)
            self.net.dictionary._weights[dead_idx, :] = sampled_input # Based on Overcomplete Framework
            alive_norm = self.net.encoder.final_block[0].weight[self.train_dead_tracker.alive_features, :].norm(dim=1) # Dimension should be n_concept, last_dimension
            mean_alive_norm = alive_norm.mean()
            target_norm = mean_alive_norm * 0.2
            self.net.encoder.final_block[0].weight[dead_idx, :] = sampled_input * target_norm
            self.net.encoder.final_block[0].bias[dead_idx] = 0.

            optimizer = self.optimizers()
            if isinstance(optimizer, torch.optim.Adam):
                # Handle encoder weight row
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