"""Script for SITS only lightning module"""

from math import ceil
from typing import Any, Dict

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn

from DeepSatModels.metrics.loss_functions import get_loss
from DeepSatModels.metrics.numpy_metrics import get_classification_metrics
from DeepSatModels.metrics.torch_metrics import get_binary_metrics, get_mean_metrics
from DeepSatModels.models.TSViT.TSViTdense import TSViTDown
from DeepSatModels.utils.lr_scheduler import build_scheduler_pytorch
from src.data.utils import segmentation_ground_truths
from src.eval.utils import get_pr_auc_scores
from src.models.convlstm import ConvLSTMNet
from src.models.resnet import ResNetConvLSTM
from src.models.vit import ViTFactorizeModel, ViTModel
from src.models.msclip_factorize_model import MSClipFactorizeModel
from src.utils.torch_utils import get_alpha, get_trainable_params
from src.models.l1c2l2a_adapter import L1C2L2AAdapterModel
import matplotlib.pyplot as plt



class ImgModule(LightningModule):
    """Lightning Module for SITS only training."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model_type = config["MODEL"]["architecture"]
        self.model = self.model_factory(config["MODEL"])
        self.multi_head = config["MODEL"]["multi_head"] if "multi_head" in config["MODEL"] else False
        self.loss_input_fn = segmentation_ground_truths
        self.loss_fn = {"mean": get_loss(config, "cuda" if torch.cuda.is_available() else "cpu", reduction="mean")}

        if self.multi_head:
            self.loss_fn["class"] = nn.CrossEntropyLoss()

        self.train_metrics_steps = config["CHECKPOINT"]["train_metrics_steps"]
        self.num_classes = config["MODEL"]["num_classes"]
        self.threshold = config["MODEL"]["threshold"]
        self.lr = config["SOLVER"]["lr_base"]
        self.lr_ratio = config["SOLVER"]["lr_ratio"] if "lr_ratio" in config["SOLVER"] else 1.0
        self.lr_mode = config["SOLVER"]["lr_mode"] if "lr_mode" in config["SOLVER"] else "full"
        self.weight_decay = config["SOLVER"]["weight_decay"]
        self.scheduler_config = {"SOLVER": config["SOLVER"]}
        self.num_steps_train = ceil(config["SOLVER"]["num_steps_train"] / config["SOLVER"]["accumulate_grad_batches"])
        self.interval = config["SOLVER"]["interval"] if "interval" in config["SOLVER"] else "step"
        self.save_hyperparameters()
        self.validation_step_outputs = []

        # --- ABMIL validation logging (flags) ---
        self.log_abmil = bool(config["MODEL"].get("log_abmil", False))
        self.log_abmil_fold_mixer = bool(config["MODEL"].get("log_abmil_fold_mixer", True))
        self.use_mixer = bool(config["MODEL"].get("use_mixer", True))
        self.abmil_fire_class_id = int(config["MODEL"].get("fire_class_id", 1))

        # --- accumulators (overall + per-class) ---
        self._abmil_bins = 36
        self._abmil_hist        = torch.zeros(self._abmil_bins, dtype=torch.float64)
        self._abmil_hist_pos    = torch.zeros(self._abmil_bins, dtype=torch.float64)
        self._abmil_hist_neg    = torch.zeros(self._abmil_bins, dtype=torch.float64)
        self._abmil_idx_sum     = None  # [T]
        self._abmil_idx_sum_pos = None  # [T]
        self._abmil_idx_sum_neg = None  # [T]
        self._abmil_rows        = 0
        self._abmil_rows_pos    = 0
        self._abmil_rows_neg    = 0
        self._abmil_eff_frames_sum     = 0.0
        self._abmil_eff_frames_sum_pos = 0.0
        self._abmil_eff_frames_sum_neg = 0.0
        self._abmil_T = None




    @staticmethod
    def model_factory(model_config: Dict[str, Any]) -> nn.Module:
        """Callable function for model initialization"""
        if model_config["architecture"] == "TSViT":
            return TSViTDown(model_config)
        if model_config["architecture"] == "ConvLSTM":
            return ConvLSTMNet(**model_config)
        if model_config["architecture"] == "ResNet":
            return ResNetConvLSTM(**model_config)
        if model_config["architecture"] == "ViT":
            return ViTModel(**model_config)
        if model_config["architecture"] == "ViTFacto":
            return ViTFactorizeModel(**model_config)
        if model_config["architecture"] == "MSClipFacto":
            return MSClipFactorizeModel(**model_config)
        if model_config["architecture"] == "L1C2L2AAdapterModel":
            return L1C2L2AAdapterModel(**model_config)
        raise NameError(f"Model architecture {model_config['architecture']} not found")

    def forward(self, x: torch.Tensor,doy:None,seq_len:None) -> torch.Tensor:
        """Forward call for the module"""
        return self.model(x,doy,seq_len)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Step for training"""

        ground_truth = self.loss_input_fn(batch)
        loss = 0

        if self.multi_head:
            outputs, class_outputs = self.model(batch["inputs"])
            class_target = ground_truth[0].long().amax(dim=(1, 2)).reshape(-1)
            class_loss = self.loss_fn["class"](class_outputs, class_target)
            alpha = get_alpha(self.current_epoch)
            loss += alpha * class_loss
            seg_weight = 1 - alpha
            self.log("train_class_loss", class_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log("alpha", alpha, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            outputs = self.model(batch["inputs"],batch["doy"],batch["seq_lengths"] )
            seg_weight = 1

        outputs = outputs.permute(0, 2, 3, 1)
        seg_loss = self.loss_fn["mean"](outputs.reshape(-1, self.num_classes), ground_truth[0].reshape(-1).long())
        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        loss += seg_weight * seg_loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        logits = outputs
        if self.global_step % self.train_metrics_steps == 0:
            labels, unk_masks = ground_truth
            logits = logits.permute(0, 3, 1, 2)  # Logits need to be (N, D, H, W)
            if self.num_classes == 1:
                batch_metrics = get_binary_metrics(logits=logits, labels=labels, thresh=self.threshold, name="step_")
            else:
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, n_classes=self.num_classes, unk_masks=unk_masks, name="step_"
                )

            self.log_dict(batch_metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        w = getattr(self.model.temp_enc, "last_attn_weights", None)
        if w is not None:
            H = -(w * (w.clamp_min(1e-8).log())).sum(dim=-1).mean()  # Shannon entropy
            loss = loss + 1e-3 * H


        return {"loss": loss, "logits": outputs, "ground_truths": ground_truth}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Step for validation"""

        ground_truth = self.loss_input_fn(batch)
        loss = 0

        if self.multi_head:
            logits, class_logits = self.model(batch["inputs"])
            class_target = ground_truth[0].long().amax(dim=(1, 2)).reshape(-1)
            class_loss = self.loss_fn["class"](class_logits, class_target)
            alpha = get_alpha(self.current_epoch)
            loss += alpha * class_loss
            seg_weight = 1 - alpha
        else:
            logits = self.model(batch["inputs"],batch["doy"],batch["seq_lengths"])
            seg_weight = 1
            class_loss = torch.tensor(0)

        logits = logits.permute(0, 2, 3, 1)

        if self.num_classes == 1:
            probs = torch.nn.functional.sigmoid(logits)
            predicted = (probs > self.threshold).to(torch.float32)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, predicted = torch.max(logits.data, -1)

        seg_loss = self.loss_fn["mean"](logits.reshape(-1, self.num_classes), ground_truth[0].reshape(-1).long())
        loss += seg_weight * seg_loss
        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        labels, unk_masks = ground_truth
        if unk_masks is not None:
            preds = predicted.view(-1)[unk_masks.view(-1)].cpu().numpy()
            probs = probs.view(-1, self.num_classes)[unk_masks.view(-1)].cpu().numpy()
            labels = labels.view(-1)[unk_masks.view(-1)].cpu().numpy()
        else:
            preds = predicted.view(-1).cpu().numpy()
            probs = probs.view(-1, self.num_classes).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()

        loss = loss.view(-1).cpu().detach().numpy()
        class_loss = class_loss.view(-1).cpu().detach().numpy()
        self.validation_step_outputs.append(
            {"predictions": preds, "labels": labels, "loss": loss, "probs": probs, "class_loss": class_loss}
        )

        # --- ABMIL VALID ACCUMULATION (fold mixer optional) ---
        if self.log_abmil and (self.model.temp_enc_type != "attention") and self.model.ABMIL:
            B, T = batch["inputs"].shape[0], batch["inputs"].shape[1]
            P = self.model.num_patches
            BPT = B * P

            # (Re)init [T]-shaped accumulators when T changes
            if (self._abmil_T is None) or (self._abmil_T != T):
                self._abmil_T = T
                self._abmil_idx_sum     = torch.zeros(T, dtype=torch.float64)
                self._abmil_idx_sum_pos = torch.zeros(T, dtype=torch.float64)
                self._abmil_idx_sum_neg = torch.zeros(T, dtype=torch.float64)

            # 1) ABMIL weights from pooler (float32)
            w_eff = self.model.temp_enc.last_attn_weights  # expected [B*P, T] or [T, B*P]

            # ---- force last dim to be T (no B*P guessing) ----
            if w_eff.dim() != 2:
                raise RuntimeError(f"ABMIL weights must be 2D, got {tuple(w_eff.shape)}")
            if w_eff.shape[1] != T:
                if w_eff.shape[0] == T:
                    w_eff = w_eff.transpose(0, 1)  # -> [B*P, T]
                else:
                    raise RuntimeError(
                        f"ABMIL weights has shape {tuple(w_eff.shape)}; "
                        f"neither dim equals T={T}. Cannot orient."
                    )
            # at this point w_eff is [B*P, T]

            # 2) DOY -> [B*P, T] in days (double for stable histogramming)
            doy_bt = batch["doy"]
            if doy_bt.ndim > 2:  # [B,T,H,W,1] -> [B,T]
                doy_bt = doy_bt.view(B, T, -1)[:, :, 0]
            doy_bt = doy_bt.clamp(0, 1)
            doy_bp = doy_bt.unsqueeze(1).expand(B, P, T).reshape(BPT, T)
            doy_days = (doy_bp * 365.0).to(torch.float64)

            # 3) Optional: fold mixer attention for frame-level attribution
            if self.log_abmil_fold_mixer and self.use_mixer:
                A = self.model.temporal_mixer.last_attn  # [B*P, T, T] (float32)
                I = torch.eye(T, device=A.device, dtype=A.dtype).unsqueeze(0)
                A_aug   = A + I
                A_tilde = A_aug / (A_aug.sum(dim=-1, keepdim=True) + 1e-12)
                w_eff   = torch.einsum("bt,bts->bs", w_eff.to(A_tilde.dtype), A_tilde)    # still float32, now frame-level
            # 4) From here: stats in float64
            w64 = w_eff.to(torch.float64)                 # [B*P, T]

            # ---- DOY histogram (0..365), attention-weighted ----
            edges = torch.linspace(0.0, 365.0, steps=self._abmil_bins + 1,
                                device=doy_days.device, dtype=doy_days.dtype)
            bin_idx = torch.bucketize(doy_days.reshape(-1), edges) - 1
            bin_idx = bin_idx.clamp_(0, self._abmil_bins - 1)
            hist_batch = torch.zeros(self._abmil_bins, dtype=torch.float64, device=doy_days.device)
            hist_batch.scatter_add_(0, bin_idx, w64.reshape(-1))
            self._abmil_hist = self._abmil_hist + hist_batch.cpu()

            # ---- average weight per time index (strictly [T]) ----
            idx_add = w64.sum(dim=0).to(torch.float64).cpu()  # MUST be [T]
            if idx_add.shape[0] != self._abmil_T:
                # If this ever triggers again, w was [T, B*P]; we guard above, so this should not happen.
                raise RuntimeError(f"idx_add has length {idx_add.shape[0]} but T={self._abmil_T}")
            self._abmil_idx_sum = self._abmil_idx_sum + idx_add

            # effective frames
            self._abmil_rows += w64.shape[0]
            self._abmil_eff_frames_sum += (1.0 / (w64.pow(2).sum(dim=1) + 1e-12)).sum().item()

            # ---- per-class split at PATCH resolution (labels shape-safe) ----
            lab = ground_truth[0]                          # [B,H,W] or [B,H,W,1] or [B,1,H,W]
            if lab.ndim == 4 and lab.shape[-1] == 1:       # [B,H,W,1] -> [B,1,H,W]
                lab = lab.permute(0, 3, 1, 2)
            elif lab.ndim == 3:                            # [B,H,W] -> [B,1,H,W]
                lab = lab.unsqueeze(1)
            labels_patch = torch.nn.functional.interpolate(
                lab.float(), size=(self.model.H_patch, self.model.W_patch), mode="nearest"
            ).squeeze(1).long()                            # [B, H_patch, W_patch]
            labels_flat = labels_patch.view(B, -1)         # [B, P]
            fire_mask = (labels_flat.reshape(-1) == self.abmil_fire_class_id).to(w_eff.device)
            non_mask  = ~fire_mask

            if fire_mask.any():
                w_pos = w64[fire_mask]; d_pos = doy_days[fire_mask]
                bin_pos = torch.bucketize(d_pos.reshape(-1), edges) - 1
                bin_pos = bin_pos.clamp_(0, self._abmil_bins - 1)
                h_pos = torch.zeros(self._abmil_bins, dtype=torch.float64, device=doy_days.device)
                h_pos.scatter_add_(0, bin_pos, w_pos.reshape(-1))
                self._abmil_hist_pos = self._abmil_hist_pos + h_pos.cpu()

                pos_add = w_pos.sum(dim=0).to(torch.float64).cpu()   # [T]
                if pos_add.shape[0] != self._abmil_T:
                    raise RuntimeError(f"pos_add len {pos_add.shape[0]} != T={self._abmil_T}")
                self._abmil_idx_sum_pos = self._abmil_idx_sum_pos + pos_add
                self._abmil_rows_pos += w_pos.shape[0]
                self._abmil_eff_frames_sum_pos += (1.0 / (w_pos.pow(2).sum(dim=1) + 1e-12)).sum().item()

            if non_mask.any():
                w_neg = w64[non_mask]; d_neg = doy_days[non_mask]
                bin_neg = torch.bucketize(d_neg.reshape(-1), edges) - 1
                bin_neg = bin_neg.clamp_(0, self._abmil_bins - 1)
                h_neg = torch.zeros(self._abmil_bins, dtype=torch.float64, device=doy_days.device)
                h_neg.scatter_add_(0, bin_neg, w_neg.reshape(-1))
                self._abmil_hist_neg = self._abmil_hist_neg + h_neg.cpu()

                neg_add = w_neg.sum(dim=0).to(torch.float64).cpu()   # [T]
                if neg_add.shape[0] != self._abmil_T:
                    raise RuntimeError(f"neg_add len {neg_add.shape[0]} != T={self._abmil_T}")
                self._abmil_idx_sum_neg = self._abmil_idx_sum_neg + neg_add
                self._abmil_rows_neg += w_neg.shape[0]
                self._abmil_eff_frames_sum_neg += (1.0 / (w_neg.pow(2).sum(dim=1) + 1e-12)).sum().item()


        return {"predictions": preds, "labels": labels, "loss": loss, "probs": probs}

    def on_validation_epoch_start(self):
        # Reset ABMIL accumulators each val epoch (so last-epoch logs are clean)
        if self.log_abmil and (self.model.temp_enc_type != "attention") and self.model.ABMIL:
            self._abmil_hist.zero_()
            self._abmil_hist_pos.zero_()
            self._abmil_hist_neg.zero_()
            if self._abmil_T is not None:
                self._abmil_idx_sum     = torch.zeros(self._abmil_T, dtype=torch.float64)
                self._abmil_idx_sum_pos = torch.zeros(self._abmil_T, dtype=torch.float64)
                self._abmil_idx_sum_neg = torch.zeros(self._abmil_T, dtype=torch.float64)
            self._abmil_rows = self._abmil_rows_pos = self._abmil_rows_neg = 0
            self._abmil_eff_frames_sum = self._abmil_eff_frames_sum_pos = self._abmil_eff_frames_sum_neg = 0.0


    def on_validation_epoch_end(self):
        """Epoch end evaluation for validation"""
        outputs = self.validation_step_outputs
        predicted_classes = np.concatenate([out["predictions"] for out in outputs])
        target_classes = np.concatenate([out["labels"] for out in outputs])
        probs_classes = np.concatenate([out["probs"] for out in outputs])

        # losses = np.concatenate([out["losses"] for out in outputs])
        loss = np.array([out["loss"] for out in outputs])
        class_loss = np.array([out["class_loss"] for out in outputs])
        print(predicted_classes.shape, target_classes.shape, probs_classes.shape, loss.shape)
        self.validation_step_outputs.clear()

        eval_metrics = get_classification_metrics(
            predicted=predicted_classes,
            labels=target_classes,
            n_classes=self.num_classes + 1 if self.num_classes == 1 else self.num_classes,
            unk_masks=None,
        )

        micro_auc, macro_auc, class_auc = get_pr_auc_scores(
            scores=probs_classes,
            labels=target_classes,
            n_classes=self.num_classes + 1 if self.num_classes == 1 else self.num_classes,
        )

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics["micro"]
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics["macro"]
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics["class"]

        val_metrics = {
            "val_loss_epoch": loss.mean(),
            "macro_Accuracy": macro_acc,
            "macro_Precision": macro_precision,
            "macro_Recall": macro_recall,
            "macro_F1": macro_F1,
            "macro_IOU": macro_IOU,
            "macro_AUC": macro_auc,
            "micro_Accuracy": micro_acc,
            "micro_Precision": micro_precision,
            "micro_Recall": micro_recall,
            "micro_F1": micro_F1,
            "micro_IOU": micro_IOU,
            "micro_AUC": micro_auc,
            "fire_Accuracy": class_acc[1],
            "fire_Precision": class_precision[1],
            "fire_Recall": class_recall[1],
            "fire_F1": class_F1[1],
            "fire_IOU": class_IOU[1],
            "fire_AUC": class_auc[1],
        }

        if class_loss.mean() > 0:
            val_metrics["class_loss"] = class_loss.mean()

        self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # --- ABMIL (final epoch only) -> W&B tables ---
        is_last_epoch = (self.current_epoch + 1) == self.trainer.max_epochs
        if is_last_epoch and self.log_abmil and (self.model.temp_enc_type != "attention") and self.model.ABMIL and (self._abmil_rows > 0):
            import wandb
            run = self.logger.experiment
            centers = (torch.linspace(0.0, 365.0, steps=self._abmil_bins + 1)[:-1] + 0.5 * (365.0 / self._abmil_bins)).tolist()

            def _safe_avg(x_sum, n_rows):
                return (x_sum / float(n_rows)).tolist() if n_rows > 0 else [0.0] * self._abmil_T

            # overall
            avg_idx_all = _safe_avg(self._abmil_idx_sum, self._abmil_rows)
            eff_all = self._abmil_eff_frames_sum / float(self._abmil_rows)
            run.log({"abmil/fold_mixer": int(self.log_abmil_fold_mixer), "abmil/eff_frames_mean": eff_all})
            run.log({"abmil/doy_hist_table": wandb.Table(columns=["day_of_year","attn_weight"],
                                    data=[[float(c), float(h)] for c,h in zip(centers, self._abmil_hist.tolist())])})
            run.log({"abmil/avg_weight_per_index": wandb.Table(columns=["time_index","avg_weight"],
                                    data=[[int(i), float(v)] for i,v in enumerate(avg_idx_all)])})

            # positive (fire)
            if self._abmil_rows_pos > 0:
                avg_idx_pos = _safe_avg(self._abmil_idx_sum_pos, self._abmil_rows_pos)
                eff_pos = self._abmil_eff_frames_sum_pos / float(self._abmil_rows_pos)
                run.log({"abmil_pos/eff_frames_mean": eff_pos})
                run.log({"abmil_pos/doy_hist_table": wandb.Table(columns=["day_of_year","attn_weight"],
                                        data=[[float(c), float(h)] for c,h in zip(centers, self._abmil_hist_pos.tolist())])})
                run.log({"abmil_pos/avg_weight_per_index": wandb.Table(columns=["time_index","avg_weight"],
                                        data=[[int(i), float(v)] for i,v in enumerate(avg_idx_pos)])})

            # negative (non-fire)
            if self._abmil_rows_neg > 0:
                avg_idx_neg = _safe_avg(self._abmil_idx_sum_neg, self._abmil_rows_neg)
                eff_neg = self._abmil_eff_frames_sum_neg / float(self._abmil_rows_neg)
                run.log({"abmil_neg/eff_frames_mean": eff_neg})
                run.log({"abmil_neg/doy_hist_table": wandb.Table(columns=["day_of_year","attn_weight"],
                                        data=[[float(c), float(h)] for c,h in zip(centers, self._abmil_hist_neg.tolist())])})
                run.log({"abmil_neg/avg_weight_per_index": wandb.Table(columns=["time_index","avg_weight"],
                                        data=[[int(i), float(v)] for i,v in enumerate(avg_idx_neg)])})



    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Function to set-up the optimizer and scheduler"""
        pool_params, base_params = [], []
        for n,p in self.model.named_parameters():
            if not p.requires_grad: 
                continue
            (pool_params if "temp_enc" in n else base_params).append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": self.lr, "weight_decay": self.weight_decay},
                {"params": pool_params, "lr": self.lr * 0.1, "weight_decay": 0.0},
            ]
        )
        scheduler = build_scheduler_pytorch(config=self.scheduler_config, optimizer=optimizer,
                                            n_iter_per_epoch=self.num_steps_train, interval=self.interval)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": self.interval}}

