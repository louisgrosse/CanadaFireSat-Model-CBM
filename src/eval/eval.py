"""Evaluation Script for Validation, Test, and Test Hard"""
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

import torch.nn.functional as F      
import json                         
import csv
import pandas as pd
from collections import defaultdict
import numpy as np

from DeepSatModels.metrics.numpy_metrics import get_classification_metrics
from src.constants import CONFIG_PATH
from src.data import get_data
from src.data.utils import segmentation_ground_truths
from src.eval.utils import get_pr_auc_scores
from src.models.module_img import ImgModule
from src.models.module_tab import TabModule




@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="eval")
@torch.no_grad()
def evaluate(cfg: DictConfig):
    """Evaluation Endpoint"""

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    seed_everything(cfg["seed"])

    # Load the model
    if cfg["mode"] == "image":
        try:
            model = ImgModule.load_from_checkpoint(cfg["model_path"])
        except (KeyError, RuntimeError):
            with open(Path(cfg["model_path"]).parent / cfg["config_name"], "r") as f:
                model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
            model = ImgModule(model_cfg)
            mis_keys, un_keys = model.load_state_dict(torch.load(cfg["model_path"]), strict=True)
            print("Missing keys:", mis_keys)

        # Test different image sizes
        if cfg["MODEL"]["out_H"] != model.model.out_H or cfg["MODEL"]["out_W"] != model.model.out_W:
            model.model.out_H = cfg["MODEL"]["out_H"]
            model.model.out_W = cfg["MODEL"]["out_W"]

    else:
        try:
            model = TabModule.load_from_checkpoint(cfg["model_path"])
        except KeyError or RuntimeError:
            with open(Path(cfg["model_path"]).parent / cfg["config_name"], "r") as f:
                model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
            model = TabModule(model_cfg)
            mis_keys, un_keys = model.load_state_dict(torch.load(cfg["model_path"]), strict=True)
            print("Missing keys:", mis_keys)
            print("Unexpected keys:", un_keys)

        # Test different image sizes
        if model.model_type in ["TabTSViT", "TabConvLSTM"]:

            if cfg["MODEL"]["out_H"] != model.model.out_H or cfg["MODEL"]["out_W"] != model.model.out_W:
                model.model.out_H = cfg["MODEL"]["out_H"]
                model.model.out_W = cfg["MODEL"]["out_W"]

        elif model.model_type in ["EnvResNet", "EnvViTFactorizeModel"]:
            if (
                cfg["MODEL"]["out_H"] != model.model.mid_model.out_H
                or cfg["MODEL"]["out_W"] != model.model.mid_model.out_W
            ):
                model.model.mid_model.out_H = cfg["MODEL"]["out_H"]
                model.model.mid_model.out_W = cfg["MODEL"]["out_W"]

        else:
            if (
                cfg["MODEL"]["out_H"] != model.model.sat_model.out_H
                or cfg["MODEL"]["out_W"] != model.model.sat_model.out_W
            ):
                model.model.sat_model.out_H = cfg["MODEL"]["out_H"]
                model.model.sat_model.out_W = cfg["MODEL"]["out_W"]

        if "ViT" in model.model_type and model.model_type != "EnvViTFactorizeModel":

            if model.model.sat_model.features.patch_embed.img_size != (
                cfg["MODEL"]["img_res"],
                cfg["MODEL"]["img_res"],
            ):
                model.model.sat_model.features.patch_embed.img_size = (cfg["MODEL"]["img_res"], cfg["MODEL"]["img_res"])

    # Set model to evaluation mode
    model.eval()
    model.to(device)

    # Load the dataset need to extend a possibility of no augmentation
    datamodule = get_data(cfg)
    dataset = datamodule.test_dataloader(split=cfg["split"]).dataset

    # Create output directory
    if "test_max_seq_len" in cfg["MODEL"]:
        temp_size = str(cfg["MODEL"]["test_max_seq_len"])
    elif "env_val_max_seq_len" in cfg["MODEL"]:
        temp_size = str(cfg["MODEL"]["env_val_max_seq_len"])
    else:
        temp_size = "adapt"

    temp_size = (
        temp_size + f"_{cfg['DATASETS']['kwargs']['eval_sampling']}"
        if "eval_sampling" in cfg["DATASETS"]["kwargs"]
        else temp_size
    )
    spa_size = str(cfg["MODEL"]["img_res"]) if "img_res" in cfg["MODEL"] else str(cfg["MODEL"]["mid_input_res"])

    if cfg["DATASETS"]["eval"].get("hard"):
        output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}_hard"
    else:
        output_dir = Path(cfg["output_dir"]) / f"{cfg['split']}_temp_{temp_size}_spa_{spa_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Outputs initialization
    tot_preds = []
    tot_labels = []
    tot_losses = []
    tot_probs = []

    # Collect metadata
    tot_regions = []
    tot_fwi = []

    concept_correct = None   # TP counts per concept (fire predicted & correct)
    concept_false = None     # FP counts per concept (fire predicted but wrong)
    concept_names = None
    concept_ids = None
    concept_cos = None

    base_model = None
    text_embs = None

    if model.model.useCBM and model.model.log_concepts:
        base_model = model.model  # MSClipFactorizeModel
        sae_module = model.model.sae
        if sae_module is not None and model.model.useCBM:
            sae_config_path = None
            if "MODEL" in cfg and "sae_config" in cfg["MODEL"]:
                sae_config_path = cfg["MODEL"]["sae_config"]

            align = None
            if sae_config_path is not None:
                try:
                    cfg_sae = OmegaConf.load(sae_config_path)
                    align = cfg_sae.get("ALIGN", None)
                except Exception as e:
                    print(f"[CBM] Could not load SAE config from {sae_config_path}: {e}")

            if align is not None and align.get("csv_phrases_path") and align.get("csv_cosLoss_path"):
                try:
                    print("align csv ohrase =================>", align.get("csv_phrases_path"))
                    device_text = str(align.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
                    msclip_model_name = align["msclip_model_name"]
                    ckpt_path_msclip = align.get("msclip_ckpt", None)
                    
                    model_clip = base_model.msclip_model
                    tokenizer = base_model._tokenizer
                    msclip = model_clip.eval().to(device_text)

                    csv_name = align["csv_cosLoss_path"]
                    csv_dir = str(align["csv_phrases_path"])
                    csv_path = csv_dir + str(csv_name)

                    # Separator hack as in train_sae
                    if csv_name in ("concept_countsCLIPDATASET.csv", "concept_counts50k_yake.csv"):
                        df = pd.read_csv(csv_path, sep=';')
                    else:
                        df = pd.read_csv(csv_path)

                    phrases = df["concept"].astype(str).tolist()

                    text_embs = _encode_phrases_msclip(
                        phrases=phrases,
                        model=msclip,
                        tokenizer=tokenizer,
                        batch_size=int(align.get("text_batch_size", 512)),
                        device=device_text,
                    )

                    concept_names, concept_cos = _best_phrase_per_concept(
                        sae_module=sae_module,
                        text_embs=text_embs,
                        phrases=phrases,
                        device=device_text,
                    )

                    num_concepts = len(concept_names)
                    concept_correct = np.zeros(num_concepts, dtype=np.int64)
                    concept_correct_region = defaultdict(lambda: np.zeros(num_concepts, dtype=np.int64))
                    concept_false_region   = defaultdict(lambda: np.zeros(num_concepts, dtype=np.int64))
                    concept_false = np.zeros(num_concepts, dtype=np.int64)
                    concept_ids = np.arange(num_concepts, dtype=np.int64)

                    print(f"[CBM] Concept text labels built for {num_concepts} concepts.")

                except Exception as e:
                    print(f"[CBM] Could not build concept text mapping: {e}")
            
               


    # Evaluation loop
    for i in tqdm(range(len(dataset)), desc=f"Inferring on split {cfg['split']}", total=len(dataset)):
        data = dataset[i]
        # Extract Batch & Forward Pass
        with torch.no_grad():
            if cfg["mode"] == "image":
                sample = data[0]
                img_name_info = data[1]
                if model.model_type == 'MSClipFacto':
                    logits = model(sample["inputs"].unsqueeze(0).to(device),sample["doy"].unsqueeze(0).to(device),torch.tensor([sample["seq_lengths"]]).to(device))
                else:
                    logits = model(sample["inputs"].unsqueeze(0).to(device))
            else:

                if model.model_type in [
                    "TabTSViT",
                    "TabConvLSTM",
                    "TabResNetConvLSTM",
                    "TabViTFactorizeModel",
                    "MultiViTFactorizeModel",
                ]:

                    sample = data[0]
                    tab_sample = data[1]
                    img_name_info = data[2]
                    logits = model(
                        sample["inputs"].unsqueeze(0).to(device),
                        tab_sample["tab_inputs"].unsqueeze(0).to(device),
                        tab_sample["mask"].unsqueeze(0).to(device),
                    )

                elif model.model_type in ["EnvResNet", "EnvViTFactorizeModel"]:

                    sample = data[0]
                    img_name_info = data[1]
                    logits = model(
                        xmid=sample["mid_inputs"].unsqueeze(0).to(device),
                        xlow=sample["low_inputs"].unsqueeze(0).to(device),
                        m_mid=sample["mid_inputs_mask"].unsqueeze(0).to(device),
                        m_low=sample["low_inputs_mask"].unsqueeze(0).to(device),
                    )

                else:

                    sample = data[0]
                    env_sample = data[1]
                    img_name_info = data[2]

                    logits = model(
                        x=sample["inputs"].unsqueeze(0).to(device),
                        xmid=env_sample["mid_inputs"].unsqueeze(0).to(device),
                        xlow=env_sample["low_inputs"].unsqueeze(0).to(device),
                        m_mid=env_sample["mid_inputs_mask"].unsqueeze(0).to(device),
                        m_low=env_sample["low_inputs_mask"].unsqueeze(0).to(device),
                    )

        logits = logits.permute(0, 2, 3, 1)
        ground_truth = segmentation_ground_truths(sample)
        labels, unk_masks = ground_truth


        if cfg["MODEL"]["num_classes"] == 1:
            probs = torch.nn.functional.sigmoid(logits)
            predicted = (probs > cfg["MODEL"]["threshold"]).to(torch.float32)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, predicted = torch.max(logits.data, -1)

                # ---- CBM per-concept TP/FP counting ----
        if (model.model.useCBM and model.model.log_concepts):
            concept_map = getattr(model.model, "last_concept_map", None)
            if concept_map is not None:
                # concept_map: [1, C, Hc, Wc]
                Bc, Cc, Hc, Wc = concept_map.shape

                # Upsample concept map to logits size (out_H, out_W)
                H_out, W_out = logits.shape[1], logits.shape[2]
                concept_map_up = F.interpolate(
                    concept_map,
                    size=(H_out, W_out),
                    mode="bilinear",
                    align_corners=False,
                )[0]  # [C, H, W]

                # concept is "used" at a pixel if its activation > 0
                concept_active = (concept_map_up > 0)  # [C, H, W]

                if cfg["MODEL"]["num_classes"] == 1:
                    # probs: [1, H, W, 1]
                    fire_pred_mask = (probs[0, ..., 0] > cfg["MODEL"]["threshold"])
                    fire_label_mask = (ground_truth[0].to(device) > 0.5)
                else:
                    fire_class_id = int(cfg["MODEL"].get("fire_class_id", 1))
                    fire_pred_mask = (predicted[0] == fire_class_id)
                    fire_label_mask = (ground_truth[0].to(device) == fire_class_id)

                if unk_masks is not None:
                    valid_mask = unk_masks[0].to(device).bool()
                    fire_pred_mask = fire_pred_mask & valid_mask
                    fire_label_mask = fire_label_mask & valid_mask

                tp_mask = fire_pred_mask & fire_label_mask        # correct fire prediction
                fp_mask = fire_pred_mask & (~fire_label_mask)     # fire predicted, but not fire in GT

                # Flatten spatial dims
                tp_flat = tp_mask.view(-1)   # [H*W]
                fp_flat = fp_mask.view(-1)

                # concept_active: [C, H, W] -> [C, H*W]
                concept_active_flat = concept_active.view(Cc, -1)

                # Count TP/FP per concept where concept is active
                tp_counts = (concept_active_flat & tp_flat.unsqueeze(0)).sum(dim=1).cpu().numpy()
                fp_counts = (concept_active_flat & fp_flat.unsqueeze(0)).sum(dim=1).cpu().numpy()

                region_name = img_name_info["region"]
                concept_correct_region[region_name] += tp_counts
                concept_false_region[region_name]   += fp_counts

                concept_correct += tp_counts.astype(np.int64)
                concept_false += fp_counts.astype(np.int64)


        loss = model.loss_fn["mean"](
            logits.reshape(-1, cfg["MODEL"]["num_classes"]), ground_truth[0].to(device).reshape(-1).long()
        )

        if unk_masks is not None:
            preds = predicted.view(-1)[unk_masks.view(-1)].cpu().numpy()
            probs = probs.view(-1, cfg["MODEL"]["num_classes"])[unk_masks.view(-1)].cpu().numpy()
            labels = labels.view(-1)[unk_masks.view(-1)].cpu().numpy()
        else:
            preds = predicted.view(-1).cpu().numpy()
            probs = probs.view(-1, cfg["MODEL"]["num_classes"]).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()

        loss = loss.view(-1).cpu().detach().numpy()

        tot_preds.append(preds)
        tot_labels.append(labels)
        tot_losses.append(loss)
        tot_probs.append(probs)

        region = [img_name_info["region"]] * len(preds)
        fwi = [img_name_info["fwinx_mean"]] * len(preds)
        # fwi = [1.]*len(preds)
        tot_regions.append(region)
        tot_fwi.append(fwi)

    # Concatenate all predictions
    predicted_classes = np.concatenate(tot_preds)
    target_classes = np.concatenate(tot_labels)
    losses = np.concatenate(tot_losses)
    probs_classes = np.concatenate(tot_probs)
    #np.save(output_dir / f"{cfg['split']}_probs.npy", probs_classes)
    #np.save(output_dir / f"{cfg['split']}_target.npy", target_classes)
    #np.save(output_dir / f"{cfg['split']}_preds.npy", predicted_classes)
    #np.save(output_dir / f"{cfg['split']}_losses_debug.npy", predicted_classes)

    # Concatenate all metadata
    regions = np.concatenate(tot_regions)
    fwis = np.concatenate(tot_fwi)
    #np.save(output_dir / f"{cfg['split']}_regions.npy", regions)
    #np.save(output_dir / f"{cfg['split']}_fwi.npy", fwis)

    # Compute metrics
    eval_metrics = get_classification_metrics(
        predicted=predicted_classes,
        labels=target_classes,
        n_classes=cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
        unk_masks=None,
    )

    micro_auc, macro_auc, class_auc = get_pr_auc_scores(
        scores=probs_classes,
        labels=target_classes,
        n_classes=cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
        output_dir=output_dir,
    )

    micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics["micro"]
    macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics["macro"]
    class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics["class"]

    metrics = {
        "loss": losses.mean(),
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

    with open(output_dir / f"{cfg['split']}_metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    if cfg["with_region"]:
        for region in np.unique(regions):
            idx = np.where(regions == region)

            region_preds = predicted_classes[idx]
            region_labels = target_classes[idx]
            region_probs = probs_classes[idx]

            region_metrics = get_classification_metrics(
                predicted=region_preds,
                labels=region_labels,
                n_classes=(
                    cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
                ),
                unk_masks=None,
            )
            region_micro_auc, region_macro_auc, region_class_auc = get_pr_auc_scores(
                scores=region_probs,
                labels=region_labels,
                n_classes=(
                    cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"]
                ),
            )

            region_micro_acc, region_micro_precision, region_micro_recall, region_micro_F1, region_micro_IOU = (
                region_metrics["micro"]
            )
            region_macro_acc, region_macro_precision, region_macro_recall, region_macro_F1, region_macro_IOU = (
                region_metrics["macro"]
            )
            region_class_acc, region_class_precision, region_class_recall, region_class_F1, region_class_IOU = (
                region_metrics["class"]
            )

            region_metrics = {
                "macro_Accuracy": region_macro_acc,
                "macro_Precision": region_macro_precision,
                "macro_Recall": region_macro_recall,
                "macro_F1": region_macro_F1,
                "macro_IOU": region_macro_IOU,
                "macro_AUC": region_macro_auc,
                "micro_Accuracy": region_micro_acc,
                "micro_Precision": region_micro_precision,
                "micro_Recall": region_micro_recall,
                "micro_F1": region_micro_F1,
                "micro_IOU": region_micro_IOU,
                "micro_AUC": region_micro_auc,
                "fire_Accuracy": region_class_acc[1],
                "fire_Precision": region_class_precision[1],
                "fire_Recall": region_class_recall[1],
                "fire_F1": region_class_F1[1],
                "fire_IOU": region_class_IOU[1],
                "fire_AUC": region_class_auc[1],
            }

            with open(output_dir / f"{cfg['split']}_{region}_metrics.txt", "w") as f:
                for key, value in region_metrics.items():
                    f.write(f"{key}: {value}\n")

    if model.model.useCBM and model.model.log_concepts:
        rows = []
        for region_name, corr_arr in concept_correct_region.items():
            fp_arr = concept_false_region[region_name]
            for cid in range(num_concepts):
                rows.append({
                    "concept_id": int(cid),
                    "name": concept_names[cid],
                    "region": region_name,
                    "correct_fire": int(corr_arr[cid]),
                    "false_fire": int(fp_arr[cid]),
                })

        df_region = pd.DataFrame(rows)
        df_region.to_csv(output_dir / f"{cfg['split']}_concept_usage_by_region.csv", index=False)

        # ---- Save CBM concept usage for later analysis / plotting ----
        if concept_correct is not None and concept_names is not None:
            concept_usage = {
                "concept_id": concept_ids,
                "name": np.array(concept_names),
                "correct_fire": concept_correct,
                "false_fire": concept_false,
            }
            if concept_cos is not None:
                concept_usage["best_cosine"] = concept_cos

            np.savez(output_dir / f"{cfg['split']}_concept_usage.npz", **concept_usage)

            import csv
            csv_path = output_dir / f"{cfg['split']}_concept_usage.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["concept_id", "name", "correct_fire", "false_fire"]
                if concept_cos is not None:
                    header.append("best_cosine_to_phrase")
                writer.writerow(header)
                for i in range(len(concept_ids)):
                    row = [
                        int(concept_ids[i]),
                        concept_names[i],
                        int(concept_correct[i]),
                        int(concept_false[i]),
                    ]
                    if concept_cos is not None:
                        row.append(float(concept_cos[i]))
                    writer.writerow(row)
            print(f"[CBM] Saved concept usage stats to {csv_path}.")

    print("Evaluation completed!")
    print(metrics)


def _update_concept_usage(concept_map, predicted, labels, unk_masks, stats, fire_cls_id: int):
        """
        concept_map: [B, C, H_p, W_p] concept activations (from SAE bottleneck)
        predicted : [B, H_out, W_out] (class indices or 0/1)
        labels    : [B, H_out, W_out] (ground-truth class indices)
        unk_masks : None or [B, H_out, W_out] (1=valid, 0=unknown)
        stats     : dict with "correct" and "false" (1D tensors length C)
        fire_cls_id: int, index of the 'fire' class
        """
        if concept_map is None:
            return

        device = concept_map.device
        concept_map = concept_map.to(device)

        # predicted → [B, H_out, W_out]
        if predicted.ndim == 4:
            pred_map = predicted.squeeze(-1)
        else:
            pred_map = predicted
        pred_map = pred_map.to(device)

        # labels → [B, H_out, W_out]
        lab = labels
        if lab.ndim == 4 and lab.shape[1] == 1:
            lab = lab[:, 0]
        elif lab.ndim == 2:
            lab = lab.unsqueeze(0)
        lab = lab.to(device)

        B, H_out, W_out = pred_map.shape
        _, C, H_p, W_p = concept_map.shape

        # map labels/preds/unk to patch grid so shapes match concept_map
        lab4 = lab.unsqueeze(1).float()          # [B,1,H_out,W_out]
        pred4 = pred_map.unsqueeze(1).float()    # [B,1,H_out,W_out]

        lab_patch = F.interpolate(lab4, size=(H_p, W_p), mode="nearest").long()[:, 0]   # [B,H_p,W_p]
        pred_patch = F.interpolate(pred4, size=(H_p, W_p), mode="nearest").long()[:, 0]

        if unk_masks is not None:
            unk = unk_masks
            if unk.ndim == 4 and unk.shape[1] == 1:
                unk = unk[:, 0]
            elif unk.ndim == 2:
                unk = unk.unsqueeze(0)
            unk4 = unk.unsqueeze(1).float()     # [B,1,H_out,W_out]
            unk_patch = F.interpolate(unk4, size=(H_p, W_p), mode="nearest")[:, 0] > 0.5
        else:
            unk_patch = torch.ones_like(lab_patch, dtype=torch.bool, device=device)
            print("Careful: Unk mask is None")

        valid_mask = unk_patch

        fire_cls_id = int(fire_cls_id).to(device)
        fire_pred_mask = (pred_patch == fire_cls_id) & valid_mask      # predicted fire & valid
        TP_mask = fire_pred_mask & (lab_patch == fire_cls_id)          # true positives
        FP_mask = fire_pred_mask & (lab_patch != fire_cls_id)          # false positives

        # concept is "used" at a patch if its activation > 0 at that patch
        act_pos = concept_map > 0                                       # [B,C,H_p,W_p]
        tp_mask_exp = TP_mask.unsqueeze(1)                              # [B,1,H_p,W_p]
        fp_mask_exp = FP_mask.unsqueeze(1)

        used_TP = act_pos & tp_mask_exp
        used_FP = act_pos & fp_mask_exp

        add_correct = used_TP.sum(dim=(0, 2, 3))                        # [C]
        add_false = used_FP.sum(dim=(0, 2, 3))                          # [C]

        if stats["correct"] is None:
            stats["correct"] = add_correct.to(dtype=torch.long)
            stats["false"] = add_false.to(dtype=torch.long)
        else:
            stats["correct"] += add_correct.to(dtype=stats["correct"].dtype)
            stats["false"] += add_false.to(dtype=stats["false"].dtype)

@torch.no_grad()
def _encode_phrases_msclip(phrases, model, tokenizer, batch_size, device):
    embs = []
    for i in range(0, len(phrases), batch_size):
        toks = tokenizer(phrases[i:i + batch_size]).to(device)
        z = model.inference_text(toks)  # [B, Dt]
        embs.append(F.normalize(z, dim=-1))
    return torch.cat(embs, dim=0)  # [N, Dt]


@torch.no_grad()
def _best_phrase_per_concept(sae_module, text_embs, phrases, device: str):
    """
    For each SAE dictionary atom, find the phrase with max cosine similarity.
    Returns:
        concept_names: list[str] of length C
        best_cos: np.ndarray [C] of cosine scores
    """
    D = sae_module.net.get_dictionary().to(device).float()  # [C, D]
    D = F.normalize(D, dim=1)
    T = F.normalize(text_embs.to(device).float(), dim=1)    # [N, D]

    sims = D @ T.t()  # [C, N_phrases]
    best_vals, best_idx = sims.max(dim=1)  # each concept → best phrase

    concept_names = [phrases[i] for i in best_idx.tolist()]
    best_cos = best_vals.detach().cpu().numpy()
    return concept_names, best_cos


if __name__ == "__main__":
    evaluate()