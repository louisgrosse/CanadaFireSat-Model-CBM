"""Evaluation Script for Validation, Test, and Test Hard"""
from pathlib import Path

import hydra
import numpy as np
import sys
import torch
import math
from pathlib import Path
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
            model = ImgModule.load_from_checkpoint(cfg["model_path"], strict=False)
            if cfg.get("log_concepts", False):
                model.model.log_concepts = True
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
        except (KeyError, RuntimeError):
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

    # ---- CBM / Concept stats init ----
    base_model = None
    sae_module = None
    concept_names_list = []
    concept_cos_list = []
    dict_cols = []
    n_dicts = 0
    num_concepts = None

    # Per-embedding stats containers
    emb_keys = ["post"]  # default
    concept_correct = {}
    concept_false = {}
    concept_correct_region = {}
    concept_false_region = {}
    concept_contrib_fire_label = {}
    concept_contrib_bg_label = {}
    concept_energy_fire_label = {}
    concept_energy_bg_label = {}
    concept_count_fire_label = {}
    concept_count_bg_label = {}

    # Weights per embedding (fire-vs-bg margin), length C each
    w_eff_by_emb = {}

    # Ablation
    do_ablation = False
    abl_chunk = 1024
    delta_TP = delta_FP = delta_FN = None
    base_TP = base_FP = base_FN = None
    logit_thr = 0.0

    is_pre = False  # sae_before_attention variant

    if getattr(model, "model", None) is not None and model.model.useCBM and getattr(model.model, "log_concepts", False):
        print("loading sae config...")
        base_model = model.model  # MSClipFactorizeModel
        sae_module = base_model.sae

        is_pre = bool(getattr(base_model, "sae_before_attention", False))
        emb_keys = ["last", "mean", "delta"] if is_pre else ["post"]

        # Build concept text labels (optional)
        sae_config_path = cfg.get("MODEL", {}).get("sae_config", None)
        align = None
        if sae_config_path is not None:
            try:
                cfg_sae = OmegaConf.load(sae_config_path)
                align = cfg_sae.get("ALIGN", None)
            except Exception as e:
                print(f"[CBM] Could not load SAE config from {sae_config_path}: {e}")

        if align is not None and align.get("csv_phrases_path") and align.get("csv_names"):
            device_text = str(align.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            model_clip = base_model.msclip_model
            tokenizer = base_model._tokenizer
            msclip = model_clip.eval().to(device_text)

            csv_names = align["csv_names"]
            csv_dir = str(align["csv_phrases_path"])

            dict_cols = [Path(str(n)).stem for n in csv_names]  # removes ".csv"
            n_dicts = len(dict_cols)

            print("loading dictionaries...")
            for csv_name in csv_names:
                csv_path = csv_dir + str(csv_name)
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

                concept_names_list.append(concept_names)
                concept_cos_list.append(concept_cos)
            print("loaded dictionaries")

            num_concepts = len(concept_names_list[0])
        else:
            # Fall back to concept count from model/SAE
            if hasattr(base_model, "concept_dim"):
                num_concepts = int(base_model.concept_dim)
            else:
                try:
                    num_concepts = int(sae_module.net.get_dictionary().shape[0])
                except Exception:
                    num_concepts = int(getattr(base_model.head, "in_channels", 0))

        # Init per-embedding stats arrays
        for k in emb_keys:
            concept_correct[k] = np.zeros(num_concepts, dtype=np.int64)
            concept_false[k] = np.zeros(num_concepts, dtype=np.int64)
            concept_correct_region[k] = defaultdict(lambda: np.zeros(num_concepts, dtype=np.int64))
            concept_false_region[k] = defaultdict(lambda: np.zeros(num_concepts, dtype=np.int64))
            concept_contrib_fire_label[k] = np.zeros(num_concepts, dtype=np.float64)
            concept_contrib_bg_label[k] = np.zeros(num_concepts, dtype=np.float64)
            concept_energy_fire_label[k] = np.zeros(num_concepts, dtype=np.float64)
            concept_energy_bg_label[k] = np.zeros(num_concepts, dtype=np.float64)
            concept_count_fire_label[k] = np.zeros(num_concepts, dtype=np.int64)
            concept_count_bg_label[k] = np.zeros(num_concepts, dtype=np.int64)

        # Prepare effective weights per embedding (for contribution stats & ablation)
        head_w = base_model.head.weight.squeeze(-1).squeeze(-1).detach()  # [K, Cin]
        if cfg["MODEL"]["num_classes"] == 1:
            w_margin = head_w[0].to(device)
            thr = float(cfg["MODEL"]["threshold"])
            logit_thr = math.log(thr / (1.0 - thr))
        else:
            fire_id = int(cfg["MODEL"].get("fire_class_id", 1))
            bg_id = 1 - fire_id
            w_margin = (head_w[fire_id] - head_w[bg_id]).to(device)
            logit_thr = 0.0

        if is_pre:
            # Cin = 3 * Cp where Cp = C + (doy extra)
            Cp = int(w_margin.numel() // 3)
            C = int(num_concepts)
            w_eff_by_emb["last"]  = w_margin[0:C]
            w_eff_by_emb["mean"]  = w_margin[Cp:Cp + C]
            w_eff_by_emb["delta"] = w_margin[2 * Cp:2 * Cp + C]
        else:
            w_eff_by_emb["post"] = w_margin[: int(num_concepts)]

        # Ablation config
        do_ablation = bool(cfg.get("CBM_ABLATION", {}).get("enabled", False))
        abl_chunk = int(cfg.get("CBM_ABLATION", {}).get("chunk_size", 1024))
        if do_ablation:
            delta_TP = torch.zeros(num_concepts, device=device, dtype=torch.long)
            delta_FP = torch.zeros(num_concepts, device=device, dtype=torch.long)
            delta_FN = torch.zeros(num_concepts, device=device, dtype=torch.long)
            base_TP = torch.zeros((), device=device, dtype=torch.long)
            base_FP = torch.zeros((), device=device, dtype=torch.long)
            base_FN = torch.zeros((), device=device, dtype=torch.long)

        print(f"[CBM] Ready for concept logging with {num_concepts} concepts; mode={'pre' if is_pre else 'post'}.")

    # --- Apply CBM ablation/enhancement gate (editing_vector) ---
    # remove_ids: list of concept ids to zero out
    # enhance_ids: list of concept ids to scale by enhance_factor (can be negative)
    remove_ids = cfg.get("CBM_ABLATION", {}).get("remove_ids", [])
    if isinstance(remove_ids, (int, np.integer)):
        remove_ids = [int(remove_ids)]
    elif isinstance(remove_ids, str):
        remove_ids = [int(remove_ids)]
    elif remove_ids is None:
        remove_ids = []

    enhance_ids = cfg.get("CBM_ABLATION", {}).get("enhance_ids", cfg.get("CBM_ABLATION", {}).get("enhance_id", []))
    if isinstance(enhance_ids, (int, np.integer)):
        enhance_ids = [int(enhance_ids)]
    elif isinstance(enhance_ids, str):
        enhance_ids = [int(enhance_ids)]
    elif enhance_ids is None:
        enhance_ids = []

    enhance_factor = None
    if enhance_ids:
        enhance_factor = float(cfg.get("CBM_ABLATION", {}).get("enhance_factor", 2.0))

    focus_cid = cfg.get("CBM_ABLATION", {}).get("focus_id", None)
    tot_preds_focus, tot_labels_focus, tot_probs_focus = [], [], []
    n_focus_images = 0

    print("focus_cid:", focus_cid)
    print("remove_ids:", remove_ids)
    print("enhance_ids:", enhance_ids, "enhance_factor:", enhance_factor)

    if getattr(model, "model", None) is not None and model.model.useCBM and (remove_ids or enhance_ids):
        # editing_vector is defined over concepts only (length = C), independent of head.in_channels
        C_gate = int(getattr(model.model, "concept_dim", num_concepts if num_concepts is not None else 0))
        if C_gate <= 0:
            raise ValueError("Could not infer concept dimension for editing_vector (C_gate).")

        gate = torch.ones(C_gate, device=device)

        # Removal first (wins over enhancement)
        for cid in remove_ids:
            if 0 <= int(cid) < C_gate:
                gate[int(cid)] = 0.0

        # Enhancement (scales activation; can be negative)
        for cid in enhance_ids:
            if 0 <= int(cid) < C_gate and gate[int(cid)] != 0.0:
                gate[int(cid)] = gate[int(cid)] * float(enhance_factor)

        model.model.editing_vector = gate.detach()

        print("[CBM] gate sum:", float(model.model.editing_vector.sum().item()), "C:", int(model.model.editing_vector.numel()))
        if remove_ids:
            print("[CBM] gate removed check:", {int(i): float(model.model.editing_vector[int(i)].item()) for i in remove_ids if 0 <= int(i) < C_gate})
        if enhance_ids:
            print("[CBM] gate enhanced check:", {int(i): float(model.model.editing_vector[int(i)].item()) for i in enhance_ids if 0 <= int(i) < C_gate})

    visu = 0

    # Evaluation loop
    for i in tqdm(range(len(dataset)), desc=f"Inferring on split {cfg['split']}", total=len(dataset)):
        data = dataset[i]

        with torch.no_grad():
            if cfg["mode"] == "image":
                sample = data[0]
                img_name_info = data[1]
                if model.model_type == 'MSClipFacto':
                    logits = model(
                        sample["inputs"].unsqueeze(0).to(device),
                        sample["doy"].unsqueeze(0).to(device),
                        torch.tensor([sample["seq_lengths"]]).to(device)
                    )
                else:
                    logits = model(sample["inputs"].unsqueeze(0).to(device))
            else:
                # tabular cases unchanged
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

        # probs/preds
        if cfg["MODEL"]["num_classes"] == 1:
            probs = torch.nn.functional.sigmoid(logits)
            predicted = (probs > cfg["MODEL"]["threshold"]).to(torch.float32)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, predicted = torch.max(logits.data, -1)

        # ---- Concept logging per embedding (post-SAE or pre-SAE summaries) ----
        if getattr(model, "model", None) is not None and model.model.useCBM and getattr(model.model, "log_concepts", False):
            # Gather concept maps
            concept_maps = {}
            concept_maps_raw = {}

            if bool(getattr(model.model, "sae_before_attention", False)):
                concept_maps["last"]  = getattr(model.model, "last_concept_map_last", None)
                concept_maps["mean"]  = getattr(model.model, "last_concept_map_mean", None)
                concept_maps["delta"] = getattr(model.model, "last_concept_map_delta", None)

                concept_maps_raw["last"]  = getattr(model.model, "last_concept_map_last_raw", None)
                concept_maps_raw["mean"]  = getattr(model.model, "last_concept_map_mean_raw", None)
                concept_maps_raw["delta"] = getattr(model.model, "last_concept_map_delta_raw", None)
            else:
                concept_maps["post"] = getattr(model.model, "last_concept_map", None)
                concept_maps_raw["post"] = getattr(model.model, "last_concept_map_raw", None)

            focus_mask = None
            if focus_cid is not None:
                cid = int(focus_cid)

                raw_maps = []
                for cmr in concept_maps_raw.values():
                    if cmr is not None and cmr.ndim == 4 and 0 <= cid < cmr.shape[1]:
                        raw_maps.append(cmr[0, cid])  # [Hc, Wc] at concept-map resolution

                if len(raw_maps) > 0:
                    focus_patch_mask = torch.stack(raw_maps, dim=0).amax(dim=0) > 0  # [Hc, Wc] bool

                    ph, pw = predicted.shape[-2], predicted.shape[-1]
                    if focus_patch_mask.shape != (ph, pw):
                        focus_mask = F.interpolate(
                            focus_patch_mask[None, None].float(),
                            size=(ph, pw),
                            mode="nearest",
                        )[0, 0].bool()
                    else:
                        focus_mask = focus_patch_mask


            # Fire masks + valid patches
            if cfg["MODEL"]["num_classes"] == 1:
                fire_pred_mask = (probs[0, ..., 0] > cfg["MODEL"]["threshold"])
                fire_label_mask = (ground_truth[0].to(device) > 0.5)
                base_field = logits[0, ..., 0]
            else:
                fire_class_id = int(cfg["MODEL"].get("fire_class_id", 1))
                fire_pred_mask = (predicted[0] == fire_class_id)
                fire_label_mask = (ground_truth[0].to(device) == fire_class_id)
                bg_id = 1 - fire_class_id
                base_field = logits[0, ..., fire_class_id] - logits[0, ..., bg_id]  # margin

            if unk_masks is not None:
                valid_mask = unk_masks[0].to(device).bool()
            else:
                valid_mask = torch.ones_like(fire_label_mask, dtype=torch.bool, device=device)

            if focus_mask is not None:
                focus_mask = focus_mask & valid_mask

            fire_pred_mask = fire_pred_mask & valid_mask
            fire_label_mask = fire_label_mask & valid_mask

            fire_label_only = fire_label_mask
            bg_label_only = (~fire_label_mask) & valid_mask

            # For ablation baseline confusion
            if do_ablation:
                lab = fire_label_only
                pred0 = (base_field > logit_thr) & valid_mask
                tp0 = (pred0 & lab).sum(dtype=torch.int64)
                fp0 = (pred0 & (~lab)).sum(dtype=torch.int64)
                fn0 = ((~pred0) & lab).sum(dtype=torch.int64)
                base_TP += tp0
                base_FP += fp0
                base_FN += fn0

            # Upsample and accumulate stats for each embedding
            H_out, W_out = logits.shape[1], logits.shape[2]
            region_name = img_name_info["region"]

            # Prepare for ablation: choose active concepts based on any embedding's activity
            if do_ablation:
                act_list = []
                for k in emb_keys:
                    cm = concept_maps.get(k, None)
                    if cm is not None:
                        act_list.append(cm[0].abs().amax(dim=(1, 2)))
                if len(act_list) > 0:
                    act = torch.stack(act_list, dim=0).max(dim=0).values
                    active_ids = (act > 0).nonzero(as_tuple=False).squeeze(1)
                else:
                    active_ids = torch.empty((0,), device=device, dtype=torch.long)

            # Compute per-embedding usage/contrib stats
            for k in emb_keys:
                cm = concept_maps.get(k, None)
                if cm is None:
                    continue

                # cm: [1, C, Hc, Wc]
                concept_map_up = F.interpolate(cm, size=(H_out, W_out), mode="bilinear", align_corners=False)[0]  # [C,H,W]

                # TP/FP usage (concept active => activation > 0)
                concept_active = (concept_map_up > 0)
                tp_mask = fire_pred_mask & fire_label_mask
                fp_mask = fire_pred_mask & (~fire_label_mask)

                tp_flat = tp_mask.view(-1)
                fp_flat = fp_mask.view(-1)
                concept_active_flat = concept_active.view(concept_active.shape[0], -1)

                tp_counts = (concept_active_flat & tp_flat.unsqueeze(0)).sum(dim=1).cpu().numpy()
                fp_counts = (concept_active_flat & fp_flat.unsqueeze(0)).sum(dim=1).cpu().numpy()

                concept_correct_region[k][region_name] += tp_counts.astype(np.int64)
                concept_false_region[k][region_name] += fp_counts.astype(np.int64)
                concept_correct[k] += tp_counts.astype(np.int64)
                concept_false[k] += fp_counts.astype(np.int64)

                # Label-based energy / contribution / counts
                w_eff = w_eff_by_emb[k].to(concept_map_up.device)  # [C]
                Cc = concept_map_up.shape[0]
                z_flat = concept_map_up.view(Cc, -1)
                fire_flat = fire_label_only.view(-1).float().to(z_flat.device)
                bg_flat = bg_label_only.view(-1).float().to(z_flat.device)

                energy_fire_inc = (z_flat * fire_flat.unsqueeze(0)).sum(dim=1)
                energy_bg_inc = (z_flat * bg_flat.unsqueeze(0)).sum(dim=1)

                concept_energy_fire_label[k] += energy_fire_inc.detach().cpu().numpy().astype(np.float64)
                concept_energy_bg_label[k] += energy_bg_inc.detach().cpu().numpy().astype(np.float64)

                w_eff_dev = w_eff.view(-1, 1)
                contrib_fire_inc = (w_eff_dev * z_flat * fire_flat.unsqueeze(0)).sum(dim=1)
                contrib_bg_inc = (w_eff_dev * z_flat * bg_flat.unsqueeze(0)).sum(dim=1)

                concept_contrib_fire_label[k] += contrib_fire_inc.detach().cpu().numpy().astype(np.float64)
                concept_contrib_bg_label[k] += contrib_bg_inc.detach().cpu().numpy().astype(np.float64)

                active_flat = (z_flat > 0).float()
                count_fire_inc = (active_flat * fire_flat.unsqueeze(0)).sum(dim=1)
                count_bg_inc = (active_flat * bg_flat.unsqueeze(0)).sum(dim=1)

                concept_count_fire_label[k] += count_fire_inc.detach().cpu().numpy().astype(np.int64)
                concept_count_bg_label[k] += count_bg_inc.detach().cpu().numpy().astype(np.int64)

            # Per-concept ablation deltas (single CSV, uses correct weights for pre/post)
            if do_ablation and active_ids.numel() > 0:
                if not bool(getattr(model.model, "sae_before_attention", False)):
                    # POST-SAE: use the single concept map
                    cm_post = concept_maps["post"]
                    for s in range(0, active_ids.numel(), abl_chunk):
                        e = min(active_ids.numel(), s + abl_chunk)
                        ids = active_ids[s:e]
                        a = F.interpolate(cm_post[:, ids], size=(H_out, W_out), mode="bilinear", align_corners=False)[0]  # [chunk,H,W]
                        w = w_eff_by_emb["post"][ids].view(-1, 1, 1).to(a.device)
                        ablated_field = base_field.unsqueeze(0) - w * a
                        pred = (ablated_field > logit_thr) & valid_mask.unsqueeze(0)
                        lab = fire_label_only
                        tp1 = (pred & lab.unsqueeze(0)).flatten(1).sum(dim=1, dtype=torch.int64)
                        fp1 = (pred & (~lab).unsqueeze(0)).flatten(1).sum(dim=1, dtype=torch.int64)
                        fn1 = ((~pred) & lab.unsqueeze(0)).flatten(1).sum(dim=1, dtype=torch.int64)
                        delta_TP[ids] += (tp1 - tp0)
                        delta_FP[ids] += (fp1 - fp0)
                        delta_FN[ids] += (fn1 - fn0)
                else:
                    # PRE-SAE summaries: combine last/mean/delta contributions
                    cm_last = concept_maps["last"]
                    cm_mean = concept_maps["mean"]
                    cm_delta = concept_maps["delta"]
                    w_last = w_eff_by_emb["last"]
                    w_mean = w_eff_by_emb["mean"]
                    w_delta = w_eff_by_emb["delta"]

                    for s in range(0, active_ids.numel(), abl_chunk):
                        e = min(active_ids.numel(), s + abl_chunk)
                        ids = active_ids[s:e]

                        a_last = F.interpolate(cm_last[:, ids], size=(H_out, W_out), mode="bilinear", align_corners=False)[0]
                        a_mean = F.interpolate(cm_mean[:, ids], size=(H_out, W_out), mode="bilinear", align_corners=False)[0]
                        a_del  = F.interpolate(cm_delta[:, ids], size=(H_out, W_out), mode="bilinear", align_corners=False)[0]

                        wl = w_last[ids].view(-1, 1, 1).to(a_last.device)
                        wm = w_mean[ids].view(-1, 1, 1).to(a_last.device)
                        wd = w_delta[ids].view(-1, 1, 1).to(a_last.device)

                        delta_field = wl * a_last + wm * a_mean + wd * a_del
                        ablated_field = base_field.unsqueeze(0) - delta_field

                        pred = (ablated_field > logit_thr) & valid_mask.unsqueeze(0)
                        lab = fire_label_only
                        tp1 = (pred & lab.unsqueeze(0)).flatten(1).sum(dim=1, dtype=torch.int64)
                        fp1 = (pred & (~lab).unsqueeze(0)).flatten(1).sum(dim=1, dtype=torch.int64)
                        fn1 = ((~pred) & lab.unsqueeze(0)).flatten(1).sum(dim=1, dtype=torch.int64)

                        delta_TP[ids] += (tp1 - tp0)
                        delta_FP[ids] += (fp1 - fp0)
                        delta_FN[ids] += (fn1 - fn0)

        # ---- Loss & metric buffers ----
        loss = model.loss_fn["mean"](
            logits.reshape(-1, cfg["MODEL"]["num_classes"]), ground_truth[0].to(device).reshape(-1).long()
        )

        if unk_masks is not None:
            preds = predicted.view(-1)[unk_masks.view(-1)].cpu().numpy()
            probs_flat = probs.view(-1, cfg["MODEL"]["num_classes"])[unk_masks.view(-1)].cpu().numpy()
            labels_flat = labels.view(-1)[unk_masks.view(-1)].cpu().numpy()
        else:
            preds = predicted.view(-1).cpu().numpy()
            probs_flat = probs.view(-1, cfg["MODEL"]["num_classes"]).cpu().numpy()
            labels_flat = labels.view(-1).cpu().numpy()

        if focus_mask is not None:
            sel = focus_mask.view(-1)

            if unk_masks is not None:
                sel = sel[unk_masks.view(-1).to(device)]

            sel = sel.detach().cpu().numpy().astype(bool)

            if sel.any():
                tot_preds_focus.append(preds[sel])
                tot_labels_focus.append(labels_flat[sel])
                tot_probs_focus.append(probs_flat[sel])
                n_focus_images += 1


        loss_np = loss.view(-1).cpu().detach().numpy()

        tot_preds.append(preds)
        tot_labels.append(labels_flat)
        tot_losses.append(loss_np)
        tot_probs.append(probs_flat)

        region = [img_name_info["region"]] * len(preds)
        fwi = [img_name_info["fwinx_mean"]] * len(preds)
        tot_regions.append(region)
        tot_fwi.append(fwi)

    # Concatenate all predictions
    predicted_classes = np.concatenate(tot_preds)
    target_classes = np.concatenate(tot_labels)
    losses = np.concatenate(tot_losses)
    probs_classes = np.concatenate(tot_probs)

    # Concatenate all metadata
    regions = np.concatenate(tot_regions)
    fwis = np.concatenate(tot_fwi)

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

    if focus_cid is not None and len(tot_preds_focus) > 0:
        pred_focus = np.concatenate(tot_preds_focus)
        lab_focus = np.concatenate(tot_labels_focus)

        focus_metrics = get_classification_metrics(
            predicted=pred_focus,
            labels=lab_focus,
            n_classes=cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
            unk_masks=None,
        )
        f_class_acc, f_class_precision, f_class_recall, f_class_F1, f_class_IOU = focus_metrics["class"]

        metrics.update({
            "concept_subset_concept_id": int(focus_cid),
            "concept_subset_num_images": int(n_focus_images),
            "concept_subset_num_patches": int(pred_focus.size),
            "concept_fire_Precision_subset": float(f_class_precision[1]),
            "concept_fire_Recall_subset": float(f_class_recall[1]),
            "concept_fire_F1_subset": float(f_class_F1[1]),
        })
    elif focus_cid is not None:
        metrics.update({
            "concept_subset_concept_id": int(focus_cid),
            "concept_subset_num_images": int(n_focus_images),
            "concept_fire_F1_subset": float("nan"),
        })

    with open(output_dir / f"{cfg['split']}_metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    # Region metrics unchanged (reuse original behavior)
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

            region_micro_acc, region_micro_precision, region_micro_recall, region_micro_F1, region_micro_IOU = region_metrics["micro"]
            region_macro_acc, region_macro_precision, region_macro_recall, region_macro_F1, region_macro_IOU = region_metrics["macro"]
            region_class_acc, region_class_precision, region_class_recall, region_class_F1, region_class_IOU = region_metrics["class"]

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

    # ---- Save concept CSVs (post: single; pre: three embeddings) ----
    if base_model is not None and base_model.useCBM and getattr(base_model, "log_concepts", False):
        # Usage-by-region
        for k in emb_keys:
            rows = []
            for region_name, corr_arr in concept_correct_region[k].items():
                fp_arr = concept_false_region[k][region_name]
                for cid in range(num_concepts):
                    row = {
                        "concept_id": int(cid),
                        "region": region_name,
                        "correct_fire": int(corr_arr[cid]),
                        "false_fire": int(fp_arr[cid]),
                    }
                    for j in range(n_dicts):
                        row[dict_cols[j]] = concept_names_list[j][cid] if concept_names_list else ""
                        if concept_cos_list:
                            row[f"{dict_cols[j]}_best_cosine"] = float(concept_cos_list[j][cid])
                    rows.append(row)

            df_region = pd.DataFrame(rows)
            if is_pre:
                df_region.to_csv(output_dir / f"{cfg['split']}_concept_usage_by_region_{k}.csv", index=False)
            else:
                df_region.to_csv(output_dir / f"{cfg['split']}_concept_usage_by_region.csv", index=False)

        # Concept usage main CSV (per embedding)
        for k in emb_keys:
            rows = []
            for cid in range(num_concepts):
                r = {
                    "concept_id": int(cid),
                    "correct_fire": int(concept_correct[k][cid]),
                    "false_fire": int(concept_false[k][cid]),
                    "contrib_fire_label": float(concept_contrib_fire_label[k][cid]),
                    "contrib_bg_label": float(concept_contrib_bg_label[k][cid]),
                    "energy_fire_label": float(concept_energy_fire_label[k][cid]),
                    "energy_bg_label": float(concept_energy_bg_label[k][cid]),
                    "count_fire_label": int(concept_count_fire_label[k][cid]),
                    "count_bg_label": int(concept_count_bg_label[k][cid]),
                }
                for j in range(n_dicts):
                    r[dict_cols[j]] = concept_names_list[j][cid] if concept_names_list else ""
                    if concept_cos_list:
                        r[f"{dict_cols[j]}_best_cosine"] = float(concept_cos_list[j][cid])
                rows.append(r)

            if is_pre:
                csv_path = output_dir / f"{cfg['split']}_concept_usage_{k}.csv"
            else:
                csv_path = output_dir / f"{cfg['split']}_concept_usage.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"[CBM] Saved concept usage stats to {csv_path}.")

        # Ablation CSV (single)
        if do_ablation:
            eps = 1e-9
            abl_TP = base_TP + delta_TP
            abl_FP = base_FP + delta_FP
            abl_FN = base_FN + delta_FN

            base_iou = (base_TP.float() / (base_TP + base_FP + base_FN).float().clamp_min(1)).item()
            base_f1 = (2 * base_TP.float() / (2 * base_TP + base_FP + base_FN).float().clamp_min(1)).item()

            iou = (abl_TP.float() / (abl_TP + abl_FP + abl_FN).float().clamp_min(1)).detach().cpu().numpy()
            f1 = (2 * abl_TP.float() / (2 * abl_TP + abl_FP + abl_FN).float().clamp_min(1)).detach().cpu().numpy()

            out_rows = []
            for cid in range(num_concepts):
                row = {
                    "concept_id": int(cid),
                    "abl_iou": float(iou[cid]),
                    "abl_f1": float(f1[cid]),
                    "delta_iou": float(iou[cid] - base_iou),
                    "delta_f1": float(f1[cid] - base_f1),
                    "abl_TP": int(abl_TP[cid].item()),
                    "abl_FP": int(abl_FP[cid].item()),
                    "abl_FN": int(abl_FN[cid].item()),
                }
                for j in range(n_dicts):
                    row[dict_cols[j]] = concept_names_list[j][cid] if concept_names_list else ""
                out_rows.append(row)

            split = cfg.get("split", cfg.get("SPLIT", "val"))
            csv_path = output_dir / f"{split}_concept_ablation.csv"
            pd.DataFrame(out_rows).to_csv(csv_path, index=False)
            print(f"[CBM] Saved per-concept ablation to {csv_path}")

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