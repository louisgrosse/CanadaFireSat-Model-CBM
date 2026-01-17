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
import matplotlib.pyplot as plt
import textwrap
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
            model = ImgModule.load_from_checkpoint(cfg["model_path"],strict=False)
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
    concept_names_list = []
    concept_ids = None
    concept_cos = None
    concept_correct_region = None
    concept_cos_list = []

    base_model = None
    text_embs = None
    do_ablation = False


    if model.model.useCBM and model.model.log_concepts:
        print("loading sae config...")
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

            if align is not None and align.get("csv_phrases_path") and align.get("csv_names"):
                device_text = str(align.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
                msclip_model_name = align["msclip_model_name"]
                ckpt_path_msclip = align.get("msclip_ckpt", None)
                
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
                
                num_concepts = len(concept_names)
                concept_correct = np.zeros(num_concepts, dtype=np.int64)
                concept_correct_region = defaultdict(lambda: np.zeros(num_concepts, dtype=np.int64))
                concept_false_region   = defaultdict(lambda: np.zeros(num_concepts, dtype=np.int64))
                concept_false = np.zeros(num_concepts, dtype=np.int64)
                concept_ids = np.arange(num_concepts, dtype=np.int64)
                concept_energy_fire  = np.zeros(num_concepts, dtype=np.float64)
                concept_contrib_fire = np.zeros(num_concepts, dtype=np.float64)

                concept_contrib_fire_label = np.zeros(num_concepts, dtype=np.float64)
                concept_contrib_bg_label   = np.zeros(num_concepts, dtype=np.float64)

                concept_energy_fire_label = np.zeros(num_concepts, dtype=np.float64)
                concept_energy_bg_label   = np.zeros(num_concepts, dtype=np.float64)

                concept_count_fire_label = np.zeros(num_concepts, dtype=np.int64)
                concept_count_bg_label   = np.zeros(num_concepts, dtype=np.int64)



                do_ablation = bool(cfg.get("CBM_ABLATION", {}).get("enabled", False))
                abl_chunk = int(cfg.get("CBM_ABLATION", {}).get("chunk_size", 1024))

                if do_ablation:
                    # head weights: [K, C]
                    head_w = model.model.head.weight.squeeze(-1).squeeze(-1).detach()  # [num_classes, C]

                    if cfg["MODEL"]["num_classes"] == 1:
                        w_eff = head_w[0].to(device)   # [C]
                        thr = float(cfg["MODEL"]["threshold"])
                        logit_thr = math.log(thr / (1.0 - thr))
                    else:
                        # if you have 2 classes, use margin trick (fire vs bg) to avoid softmax recompute
                        fire_id = int(cfg["MODEL"].get("fire_class_id", 1))
                        bg_id = 1 - fire_id
                        w_eff = (head_w[fire_id] - head_w[bg_id]).to(device)  # [C]
                        logit_thr = 0.0  # margin > 0 == predict fire

                    # Per-concept confusion under ablation
                    C = num_concepts
                    
                    delta_TP = torch.zeros(num_concepts, device=device, dtype=torch.long)
                    delta_FP = torch.zeros(num_concepts, device=device, dtype=torch.long)
                    delta_FN = torch.zeros(num_concepts, device=device, dtype=torch.long)

                    # Baseline confusion (for delta metrics)
                    base_TP = torch.zeros((), device=device, dtype=torch.long)
                    base_FP = torch.zeros((), device=device, dtype=torch.long)
                    base_FN = torch.zeros((), device=device, dtype=torch.long)


                print(f"[CBM] Concept text labels built for {num_concepts} concepts.")


            
    remove_ids = cfg.get("CBM_ABLATION", {}).get("remove_ids", [])

    if isinstance(remove_ids, (int, np.integer)):
        remove_ids = [int(remove_ids)]
    elif isinstance(remove_ids, str):
        remove_ids = [int(remove_ids)]
    elif remove_ids is None:
        remove_ids = []

    focus_cid = cfg.get("CBM_ABLATION", {}).get("focus_id", [])
    tot_preds_focus, tot_labels_focus, tot_probs_focus = [], [], []
    n_focus_images = 0

    if remove_ids:
        C = model.model.head.in_channels
        gate = torch.ones(C, device=device)
        gate[torch.tensor(remove_ids, device=device)] = 0.0
        model.model.editing_vector = gate
    else:
        model.model.editing_vector = None

    print("focus_cid:", focus_cid)
    print("remove_ids:", remove_ids)
    if remove_ids:
        print("gate sum:", float(model.model.editing_vector.sum().item()), "C:", int(model.model.editing_vector.numel()))
        print("gate removed check:", [float(model.model.editing_vector[i].item()) for i in remove_ids])


    visu = 0
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

        focus_active = False


        if cfg["MODEL"]["num_classes"] == 1:
            probs = torch.nn.functional.sigmoid(logits)
            predicted = (probs > cfg["MODEL"]["threshold"]).to(torch.float32)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, predicted = torch.max(logits.data, -1)


        if (model.model.useCBM and model.model.log_concepts):
            concept_map = getattr(model.model, "last_concept_map", None)
            if concept_map is not None:
                # concept_map: [1, C, Hc, Wc]
                Bc, Cc, Hc, Wc = concept_map.shape

                concept_map_raw = getattr(model.model, "last_concept_map_raw", None)
                if focus_cid is not None and 0 <= focus_cid < concept_map_raw.shape[1]:
                    focus_active = bool((concept_map_raw[0, focus_cid] > 0).any().item())


                # Upsample concept map to logits size (out_H, out_W)
                H_out, W_out = logits.shape[1], logits.shape[2]
                concept_map_up = F.interpolate(
                    concept_map,
                    size=(H_out, W_out),
                    mode="bilinear",
                    align_corners=False,
                )[0]  # [C, H, W]

                # ---- CBM per-concept TP/FP counting ----
                if do_ablation and concept_map is not None:
                    H_out, W_out = logits.shape[1], logits.shape[2]

                    if cfg["MODEL"]["num_classes"] == 1:
                        base_field = logits[0, ..., 0]
                        fire_label = (ground_truth[0].to(device) > 0.5)
                        thr = float(cfg["MODEL"]["threshold"])
                        logit_thr = math.log(thr / (1.0 - thr))
                        base_pred = (base_field > logit_thr)
                    else:
                        fire_id = int(cfg["MODEL"].get("fire_class_id", 1))
                        bg_id = 1 - fire_id
                        base_field = logits[0, ..., fire_id] - logits[0, ..., bg_id]
                        fire_label = (ground_truth[0].to(device) == fire_id)
                        logit_thr = 0.0
                        base_pred = (base_field > logit_thr)

                    if unk_masks is not None:
                        valid = unk_masks[0].to(device).bool()
                    else:
                        valid = torch.ones_like(fire_label, dtype=torch.bool, device=device)

                    lab = fire_label & valid
                    pred0 = base_pred & valid

                    tp0 = (pred0 & lab).sum(dtype=torch.int64)
                    fp0 = (pred0 & (~lab)).sum(dtype=torch.int64)
                    fn0 = ((~pred0) & lab).sum(dtype=torch.int64)

                    base_TP += tp0
                    base_FP += fp0
                    base_FN += fn0

                    act = concept_map[0].abs().amax(dim=(1, 2))
                    active_ids = (act > 0).nonzero(as_tuple=False).squeeze(1)

                    if i == 0:
                        nz_per_patch = (concept_map[0] != 0).sum(dim=0).to(torch.int32)
                        print("logits", tuple(logits.shape))
                        print("concept_map", tuple(concept_map.shape))
                        print("nonzero per patch min/mean/max",
                        int(nz_per_patch.min().item()),
                        float(nz_per_patch.float().mean().item()),
                        int(nz_per_patch.max().item()))
                        print("active concepts in sample", int(active_ids.numel()))
                        print("w_eff abs mean", float(w_eff.abs().mean().item()), "max", float(w_eff.abs().max().item()))
                        if active_ids.numel() > 0:
                            cid = int(active_ids[0].item())
                            a1 = F.interpolate(concept_map[:, cid:cid+1], size=(H_out, W_out), mode="bilinear", align_corners=False)[0, 0]
                            delta1 = (w_eff[cid] * a1).abs()
                            pred1 = ((base_field - w_eff[cid] * a1) > logit_thr) & valid
                            print("example cid", cid,
                                "delta abs max", float(delta1.max().item()),
                                "delta abs mean", float(delta1.mean().item()),
                                "pixel flips", int((pred1 != pred0).sum().item()))

                    for s in range(0, active_ids.numel(), abl_chunk):
                        e = min(active_ids.numel(), s + abl_chunk)
                        ids = active_ids[s:e]

                        a = F.interpolate(concept_map[:, ids], size=(H_out, W_out), mode="bilinear", align_corners=False)[0]
                        w = w_eff[ids].view(-1, 1, 1)

                        ablated_field = base_field.unsqueeze(0) - w * a
                        pred = (ablated_field > logit_thr) & valid.unsqueeze(0)

                        tp1 = (pred & lab.unsqueeze(0)).flatten(1).sum(dim=1, dtype=torch.int64)
                        fp1 = (pred & (~lab).unsqueeze(0)).flatten(1).sum(dim=1, dtype=torch.int64)
                        fn1 = ((~pred) & lab.unsqueeze(0)).flatten(1).sum(dim=1, dtype=torch.int64)

                        delta_TP[ids] += (tp1 - tp0)
                        delta_FP[ids] += (fp1 - fp0)
                        delta_FN[ids] += (fn1 - fn0)

                    if i in (0, 10, 100):
                        print("baseline TP/FP/FN so far",
                            int(base_TP.item()), int(base_FP.item()), int(base_FN.item()))
                        print("delta sums so far",
                            int(delta_TP.sum().item()), int(delta_FP.sum().item()), int(delta_FN.sum().item()))

                
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

                # valid pixels (mask out unknowns if provided)
                if unk_masks is not None:
                    valid_mask = unk_masks[0].to(device).bool()
                    fire_pred_mask = fire_pred_mask & valid_mask
                    fire_label_mask = fire_label_mask & valid_mask
                    label_valid_mask = valid_mask
                else:
                    label_valid_mask = torch.ones_like(fire_label_mask, dtype=torch.bool, device=fire_label_mask.device)

                # --- NEW: label-based contributions (fire vs non-fire) ---
                # Masks on labels only (not conditioned on prediction)
                fire_label_only = fire_label_mask & label_valid_mask
                bg_label_only   = (~fire_label_mask) & label_valid_mask

                # Effective fire weight per concept (fire logit margin)
                head_w = model.model.head.weight.squeeze(-1).squeeze(-1)  # [num_classes, C]
                if cfg["MODEL"]["num_classes"] == 1:
                    w_eff = head_w[0]  # [C]
                else:
                    fire_id = int(cfg["MODEL"].get("fire_class_id", 1))
                    bg_id = 1 - fire_id
                    w_eff = head_w[fire_id] - head_w[bg_id]  # [C]

                Cc, H_out, W_out = concept_map_up.shape
                z_flat    = concept_map_up.view(Cc, -1)  # [C, H*W]
                fire_flat = fire_label_only.view(-1).float().to(z_flat.device)
                bg_flat   = bg_label_only.view(-1).float().to(z_flat.device)

                energy_fire_inc = (z_flat * fire_flat.unsqueeze(0)).sum(dim=1)  # [C]
                energy_bg_inc   = (z_flat * bg_flat.unsqueeze(0)).sum(dim=1)    # [C]

                concept_energy_fire_label += energy_fire_inc.detach().cpu().numpy().astype(np.float64)
                concept_energy_bg_label   += energy_bg_inc.detach().cpu().numpy().astype(np.float64)

                w_eff_dev = w_eff.to(z_flat.device).view(-1, 1)  # [C, 1]

                contrib_fire_inc = (w_eff_dev * z_flat * fire_flat.unsqueeze(0)).sum(dim=1)  # [C]
                contrib_bg_inc   = (w_eff_dev * z_flat * bg_flat.unsqueeze(0)).sum(dim=1)    # [C]

                concept_contrib_fire_label += contrib_fire_inc.detach().cpu().numpy().astype(np.float64)
                concept_contrib_bg_label   += contrib_bg_inc.detach().cpu().numpy().astype(np.float64)

                active_flat = (z_flat > 0).float()  # [C, H*W]

                count_fire_inc = (active_flat * fire_flat.unsqueeze(0)).sum(dim=1)  # [C]
                count_bg_inc   = (active_flat * bg_flat.unsqueeze(0)).sum(dim=1)    # [C]

                concept_count_fire_label += count_fire_inc.detach().cpu().numpy().astype(np.int64)
                concept_count_bg_label   += count_bg_inc.detach().cpu().numpy().astype(np.int64)

                # -------------------- Visualisation / report demo --------------------
                conceptVis = cfg.get("ConceptToVisualise", 0)
                if np.count_nonzero(concept_active[conceptVis].cpu())>200 and visu < 18 and cfg.get("visualise", False):
                    visu += 1
                    np.save(arr=sample["inputs"],file=f"/home/grosse/CanadaFireSat-Model-CBM/results/visualisation/visu{str(visu)}")
                    np.save(arr = concept_active[conceptVis].cpu(),file=f"/home/grosse/CanadaFireSat-Model-CBM/results/visualisation/visuLoc{str(visu)}")
                    np.save(arr = sample["labels"],file=f"/home/grosse/CanadaFireSat-Model-CBM/results/visualisation/visuGt{str(visu)}")
                    print("saved some files")
                if visu>=18:
                    print("Finished!!!!!!!!!!")
                    sys.exit(0)

                if cfg.get("visualise", False) and visu < int(cfg.get("n_plots", 3)) and False:
                    # pick concept id
                    conceptVis = cfg.get("ConceptToVisualise", None)
                    if conceptVis in (None, "", []):
                        conceptVis = cfg.get("CBM_ABLATION", {}).get("focus_id", None)
                        if conceptVis in (None, "", []):
                            conceptVis = remove_ids[0] if (isinstance(remove_ids, (list, tuple)) and len(remove_ids) > 0) else 0
                    conceptVis = int(conceptVis)
                
                    # selection criterion: RAW activation fraction over patches
                    cm_raw_all = getattr(model.model, "last_concept_map_raw", None)
                    if cm_raw_all is None:
                        cm_raw_all = concept_map  # fallback (already edited)
                    patch_frac = 0.0
                    if cm_raw_all is not None and cm_raw_all.ndim == 4 and 0 <= conceptVis < cm_raw_all.shape[1]:
                        patch_frac = float((cm_raw_all[0, conceptVis] > 0).float().mean().item())
                
                    min_frac = float(cfg.get("visualise_min_patch_frac", 0.30))
                    target_idx = cfg.get("visualise_idx", None)
                    try:
                        target_idx = int(target_idx) if target_idx is not None else None
                    except Exception:
                        target_idx = None
                
                    if (target_idx is None or i == target_idx) and patch_frac >= min_frac:
                        vis_dir = output_dir / "visualisation"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                
                        # helper: forward with a temporary editing gate (None == base)
                        def _forward_with_gate(gate_vec):
                            prev_gate = getattr(model.model, "editing_vector", None)
                            model.model.editing_vector = gate_vec
                            if model.model_type == "MSClipFacto":
                                lg = model(
                                    sample["inputs"].unsqueeze(0).to(device),
                                    sample["doy"].unsqueeze(0).to(device),
                                    torch.tensor([sample["seq_lengths"]]).to(device),
                                )
                            else:
                                lg = model(sample["inputs"].unsqueeze(0).to(device))
                            model.model.editing_vector = prev_gate
                            lg = lg.permute(0, 2, 3, 1)  # [1,H,W,K]
                            if cfg["MODEL"]["num_classes"] == 1:
                                pr = torch.sigmoid(lg)
                                fire_prob = pr[0, ..., 0]
                            else:
                                pr = torch.softmax(lg, dim=-1)
                                fire_id_ = int(cfg["MODEL"].get("fire_class_id", 1))
                                fire_prob = pr[0, ..., fire_id_]
                            return fire_prob.detach().cpu().numpy().astype(np.float32)
                
                        compare = bool(cfg.get("visualise_compare", True))
                        flip_factor = float(cfg.get("visualise_flip_factor", -1.0))
                
                        # gates are concept-length (not head.in_channels)
                        C_gate = int(getattr(model.model, "concept_dim", Cc))
                        base_gate = None
                        remove_gate = torch.ones(C_gate, device=device); remove_gate[conceptVis] = 0.0
                        flip_gate   = torch.ones(C_gate, device=device); flip_gate[conceptVis] = flip_factor
                
                        fire_prob_base = _forward_with_gate(base_gate)
                        fire_prob_remove = _forward_with_gate(remove_gate) if compare else None
                        fire_prob_flip = _forward_with_gate(flip_gate) if compare else None
                
                        # upsample RAW focus map to output size
                        def _upsample_focus(raw_hw: torch.Tensor):
                            t = raw_hw.float().unsqueeze(0).unsqueeze(0)
                            return F.interpolate(t, size=(H_out, W_out), mode="nearest")[0, 0].cpu().numpy().astype(np.float32)
                
                        raw_focus_up = _upsample_focus(cm_raw_all[0, conceptVis].detach())
                        w_focus = float(w_eff[conceptVis].detach().cpu().item())
                        contrib_base = (w_focus * raw_focus_up).astype(np.float32)
                        contrib_remove = (w_focus * raw_focus_up * 0.0).astype(np.float32)
                        contrib_flip = (w_focus * raw_focus_up * flip_factor).astype(np.float32)
                
                        # rank top concepts by Σ|w·z| over patches (RAW)
                        raw_flat = cm_raw_all[0].float().view(Cc, -1)
                        score = (w_eff.abs().to(raw_flat.device).view(-1, 1) * raw_flat.abs()).sum(dim=1)
                        topk = min(12, int(score.numel()))
                        _, top_ids = torch.topk(score, k=topk)
                        top_ids = top_ids.detach().cpu().tolist()
                        lines = []
                        for r, cid_ in enumerate(top_ids, start=1):
                            name_ = ""
                            try:
                                if concept_names_list is not None:
                                    name_ = str(concept_names_list[0][int(cid_)])
                            except Exception:
                                name_ = ""
                            lines.append(f"{r:02d}. {cid_}: {name_}")
                        top_text = "\n".join(lines)
                
                        # build RGB from last valid timestep (assumes B4,B3,B2 are indices 2,1,0)
                        def _rgb_from_inputs(x_seq: np.ndarray, seq_len: int, band_idxs=(2,1,0)):
                            t_last = max(int(seq_len) - 1, 0)
                            x = x_seq[t_last]  # [C,H,W]
                            C_in = x.shape[0]
                            idxs = [b for b in band_idxs if b < C_in]
                            if len(idxs) < 3:
                                idxs = list(range(min(3, C_in)))
                            rgb = x[idxs].transpose(1,2,0).astype(np.float32)
                            mn = np.percentile(rgb, 2, axis=(0,1), keepdims=True)
                            mx = np.percentile(rgb, 98, axis=(0,1), keepdims=True)
                            rgb = (rgb - mn) / (mx - mn + 1e-6)
                            return np.clip(rgb, 0, 1)
                
                        rgb = _rgb_from_inputs(sample["inputs"].cpu().numpy(), sample.get("seq_lengths", 1))
                
                        if cfg["MODEL"]["num_classes"] == 1:
                            gt_fire = (labels[0].cpu().numpy() > 0.5).astype(np.float32)
                        else:
                            fire_id_ = int(cfg["MODEL"].get("fire_class_id", 1))
                            gt_fire = (labels[0].cpu().numpy() == fire_id_).astype(np.float32)
                
                        # Save NPZ (for custom plotting / report)
                        stem = f"demo_idx{i}_cid{conceptVis}_visu{visu}"
                        npz_path = vis_dir / f"{stem}.npz"
                        np.savez_compressed(
                            npz_path,
                            idx=i,
                            concept_id=int(conceptVis),
                            patch_frac=float(patch_frac),
                            flip_factor=float(flip_factor),
                            inputs=sample["inputs"].cpu().numpy(),
                            doy=sample.get("doy", None).cpu().numpy() if isinstance(sample.get("doy", None), torch.Tensor) else sample.get("doy", None),
                            seq_len=int(sample.get("seq_lengths", 0)),
                            gt_fire=gt_fire,
                            fire_prob_base=fire_prob_base,
                            fire_prob_remove=(fire_prob_remove if fire_prob_remove is not None else fire_prob_base),
                            fire_prob_flip=(fire_prob_flip if fire_prob_flip is not None else fire_prob_base),
                            raw_concept_map=cm_raw_all[0].detach().cpu().to(torch.float16).numpy(),  # [C,Hc,Wc]
                            w_eff=w_eff.detach().cpu().to(torch.float16).numpy(),                    # [C]
                            focus_raw_up=raw_focus_up,
                            contrib_base=contrib_base,
                            contrib_remove=contrib_remove,
                            contrib_flip=contrib_flip,
                            top_ids=np.array(top_ids, dtype=np.int32),
                        )
                
                        # Save PNG overview
                        fig, axes = plt.subplots(2, 5, figsize=(cfg.get("size_plots", 6) * 5, cfg.get("size_plots", 6) * 2))
                        for ax in axes.ravel():
                            ax.axis("off")
                
                        axes[0,0].imshow(rgb); axes[0,0].set_title("RGB (last timestep)")
                        axes[0,1].imshow(gt_fire); axes[0,1].set_title("GT fire")
                        axes[0,2].imshow(fire_prob_base); axes[0,2].set_title("Fire prob (base)")
                
                        if fire_prob_remove is not None:
                            axes[0,3].imshow(fire_prob_remove); axes[0,3].set_title("Fire prob (remove)")
                            axes[1,3].imshow(fire_prob_remove - fire_prob_base); axes[1,3].set_title("Δ prob (remove-base)")
                        else:
                            axes[0,3].text(0.05, 0.5, "remove: not run", transform=axes[0,3].transAxes)
                            axes[1,3].text(0.05, 0.5, "Δ remove: n/a", transform=axes[1,3].transAxes)
                
                        if fire_prob_flip is not None:
                            axes[0,4].imshow(fire_prob_flip); axes[0,4].set_title("Fire prob (flip)")
                            axes[1,4].imshow(fire_prob_flip - fire_prob_base); axes[1,4].set_title("Δ prob (flip-base)")
                        else:
                            axes[0,4].text(0.05, 0.5, "flip: not run", transform=axes[0,4].transAxes)
                            axes[1,4].text(0.05, 0.5, "Δ flip: n/a", transform=axes[1,4].transAxes)
                
                        axes[1,0].imshow(raw_focus_up); axes[1,0].set_title(f"Concept {conceptVis} RAW\n(frac={patch_frac:.2f})")
                        axes[1,1].imshow(contrib_base); axes[1,1].set_title(f"Contribution (base)\nw={w_focus:+.3f}")
                        axes[1,2].text(0.01, 0.99, top_text, va="top", ha="left", transform=axes[1,2].transAxes, fontsize=10)
                        axes[1,2].set_title("Top concepts by Σ|w·z|")
                
                        fig.suptitle(f"{stem} | flip_factor={flip_factor:+.2f}")
                        png_path = vis_dir / f"{stem}.png"
                        fig.tight_layout()
                        fig.savefig(png_path, dpi=150)
                        plt.close(fig)
                
                        print(f"[VIS] Saved {png_path} and {npz_path}")
                        visu += 1
                else:
                    continue
                    #sys.exit(0)
                # -------------------------------------------------------------------

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

        if focus_cid is not None and focus_active:
            tot_preds_focus.append(preds)
            tot_labels_focus.append(labels)
            tot_probs_focus.append(probs)
            n_focus_images += 1

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

    if focus_cid is not None and len(tot_preds_focus) > 0:
        pred_focus = np.concatenate(tot_preds_focus)
        lab_focus  = np.concatenate(tot_labels_focus)

        focus_metrics = get_classification_metrics(
            predicted=pred_focus,
            labels=lab_focus,
            n_classes=cfg["MODEL"]["num_classes"] + 1 if cfg["MODEL"]["num_classes"] == 1 else cfg["MODEL"]["num_classes"],
            unk_masks=None,
        )
        _, _, _, _, _ = focus_metrics["micro"]
        f_class_acc, f_class_precision, f_class_recall, f_class_F1, f_class_IOU = focus_metrics["class"]

        metrics.update({
            "concept_subset_concept_id": int(focus_cid),
            "concept_subset_num_images": int(n_focus_images),
            "concept_subset_num_pixels": int(pred_focus.size),
            "concept_fire_Precision_subset": float(f_class_precision[1]),
            "concept_fire_Recall_subset": float(f_class_recall[1]),
            "concept_fire_F1_subset": float(f_class_F1[1]),
        })
    else:
        if focus_cid is not None:
            metrics.update({
                "concept_subset_concept_id": int(focus_cid),
                "concept_subset_num_images": int(n_focus_images),
                "concept_fire_F1_subset": float("nan"),
            })


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
                    **{dict_cols[j]: concept_names_list[j][cid] for j in range(n_dicts)},
                    "region": region_name,
                    "correct_fire": int(corr_arr[cid]),
                    "false_fire": int(fp_arr[cid]),
                })

        df_region = pd.DataFrame(rows)
        df_region.to_csv(output_dir / f"{cfg['split']}_concept_usage_by_region.csv", index=False)
        
        if do_ablation:
            eps = 1e-9
            # baseline
            abl_TP = base_TP + delta_TP
            abl_FP = base_FP + delta_FP
            abl_FN = base_FN + delta_FN

            base_iou = (base_TP.float() / (base_TP + base_FP + base_FN).float().clamp_min(1)).item()
            base_f1  = (2*base_TP.float() / (2*base_TP + base_FP + base_FN).float().clamp_min(1)).item()

            iou = (abl_TP.float() / (abl_TP + abl_FP + abl_FN).float().clamp_min(1)).detach().cpu().numpy()
            f1  = (2*abl_TP.float() / (2*abl_TP + abl_FP + abl_FN).float().clamp_min(1)).detach().cpu().numpy()


            out_rows = []
            for cid in range(num_concepts):
                out_rows.append({
                    "concept_id": int(cid),
                    **{dict_cols[j]: concept_names_list[j][cid] for j in range(len(concept_names_list))},
                    "abl_iou": float(iou[cid]),
                    "abl_f1": float(f1[cid]),
                    "delta_iou": float(iou[cid] - base_iou),
                    "delta_f1": float(f1[cid] - base_f1),
                    "abl_TP": int(abl_TP[cid].item()),
                    "abl_FP": int(abl_FP[cid].item()),
                    "abl_FN": int(abl_FN[cid].item()),
                })

            split = cfg.get("split", None)
            if split is None:
                split = cfg.get("SPLIT", "val")  # fallback if your config uses another key
            csv_path = output_dir / f"{split}_concept_ablation.csv"

            pd.DataFrame(out_rows).to_csv(csv_path, index=False)
            print(f"[CBM] Saved per-concept ablation to {csv_path}")


        # ---- Save CBM concept usage for later analysis / plotting ----
        if concept_correct is not None and concept_names is not None:
            concept_usage = {
                "concept_id": concept_ids,
                "names": np.array(concept_names),
                "correct_fire": concept_correct,
                "false_fire": concept_false,
            }
            if concept_cos_list:
                for j in range(n_dicts):
                    concept_usage[f"best_cosine_{dict_cols[j]}"] = concept_cos_list[j]


            #np.savez(output_dir / f"{cfg['split']}_concept_usage.npz", **concept_usage)

            csv_path = output_dir / f"{cfg['split']}_concept_usage.csv"

            rows = []
            for cid in range(num_concepts):
                r = {
                    "concept_id": int(cid),
                    "correct_fire": int(concept_correct[cid]),
                    "false_fire": int(concept_false[cid]),
                    "contrib_fire_label": float(concept_contrib_fire_label[cid]),
                    "contrib_bg_label":   float(concept_contrib_bg_label[cid]),
                    "energy_fire_label":  float(concept_energy_fire_label[cid]),
                    "energy_bg_label":    float(concept_energy_bg_label[cid]),
                    "count_fire_label":   int(concept_count_fire_label[cid]),
                    "count_bg_label":     int(concept_count_bg_label[cid]),
                }

                for j in range(n_dicts):
                    r[dict_cols[j]] = concept_names_list[j][cid]
                if concept_cos_list:
                    for j in range(n_dicts):
                        r[f"{dict_cols[j]}_best_cosine"] = float(concept_cos_list[j][cid])

                rows.append(r)

            pd.DataFrame(rows).to_csv(csv_path, index=False)
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