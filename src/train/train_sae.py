from typing import List, Optional, Dict, Any

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import hydra
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from overcomplete.sae.archetypal_dictionary import RelaxedArchetypalDictionary
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from src.models.sae import plSAE
from src.utils.process_utils import save_activations_to_npy, save_labels_to_npy
import numpy as np
from src.constants import CONFIG_PATH
import sys
from src.data.hf_Canada.ssl4eos12_dataset import SSL4EOS12Dataset, collate_fn, S2L1C_MEAN, S2L1C_STD,S2L2A_MEAN, S2L2A_STD

sys.path.append('MS-CLIP')
from msclip.inference.utils import build_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train(cfg: DictConfig) -> Dict[Any, Any]:
    """Training Script for SAE training [with black-box model inference prior]

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    
    if not HydraConfig.initialized():
        HydraConfig.instance().clear()
        HydraConfig().set_config(cfg)
    hydra_run_dir = HydraConfig.get().run.dir
    #hydra_run_dir = cfg.paths.output_dir
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_logger  = WandbLogger(
                project=cfg["CHECKPOINT"]["wandb_project"],
                entity=cfg["CHECKPOINT"]["wandb_user"],
                name=cfg["CHECKPOINT"]["experiment_name"],
                save_dir="/home/louis/Code/CanadaFireSat-Model-CBM/wandb/sae",
                group=cfg["CHECKPOINT"]["group"]
            )

    try:
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        log.warning(f"Could not push config to WandB: {e}")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    
    #wandb_logger.log_hyperparams(cfg)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating SAE <{cfg.sae._target_}>")
    sae: plSAE = hydra.utils.instantiate(cfg.sae)

    if cfg["model"]["net"]["from_msclip"]:
    #If someday we want to train the sae directly after computing the npy files.
        log.info(f"Instantiating Backbone <{cfg.model.net._target_}>")
        net: nn.Module = hydra.utils.instantiate(cfg.model.net)

        sae.msclip_model = net
        sae.from_msclip = True

    align = cfg.get("ALIGN")
    device = str(align.get("device", "cuda"))

    model_, _, tokenizer = build_model(
            model_name=align["msclip_model_name"],
            pretrained=bool(align.get("pretrained", True)),
            ckpt_path=align.get("msclip_ckpt", None),
            device=device,
            channels=10,
        )
    msclip = model_.eval().to(device)

    if align is not None and align.get("enabled") and align.get("csv_phrases_path") and align.get("align_loss_coeff", 0) > 0:

        all_phrases = []
        csv_name = align["csv_cosLoss_path"]
        if csv_name in ("concept_countsCLIPDATASET.csv", "concept_counts50k_yake.csv"):
            df = pd.read_csv(str(align["csv_phrases_path"]) + str(csv_name), sep=';')
        else:
            df = pd.read_csv(str(align["csv_phrases_path"]) + str(csv_name))
        all_phrases.extend(df["concept"].astype(str).tolist())

        # Encode & normalize (reuses your helper)
        text_embs = _encode_phrases_msclip(
            phrases=all_phrases,
            model=msclip,
            tokenizer=tokenizer,
            batch_size=int(align.get("text_batch_size", 512)),
            device=device,
        )  # [N, D]

        sae.align_text_embs=text_embs.cpu()
        sae.align_loss_coeff = float(align["align_loss_coeff"])


    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=[wandb_logger])

    callbacks_cfg = cfg.get("callbacks") or []
    callbacks: List[Callback] = [hydra.utils.instantiate(cb) for cb in callbacks_cfg]

    log.info(f"Instantiating trainer <{cfg.trainer_sae._target_}>")
    trainer_sae: Trainer = hydra.utils.instantiate(cfg.trainer_sae, callbacks=callbacks, logger=[wandb_logger])

    datamodule.setup()
    
    act_datamodule = None
    if not cfg["model"]["net"]["from_msclip"]:
        act_datamodule = NpyActDataModule(batch_size=cfg.sae_batch_size, train_npy_path=cfg.datamodule["train_path"], val_npy_path=cfg.datamodule["train_path"], test_npy_path=cfg.datamodule["train_path"])

    if cfg["use_archetypal"]["enabled"]:
        if cfg["use_archetypal"]["uniform"]:
            print("Using unisform sampling to sample points")
            points = sample_points_uniform(dl = act_datamodule.train_dataloader(),n_points=16000)
        else:
            print("Using k-means to sample points")
            points = sample_points_kmeans(dl = act_datamodule.train_dataloader(),n_centers=16_000)
            
        archetypal_dict = RelaxedArchetypalDictionary(
            in_dimensions=cfg.sae.sae_kwargs["input_shape"],
            nb_concepts=cfg.sae.sae_kwargs["nb_concepts"],
            points=points,
            delta=1.0,
        )
        sae.net.dictionary = archetypal_dict
        if cfg.sae["bind_init"]:
            sae._initialize_encoder_from_decoder()

    # Training the SAE
    if not cfg["model"]["net"]["from_msclip"]:
        trainer_sae.fit(model=sae, datamodule=act_datamodule, ckpt_path=cfg.get("sae_ckpt_path"))
    else:
        trainer_sae.fit(model=sae, datamodule=datamodule)


    align = cfg.get("ALIGN")
    if align is not None and align.get("csv_phrases_path") and align["enabled"]:
        device = str(align.get("device", "cuda"))
        columns = ["dict", "mean", "std", "N", "max"]
        data = []
        dataNames = []

        for csv_name in tqdm(align["csv_names"]):
            if csv_name == "concept_countsCLIPDATASET.csv" or csv_name == "concept_counts50k_yake.csv":
                df = pd.read_csv(str(align["csv_phrases_path"])+str(csv_name),sep=';')
            else:
                df = pd.read_csv(str(align["csv_phrases_path"])+str(csv_name))
            phrases = df["concept"].astype(str).tolist()

            text_embs = _encode_phrases_msclip(
                phrases=phrases,
                model=msclip,
                tokenizer=tokenizer,
                batch_size=int(align.get("text_batch_size", 512)),
                device=device,
            )  # [N, Dt]

            mean_c, std_c, max_c, top_rows = _alignment_stats_and_topk(
                sae_module=sae,
                text_embs=text_embs,
                device=device,
                phrases=phrases,
                k=5,
                name = csv_name,
            )

            max_v = float(max_c)
            mean = float(mean_c)
            std  = float(std_c)
            data.append([csv_name, mean, std, len(phrases), max_v])
            print(f"[ALIGN] top1 cosine — mean={mean_c:.4f}, std={std_c:.4f} {csv_name}")

            wandb_logger.log_table(
                key=f"dictionnary_{csv_name}",
                columns=["rank", "concept", "best_sae_concept_id", "cosine"],
                data=top_rows,
            )

        wandb_logger.log_table(key="alignment", columns=columns, data=data)


    ckpt_dir = callbacks_cfg[0]["dirpath"]
    OmegaConf.save(cfg, os.path.join(ckpt_dir, "config.yaml"))

    # Test the SAE
    if cfg["datamodule"]["test_path"] is not None:
        trainer_sae.test(
            model=sae,
            datamodule=act_datamodule,
            ckpt_path="best"
        )


def sample_points_uniform(n_points=16000, dl =None, cap_tokens_per_batch=4096, device="cpu"):

    points = []
    with torch.no_grad():
        for batch in dl:

            x = batch["inputs"]  
            B, D, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, D)   # [B*H*W, D]    

            x = x.float()
            #x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, unbiased=False, keepdim=True) + 1e-6) 

            if x.shape[0] > cap_tokens_per_batch:
                idx = torch.randperm(x.shape[0])[:cap_tokens_per_batch]
                x = x[idx]

            points.append(x.cpu())
            if sum(p.shape[0] for p in points) >= n_points:
                break

    points = torch.cat(points, dim=0)
    if points.shape[0] > n_points:
        idx = torch.randperm(points.shape[0])[:n_points]
        points = points[idx]
    return points.to(device)

@torch.no_grad()
def sample_points_kmeans(dl=None, n_centers=16_000, cap_tokens_per_batch=8192):
    kmeans = MiniBatchKMeans(
        n_clusters=n_centers,
        batch_size=cap_tokens_per_batch,
        init="k-means++",
        n_init=1,
    )

    init_needed = n_centers
    init_chunks = []

    def prep_tokens(x):
        B, D, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, D).float()
        if x.shape[0] > cap_tokens_per_batch:
            idx = torch.randperm(x.shape[0])[:cap_tokens_per_batch]
            x = x[idx]
        return x.cpu().numpy()

    for batch in dl:
        x = batch["inputs"]
        arr = prep_tokens(x)
        if init_needed > 0:
            take = min(init_needed, arr.shape[0])
            init_chunks.append(arr[:take])
            init_needed -= take
            if init_needed == 0:
                init_buf = np.concatenate(init_chunks, axis=0)  # [n_centers, D]
                kmeans.partial_fit(init_buf)
                break
    else:
        raise ValueError(f"Only collected {n_centers - init_needed} samples < n_centers={n_centers}. "
                         f"Lower n_centers or raise cap_tokens_per_batch.")

    for batch in dl:
        x = batch["inputs"]
        arr = prep_tokens(x)
        kmeans.partial_fit(arr)

    centers = torch.from_numpy(kmeans.cluster_centers_).float()  # [n_centers, D]
    return centers

class NpyActivationDataset(Dataset):
    def __init__(self, npy_path):
        self.data = np.load(npy_path, mmap_mode='r')
        lbl_path = npy_path.replace("features.npy", "labels.npy")                  
        self.labels = np.load(lbl_path, mmap_mode='r') if os.path.exists(lbl_path) else None  

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx].copy())
        if self.labels is None:    
            return {"inputs": x} 
        y = torch.from_numpy(self.labels[idx].astype(np.uint8).copy())                        
        return {"inputs": x, "label": y}  

class NpyActDataModule(LightningDataModule):
    def __init__(self, batch_size: int, train_npy_path: str,
                 val_npy_path: Optional[str] = None, test_npy_path: Optional[str] = None):
        super().__init__()
        self.batch_size = batch_size
        self.train_npy_path = train_npy_path
        self.val_npy_path = val_npy_path
        self.test_npy_path = test_npy_path


    def train_dataloader(self, num_workers: int = 8):
        dataset = NpyActivationDataset(self.train_npy_path)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self, num_workers: int = 8):
        if self.val_npy_path is not None:
            dataset = NpyActivationDataset(self.val_npy_path)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        else:
            return []

    def test_dataloader(self, num_workers: int = 8):
        if self.test_npy_path is not None:
            dataset = NpyActivationDataset(self.test_npy_path)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        else:
            return []

@torch.no_grad()
def _encode_phrases_msclip(phrases, model, tokenizer, batch_size, device):
    embs = []
    for i in range(0, len(phrases), batch_size):
        toks = tokenizer(phrases[i:i+batch_size]).to(device)
        z = model.inference_text(toks)          # [B, Dt]
        embs.append(F.normalize(z, dim=-1))
    return torch.cat(embs, dim=0)               # [N, Dt]

@torch.no_grad()
def _alignment_stats_and_topk(sae_module, text_embs, device: str, phrases, k: int = 5,name:str = None):
    alive_mask = sae_module.val_alive_mask 
    print("=============> Alive ?", alive_mask.shape)

    D = sae_module.net.get_dictionary().to(device).float()
    D = F.normalize(D, dim=1)
    if alive_mask is not None:
        print("Keeping only live atoms!!!")
        print("before : ",D.shape)
        D = D[alive_mask.to(D.device)]
        print("after : ", D.shape)

    T = F.normalize(text_embs.to(device).float(), dim=1)  # [N, Dt]

    sims = D @ T.t()                              # [C, N] = concept x phrase cosine
    top1_concept_vals = sims.max(dim=1).values    # [C]
    mean_c = top1_concept_vals.mean().item()
    std_c  = top1_concept_vals.std(unbiased=False).item()
    max_c  = top1_concept_vals.max().item()

    phrase_best_vals, phrase_best_concepts = sims.max(dim=0)      # [N], [N]
    k = min(k, phrase_best_vals.numel())
    top_vals, top_phrase_idx = torch.topk(phrase_best_vals, k)    # [k], [k]

    top_rows = []
    for rank, (pidx, val) in enumerate(zip(top_phrase_idx.tolist(), top_vals.tolist()), start=1):
        top_rows.append([rank, phrases[pidx], int(phrase_best_concepts[pidx].item()), float(val)])

    return mean_c, std_c, max_c, top_rows


@hydra.main(version_base="1.2", config_path=str(CONFIG_PATH), config_name="sae_config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()