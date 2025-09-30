"""Script for downloading CanadaFireSat locally"""
import os
import re

import hydra
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from omegaconf import DictConfig
from tqdm import tqdm

from src.constants import CONFIG_PATH


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="download")
def download(cfg: DictConfig):
    """Download the main parquet files"""

    dataset_repo = "EPFL-ECEO/CanadaFireSat"
    if cfg.split is not None:
        filename_pattern = re.compile(rf"{cfg.split}-.*\.parquet")  # Customize your pattern
    else:
        filename_pattern = re.compile(r".*\.parquet")

    # List files in dataset repo
    all_files = list_repo_files(repo_id=dataset_repo, repo_type="dataset")

    if cfg.regions is not None:
        tot_files = []
        for region in cfg.regions:
            config_name = region
            matching_files = [
                f for f in all_files if f.startswith(f"{config_name}/") and filename_pattern.search(os.path.basename(f))
            ]
            tot_files.extend(matching_files)

    else:
        tot_files = [f for f in all_files if filename_pattern.search(os.path.basename(f))]

    os.makedirs(cfg.output_dir, exist_ok=True)

    local_files = []
    for file in tqdm(tot_files, total=len(tot_files)):
        local_path = hf_hub_download(repo_id=dataset_repo, filename=file, repo_type="dataset", local_dir=cfg.output_dir)
        local_files.append(local_path)
        print(f"Downloaded: {local_path}")

    return local_files


if __name__ == "__main__":
    download()
