import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from datetime import datetime
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import random
import pickle

# =====================================================
# CONFIG
# =====================================================
L1C_DIR = "/home/louis/Code/wildfire-forecast/worldstrat/l1c"
L2A_DIR = "/home/louis/Code/wildfire-forecast/worldstrat/l2a"
OUT_ROOT = "/home/louis/Code/wildfire-forecast/worldstrat/"

RES_10 = (264, 264)
RES_20 = (132, 132)
RES_60 = (44, 44)
BATCH_SIZE = 1000  # write to disk every 1000 pairs

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

os.makedirs(os.path.join(OUT_ROOT, "data"), exist_ok=True)


def serialize_array(arr):
    """Serialize a numpy array safely for Parquet."""
    return pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)


def read_multiband(path, out_res=(264, 264)):
    with rasterio.open(path) as src:
        data = src.read(
            out_shape=(src.count, out_res[0], out_res[1]),
            resampling=Resampling.bilinear
        ).astype(np.float32)
        lon, lat = np.meshgrid(
            np.linspace(src.bounds.left, src.bounds.right, out_res[1]),
            np.linspace(src.bounds.top, src.bounds.bottom, out_res[0])
        )
        loc = np.stack([lat, lon], axis=0).astype(np.float32)
    return data, loc


def split_s2_bands(full_img):
    n_bands = full_img.shape[0]
    idx_10 = [1, 2, 3, 7]
    idx_20 = [4, 5, 6, 8, 11 if n_bands == 13 else 10, 12 if n_bands == 13 else 11]
    idx_60 = [0, 9] if n_bands == 12 else [0, 9, 10]
    return full_img[idx_10], full_img[idx_20], full_img[idx_60]


def parse_date_from_filename(fname):
    parts = os.path.basename(fname).split("_")
    for p in parts:
        if p.startswith("20") and "T" in p:
            try:
                return datetime.strptime(p.split("T")[0], "%Y%m%d")
            except ValueError:
                continue
    return None


def assign_split():
    r = random.random()
    if r < TRAIN_RATIO:
        return "data"
    elif r < TRAIN_RATIO + VAL_RATIO:
        return "data"
    else:
        return "data"


def create_parquet():
    pattern = os.path.join(L1C_DIR, "**/*-L1C_data.tiff")
    l1c_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(l1c_files)} L1C scenes")

    batch_records = {"train": [], "val": [], "test": []}
    batch_count = {"train": 0, "val": 0, "test": 0}

    for i, l1c_path in enumerate(tqdm(l1c_files, desc="Pairing L1C/L2A")):
        rel_path = os.path.relpath(l1c_path, L1C_DIR)
        l2a_path = rel_path.replace("-L1C_data.tiff", "-L2A_data.tiff")
        l2a_path = os.path.join(L2A_DIR, l2a_path.replace("/L1C/", "/L2A/"))

        if not os.path.exists(l2a_path):
            continue

        try:
            l1c_data, loc = read_multiband(l1c_path, out_res=RES_10)
            l2a_data, _ = read_multiband(l2a_path, out_res=RES_10)
        except Exception as e:
            print(f"⚠️  Skipping {l1c_path} ({e})")
            continue

        ten_L1C, twenty_L1C, sixty_L1C = split_s2_bands(l1c_data)
        ten_L2A, twenty_L2A, sixty_L2A = split_s2_bands(l2a_data)

        date = parse_date_from_filename(l1c_path)
        doy = date.timetuple().tm_yday if date else -1
        file_id = os.path.splitext(os.path.basename(l1c_path))[0]

        record = {
            "date": np.int64(date.timestamp()) if date else np.int64(0),
            "doy": np.int64(doy),
            "10x_L1C": serialize_array(ten_L1C),
            "20x_L1C": serialize_array(twenty_L1C),
            "60x_L1C": serialize_array(sixty_L1C),
            "10x_L2A": serialize_array(ten_L2A),
            "20x_L2A": serialize_array(twenty_L2A),
            "60x_L2A": serialize_array(sixty_L2A),
            "loc": serialize_array(loc),
            "region": os.path.basename(os.path.dirname(os.path.dirname(l1c_path))),
            "tile_id": np.int32(0),
            "file_id": file_id,
            "fwi": np.nan,
        }


        split = assign_split()
        batch_records[split].append(record)

        # --- Write every BATCH_SIZE samples ---
        if len(batch_records[split]) >= BATCH_SIZE:
            out_file = os.path.join(
                OUT_ROOT, split, f"worldstrat_{split}_{batch_count[split]:04d}.parquet"
            )
            df = pd.DataFrame(batch_records[split])
            pq.write_table(pa.Table.from_pandas(df), out_file)
            print(f"💾 Wrote {len(batch_records[split])} records → {out_file}")
            batch_records[split].clear()
            batch_count[split] += 1

    # --- Write any remaining batches ---
    for split in ["train", "val", "test"]:
        if batch_records[split]:
            out_file = os.path.join(
                OUT_ROOT, split, f"worldstrat_{split}_{batch_count[split]:04d}.parquet"
            )
            df = pd.DataFrame(batch_records[split])
            pq.write_table(pa.Table.from_pandas(df), out_file)
            print(f"Wrote {len(batch_records[split])} records → {out_file}")

    print("\n Done — all split Parquet files saved under:", OUT_ROOT)


if __name__ == "__main__":
    create_parquet()
