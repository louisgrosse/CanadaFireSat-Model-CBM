import os, re, json
from datetime import datetime, timedelta, date
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets import Dataset, Features, Value, Sequence, Array3D
from pystac_client import Client
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling

# ---------------- CONFIG ----------------
PARQUET_ROOT = "/home/louis/Code/wildfire-forecast/hug_data/Alberta" 
OUT_DIR      = "/home/louis/Code/wildfire-forecast/hug_data_mixed/Alberta-2018"
TARGET_YEAR  = 2018
REGION_NAME  = "Alberta"
MAX_PER_SPLIT = 20         # e.g., 200 for a quick test; None for all
SEARCH_PAD_DAYS = 5          # +/- days around DOY to find acquisitions
CLOUD_COVER_MAX = 85         # skip super-cloudy L2A scenes
EARTH_SEARCH   = "https://earth-search.aws.element84.com/v1"
COL_L2A, COL_L1C = "sentinel-2-l2a", "sentinel-2-l1c"

BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
BANDS_60M = ["B01", "B09", "B10"]

SAFE_COLS = ["region", "date", "doy", "loc", "file_id"]

def to_float01(x): return (x.astype("float32") / 10000.0)

# ---------------------------------------

def list_parquets_by_split(root):
    files = []
    for p in glob(os.path.join(root, "**", "*.parquet"), recursive=True):
        base = os.path.basename(p).lower()
        if re.search(r"\btrain-.*\.parquet$", base):
            files.append(("train", p))
        elif re.search(r"\bval-.*\.parquet$", base) or re.search(r"\bvalid-.*\.parquet$", base):
            files.append(("val", p))
        elif re.search(r"\btest-.*\.parquet$", base):
            files.append(("test", p))
        else:
            # If split not in filename, treat as train by default
            files.append(("train", p))
    return files

def same_year_doys(target_year, doys):
    out = []
    for d in doys:
        d = int(d)
        if d == 366:
            # clamp leap DOY
            d = 365
        out.append(datetime.strptime(f"{target_year}-{d:03d}", "%Y-%j").date())
    # de-dup but keep original order
    seen, seq = set(), []
    for t in out:
        if t not in seen:
            seq.append(t); seen.add(t)
    return seq

def stac_items_for_window(cat, collection, bbox4326, center_date, pad_days=5, limit=50, query=None):
    start = (center_date - timedelta(days=pad_days)).isoformat() + "T00:00:00Z"
    end   = (center_date + timedelta(days=pad_days)).isoformat() + "T23:59:59Z"
    search = cat.search(
        collections=[collection],
        bbox=bbox4326,
        datetime=f"{start}/{end}",
        limit=limit,
        query=query
    )
    return list(search.get_items())

def pick_best_l2a(items):
    if not items: return None
    def cloud(p):
        for k in ("eo:cloud_cover","s2:cloud_cover"):
            v = p.get(k, None)
            if v is not None: return v
        return 1000
    # bound cloud cover
    items = [it for it in items if cloud(it.properties) <= CLOUD_COVER_MAX]
    if not items: return None
    # lowest cloud, then earliest
    items.sort(key=lambda it: (cloud(it.properties), it.datetime))
    return items[0]

def find_matching_l1c(cat, l2a_item, pad_hours=3):
    t = l2a_item.datetime
    bbox = l2a_item.bbox
    mgrs = l2a_item.properties.get("s2:mgrs_tile") or l2a_item.properties.get("mgrs:tile")
    query = {"s2:mgrs_tile": {"eq": mgrs}} if mgrs else None
    start = (t - timedelta(hours=pad_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end   = (t + timedelta(hours=pad_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    items = list(Client.open(EARTH_SEARCH).search(
        collections=[COL_L1C], bbox=bbox, datetime=f"{start}/{end}", limit=10, query=query
    ).get_items())
    if not items:
        # fallback same day
        d = t.date()
        items = list(Client.open(EARTH_SEARCH).search(
            collections=[COL_L1C], bbox=bbox, datetime=f"{d}T00:00:00Z/{d}T23:59:59Z", limit=10, query=query
        ).get_items())
    if not items: return None
    items.sort(key=lambda it: abs((it.datetime - t).total_seconds()))
    return items[0]

def crop_resample_single(href, bbox4326, out_h, out_w):
    with rasterio.Env(AWS_REQUEST_PAYER="requester"):
        with rasterio.open(href) as src:
            bbox_src = transform_bounds("EPSG:4326", src.crs, *bbox4326, densify_pts=21)
            win = from_bounds(*bbox_src, transform=src.transform)
            data = src.read(1, window=win, boundless=True, fill_value=0)
            dst_transform, _, _ = calculate_default_transform(
                src.crs, src.crs,
                width=data.shape[1], height=data.shape[0],
                left=bbox_src[0], bottom=bbox_src[1], right=bbox_src[2], top=bbox_src[3],
                dst_width=out_w, dst_height=out_h
            )
            dst = np.zeros((out_h, out_w), dtype=np.float32)
            reproject(
                source=data,
                destination=dst,
                src_transform=src.window_transform(win),
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear
            )
            return dst  # float32

def read_stack(item, bands, bbox4326, out_shape):
    # Try official band code first; for L2A, fall back to common aliases
    alias = {"B02":"blue", "B03":"green", "B04":"red", "B08":"nir"}
    H, W = out_shape
    arrs = []
    for b in bands:
        asset = item.assets.get(b)
        if not asset and b in alias:
            asset = item.assets.get(alias[b])
        if not asset:
            # missing band -> zero
            arrs.append(np.zeros((H, W), dtype=np.float32))
            continue
        arrs.append(crop_resample_single(asset.href, bbox4326, H, W))
    # scale to [0,1] reflectance convention
    stack = np.stack(arrs, axis=0)
    return to_float01(stack)

def bbox_from_loc(loc):
    lat = np.array(loc[0], dtype=float)
    lon = np.array(loc[1], dtype=float)
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())

def build_features():
    # L1C kept under canonical names for drop-in
    feats = {
        "date": Value("timestamp[s]"),
        "region": Value("string"),
        "file_id": Value("string"),
        "doy": Sequence(Value("int64")),
        "loc": Array3D(shape=(2,264,264), dtype="float32"),

        "10x": Sequence(Array3D(shape=(4,264,264), dtype="float32")),
        "20x": Sequence(Array3D(shape=(6,132,132), dtype="float32")),
        "60x": Sequence(Array3D(shape=(3,44,44),  dtype="float32")),

        # Parallel L2A sequences
        "10x_l2a": Sequence(Array3D(shape=(4,264,264), dtype="float32")),
        "20x_l2a": Sequence(Array3D(shape=(6,132,132), dtype="float32")),
        "60x_l2a": Sequence(Array3D(shape=(3,44,44),  dtype="float32")),

        # Provenance
        "l2a_item_ids": Sequence(Value("string")),
        "l1c_item_ids": Sequence(Value("string")),
    }
    return Features(feats)

def safe_read_source_parquet(fpath):
    """
    Read only the columns we need. If pandas/pyarrow hits a HF extension
    decoding issue, fall back to 🤗 datasets which knows how to handle them.
    Returns a pandas.DataFrame with SAFE_COLS.
    """
    try:
        # read only the necessary columns to bypass problematic env_* arrays
        return pd.read_parquet(fpath, columns=SAFE_COLS)
    except Exception as e:
        # Fallback: use HuggingFace datasets to read and select columns
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=fpath, split="train")
        keep = [c for c in SAFE_COLS if c in ds.column_names]
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        return ds.to_pandas()

def process_split(split_name, files, out_parquet):
    cat = Client.open(EARTH_SEARCH)
    rows = []

    total = 0
    for _, fpath in files:
        if MAX_PER_SPLIT and total >= MAX_PER_SPLIT:
            break
        df = safe_read_source_parquet(fpath)
        # Expect columns: 'region','date'(epoch sec),'doy' (sequence),'loc' ((2,264,264)), 'file_id'
        # Filter Alberta + pre-2017
        if "region" in df.columns:
            df = df[df["region"] == REGION_NAME]
        # convert epoch seconds to UTC date
        df["date"] = pd.to_datetime(df["date"], unit="s", utc=True).dt.tz_convert("UTC").dt.date
        df = df[pd.DatetimeIndex(df["date"]).year < 2017]
        if df.empty: 
            continue

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name}:{os.path.basename(fpath)}"):
            if MAX_PER_SPLIT and total >= MAX_PER_SPLIT:
                break

            loc = np.array(row["loc"], dtype=np.float32)  # (2,264,264)
            bbox4326 = bbox_from_loc(loc)

            if not isinstance(row["doy"], (list, tuple, np.ndarray)):
                continue
            doys = list(map(int, row["doy"]))
            target_dates = same_year_doys(TARGET_YEAR, doys)

            # For each target date: pick best L2A, then nearest L1C
            l2a_seq_10, l2a_seq_20, l2a_seq_60 = [], [], []
            l1c_seq_10, l1c_seq_20, l1c_seq_60 = [], [], []
            l2a_ids, l1c_ids, tgt_doys = [], [], []

            complete = True
            for td in target_dates:
                # L2A
                l2a_cands = stac_items_for_window(cat, COL_L2A, bbox4326, td, SEARCH_PAD_DAYS, 50)
                best_l2a = pick_best_l2a(l2a_cands)
                if not best_l2a:
                    complete = False
                    break

                # L1C matched
                match_l1c = find_matching_l1c(cat, best_l2a, pad_hours=3)
                if not match_l1c:
                    complete = False
                    break

                # read stacks
                l2a_10 = read_stack(best_l2a, BANDS_10M, bbox4326, (264,264))
                l2a_20 = read_stack(best_l2a, BANDS_20M, bbox4326, (132,132))
                l2a_60 = read_stack(best_l2a, BANDS_60M, bbox4326, (44,44))

                l1c_10 = read_stack(match_l1c, BANDS_10M, bbox4326, (264,264))
                l1c_20 = read_stack(match_l1c, BANDS_20M, bbox4326, (132,132))
                l1c_60 = read_stack(match_l1c, BANDS_60M, bbox4326, (44,44))

                l2a_seq_10.append(l2a_10); l2a_seq_20.append(l2a_20); l2a_seq_60.append(l2a_60)
                l1c_seq_10.append(l1c_10); l1c_seq_20.append(l1c_20); l1c_seq_60.append(l1c_60)

                l2a_ids.append(best_l2a.id); l1c_ids.append(match_l1c.id)
                tgt_doys.append(int(td.strftime("%j")))

            if not complete:
                # skip samples where we couldn't assemble the full time series
                continue

            # build row (L1C under canonical keys; L2A under *_l2a)
            rows.append({
                "date": pd.Timestamp(target_dates[0].isoformat()),  # pick first target date
                "region": REGION_NAME,
                "file_id": str(row.get("file_id", f"{split_name}_{os.path.basename(fpath)}_{idx}")),
                "doy": tgt_doys,
                "loc": loc.astype("float32"),

                "10x":  l1c_seq_10,
                "20x":  l1c_seq_20,
                "60x":  l1c_seq_60,

                "10x_l2a": l2a_seq_10,
                "20x_l2a": l2a_seq_20,
                "60x_l2a": l2a_seq_60,

                "l2a_item_ids": l2a_ids,
                "l1c_item_ids": l1c_ids,
            })
            total += 1

    if not rows:
        print(f"[{split_name}] No rows produced.")
        return

    ds = Dataset.from_list(rows, features=build_features())
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{split_name}.parquet")
    ds.to_parquet(out_path)
    print(f"[{split_name}] wrote {out_path} with {len(rows)} rows.")

def main():
    files_by_split = list_parquets_by_split(PARQUET_ROOT)
    splits = {"train": [], "val": [], "test": []}
    for split, path in files_by_split:
        splits[split].append((split, path))

    for split in ["train", "val", "test"]:
        if not splits[split]:
            print(f"Skipping {split}: no source files detected.")
            continue
        process_split(split, splits[split], os.path.join(OUT_DIR, f"{split}.parquet"))

if __name__ == "__main__":
    main()
