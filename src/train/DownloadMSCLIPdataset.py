import os, time
from pathlib import Path
from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import HfHubHTTPError


DST = Path("/work/eceo/grosse/ssl4eo-s12")
DST.mkdir(parents=True, exist_ok=True)

# (optional) login via env var if needed
tok = os.environ.get("HF_TOKEN")
if tok:
    login(token=tok)

# ensure caches exist (fast scratch)
os.makedirs(os.environ.get("HF_HOME", str(DST / ".hf_home")), exist_ok=True)
os.makedirs(os.environ.get("HUGGINGFACE_HUB_CACHE", str(DST / ".hf_cache")), exist_ok=True)

def safe_snapshot(*args, **kwargs):
    backoff = 10
    for attempt in range(6):
        try:
            return snapshot_download(*args, **kwargs)
        except HfHubHTTPError as e:
            s = str(e)
            if "429" in s or "Read timed out" in s:
                print(f"[warn] {s}\nSleeping {backoff}s (attempt {attempt+1}/6)")
                time.sleep(backoff); backoff *= 2
                continue
            raise

safe_snapshot(
    repo_id="embed2scale/SSL4EO-S12-v1.1",
    repo_type="dataset",
    local_dir=str(DST),
    allow_patterns=[
        "splits/*",
        "train/S2L1C/*",
        "val/S2L1C/*",
    ],
    resume_download=True,
    local_dir_use_symlinks=False,
    max_workers=4,   # keep this modest to avoid rate limits
)
print("Done.")
