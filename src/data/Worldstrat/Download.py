import os, tarfile, urllib.request

def download_and_extract(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.join(dest_dir, os.path.basename(url))
    if not os.path.exists(fname):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, fname)
    print("Extracting ...")
    with tarfile.open(fname, "r:gz") as tar:
        tar.extractall(dest_dir)
    print(f"Done: {dest_dir}")

download_and_extract("https://zenodo.org/records/6810792/files/lr_dataset_l1c.tar.gz?download=1", "/home/louis/Code/wildfire-forecast/worldstrat/l1c")
download_and_extract("https://zenodo.org/records/6810792/files/lr_dataset_l2a.tar.gz?download=1", "/home/louis/Code/wildfire-forecast/worldstrat/l2a")
