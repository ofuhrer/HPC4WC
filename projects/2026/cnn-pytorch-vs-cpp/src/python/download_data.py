"""
download_data.py

Fetch the MNIST dataset into ``data/MNIST/raw/`` (relative to the repo root),
matching the on-disk layout torchvision expects. Standard library only, so it
runs with a bare login-node ``python3`` (no torch / torchvision needed).

Idempotent: an archive whose extracted idx file is already present is skipped,
so this is a cheap no-op on repeat runs and on machines that already have the
data. Needs internet the first time — on CSCS Santis run it from a login node.

Used automatically by the Makefile's ``run`` / ``verify`` / ``train`` targets;
can also be run directly:

    python3 src/python/download_data.py
"""

import gzip
import hashlib
import os
import sys
import urllib.request

# The mirror torchvision itself uses, with its published md5 checksums.
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
ARCHIVES = {
    "train-images-idx3-ubyte": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train-labels-idx1-ubyte": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "t10k-images-idx3-ubyte": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "t10k-labels-idx1-ubyte": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}

RAW_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../data/MNIST/raw")
)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    fetched = 0

    for idx_name, (gz_name, md5) in ARCHIVES.items():
        idx_path = os.path.join(RAW_DIR, idx_name)
        if os.path.exists(idx_path):
            print(f"  ok    {idx_name} (already present)")
            continue

        gz_path = os.path.join(RAW_DIR, gz_name)
        url = MIRROR + gz_name
        print(f"  fetch {gz_name} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, gz_path)
        except Exception as e:  # noqa: BLE001 - report any network/URL failure the same way
            print(
                f"\nERROR: could not download {url}\n       {e}\n"
                f"       MNIST is required at {RAW_DIR}.\n"
                f"       Run this from a machine with internet "
                f"(on CSCS Santis: a login node).",
                file=sys.stderr,
            )
            sys.exit(1)

        if _md5(gz_path) != md5:
            os.remove(gz_path)
            print(f"\nERROR: checksum mismatch for {gz_name}", file=sys.stderr)
            sys.exit(1)

        with gzip.open(gz_path, "rb") as src, open(idx_path, "wb") as dst:
            dst.write(src.read())
        fetched += 1

    print(f"MNIST ready in {RAW_DIR} ({fetched} archive(s) downloaded).")


if __name__ == "__main__":
    main()
