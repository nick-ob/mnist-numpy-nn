"""File for loading MNIST dataset logic.
"""

import os
import urllib.request

BASE_URL: str = "http://yann.lecun.com/exdb/mnist/"

FILES: list[str] = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

# get the folder relative to this file
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
OUT_DIR: str = os.path.join(SCRIPT_DIR, "data")

def download(filename:str) -> None:
    url = BASE_URL + filename

    out_path: str = os.path.join(OUT_DIR, filename)

    # skip if already downloaded
    if os.path.exists(out_path):
        print(f"{filename} already exists")
        return

    urllib.request.urlretrieve(url, out_path)
    print(f"{filename} saved to {out_path}")

def main() -> None:
    for file in FILES:
        download(file)

    print(f"\nDone. All MNIST dataset saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
