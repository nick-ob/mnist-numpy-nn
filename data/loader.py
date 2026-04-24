"""File for loading MNIST dataset logic.
"""
import os
import urllib.request

BASE_URL: str = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES: list[str] = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

# get the folder relative to this file
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR: str = os.path.join(SCRIPT_DIR, "mnist")
os.makedirs(MNIST_DIR, exist_ok=True)

def download(filename:str) -> None:
    """Download a file from the BASE_URL and save it to the folder that contains this script.

    Args:
        filename: The name of the file that should be downloaded.
    """
    url = BASE_URL + filename

    out_path: str = os.path.join(MNIST_DIR, filename)

    # skip if already downloaded
    if os.path.exists(out_path):
        print(f"{filename} already exists")
        return
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        raise Exception(f"Failed to download {filename}: {e}")

    print(f"{filename} saved to {out_path}")

def main() -> None:
    """Download all of the MNIST data from the BASE_URL.
    """
    print(f"Download MNIST dataset from {BASE_URL}\n")

    for file in FILES:
        download(file)

    print(f"\nDone. All MNIST dataset saved in {SCRIPT_DIR}")

if __name__ == "__main__":
    main()
