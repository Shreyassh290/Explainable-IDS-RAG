"""Download the official pre-split UNSW-NB15 dataset files.

Downloads:
1. UNSW_NB15_training-set.csv (~30 MB, 175,341 flows)
2. UNSW_NB15_testing-set.csv (~15 MB, 82,332 flows)

Includes multiple fallback mirrors (Hugging Face Mouwiya-small, Mari4Z, GitHub raw).
Saves directly into data/unsw/ without touching any baseline data.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ids_rag.config import get_project_root

FILES = {
    "UNSW_NB15_training-set.csv": [
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_training-set.csv",
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
    ],
    "UNSW_NB15_testing-set.csv": [
        "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_testing-set.csv",
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
        "https://huggingface.co/datasets/Mari4Z/unsw-nb15-dvc-storage/resolve/main/UNSW_NB15_testing-set.csv",
    ],
}


def download_with_fallbacks(filename: str, urls: list[str], dest_path: Path) -> bool:
    """Download file trying fallback URLs if needed."""
    if dest_path.exists() and dest_path.stat().st_size > 1_000_000:
        print(f"File already exists: {dest_path.name} ({dest_path.stat().st_size:,} bytes). Skipping.")
        return True

    for idx, url in enumerate(urls, 1):
        print(f"Attempting download for {filename} (Mirror {idx}/{len(urls)}: {url[:60]}...)...")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )

            def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100.0, downloaded * 100.0 / total_size)
                    mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  Progress: {mb:.1f}/{tot_mb:.1f} MB ({percent:.1f}%)")
                    sys.stdout.flush()

            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                block_size = 65536
                downloaded = 0
                block_num = 0
                while True:
                    data = response.read(block_size)
                    if not data:
                        break
                    out_file.write(data)
                    downloaded += len(data)
                    block_num += 1
                    progress_callback(block_num, block_size, total_size)

            print(f"\nSuccessfully downloaded {filename} ({dest_path.stat().st_size:,} bytes).")
            return True

        except Exception as e:
            print(f"\nMirror {idx} failed: {e}")
            if dest_path.exists():
                dest_path.unlink()

    return False


def main() -> None:
    root = get_project_root()
    unsw_dir = root / "data" / "unsw"
    unsw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STARTING UNSW-NB15 DATASET DOWNLOAD")
    print(f"Destination: {unsw_dir}")
    print("=" * 60)

    for filename, urls in FILES.items():
        dest = unsw_dir / filename
        success = download_with_fallbacks(filename, urls, dest)
        if not success:
            print(f"Warning: Could not download {filename}. Fallback split will be used if training set is present.")

    print("\nDownload process completed.")


if __name__ == "__main__":
    main()
