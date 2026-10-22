"""Download and extract the complete official FEVER dataset and Wikipedia corpus with robust resumption."""
import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config_loader import get_storage_dir

def download_resumable(url: str, output_path: Path, chunk_size: int = 1024 * 1024):
    """Downloads a file with HTTP Range resumption support, retry loop, and progress reporting."""
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req) as resp:
        total_bytes = int(resp.headers.get('Content-Length', 0))

    part_path = output_path.with_suffix(output_path.suffix + ".part")

    # If the output file already exists, check if it's complete
    if output_path.exists():
        if output_path.stat().st_size >= total_bytes:
            print(f"File already exists and is complete at {output_path} ({output_path.stat().st_size / (1024*1024):.1f} MB). Skipping.")
            return
        else:
            print(f"Existing file {output_path.name} is incomplete ({output_path.stat().st_size} < {total_bytes}). Renaming to .part to resume...")
            if part_path.exists():
                part_path.unlink()
            output_path.rename(part_path)

    start_time = time.time()
    last_log_time = start_time

    while True:
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0
        if existing_bytes >= total_bytes and total_bytes > 0:
            part_path.rename(output_path)
            print(f"Download complete: {output_path.name} ({output_path.stat().st_size / (1024*1024):.1f} MB).")
            return

        headers = {}
        mode = "wb"
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            mode = "ab"
            print(f"Resuming download from byte {existing_bytes:,} ({existing_bytes / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB)...")
        else:
            print(f"Starting download from {url} to {output_path} ({total_bytes / (1024*1024):.1f} MB)...")

        req = urllib.request.Request(url, headers=headers)
        downloaded = existing_bytes

        try:
            with urllib.request.urlopen(req) as resp, open(part_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log_time >= 5.0 or downloaded >= total_bytes:
                        elapsed = now - start_time
                        speed = (downloaded - existing_bytes) / elapsed / (1024 * 1024) if elapsed > 0 else 0
                        pct = (downloaded / total_bytes) * 100 if total_bytes > 0 else 0
                        print(f"  Progress: {downloaded / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", flush=True)
                        last_log_time = now

            if downloaded >= total_bytes:
                part_path.rename(output_path)
                total_time = time.time() - start_time
                print(f"Successfully downloaded {output_path.name} ({total_bytes / (1024*1024):.1f} MB) in {total_time:.1f}s.")
                break
            else:
                print(f"Connection ended early at {downloaded:,} of {total_bytes:,} bytes. Reconnecting in 3s...", flush=True)
                time.sleep(3)
        except Exception as e:
            print(f"Network error during download: {e}. Reconnecting in 5s...", flush=True)
            time.sleep(5)

def main():
    target_dir = get_storage_dir("data/raw", for_writing=True)
    print(f"Target directory: {target_dir}")

    # 1. Download official claims
    urls = {
        "train.jsonl": "https://fever.ai/download/fever/train.jsonl",
        "shared_task_dev.jsonl": "https://fever.ai/download/fever/shared_task_dev.jsonl",
        "shared_task_test.jsonl": "https://fever.ai/download/fever/shared_task_test.jsonl",
    }
    for fname, url in urls.items():
        download_resumable(url, target_dir / fname)

    # 2. Download official Wikipedia dump
    wiki_zip = target_dir / "wiki-pages.zip"
    wiki_url = "https://fever.ai/download/fever/wiki-pages.zip"
    download_resumable(wiki_url, wiki_zip)

    # 3. Extract if zip is downloaded
    wiki_extract_dir = target_dir / "wiki-pages"
    if wiki_zip.exists() and not (wiki_extract_dir / "wiki-109.jsonl").exists():
        print(f"Extracting {wiki_zip} into {target_dir}...")
        with zipfile.ZipFile(wiki_zip, 'r') as zf:
            zf.extractall(target_dir)
        print("Extraction complete.")

if __name__ == "__main__":
    main()
