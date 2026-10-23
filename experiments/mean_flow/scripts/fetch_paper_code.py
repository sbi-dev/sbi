"""Download the code of Arruda et al. (2026) from anonymous.4open.science.

The repository page needs JavaScript, so this walks the site's file API instead:
`/api/repo/<name>/files/?path=<dir>` lists a directory and
`/api/repo/<name>/file/<path>` returns a file. A few files differ from the listed
size by some bytes (the site anonymizes names), which is reported but harmless.

    python experiments/mean_flow/scripts/fetch_paper_code.py --out ~/meanflow-paper
"""

import argparse
import json
import subprocess
import time
import urllib.parse
from pathlib import Path

BASE = "https://anonymous.4open.science/api/repo/MeanFlow-AISTAST"


def get(url: str, out: Path | None = None) -> bytes:
    command = ["curl", "-sfL", "--max-time", "60", url]
    if out is not None:
        command += ["-o", str(out)]
    for _ in range(4):
        result = subprocess.run(command, capture_output=True)
        if result.returncode == 0:
            return result.stdout
        time.sleep(3)
    raise RuntimeError(f"Download failed: {url}")


def walk(path: str, files: list) -> None:
    query = f"?path={urllib.parse.quote(path)}" if path else ""
    for entry in json.loads(get(f"{BASE}/files/{query}")):
        child = f"{path}/{entry['name']}" if path else entry["name"]
        if "size" in entry:
            files.append((child, entry["size"]))
        else:
            walk(child, files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files: list = []
    walk("", files)
    for path, size in files:
        target = args.out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        get(f"{BASE}/file/{urllib.parse.quote(path)}", out=target)
        if target.stat().st_size != size:
            print(f"size differs: {path} ({target.stat().st_size} vs {size})")
    print(f"{len(files)} files in {args.out}")


if __name__ == "__main__":
    main()
