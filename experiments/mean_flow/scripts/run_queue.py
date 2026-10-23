"""Run the shell commands of a file, N at a time.

python experiments/mean_flow/scripts/run_queue.py jobs.txt 6 > queue.log
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run(command: str) -> None:
    start = time.time()
    result = subprocess.run(command, shell=True)
    label = command.split("--out")[0][-120:]
    print(f"{result.returncode} {time.time() - start:.0f}s {label}", flush=True)


if __name__ == "__main__":
    jobs = [line for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    with ThreadPoolExecutor(int(sys.argv[2])) as pool:
        list(pool.map(run, jobs))
    print("ALL DONE", flush=True)
