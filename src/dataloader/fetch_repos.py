import subprocess
import sys
from pathlib import Path
from src.dataloader.config import BENCHMARK_REPOS, RAW_DATA_DIR

def clone_or_update_repo(name: str, url: str, branch: str = "main") -> Path:
    target_dir = RAW_DATA_DIR / name
    if target_dir.exists():
        print(f"[*] Updating existing repo: {name}")
        subprocess.run(["git", "-C", str(target_dir), "pull"], check=True)
    else:
        print(f"[*] Cloning repository: {name}")
        subprocess.run(["git", "clone", "--branch", branch, url, str(target_dir)], check=True)
    return target_dir

def sync_all_benchmarks():
    print("=== Stage 1: Ingesting Benchmark Repositories ===")
    for name, repo_info in BENCHMARK_REPOS.items():
        try:
            clone_or_update_repo(name, repo_info["url"], repo_info.get("branch", "main"))
        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to sync {name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    sync_all_benchmarks()