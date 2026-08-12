import json
from pathlib import Path
from typing import List, Dict, Any
from src.dataloader.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.dataloader.static_analysers import CodeAnalyzer
from src.dataloader.fetch_repos import sync_all_benchmarks

# Default Task Descriptions for Known Mini-Apps
DEFAULT_DESCRIPTIONS = {
    "BabelStream": "Measure maximum achievable memory bandwidth using vector operation kernels (Copy, Mul, Add, Triad, Dot).",
    "CloverLeaf": "Solve 2D compressible Euler equations using a staggered-grid hydrodynamics scheme.",
}

def detect_paradigm(file_path: Path) -> str:
    path_str = str(file_path).lower()
    if "cuda" in path_str or file_path.suffix == ".cu":
        return "CUDA"
    elif "sycl" in path_str:
        return "SYCL"
    elif "openmp" in path_str or "omp" in path_str:
        return "OpenMP-Offload"
    elif "kokkos" in path_str:
        return "Kokkos"
    elif "hip" in path_str:
        return "HIP"
    return "C++ Sequential"

def build_dataset() -> Path:
    # 1. Ensure raw repos are present
    sync_all_benchmarks()
    
    dataset: List[Dict[str, Any]] = []
    supported_extensions = {".cu", ".cpp", ".c", ".hpp", ".h"}

    print("\n=== Stage 2 & 3: Analyzing Code & Building Dataset ===")
    for repo_dir in RAW_DATA_DIR.iterdir():
        if not repo_dir.is_dir():
            continue

        app_name = repo_dir.name
        problem_desc = DEFAULT_DESCRIPTIONS.get(app_name, "GPU parallel execution kernel.")

        for file_path in repo_dir.rglob("*"):
            if file_path.suffix in supported_extensions and not file_path.name.startswith("."):
                try:
                    metrics = CodeAnalyzer.analyze_file(str(file_path))
                    if metrics["sloc"] == 0:
                        continue  # Skip empty or non-executable headers

                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        source_code = f.read()

                    entry = {
                        "id": f"{app_name}_{file_path.stem}_{detect_paradigm(file_path)}",
                        "app_name": app_name,
                        "file_name": file_path.name,
                        "paradigm": detect_paradigm(file_path),
                        "problem_description": problem_desc,
                        "source_code": source_code,
                        "metrics": metrics,
                    }
                    dataset.append(entry)
                except Exception as e:
                    print(f"[!] Warning: Could not process {file_path}: {e}")

    # Output processed dataset
    output_path = PROCESSED_DATA_DIR / "complexity_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n[✓] Dataset built successfully with {len(dataset)} entries.")
    print(f"[✓] Saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    build_dataset()