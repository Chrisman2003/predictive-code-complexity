from pathlib import Path

# Base Paths relative to the repository root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Benchmark Target Repositories
BENCHMARK_REPOS = {
    "BabelStream": {
        "url": "https://github.com/UoB-HPC/BabelStream.git",
        "branch": "main", # Current Branch developed on in BabelStream
    },
    "CloverLeaf": {
        "url": "https://github.com/UK-MAC/CloverLeaf_CUDA.git",
        "branch": "master", # Current Branch developed on in CloverLeaf
    },
}
