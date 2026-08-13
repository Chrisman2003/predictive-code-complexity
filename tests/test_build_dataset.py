from pathlib import Path
from src.dataloader.build_dataset import detect_paradigm

def test_detect_paradigm():
    """Ensure the pipeline accurately tags the programming paradigm based on path."""
    assert detect_paradigm(Path("src/cuda/triad.cu")) == "CUDA"
    assert detect_paradigm(Path("src/sycl/triad.cpp")) == "SYCL"
    assert detect_paradigm(Path("src/kokkos/triad.cpp")) == "Kokkos"
    assert detect_paradigm(Path("src/omp/triad.c")) == "OpenMP-Offload"
    assert detect_paradigm(Path("src/sequential/triad.cpp")) == "C++ Sequential"