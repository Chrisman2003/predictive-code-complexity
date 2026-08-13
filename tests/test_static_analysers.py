import pytest
from src.dataloader.static_analysers import CodeAnalyzer

def test_metric_extraction(tmp_path):
    """Verify SLOC and Cyclomatic complexity on a known dummy function."""
    
    # Create a dummy CUDA kernel with exactly 1 if-statement (Cyclomatic = 2)
    dummy_code = """
    __global__ void simple_kernel(float *data, int n) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) {
            data[idx] = data[idx] * 2.0f;
        }
    }
    """
    
    # Write to a temporary file provided by pytest
    test_file = tmp_path / "dummy_kernel.cu"
    test_file.write_text(dummy_code)
    
    # Analyze
    metrics = CodeAnalyzer.analyze_file(str(test_file))
    
    # Assertions
    assert metrics["function_count"] == 1
    assert metrics["max_cyclomatic_complexity"] == 2  # 1 function + 1 'if' condition
    assert metrics["sloc"] > 0
    assert "halstead_effort" in metrics