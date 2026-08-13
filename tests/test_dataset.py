import json
import torch
from src.dataloader.dataset import CodeComplexityDataset

def test_pytorch_dataset_formatting(tmp_path):
    """Ensure the dataset correctly formats NLP text and tensor targets."""
    
    # Mock a processed dataset file
    mock_data = [
        {
            "id": "mock_01",
            "problem_description": "Vector addition",
            "paradigm": "CUDA",
            "source_code": "void add() {}",
            "metrics": {"halstead_effort": 150.5}
        }
    ]
    
    mock_file = tmp_path / "complexity_dataset.json"
    with open(mock_file, "w") as f:
        json.dump(mock_data, f)
        
    # Initialize the dataset
    dataset = CodeComplexityDataset(dataset_path=mock_file, target_metric="halstead_effort")
    
    # Test length
    assert len(dataset) == 1
    
    # Test item retrieval and types
    item = dataset[0]
    assert item["id"] == "mock_01"
    
    # Verify NLP Context string was built correctly
    assert "Vector addition" in item["input_text"]
    assert "CUDA" in item["input_text"]
    
    # Verify the target is a valid FloatTensor
    assert isinstance(item["target"], torch.Tensor)
    assert item["target"].dtype == torch.float32
    assert item["target"].item() == 150.5