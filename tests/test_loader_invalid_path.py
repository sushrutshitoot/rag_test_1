import pytest
from src.rag_app.indexer.data_loader import load_and_embed_vault

def test_load_and_embed_vault_invalid_path(capsys):
    """Test that the function returns None and prints an error for an invalid path."""
    result = load_and_embed_vault("/invalid/path/that/does/not/exist")
    
    assert result is None
    captured = capsys.readouterr()
    assert "Error: Path does not exist" in captured.out
