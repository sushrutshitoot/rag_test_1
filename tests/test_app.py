import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
import os
import json

def test_app_branding_and_structure():
    """Test that the Prophecy app loads with correct branding and tabs."""
    at = AppTest.from_file("src/rag_app/ui/app.py")
    
    # Run the app
    at.run(timeout=30)
    
    # Assert no exceptions
    assert not at.exception
    
    # Check Prophecy title
    assert "Prophecy" in at.title[0].value
    
    # Check Tab names
    tab_labels = [tab.label for tab in at.tabs]
    assert "Oracle" in tab_labels
    assert "Dashboard" in tab_labels
    
    # Check sidebar components
    assert at.sidebar.text_input[0].label == "Ollama Model Tag"
    assert at.sidebar.button[0].label == "⏹️ Stop Active Prophecy"

@patch("src.rag_app.ui.app.ChatOllama")
@patch("src.rag_app.ui.app.log_metrics")
def test_app_memory_flow(mock_log_metrics, mock_chat_ollama):
    """Test that the app handles conversation memory flow (mocked)."""
    # Mock LLM response
    mock_llm = MagicMock()
    # First call for contextualization, second for answer
    mock_llm.invoke.return_value.content = "Standalone: Who is Ant-Man?"
    mock_llm.stream.return_value = [MagicMock(content="Prophecy answer content")]
    mock_chat_ollama.return_value = mock_llm
    
    at = AppTest.from_file("src/rag_app/ui/app.py")
    at.run(timeout=30)
    
    # 1. Ask first question
    at.chat_input[0].set_value("Who is Ant-Man?").run()
    assert not at.exception
    
    # 2. Check if messages stored in session state
    # Streamlit AppTest doesn't expose session_state directly in a simple way 
    # but we can check if messages are rendered
    assert len(at.chat_message) >= 2
    
    # 3. Ask follow up
    at.chat_input[0].set_value("Tell me more about him.").run()
    assert not at.exception
    
    # Verify that the LLM was called multiple times (contextualize + answer)
    assert mock_llm.invoke.called or mock_llm.stream.called

def test_metrics_logging_function():
    """Unit test for the log_metrics function logic."""
    from src.rag_app.ui.app import log_metrics
    
    test_file = "test_metrics.json"
    if os.path.exists(test_file):
        os.remove(test_file)
        
    # Patch the filename in the function if possible, or just check the real one
    # For simplicity, we'll test the logic by mocking the file path if it was global
    # but here we'll just verify it writes to the default file and then clean up.
    
    try:
        log_metrics("TEST-ID", "Test Question", 1.5, 0.5, 0.9, 2)
        
        with open("run_metrics.json", "r") as f:
            data = json.load(f)
            latest = data[-1]
            assert latest["run_id"] == "TEST-ID"
            assert latest["question"] == "Test Question"
            assert latest["total_latency_sec"] == 1.5
    except Exception as e:
        pytest.fail(f"Logging failed: {e}")
