import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.rag_app.indexer.data_loader import load_and_embed_vault

@patch("src.rag_app.indexer.data_loader.ObsidianLoader")
@patch("src.rag_app.indexer.data_loader.HuggingFaceEmbeddings")
@patch("src.rag_app.indexer.data_loader.Chroma")
@patch("src.rag_app.indexer.data_loader.LocalFileStore")
@patch("langchain_classic.storage._lc_store.create_kv_docstore")
@patch("src.rag_app.indexer.data_loader.ParentDocumentRetriever")
def test_load_and_embed_vault_success(
    mock_retriever, mock_create_kv_docstore, mock_local_store, 
    mock_chroma, mock_embeddings, mock_loader, tmp_path
):
    """Test that the vault is loaded, embedded, and indexed correctly using hierarchical strategy and cosine similarity."""
    
    # Mocking the ObsidianLoader to return dummy documents
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    dummy_docs = [
        Document(page_content="This is a test document about Iron Man.", metadata={"source": "Iron-Man.md"}),
        Document(page_content="This is another test document about Thor.", metadata={"source": "Thor.md"})
    ]
    mock_loader_instance.load.return_value = dummy_docs

    # Mocking the Retriever
    mock_retriever_instance = MagicMock()
    mock_retriever.return_value = mock_retriever_instance

    # Create a temporary directory to act as the vault
    test_vault_path = tmp_path / "KG_Test"
    test_vault_path.mkdir()

    # Run the function with mocked dependencies
    result = load_and_embed_vault(
        str(test_vault_path), 
        persist_directory=str(tmp_path / "chroma_db"), 
        docstore_dir=str(tmp_path / "docstore")
    )
    
    # Assertions
    mock_loader.assert_called_once_with(str(test_vault_path))
    mock_loader_instance.load.assert_called_once()
    mock_embeddings.assert_called_once_with(model_name="all-MiniLM-L6-v2")
    
    mock_chroma.assert_called_once()
    _, kwargs = mock_chroma.call_args
    assert kwargs.get("collection_metadata") == {"hnsw:space": "cosine"}
    
    mock_local_store.assert_called_once_with(str(tmp_path / "docstore"))
    mock_create_kv_docstore.assert_called_once()
    mock_retriever.assert_called_once()
    mock_retriever_instance.add_documents.assert_called_once()
    
    assert result == mock_retriever_instance
