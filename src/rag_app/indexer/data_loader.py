import os
from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from tqdm import tqdm

def load_and_embed_vault(vault_path, persist_directory="./chroma_db", docstore_dir="./docstore"):
    print(f"Loading Obsidian vault from: {vault_path}")
    
    if not os.path.exists(vault_path):
        print(f"Error: Path does not exist -> {vault_path}")
        return None

    # Load documents
    loader = ObsidianLoader(vault_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # Initialize Embeddings model
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create empty vectorstore with COSINE similarity
    print(f"Connecting to Chroma vector database at {persist_directory} with cosine similarity...")
    vectorstore = Chroma(
        collection_name="hierarchical_marvel_vault",
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"} # Use cosine similarity
    )
    
    from langchain_classic.storage._lc_store import create_kv_docstore
    # Create the persistent document store for parent documents
    fs = LocalFileStore(docstore_dir)
    store = create_kv_docstore(fs)

    # The Parent Splitter splits into larger chunks (for context)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    
    # The Child Splitter splits into smaller chunks (for precise matching)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # Initialize the hierarchical retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print(f"Indexing {len(docs)} documents hierarchically...")
    # Add documents in batches with tqdm progress bar
    batch_size = 5
    for i in tqdm(range(0, len(docs), batch_size), desc="Indexing docs (Hierarchical)"):
        batch = docs[i:i+batch_size]
        retriever.add_documents(batch, ids=None)
    
    print("\nVector database and Docstore updated successfully!")
    return retriever

if __name__ == "__main__":
    VAULT_PATH = "/Users/sushrutshitoot/Library/CloudStorage/GoogleDrive-sushrutshitoot@gmail.com/My Drive/KG_Test_1"
    
    retriever = load_and_embed_vault(VAULT_PATH)
    
    if retriever:
        query = "Who is Iron Man?"
        print(f"\nTesting hierarchical vector search with query: '{query}'")
        # similarity_search is on vectorstore, but invoke uses the parent retriever
        results = retriever.invoke(query)
        
        print("\nTop hierarchical result (returns the large parent context):")
        if results:
            print(f"\n--- Source: {results[0].metadata.get('source', 'Unknown')} ---")
            print(results[0].page_content[:500] + "...\n[TRUNCATED]")
