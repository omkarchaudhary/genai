"""
Vector Store Module
===================
Handles ChromaDB vector database operations.

ChromaDB is used because:
- Free and open source
- Persistent storage (data survives restarts)
- Easy to use with LangChain
- Good performance for small to medium datasets
"""

from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, RETRIEVAL_TOP_K


# =============================================================================
# EMBEDDINGS
# =============================================================================

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get HuggingFace embeddings model.
    
    Returns:
        HuggingFaceEmbeddings instance
        
    Why HuggingFace embeddings:
        - Free (no API key needed)
        - Runs locally
        - all-MiniLM-L6-v2 is fast and effective
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # Use CPU (works everywhere)
        encode_kwargs={"normalize_embeddings": True}  # Normalize for better similarity
    )
    return embeddings


# =============================================================================
# VECTOR STORE OPERATIONS
# =============================================================================

def get_vector_store() -> Chroma:
    """
    Get or create ChromaDB vector store.
    
    Returns:
        Chroma vector store instance
        
    The persist_directory ensures data is saved to disk
    and loaded on next startup.
    """
    embeddings = get_embeddings()
    
    # Create/load vector store with persistence
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    return vector_store


def add_documents_to_store(documents: List[Document]) -> int:
    """
    Add documents to the vector store.
    
    Args:
        documents: List of Document objects to add
        
    Returns:
        Number of documents added
        
    Process:
        1. Get vector store
        2. Generate embeddings for each document
        3. Store embeddings + text + metadata in ChromaDB
    """
    if not documents:
        return 0
    
    vector_store = get_vector_store()
    
    # Add documents (embeddings are generated automatically)
    vector_store.add_documents(documents)
    
    return len(documents)


def similarity_search(query: str, k: int = RETRIEVAL_TOP_K) -> List[Document]:
    """
    Search for similar documents based on query.
    
    Args:
        query: User's question or search query
        k: Number of results to return
        
    Returns:
        List of most similar Document objects
        
    How it works:
        1. Convert query to embedding
        2. Find k nearest neighbors in vector space
        3. Return documents with highest similarity
    """
    vector_store = get_vector_store()
    
    # Perform similarity search
    results = vector_store.similarity_search(query, k=k)
    
    return results


def get_retriever(k: int = RETRIEVAL_TOP_K):
    """
    Get a retriever for use in RAG chain.
    
    Args:
        k: Number of documents to retrieve
        
    Returns:
        Retriever object compatible with LangChain
    """
    vector_store = get_vector_store()
    
    # Create retriever with search parameters
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    return retriever


def get_collection_stats() -> dict:
    """
    Get statistics about the vector store collection.
    
    Returns:
        Dictionary with collection statistics
    """
    vector_store = get_vector_store()
    
    # Get collection count
    collection = vector_store._collection
    count = collection.count()
    
    return {
        "total_documents": count,
        "collection_name": COLLECTION_NAME,
        "persist_directory": CHROMA_PERSIST_DIR
    }


def clear_vector_store() -> bool:
    """
    Clear all documents from the vector store.
    
    Returns:
        True if successful
        
    Use with caution - this deletes all stored knowledge!
    """
    vector_store = get_vector_store()
    
    # Get all document IDs and delete them
    collection = vector_store._collection
    
    # Get all IDs
    all_ids = collection.get()["ids"]
    
    if all_ids:
        collection.delete(ids=all_ids)
    
    return True
