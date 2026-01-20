"""
Text Processor Module
=====================
Handles text chunking and preprocessing.

Why chunking is important:
- LLMs have token limits
- Smaller chunks = more precise retrieval
- Overlap maintains context between chunks
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with configured settings.
    
    Returns:
        RecursiveCharacterTextSplitter instance
        
    How RecursiveCharacterTextSplitter works:
        1. Tries to split on paragraphs first (\n\n)
        2. Then sentences (\n)
        3. Then words ( )
        4. Finally characters if needed
        This preserves semantic meaning better than fixed-size splits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,           # Max characters per chunk
        chunk_overlap=CHUNK_OVERLAP,      # Overlap between chunks
        length_function=len,              # Use character count
        separators=["\n\n", "\n", " ", ""]  # Split priority
    )
    return splitter


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        documents: List of Document objects to split
        
    Returns:
        List of chunked Document objects with preserved metadata
        
    Example:
        Input: 1 document with 5000 characters
        Output: ~5 documents with ~1000 characters each
        
    The metadata (source, page, type) is preserved in each chunk.
    """
    # Create splitter with our config
    splitter = create_text_splitter()
    
    # Split all documents - metadata is automatically preserved
    chunks = splitter.split_documents(documents)
    
    # Add chunk index to metadata for tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    
    return chunks


def get_chunk_info(chunks: List[Document]) -> dict:
    """
    Get information about the chunks created.
    
    Args:
        chunks: List of chunked Document objects
        
    Returns:
        Dictionary with chunk statistics
        
    Useful for debugging and displaying info to users.
    """
    if not chunks:
        return {"total_chunks": 0, "avg_length": 0, "sources": []}
    
    # Calculate statistics
    total_chunks = len(chunks)
    total_length = sum(len(chunk.page_content) for chunk in chunks)
    avg_length = total_length // total_chunks
    
    # Get unique sources
    sources = list(set(chunk.metadata.get("source", "Unknown") for chunk in chunks))
    
    return {
        "total_chunks": total_chunks,
        "avg_length": avg_length,
        "total_characters": total_length,
        "sources": sources
    }
