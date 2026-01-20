"""
Configuration Module
====================
Centralized configuration for the RAG application.
All settings and environment variables are managed here.

Supports both:
- Local development: .env file
- Streamlit Cloud: st.secrets
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()


def get_secret(key: str, default: str = None) -> str:
    """
    Get secret from Streamlit secrets or environment variables.
    
    Priority:
    1. Streamlit secrets (for cloud deployment)
    2. Environment variables (for local development)
    3. Default value
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    return os.getenv(key, default)


# =============================================================================
# API KEYS
# =============================================================================
# Groq API key for LLM access (get free key at: https://console.groq.com)
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# =============================================================================
# LLM SETTINGS
# =============================================================================
# Groq model to use (llama-3.3-70b-versatile is fast and capable)
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# Temperature controls randomness (0 = deterministic, 1 = creative)
LLM_TEMPERATURE = 0.1

# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================
# HuggingFace embedding model (free, no API key needed)
# all-MiniLM-L6-v2 is lightweight and effective
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# =============================================================================
# TEXT CHUNKING SETTINGS
# =============================================================================
# Size of each text chunk in characters
CHUNK_SIZE = 1000

# Overlap between chunks to maintain context
CHUNK_OVERLAP = 200

# =============================================================================
# VECTOR STORE SETTINGS
# =============================================================================
# Directory to persist ChromaDB data (survives app restarts)
CHROMA_PERSIST_DIR = "./chroma_db"

# Name of the collection in ChromaDB
COLLECTION_NAME = "rag_knowledge_base"

# Number of similar documents to retrieve for context
RETRIEVAL_TOP_K = 5

# =============================================================================
# FILE UPLOAD SETTINGS
# =============================================================================
# Allowed file extensions for upload
ALLOWED_EXTENSIONS = [".pdf", ".txt"]

# Maximum file size in MB
MAX_FILE_SIZE_MB = 10
