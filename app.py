"""
RAG Application - Main Streamlit App
====================================
A production-ready RAG application with:
- File upload (PDF, TXT)
- Web scraping from URLs
- Knowledge base management
- Chat interface with sources

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_loader import load_document
from text_processor import split_documents, get_chunk_info
from vector_store import add_documents_to_store, get_collection_stats, clear_vector_store
from rag_chain import ask_question
from utils import is_valid_url, is_valid_file_extension
from config import ALLOWED_EXTENSIONS, GROQ_API_KEY


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables for chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_base_ready" not in st.session_state:
        st.session_state.knowledge_base_ready = False


# =============================================================================
# SIDEBAR - KNOWLEDGE BASE MANAGEMENT
# =============================================================================

def render_sidebar():
    """Render the sidebar with knowledge base management options."""
    
    with st.sidebar:
        st.title("📚 Knowledge Base")
        st.markdown("---")
        
        # ----- FILE UPLOAD SECTION -----
        st.subheader("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT"
        )
        
        if uploaded_files:
            if st.button("📤 Process Uploaded Files", use_container_width=True):
                process_uploaded_files(uploaded_files)
        
        st.markdown("--- OR ---")
        
        # ----- URL SCRAPING SECTION -----
        st.subheader("🌐 Web Scraping")
        
        url_input = st.text_input(
            "Enter URL to scrape",
            placeholder="https://example.com/article",
            help="Enter a valid URL to extract text content"
        )
        
        if url_input:
            if st.button("🔗 Process URL", use_container_width=True):
                process_url(url_input)
        
        st.markdown("---")
        
        # ----- KNOWLEDGE BASE STATS -----
        st.subheader("📊 Statistics")
        
        try:
            stats = get_collection_stats()
            st.metric("Documents in KB", stats["total_documents"])
            
            if stats["total_documents"] > 0:
                st.session_state.knowledge_base_ready = True
            
        except Exception as e:
            st.info("No documents in knowledge base yet")
        
        st.markdown("---")
        
        # ----- CLEAR KNOWLEDGE BASE -----
        st.subheader("⚠️ Danger Zone")
        
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True, type="secondary"):
            if clear_knowledge_base():
                st.success("Knowledge base cleared!")
                st.session_state.knowledge_base_ready = False
                st.rerun()


# =============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# =============================================================================

def process_uploaded_files(files):
    """
    Process uploaded files and add to knowledge base.
    
    Args:
        files: List of uploaded file objects
    """
    with st.spinner("Processing files..."):
        total_chunks = 0
        
        for file in files:
            try:
                # Step 1: Validate file
                if not is_valid_file_extension(file.name):
                    st.error(f"❌ Unsupported file: {file.name}")
                    continue
                
                # Step 2: Load document
                st.info(f"📄 Loading: {file.name}")
                documents = load_document(file=file)
                
                # Step 3: Split into chunks
                chunks = split_documents(documents)
                chunk_info = get_chunk_info(chunks)
                
                # Step 4: Add to vector store
                added = add_documents_to_store(chunks)
                total_chunks += added
                
                st.success(f"✅ {file.name}: {added} chunks added")
                
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")
        
        if total_chunks > 0:
            st.success(f"🎉 Total: {total_chunks} chunks added to knowledge base!")
            st.session_state.knowledge_base_ready = True
            st.rerun()


def process_url(url: str):
    """
    Process URL and add scraped content to knowledge base.
    
    Args:
        url: URL to scrape
    """
    # Validate URL
    if not is_valid_url(url):
        st.error("❌ Invalid URL. Please enter a valid http/https URL.")
        return
    
    with st.spinner(f"Scraping: {url}"):
        try:
            # Step 1: Load from URL
            documents = load_document(url=url)
            
            # Step 2: Split into chunks
            chunks = split_documents(documents)
            chunk_info = get_chunk_info(chunks)
            
            # Step 3: Add to vector store
            added = add_documents_to_store(chunks)
            
            st.success(f"✅ Added {added} chunks from URL!")
            st.session_state.knowledge_base_ready = True
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error scraping URL: {str(e)}")


def clear_knowledge_base() -> bool:
    """Clear all documents from knowledge base."""
    try:
        clear_vector_store()
        st.session_state.messages = []  # Clear chat history too
        return True
    except Exception as e:
        st.error(f"Error clearing knowledge base: {str(e)}")
        return False


# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

def render_chat_interface():
    """Render the main chat interface."""
    
    # Header
    st.title("🧠 RAG Knowledge Assistant")
    st.markdown("Ask questions about your uploaded documents and web content.")
    
    # Check for API key
    if not GROQ_API_KEY:
        st.error(
            "⚠️ **GROQ_API_KEY not found!**\n\n"
            "Please create a `.env` file in the project root with:\n"
            "```\nGROQ_API_KEY=your_api_key_here\n```\n\n"
            "Get your free API key at: https://console.groq.com"
        )
        return
    
    # Check if knowledge base has documents
    try:
        stats = get_collection_stats()
        if stats["total_documents"] == 0:
            st.info(
                "📚 **No documents in knowledge base yet!**\n\n"
                "👈 Use the sidebar to:\n"
                "- Upload PDF or TXT files\n"
                "- Scrape content from URLs"
            )
            return
    except Exception:
        st.info("👈 Upload documents to get started!")
        return
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['source']}")
                        st.caption(source['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get answer from RAG chain
                    result = ask_question(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:** {source['source']}")
                                st.caption(source['content'])
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main chat interface
    render_chat_interface()


# Run the app
if __name__ == "__main__":
    main()
