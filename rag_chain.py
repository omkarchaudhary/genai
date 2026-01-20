"""
RAG Chain Module
================
Creates the RAG (Retrieval-Augmented Generation) chain
using LangChain with Groq LLM.

RAG Flow:
    User Question → Retrieve Context → Generate Answer → Return with Sources

Uses LCEL (LangChain Expression Language) - the modern recommended approach.
"""

from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import GROQ_API_KEY, GROQ_MODEL_NAME, LLM_TEMPERATURE


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

# Custom prompt template for RAG
# This guides the LLM to use the context effectively
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on the provided context.

INSTRUCTIONS:
1. Use ONLY the information from the context below to answer the question
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question based on the provided documents."
3. Be concise and accurate
4. If relevant, mention which source the information came from

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def get_llm() -> ChatGroq:
    """
    Initialize Groq LLM.
    
    Returns:
        ChatGroq instance configured for RAG
        
    Why Groq:
        - Very fast inference (uses custom hardware)
        - Free tier available
        - Supports powerful open models like Llama 3
    """
    # Check for API key
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found! "
            "Please set it in your .env file: GROQ_API_KEY=your_key_here"
        )
    
    # Initialize Groq LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=2048  # Max response length
    )
    
    return llm


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Create the RAG prompt template.
    
    Returns:
        ChatPromptTemplate for RAG chain
    """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    return prompt


def format_docs(docs: List) -> str:
    """
    Format retrieved documents into a single string for context.
    
    Args:
        docs: List of Document objects
        
    Returns:
        Formatted string with all document contents
    """
    return "\n\n".join(doc.page_content for doc in docs)


def ask_question(question: str) -> Dict[str, Any]:
    """
    Ask a question and get an answer from the RAG system.
    
    Uses LCEL (LangChain Expression Language) for the chain:
    Question → Retrieve Documents → Format Context → LLM → Parse Output
    
    Args:
        question: User's question string
        
    Returns:
        Dictionary with:
            - answer: The generated answer
            - sources: List of source documents used
            
    Usage:
        result = ask_question("What is machine learning?")
        print(result["answer"])
        print(result["sources"])
    """
    # Import here to avoid circular imports
    from vector_store import get_retriever
    
    # Get components
    llm = get_llm()
    retriever = get_retriever()
    prompt = get_rag_prompt()
    
    # First, retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    
    # Build the LCEL chain
    # RunnablePassthrough passes the input through unchanged
    rag_chain = (
        {
            "context": lambda x: format_docs(retrieved_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Run the chain
    answer = rag_chain.invoke(question)
    
    # Format sources for display
    sources = []
    for doc in retrieved_docs:
        source_info = {
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "type": doc.metadata.get("type", "Unknown")
        }
        sources.append(source_info)
    
    return {
        "answer": answer,
        "sources": sources
    }
