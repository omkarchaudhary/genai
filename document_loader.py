"""
Document Loader Module
======================
Handles loading documents from various sources:
- PDF files
- Text files  
- Web URLs (web scraping)

Returns LangChain Document objects for further processing.
"""

import requests
from typing import List
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader
from utils import clean_text, is_valid_url


def load_pdf(file) -> List[Document]:
    """
    Load content from a PDF file.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        List of Document objects (one per page)
        
    How it works:
        1. Read PDF using PdfReader
        2. Extract text from each page
        3. Create Document with page metadata
    """
    documents = []
    
    # Create PDF reader from file bytes
    pdf_reader = PdfReader(file)
    
    # Loop through each page and extract text
    for page_num, page in enumerate(pdf_reader.pages):
        # Extract text content from page
        text = page.extract_text()
        
        if text and text.strip():  # Only add non-empty pages
            # Create Document with metadata about source
            doc = Document(
                page_content=clean_text(text),
                metadata={
                    "source": file.name,
                    "page": page_num + 1,  # 1-indexed for readability
                    "type": "pdf"
                }
            )
            documents.append(doc)
    
    return documents


def load_text_file(file) -> List[Document]:
    """
    Load content from a text file.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        List containing one Document object
    """
    # Read and decode file content
    content = file.read().decode("utf-8")
    
    # Create single Document for text file
    doc = Document(
        page_content=clean_text(content),
        metadata={
            "source": file.name,
            "type": "text"
        }
    )
    
    return [doc]


def load_from_url(url: str) -> List[Document]:
    """
    Load content from a web URL using web scraping.
    
    Args:
        url: Web URL to scrape content from
        
    Returns:
        List containing one Document object
        
    How it works:
        1. Fetch HTML content using requests
        2. Parse HTML with BeautifulSoup
        3. Extract main text content
        4. Create Document with URL as source
    """
    # Validate URL first
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")
    
    # Set headers to mimic browser (some sites block bots)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    # Fetch webpage content
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # Raise error for bad status codes
    
    # Parse HTML content
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Remove script and style elements (they contain no useful text)
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    
    # Extract text from remaining content
    text = soup.get_text(separator=" ")
    
    # Create Document with URL metadata
    doc = Document(
        page_content=clean_text(text),
        metadata={
            "source": url,
            "type": "web"
        }
    )
    
    return [doc]


def load_document(file=None, url: str = None) -> List[Document]:
    """
    Main function to load documents from file or URL.
    
    Args:
        file: Uploaded file object (optional)
        url: Web URL string (optional)
        
    Returns:
        List of Document objects
        
    Usage:
        # From file
        docs = load_document(file=uploaded_file)
        
        # From URL
        docs = load_document(url="https://example.com")
    """
    if file is not None:
        # Determine file type and use appropriate loader
        filename = file.name.lower()
        
        if filename.endswith(".pdf"):
            return load_pdf(file)
        elif filename.endswith(".txt"):
            return load_text_file(file)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    
    elif url is not None:
        return load_from_url(url)
    
    else:
        raise ValueError("Either file or url must be provided")
