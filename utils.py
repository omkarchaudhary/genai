"""
Utility Functions Module
========================
Helper functions for validation and common operations.
"""

import re
from urllib.parse import urlparse
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB


def is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL.
    
    Args:
        url: String to validate as URL
        
    Returns:
        True if valid URL, False otherwise
        
    Example:
        >>> is_valid_url("https://example.com")
        True
        >>> is_valid_url("not a url")
        False
    """
    try:
        # Parse the URL and check for scheme (http/https) and netloc (domain)
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except Exception:
        return False


def is_valid_file_extension(filename: str) -> bool:
    """
    Check if file has an allowed extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if extension is allowed, False otherwise
    """
    # Get file extension (e.g., ".pdf" from "document.pdf")
    extension = "." + filename.split(".")[-1].lower() if "." in filename else ""
    return extension in ALLOWED_EXTENSIONS


def is_valid_file_size(file_size_bytes: int) -> bool:
    """
    Check if file size is within allowed limit.
    
    Args:
        file_size_bytes: Size of file in bytes
        
    Returns:
        True if within limit, False otherwise
    """
    # Convert bytes to MB and compare with limit
    file_size_mb = file_size_bytes / (1024 * 1024)
    return file_size_mb <= MAX_FILE_SIZE_MB


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text with normalized whitespace
    """
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def get_file_extension(filename: str) -> str:
    """
    Extract file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (e.g., ".pdf")
    """
    if "." in filename:
        return "." + filename.split(".")[-1].lower()
    return ""
