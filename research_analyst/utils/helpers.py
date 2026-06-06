"""
Utility helper functions for the research analyst system.
"""

import hashlib
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse


def generate_id(prefix: str = "") -> str:
    """
    Generate a unique identifier.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique identifier string
    """
    unique_id = str(uuid.uuid4())
    return f"{prefix}_{unique_id}" if prefix else unique_id


def generate_hash(text: str) -> str:
    """
    Generate a hash for a given text.
    
    Args:
        text: Input text
        
    Returns:
        MD5 hash of the text
    """
    return hashlib.md5(text.encode()).hexdigest()


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
    
    return text.strip()


def truncate_text(text: str, max_length: int = 1000, 
                 suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: Full URL
        
    Returns:
        Domain name
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return ""


def is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid URL
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple Jaccard similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def deduplicate_by_key(items: List[Dict], key: str) -> List[Dict]:
    """
    Deduplicate a list of dictionaries by a specific key.
    
    Args:
        items: List of dictionaries
        key: Key to deduplicate by
        
    Returns:
        Deduplicated list
    """
    seen = set()
    result = []
    
    for item in items:
        if key in item and item[key] not in seen:
            seen.add(item[key])
            result.append(item)
    
    return result


def merge_dicts_with_priority(base: Dict, override: Dict) -> Dict:
    """
    Merge two dictionaries with override taking priority.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    result.update(override)
    return result


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_list(nested_list: List[List]) -> List:
    """
    Flatten a nested list.
    
    Args:
        nested_list: Nested list
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def safe_divide(numerator: float, denominator: float, 
               default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        Division result or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


def parse_time_range(time_str: str) -> Optional[Dict[str, datetime]]:
    """
    Parse natural language time ranges.
    
    Args:
        time_str: Time range string (e.g., "last week", "past month")
        
    Returns:
        Dictionary with start and end datetimes, or None
    """
    # This is a simplified version - in production you'd use dateparser
    import dateparser
    
    try:
        result = dateparser.parse(time_str, settings={'RELATIVE_BASE': datetime.utcnow()})
        if result:
            return {
                'start': result,
                'end': datetime.utcnow()
            }
    except:
        pass
    
    return None


def estimate_reading_time(text: str, words_per_minute: int = 200) -> float:
    """
    Estimate reading time for text in minutes.
    
    Args:
        text: Text to analyze
        words_per_minute: Average reading speed
        
    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    return word_count / words_per_minute


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract top keywords from text (simple implementation).
    
    Args:
        text: Input text
        top_n: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Remove common stop words (simplified)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'do', 'does', 'did', 'will', 'would', 'should', 'could',
                 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'}
    
    words = text.lower().split()
    
    # Filter stop words and count frequency
    word_freq = {}
    for word in words:
        word = re.sub(r'[^\w]', '', word)
        if word and word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def validate_json_structure(data: Any, required_keys: List[str]) -> bool:
    """
    Validate that a dictionary has required keys.
    
    Args:
        data: Data to validate
        required_keys: List of required keys
        
    Returns:
        True if all keys present
    """
    if not isinstance(data, dict):
        return False
    
    return all(key in data for key in required_keys)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Limit length
    return filename[:200]


def format_duration(milliseconds: float) -> str:
    """
    Format duration in milliseconds to human-readable string.
    
    Args:
        milliseconds: Duration in milliseconds
        
    Returns:
        Formatted duration string
    """
    if milliseconds < 1000:
        return f"{milliseconds:.0f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds/1000:.2f}s"
    else:
        return f"{milliseconds/60000:.2f}min"


def estimate_token_count(text: str) -> int:
    """
    Rough estimate of token count (1 token ≈ 4 characters for English).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def batch_items(items: List, batch_size: int):
    """
    Generator to yield batches of items.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
