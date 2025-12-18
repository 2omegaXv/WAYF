"""
Utility functions for data processing.
"""

import hashlib
import re
from typing import List, Dict, Set
import unicodedata


def hash_text(text: str) -> str:
    """Generate SHA256 hash for text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_translation(text: str) -> str:
    """Clean translation output."""
    # Strip whitespace and quotes
    text = text.strip().strip('"\'')
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove @handles
    text = re.sub(r'@\w+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def check_cjk_ratio(text: str, min_ratio: float = 0.7) -> bool:
    """Check if text has sufficient CJK characters."""
    if not text:
        return False
    
    # Count CJK characters
    cjk_count = sum(1 for c in text if is_cjk_char(c))
    
    # Count non-whitespace characters
    total_count = sum(1 for c in text if not c.isspace())
    
    if total_count == 0:
        return False
    
    return (cjk_count / total_count) >= min_ratio


def is_cjk_char(char: str) -> bool:
    """Check if a character is CJK."""
    code = ord(char)
    
    # CJK Unified Ideographs
    if 0x4E00 <= code <= 0x9FFF:
        return True
    
    # CJK Extension A
    if 0x3400 <= code <= 0x4DBF:
        return True
    
    # CJK Extension B
    if 0x20000 <= code <= 0x2A6DF:
        return True
    
    # CJK Extension C-F
    if 0x2A700 <= code <= 0x2B73F:
        return True
    if 0x2B740 <= code <= 0x2B81F:
        return True
    if 0x2B820 <= code <= 0x2CEAF:
        return True
    if 0x2CEB0 <= code <= 0x2EBEF:
        return True
    
    return False


def remove_duplicates(items: List[Dict], key: str = 'text') -> List[Dict]:
    """Remove exact duplicates from list of items."""
    seen: Set[str] = set()
    unique_items = []
    
    for item in items:
        text = item.get(key, '')
        text_hash = hash_text(text)
        
        if text_hash not in seen:
            seen.add(text_hash)
            unique_items.append(item)
    
    return unique_items


def split_by_length(texts: List[str], bin_size: int = 50) -> Dict[int, List[str]]:
    """Split texts into length bins."""
    bins = {}
    
    for text in texts:
        bin_idx = len(text) // bin_size
        if bin_idx not in bins:
            bins[bin_idx] = []
        bins[bin_idx].append(text)
    
    return bins


def normalize_chinese_punctuation(text: str) -> str:
    """Normalize punctuation to Chinese conventions."""
    replacements = {
        ',': '，',
        '.': '。',
        '!': '！',
        '?': '？',
        ':': '：',
        ';': '；',
        '(': '（',
        ')': '）',
        '"': '"',
        "'": ''',
    }
    
    for eng, chn in replacements.items():
        text = text.replace(eng, chn)
    
    return text


def extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response that might have extra text."""
    # Try to find JSON block
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    
    if json_match:
        return json_match.group(0)
    
    return response
