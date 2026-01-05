"""
Utility functions for OCR extraction
Common helper functions used across document extractors.
"""

import re
from typing import List, Dict

def is_uidai_boilerplate(s: str) -> bool:
    if not s:
        return True

    s_low = s.lower()

    uidai_phrases = [
        "unique identification",
        "identification authority",
        "authority of india",
        "aadhaar is a proof",
        "proof of identity",
        "authenticate online",
        "electronically generated",
        "information",
        "government of india",
        "uidai",
        "identification",
        "authority",
        "government",
        "unique",
        "enrolment"
    ]

    return any(p in s_low for p in uidai_phrases)


def looks_like_uidai_text(s: str) -> bool:
    s = s.lower().replace(" ", "")
    bad_tokens = [
        "unique", "identification", "authority",
        "government", "india", "aadhaar", "uidai"
    ]
    return any(t in s for t in bad_tokens)


def is_english_text(text: str) -> bool:
    """Check if text is predominantly English (ASCII)."""
    if not text:
        return False
    
    # Calculate ratio of ASCII letters to total length
    # Remove spaces for accurate density check
    t_clean = text.replace(' ', '')
    if not t_clean:
        return True
        
    ascii_count = sum(1 for c in t_clean if c.isascii() and c.isalpha())
    ratio = ascii_count / len(t_clean)
    
    # Allow if at least 50% are ASCII letters
    # This handles mostly English text with some punctuation/noise
    return ratio >= 0.5


def merge_results(results: List[Dict]) -> List[Dict]:
    """Merge OCR results by y-position, keeping all texts on same line."""
    if not results:
        return []

    # Group by y-position bucket
    buckets = {}
    for r in results:
        b = int(round(r['y'] / 5.0) * 5)
        if b not in buckets:
            buckets[b] = []
        buckets[b].append(r)

    # For each bucket, sort by x-position and merge texts on same line
    ordered = []
    for b in sorted(buckets.keys()):
        items = buckets[b]
        # Sort by x-position (left to right)
        items_sorted = sorted(items, key=lambda x: x.get('x', 0))
        
        # Merge items on same line
        merged_text = ' '.join([item['text'] for item in items_sorted])
        avg_conf = sum(item['conf'] for item in items_sorted) / len(items_sorted)
        
        ordered.append({
            'text': merged_text,
            'conf': avg_conf,
            'y': b
        })

    # Merge tiny fragments
    merged = []
    buffer = None
    for rec in ordered:
        ln = rec['text']
        if buffer and len(buffer['text']) < 8 and re.match(r'^[A-Za-z]+$', ln.replace(' ', '')):
            buffer['text'] = buffer['text'] + ' ' + ln
            buffer['conf'] = max(buffer.get('conf', 0.0), rec.get('conf', 0.0))
        else:
            if buffer:
                merged.append(buffer)
            buffer = dict(text=ln, conf=rec.get('conf', 0.0), y=rec.get('y', 0))
    if buffer:
        merged.append(buffer)

    return merged


def nearest_line(lines: List[str], index: int, direction: int = -1) -> str:
    """Get nearest non-empty line."""
    i = index + direction
    while 0 <= i < len(lines):
        if lines[i].strip():
            return lines[i].strip()
        i += direction
    return ""


def clean_ocr_garbage(s: str) -> str:
    """Remove OCR garbage from name strings."""
    if not s:
        return s
    
    # First, try to extract only English/ASCII words if it's a mixed string
    # This helps when Hindi and English are merged on the same line
    s = extract_english_only(s)
    
    # Split into words
    words = s.split()
    cleaned_words = []
    for word in words:
        # Remove any non-alphabetic characters from start/end
        word = word.strip('.,-:;()[]{}')
        if not word:
            continue
        # Skip very short uppercase sequences that look like OCR garbage
        if len(word) <= 2 and word.isupper() and not any(c.isalpha() for c in word):
            continue
        # Skip words that contain both digits and letters usually OCR noise
        if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
            continue
            
        # Keep words that look like parts of a name
        if (word[0].isupper() and word.isalpha()) or (word.isupper() and len(word) >= 2 and word.isalpha()):
            cleaned_words.append(word)
    return ' '.join(cleaned_words)


def extract_english_only(s: str) -> str:
    """Extract only English/ASCII words from a potentially mixed-script string."""
    if not s:
        return ""
    # Keep only ASCII characters and spaces
    ascii_only = "".join(c if ord(c) < 128 else " " for c in s)
    # Clean up multiple spaces
    return re.sub(r'\s+', ' ', ascii_only).strip()


def is_valid_name(s: str, strict: bool = True) -> bool:
    """Check if string looks like a valid person name."""
    if not s or len(s) < 3:
        return False
    # Clean first
    cleaned = clean_ocr_garbage(s)
    if not cleaned or len(cleaned) < 3:
        return False
    
    # Check for likely garbage fragments
    if is_likely_garbage(cleaned):
        return False
        
    # Standard name check: must have some vowels
    if not has_reasonable_vowel_ratio(cleaned):
        return False

    words = cleaned.split()
    # Should have 2-4 words
    if len(words) < 2 or len(words) > 4:
        return False
    
    if strict:
        # Strict: Each word should be at least 3 characters and start with uppercase
        if not all(len(w) >= 3 and (w[0].isupper() or w.isupper()) for w in words):
            return False
    else:
        # Less strict: Each word should be at least 2 characters and start with uppercase or be all caps
        if not all(len(w) >= 2 and (w[0].isupper() or w.isupper()) for w in words):
            return False
    return True


def has_reasonable_vowel_ratio(s: str) -> bool:
    """Names usually have a decent mix of vowels (A, E, I, O, U)."""
    if not s: return False
    s_up = s.upper()
    total_letters = sum(1 for c in s_up if c.isalpha())
    if total_letters == 0: return False
    vowels = sum(1 for c in s_up if c in "AEIOUY") # Y is often vowel-ish in names
    ratio = vowels / total_letters
    # Most names have > 20% vowels. Garbage fragments like "HRG" or "JHTST" have 0%.
    return ratio >= 0.20


def is_likely_garbage(s: str) -> bool:
    """Check if a string is likely OCR garbage or vertical text fragment."""
    if not s:
        return True
    
    words = s.split()
    if not words:
        return True

    # If it's just a bunch of random caps like "IBJ" or "FABL" that aren't common names
    garbage_patterns = [
        r'\b(DOW|FABL|IBJ|HBI|TIO|LLC|PAE|HRG|HRCR|FLSH|JHTST|USIT|JHT|USI|HTS|TST)\b',  # Specific OCR noise
    ]
    
    # Consonants only check (excluding Y which is common in names)
    consonant_block = r'^[BCDFGHJKLMNPQRSTVWXZ]{2,}$'
    
    for w in words:
        w_up = w.upper()
        # 1. Check specific noise
        for p in garbage_patterns:
            if re.search(p, w_up):
                return True
        # 2. Check if a word is purely consonants and at least 2 chars (likely misread)
        if len(w) >= 2 and re.match(consonant_block, w_up):
            return True
        # 3. Check for mixed alphanumeric word which is rare in names
        if any(c.isdigit() for c in w) and any(c.isalpha() for c in w):
            return True

    return False


def looks_like_address(s: str) -> bool:
    """Check if a string looks like an address."""
    # More comprehensive address patterns and noise to exclude from names
    address_like_patterns = r'\b(STOP|BUS|ROAD|LANE|STREET|NAGAR|COLONY|SECTOR|POST\s*OFFICE|HOUSE|FLAT|VILLAGE|NEAR|BEHIND|OPP|JUNCTION|DISTRICT|TALUKA|STATE|FLOOR|BLOCK|BUILDING|APARTMENT|S\s*/\s*O|D\s*/\s*O|W\s*/\s*O|C\s*/\s*O|ENROLMENT|ENROLLMENT|NO\.|NUMBER|DATE|DOWNLOAD|ISSUE|PRINT)\b'
    if re.search(address_like_patterns, s, re.I):
        return True
    # If it ends with PIN-like 6 digit or contains many commas, likely address
    if re.search(r'\b\d{6}\b', s):
        return True
    if s.count(',') >= 2:
        return True
    return False


def contains_devanagari(text: str) -> bool:
    """Check if text contains Devanagari (Hindi/Marathi) characters.
    
    Devanagari Unicode range: U+0900 to U+097F
    Used to detect Hindi/Marathi names on Aadhaar cards.
    """
    if not text:
        return False
    return any('\u0900' <= char <= '\u097F' for char in text)


def is_title_case_name(text: str) -> bool:
    """Check if text is a valid Title Case English name (2-4 words, no digits/addresses).
    
    Used for Aadhaar cards where English name appears after Devanagari name.
    Validates:
    - 2-4 words (e.g., "Nikky Bisen" or "Ayush Kumar Sharma")
    - Each word is Title Case: First letter uppercase, rest lowercase (e.g., "Nikky" not "NIKKY")
    - No digits
    - No address keywords
    """
    if not text:
        return False
    
    # Extract English only (handles mixed Hindi+English lines)
    english_text = extract_english_only(text).strip()
    
    if not english_text or len(english_text) < 3:
        return False
    
    # Must have 2-4 words
    words = english_text.split()
    if not (2 <= len(words) <= 4):
        return False
    
    # No digits
    if re.search(r'\d', english_text):
        return False
    
    # No address keywords
    if looks_like_address(english_text):
        return False
    
    # Each word should be Title Case (First letter uppercase, rest lowercase)
    # Pattern: Nikky, Bisen (not NIKKY, not nikky, not NiKkY)
    for word in words:
        # Allow single letter initials (e.g., "A" in "A Kumar")
        if len(word) == 1:
            if not word.isupper():
                return False
        else:
            # Title Case: First char upper, rest lower
            if not (word[0].isupper() and word[1:].islower()):
                return False
    
    return True

