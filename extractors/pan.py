"""
PAN Card Extraction Logic
Extracts information from PAN cards (C).
"""

import re
from typing import Dict, List
from .utils import nearest_line


# Keywords found in PAN card headers/labels — not valid names
_PAN_SKIP_RE = re.compile(
    r'\b(INCOME|TAX|DEPARTMENT|GOVERNMENT|GOVT|INDIA|PERMANENT|ACCOUNT|NUMBER|'
    r'SIGNATURE|DATE|BIRTH|FILE|CARD|PAN|NAME|FATHER|MOTHER)\b', re.I
)


def _is_valid_pan_name(s: str) -> bool:
    """Check if a string looks like a valid name on a PAN card.

    Rejects OCR garbage by checking:
    - Minimum length
    - Alphabetic characters only (letters + spaces)
    - Reasonable vowel ratio (real names have vowels)
    - No excessive consecutive consonants (OCR noise indicator)
    """
    if not s or len(s.strip()) < 4:
        return False
    s = s.strip()
    # Must be alphabetic with spaces/dots only
    if not re.match(r'^[A-Za-z\s\.]+$', s):
        return False
    # Analyse letters only
    letters = re.sub(r'[\s\.]', '', s).upper()
    if len(letters) < 3:
        return False
    # Vowel ratio — real names have at least ~20% vowels
    vowels = sum(1 for c in letters if c in 'AEIOU')
    if vowels / len(letters) < 0.20:
        return False
    # Excessive consecutive consonants → OCR garbage (e.g. "RTST", "HRCR")
    if re.search(r'[BCDFGHJKLMNPQRSTVWXZ]{4,}', letters):
        return False
    return True


def _has_good_confidence(text: str, conf_map: Dict[str, float], min_conf: float = 0.70) -> bool:
    """Check if text has acceptable OCR confidence. Returns True if no data available."""
    if not conf_map:
        return True
    return conf_map.get(text.strip(), 1.0) >= min_conf


def _get_name_candidates(lines: List[str], conf_map: Dict[str, float] = None) -> List[str]:
    """Collect lines that look like valid person names on a PAN card.

    PAN card layout (top to bottom):
      1. Header (INCOME TAX DEPARTMENT / GOVT. OF INDIA)
      2. Person's name  (ALL CAPS)
      3. Father's name  (ALL CAPS)
      4. DOB
      5. "Permanent Account Number"
      6. PAN number

    Returns candidate names in document order.
    """
    candidates = []
    for ln in lines:
        stripped = ln.strip()
        if len(stripped) < 4:
            continue
        # Skip lines with digits (dates, PAN, etc.)
        if re.search(r'\d', stripped):
            continue
        # Skip header / label keywords
        if _PAN_SKIP_RE.search(stripped):
            continue
        # Must be uppercase English text (5-50 chars)
        if not re.match(r'^[A-Z][A-Z\s\.]{3,49}$', stripped):
            continue
        # Reject low-confidence OCR results (likely garbage from multi-pass)
        if conf_map and stripped in conf_map and conf_map[stripped] < 0.70:
            continue
        if _is_valid_pan_name(stripped):
            candidates.append(stripped)
    return candidates


def extract_pan(lines: List[str], text: str, records: List[Dict] = None) -> Dict:
    """Extract data from PAN card."""
    obj = {
        "name": "",
        "father_name": "",
        "pan_number": "",
        "dob": ""
    }

    # Build confidence map from OCR records so we can reject low-confidence garbage
    conf_map = {}
    if records:
        for rec in records:
            t = rec.get('text', '').strip()
            if t:
                conf_map[t] = rec.get('conf', 0.0)

    # --- PAN number ---
    pan_match = re.search(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b', text)
    if pan_match:
        obj['pan_number'] = pan_match.group(1)

    # --- DOB ---
    dob_match = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})', text)
    if dob_match:
        obj['dob'] = dob_match.group(1)

    # --- Name extraction (keyword-based) ---
    for i, ln in enumerate(lines):
        up = ln.upper()

        # Look for "NAME" label (but not "FATHER'S NAME")
        if 'NAME' in up and 'FATHER' not in up and 'MOTHER' not in up and 'FILE' not in up:
            if re.match(r'^(NUMBER\s+)?NAME[:\s]*$', up.strip()):
                val = nearest_line(lines, i, 1)
                if val and not re.search(r'FATHER|DATE|BIRTH|PAN|NUMBER', val, re.I):
                    if _is_valid_pan_name(val) and _has_good_confidence(val, conf_map):
                        obj['name'] = val
                        break
            elif ':' in ln:
                val = ln.split(':')[-1].strip()
                if val and _is_valid_pan_name(val) and _has_good_confidence(val, conf_map):
                    obj['name'] = val
                    break

            if not obj['name']:
                val = nearest_line(lines, i, 1)
                if val and not re.search(r'FATHER|DATE|BIRTH|PAN|NUMBER', val, re.I):
                    if _is_valid_pan_name(val) and _has_good_confidence(val, conf_map):
                        obj['name'] = val
                        break

    # --- Father name extraction (keyword-based) ---
    for i, ln in enumerate(lines):
        up = ln.upper()

        if 'FATHER' in up or 'S/O' in up or 'SON OF' in up:
            below = nearest_line(lines, i, 1)
            if below and len(below) > 2 and not re.search(r'FATHER|NAME|MOTHER|DATE|BIRTH', below, re.I):
                if _is_valid_pan_name(below) and _has_good_confidence(below, conf_map):
                    obj['father_name'] = below
                    break
            above = nearest_line(lines, i, -1)
            if above and len(above) > 2:
                if _is_valid_pan_name(above) and _has_good_confidence(above, conf_map):
                    obj['father_name'] = above
                    break

    # --- Layout-based fallback ---
    # If keyword approach failed, use document layout:
    # first valid name candidate = person, second = father
    if not obj['name'] or not obj['father_name']:
        candidates = _get_name_candidates(lines, conf_map)

        if not obj['name'] and candidates:
            obj['name'] = candidates[0]

        if not obj['father_name'] and len(candidates) >= 2:
            # Pick the second candidate that isn't the same as name
            for c in candidates[1:]:
                if c != obj['name']:
                    obj['father_name'] = c
                    break

    return obj
