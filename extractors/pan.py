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


def _is_garbage_pan_name(name: str) -> bool:
    """Stricter per-word check to detect OCR garbage that passes basic validation.

    Hindi labels like "नाम" OCR'd with an English model produce garbage such as
    "HRCOR TERCOR".  These pass the overall vowel ratio check (30%) but contain
    impossible consonant clusters at the word level (e.g. "HRC" start).

    Rules checked per word:
    - Word must contain at least one vowel (A E I O U)
    - Word must not START with 3+ consonants unless it is a well-known cluster
      (SHR, STR, SPR, SCR, CHR, THR — common in Indian names like SHRUTI)
    """
    if not name:
        return False

    # Well-known 3-consonant starts found in real Indian names
    _ALLOWED_3 = ('SHR', 'STR', 'SPR', 'SCR', 'CHR', 'THR')
    _CONSONANTS = 'BCDFGHJKLMNPQRSTVWXYZ'

    for word in name.strip().split():
        w = word.upper().strip('.')
        if len(w) < 2:
            continue

        # No vowels at all → garbage (real names always have vowels)
        if not any(c in 'AEIOU' for c in w):
            return True

        # Starts with 3+ consonants that aren't a known cluster
        m = re.match(r'^([' + _CONSONANTS + r']+)', w)
        if m and len(m.group(1)) >= 3:
            if not m.group(1).startswith(_ALLOWED_3):
                return True

    return False


def _fix_garbage_pan_names(obj: Dict, lines: List[str], conf_map: Dict) -> None:
    """Post-processing: replace garbage names with clean candidates.

    If the extracted name or father_name looks like OCR garbage, find better
    candidates from the OCR lines.  Also fixes the case where the real person
    name was wrongly assigned to father_name.
    """
    name_dirty = obj.get('name', '') and _is_garbage_pan_name(obj['name'])
    father_dirty = obj.get('father_name', '') and _is_garbage_pan_name(obj['father_name'])

    if not name_dirty and not father_dirty:
        return  # nothing to fix

    # Collect clean candidates (reuses existing helper)
    all_candidates = _get_name_candidates(lines, conf_map)
    clean = [c for c in all_candidates if not _is_garbage_pan_name(c)]

    if name_dirty:
        obj['name'] = clean[0] if clean else ''

    if father_dirty:
        others = [c for c in clean if c != obj['name']]
        obj['father_name'] = others[0] if others else ''

    # If name and father ended up the same (e.g. father was the real name),
    # reassign father to the next available clean candidate
    if obj['name'] and obj['father_name'] and obj['name'].upper() == obj['father_name'].upper():
        others = [c for c in clean if c.upper() != obj['name'].upper()]
        obj['father_name'] = others[0] if others else ''


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

    # --- Post-processing: detect and replace garbage names ---
    # Hindi labels (e.g. "नाम") OCR'd with English model produce garbage like
    # "HRCOR TERCOR" that passes basic validation but has impossible consonant
    # clusters.  Detect these and replace with clean candidates.
    _fix_garbage_pan_names(obj, lines, conf_map)

    return obj
