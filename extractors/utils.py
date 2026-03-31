import re
from typing import List, Dict

def validate_verhoeff(number: str) -> bool:
    """Validate 12-digit Aadhaar number using Verhoeff algorithm."""
    if not number or len(number) != 12 or not number.isdigit():
        return False
    
    # Verhoeff tables
    d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    p = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]
    
    c = 0
    # Process from right to left (Verhoeff is sensitive to position)
    for i, num in enumerate(number[::-1]):
        c = d[c][p[i % 8][int(num)]]
    
    return c == 0


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
            # Filter out all-uppercase garbage fragments with poor vowel ratio
            # e.g. "JTUTR", "HRG", "BKLT" — not real name words
            if word.isupper() and len(word) >= 3:
                vowels = sum(1 for c in word if c in "AEIOUY")
                if vowels / len(word) < 0.25:
                    continue
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
    # Real names have at least ~25-30% vowels. Garbage fragments like "HRG" have 0%.
    # Increased from 20% to filter out more OCR noise
    return ratio >= 0.25


def is_likely_garbage(s: str) -> bool:
    """Check if a string is likely OCR garbage or vertical text fragment."""
    if not s:
        return True
    
    words = s.split()
    if not words:
        return True

    # If it's just a bunch of random caps like "IBJ" or "FABL" that aren't common names
    garbage_patterns = [
        r'\b(DOW|FABL|IBJ|HBI|TIO|LLC|PAE|HRG|HRCR|HRCOR|FLSH|JHTST|USIT|JHT|USI|HTS|TST)\b',  # Specific OCR noise
    ]
    
    # Consonants only check (excluding Y which is common in names)
    consonant_block = r'^[BCDFGHJKLMNPQRSTVWXZ]{2,}$'
    
    # Pattern for words with too many consecutive consonants (> 3 in a row is unusual)
    excessive_consonants = r'[BCDFGHJKLMNPQRSTVWXZ]{4,}'
    
    for w in words:
        w_up = w.upper()
        # 1. Check specific noise patterns
        for p in garbage_patterns:
            if re.search(p, w_up):
                return True
        # 2. Check if a word is purely consonants and at least 2 chars (likely misread)
        if len(w) >= 2 and re.match(consonant_block, w_up):
            return True
        # 3. Check for mixed alphanumeric word which is rare in names
        if any(c.isdigit() for c in w) and any(c.isalpha() for c in w):
            return True
        # 4. Check for excessive consecutive consonants (SHRG, HRCOR pattern)
        if re.search(excessive_consonants, w_up):
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


def _safe_get(data, *keys):
    """Safely traverse nested dicts/lists. Returns None if any key is missing or types mismatch."""
    current = data
    for key in keys:
        if current is None:
            return None
        if isinstance(key, int):
            if isinstance(current, (list, tuple)) and 0 <= key < len(current):
                current = current[key]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _map_dl_fields(merged, ocr_data):
    """Map DL-specific OCR fields into the merged response.

    Mapping: ocrData.dlNo → dl_number, name → name, dob → dob,
             relationName → father (if present), address → address.
    """
    field_map = {
        "dlNo": "dl_number",
        "name": "name",
        "dob": "dob",
        "address": "address",
    }

    for ocr_key, response_key in field_map.items():
        ocr_field = _safe_get(ocr_data, ocr_key) or {}
        ocr_value = _safe_get(ocr_field, "value")

        if ocr_value is not None and ocr_value != "":
            if not isinstance(merged.get(response_key), dict):
                merged[response_key] = {"value": "", "confidence": 0}
            merged[response_key]["value"] = ocr_value
            ocr_conf = _safe_get(ocr_field, "confidence")
            if ocr_conf is not None:
                merged[response_key]["confidence"] = ocr_conf

    relation_field = _safe_get(ocr_data, "relationName") or {}
    relation_value = _safe_get(relation_field, "value")
    if relation_value is not None and relation_value != "":
        if not isinstance(merged.get("father"), dict):
            merged["father"] = {"value": "", "confidence": 0}
        merged["father"]["value"] = relation_value
        relation_conf = _safe_get(relation_field, "confidence")
        if relation_conf is not None:
            merged["father"]["confidence"] = relation_conf


def mergeOcrIntoResponse(existingResponse, ocrPayload, isDocumentUploaded):
    """Merge an existing response packet with an OCR payload.

    Rules:
    - Preserves the existing schema and all original values by default.
    - Copies ``data.result.documents[0].additionalDetails`` from *ocrPayload*
      into the merged result without modifying keys or structure.
    - Sets ``additionalDetails.faceDetected = True`` only when
      *isDocumentUploaded* is truthy **and** a face/image field is present
      in the OCR payload document; otherwise ``False``.
    - If ``documents[0].ocrData.address.value`` exists, overrides the
      response ``address.value`` and ``address.confidence`` with OCR values.
    - When ``documentType == "DL"``, maps DL-specific OCR fields
      (dlNo, name, dob, relationName, address) into the response.
    - All null / missing fields are handled safely without crashing.

    Returns the final merged dict.
    """
    import copy

    merged = copy.deepcopy(existingResponse) if existingResponse else {}

    if not ocrPayload:
        merged.setdefault("additionalDetails", {"faceDetected": False})
        return merged

    document = _safe_get(ocrPayload, "data", "result", "documents", 0)

    if document is None:
        merged.setdefault("additionalDetails", {"faceDetected": False})
        return merged

    ocr_data = _safe_get(document, "ocrData") or {}
    document_type = _safe_get(document, "documentType") or ""

    # --- additionalDetails (preserve structure exactly) ---
    additional_details = copy.deepcopy(_safe_get(document, "additionalDetails") or {})

    face_detected = False
    if isDocumentUploaded:
        face_image = (
            _safe_get(document, "faceImage")
            or _safe_get(document, "image")
            or _safe_get(document, "photo")
            or _safe_get(additional_details, "faceImage")
            or _safe_get(additional_details, "image")
            or _safe_get(ocr_data, "photo", "value")
            or _safe_get(ocr_data, "image", "value")
        )
        if face_image:
            face_detected = True

    additional_details["faceDetected"] = face_detected
    merged["additionalDetails"] = additional_details

    # --- address override ---
    ocr_address = _safe_get(ocr_data, "address") or {}
    ocr_address_value = _safe_get(ocr_address, "value")

    if ocr_address_value is not None and ocr_address_value != "":
        if not isinstance(merged.get("address"), dict):
            merged["address"] = {"value": "", "confidence": 0}
        merged["address"]["value"] = ocr_address_value
        ocr_address_conf = _safe_get(ocr_address, "confidence")
        if ocr_address_conf is not None:
            merged["address"]["confidence"] = ocr_address_conf

    # --- DL-specific mapping ---
    if document_type == "DL":
        _map_dl_fields(merged, ocr_data)

    return merged


INDIAN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "orissa", "punjab", "rajasthan",
    "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "new delhi", "chandigarh",
    "puducherry", "pondicherry", "jammu and kashmir", "jammu & kashmir",
    "ladakh", "andaman and nicobar", "dadra and nagar haveli",
    "daman and diu", "lakshadweep",
]


def split_address(address_str: str) -> dict:
    """Split an Indian address string into structured components.

    Returns a dict with keys: building, city, district, pin, floor, house,
    locality, state, street, complex, landmark, untagged.
    All values are strings (empty string when not detected).
    Pure function - never raises.
    """
    result = {
        "building": "",
        "city": "",
        "district": "",
        "pin": "",
        "floor": "",
        "house": "",
        "locality": "",
        "state": "",
        "street": "",
        "complex": "",
        "landmark": "",
        "untagged": "",
    }

    if not address_str or not address_str.strip():
        return result

    addr = address_str.strip()

    # --- PIN code (6-digit Indian postal code) ---
    pin_match = re.search(r'\b(\d{6})\b', addr)
    if pin_match:
        result["pin"] = pin_match.group(1)
        addr = addr[:pin_match.start()] + addr[pin_match.end():]

    # --- State ---
    addr_lower = addr.lower()
    for state in sorted(INDIAN_STATES, key=len, reverse=True):
        pattern = r'\b' + re.escape(state) + r'\b'
        m = re.search(pattern, addr_lower)
        if m:
            result["state"] = addr[m.start():m.end()].strip()
            addr = addr[:m.start()] + addr[m.end():]
            break

    # --- District ---
    dist_match = re.search(r'\b(?:dist(?:rict)?|distt?)[\s.:~-]*([A-Za-z\s]+)', addr, re.I)
    if dist_match:
        result["district"] = dist_match.group(1).strip().strip(',').strip()
        addr = addr[:dist_match.start()] + addr[dist_match.end():]

    # Work with comma-separated parts for the remaining fields
    parts = [p.strip() for p in re.split(r'[,\n]+', addr) if p.strip()]
    remaining = []

    for part in parts:
        part_lower = part.lower().strip()

        # --- House number (e.g., "H.No 123", "No. 45", "1-2-3/4", "#123") ---
        if not result["house"] and re.match(
            r'^(?:h\.?\s*no\.?\s*|no\.?\s*|#\s*)?\d[\d\-/\\a-zA-Z]*$', part_lower
        ):
            result["house"] = part.strip()
            continue

        # --- Floor ---
        if not result["floor"] and re.search(
            r'\b(\d+\s*(?:st|nd|rd|th)\s*floor|ground\s*floor|basement)\b', part_lower
        ):
            result["floor"] = part.strip()
            continue

        # --- Landmark ---
        if not result["landmark"] and re.match(
            r'\b(?:near|behind|opp(?:osite)?|beside|adjacent|next\s+to|in\s+front\s+of)\b',
            part_lower
        ):
            result["landmark"] = part.strip()
            continue

        # --- Street ---
        if not result["street"] and re.search(
            r'\b(?:road|rd|street|st|lane|marg|path|gali|gully|chowk|cross)\b', part_lower
        ):
            result["street"] = part.strip()
            continue

        # --- Building ---
        if not result["building"] and re.search(
            r'\b(?:building|bldg|tower|plaza|bhawan|bhavan|mansion|house)\b', part_lower
        ):
            result["building"] = part.strip()
            continue

        # --- Complex (apartment/society/complex) ---
        if not result["complex"] and re.search(
            r'\b(?:apartment|apt|society|complex|residency|enclave|heights|villa|park)\b',
            part_lower
        ):
            result["complex"] = part.strip()
            continue

        # --- Locality (nagar/colony/sector/ward/area/mohalla etc.) ---
        if not result["locality"] and re.search(
            r'\b(?:nagar|colony|sector|ward|area|mohalla|mohala|puram|puri|bagh|'
            r'vihar|kunj|block|phase|extension|extn|layout|scheme|circle|'
            r'town|village|vill|post|po|tehsil|taluka|mandal|hobli)\b',
            part_lower
        ):
            result["locality"] = part.strip()
            continue

        # --- City (capitalized word that doesn't match other patterns) ---
        # Assign first unmatched part with 2+ alpha words or a single capitalized word as city
        if not result["city"] and re.match(r'^[A-Za-z][A-Za-z\s]+$', part.strip()):
            result["city"] = part.strip()
            continue

        remaining.append(part.strip())

    # Anything left goes to untagged
    untagged = ", ".join(remaining).strip(", ").strip()
    result["untagged"] = untagged

    # Clean up trailing/leading commas and whitespace in all fields
    for key in result:
        result[key] = re.sub(r'^[\s,]+|[\s,]+$', '', result[key])

    return result


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

