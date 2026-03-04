"""
Aadhaar Card Extraction Logic - Optimized for Multiple Formats
Supports: PVC Card, e-Aadhaar, Long Aadhaar, Combined Front+Back images

Key Patterns:
- Hindi name followed by English name
- DOB pattern: "जन्म तिथि/DOB: DD/MM/YYYY" or "DOB: DD/MM/YYYY"
- Gender: "पुरुष/MALE" or "महिला/FEMALE"
- Aadhaar: 12 digits (XXXX XXXX XXXX)
- VID: 16 digits (XXXX XXXX XXXX XXXX)
"""

import re
from typing import Dict, List, Tuple, Optional
from .utils import (
    clean_ocr_garbage, is_valid_name, looks_like_address,
    contains_devanagari, is_title_case_name, is_uidai_boilerplate,
    looks_like_uidai_text, is_likely_garbage, has_reasonable_vowel_ratio,
    extract_english_only, validate_verhoeff
)


def is_aadhaar_back_side(lines: List[str], text: str) -> bool:
    """Detect if the image is the back side of an Aadhaar card.

    Back side characteristics:
    - Has "Address:" label or Hindi "पत्ता" label
    - Does NOT have DOB or gender (front side only fields)
    - Typically has bilingual text (English + Hindi/regional language) in two columns
    """
    # Back side indicators
    has_address_label = bool(
        re.search(r'\bAddress\s*:', text, re.I) or 'पत्ता' in text
    )

    # Front side indicators (should be absent on back)
    has_dob = bool(re.search(
        r'\bDOB\b|जन्म\s*तिथि|Year\s*of\s*Birth|Date\s*of\s*Birth',
        text, re.I
    ))
    has_gender = bool(re.search(
        r'\b(MALE|FEMALE|TRANSGENDER)\b|पुरुष|महिला',
        text, re.I
    ))

    # Back side: has address label but no DOB and no gender
    if has_address_label and not has_dob and not has_gender:
        print("[DEBUG] Detected Aadhaar BACK side")
        return True

    return False


def extract_aadhaar_back(lines: List[str], text: str, records: List[Dict]) -> Dict:
    """Extract data from Aadhaar card back side.

    Back side contains:
    - Address (in English + regional language)
    - Aadhaar number
    - Optionally VID

    Does NOT contain:
    - Name, DOB, Gender (these are on front side only)
    """
    obj = {
        "name": "",
        "gender": "",
        "dob": "",
        "aadhaar_number": "",
        "address": "",
        "vid": "",
        "father_name": "",
        "mother_name": "",
        "husband_name": "",
        "nationality": ""
    }

    print("\n[DEBUG] ===== Aadhaar BACK Side Extraction Started =====")
    print(f"[DEBUG] Total OCR lines: {len(lines)}")
    print("[DEBUG] OCR Lines:")
    for i, ln in enumerate(lines[:20]):
        print(f"[DEBUG]   Line {i}: '{ln}'")

    # Extract Aadhaar number (same logic as front side)
    obj['aadhaar_number'] = extract_aadhaar_number(text, lines)
    print(f"[DEBUG] Aadhaar Number: {obj['aadhaar_number']}")

    # Extract VID
    obj['vid'] = extract_vid(text, lines)
    print(f"[DEBUG] VID: {obj['vid']}")

    # Extract address (use existing logic - works well with clean left-column text)
    obj['address'] = extract_address(lines, "")
    print(f"[DEBUG] Address: {obj['address'][:100]}..." if len(obj['address']) > 100 else f"[DEBUG] Address: {obj['address']}")

    # Extract relation names (father/husband/mother) using shared logic
    father, mother, husband = extract_relation_names(lines, "")
    obj['father_name'] = father
    obj['mother_name'] = mother
    obj['husband_name'] = husband

    print(f"[DEBUG] Father: {obj['father_name']}")
    print(f"[DEBUG] Mother: {obj['mother_name']}")
    print(f"[DEBUG] Husband: {obj['husband_name']}")

    # Nationality - expanded detection
    if re.search(r'(GOVERNMENT OF INDIA|भारत सरकार|REPUBLIC OF INDIA|Unique Identification|UIDAI|Aadhar|Aadhaar)', text, re.I):
        obj['nationality'] = 'INDIAN'

    print("\n[DEBUG] ===== Back Side Extraction Results =====")
    for key, val in obj.items():
        print(f"[DEBUG]   {key}: '{val}'")
    print("[DEBUG] =========================================\n")

    return obj


def extract_aadhaar(lines: List[str], text: str, records: List[Dict]) -> Dict:
    """Extract data from Aadhaar card - handles all Indian Aadhaar formats."""

    # Check if this is the back side of an Aadhaar card
    if is_aadhaar_back_side(lines, text):
        return extract_aadhaar_back(lines, text, records)

    obj = {
        "name": "",
        "gender": "",
        "dob": "",
        "aadhaar_number": "",
        "address": "",
        "vid": "",
        "father_name": "",
        "mother_name": "",
        "husband_name": "",
        "nationality": ""
    }

    print("\n[DEBUG] ===== Aadhaar Extraction Started =====")
    print(f"[DEBUG] Total OCR lines: {len(lines)}")
    print(f"[DEBUG] Total OCR records: {len(records)}")
    print("[DEBUG] OCR Lines:")
    for i, ln in enumerate(lines[:20]):
        print(f"[DEBUG]   Line {i}: '{ln}'")
    print()

    # =========================================================================
    # STEP 1: Extract Aadhaar Number (12 digits)
    # =========================================================================
    print("[DEBUG] === Extracting Aadhaar Number ===")
    obj['aadhaar_number'] = extract_aadhaar_number(text, lines)
    print(f"[DEBUG] Aadhaar Number: {obj['aadhaar_number']}")

    # =========================================================================
    # STEP 2: Extract VID (16 digits)
    # =========================================================================
    print("\n[DEBUG] === Extracting VID ===")
    obj['vid'] = extract_vid(text, lines)
    print(f"[DEBUG] VID: {obj['vid']}")

    # =========================================================================
    # STEP 3: Extract DOB - Critical fix for wrong date extraction
    # =========================================================================
    print("\n[DEBUG] === Extracting DOB ===")
    obj['dob'], dob_line_idx = extract_dob(lines, records, text)
    print(f"[DEBUG] DOB: {obj['dob']} (found at line {dob_line_idx})")

    # =========================================================================
    # STEP 4: Extract Gender
    # =========================================================================
    print("\n[DEBUG] === Extracting Gender ===")
    obj['gender'], gender_line_idx = extract_gender(lines, records)
    print(f"[DEBUG] Gender: {obj['gender']} (found at line {gender_line_idx})")

    # =========================================================================
    # STEP 5: Extract Name - Multiple strategy approach
    # =========================================================================
    print("\n[DEBUG] === Extracting Name ===")
    obj['name'] = extract_name(lines, records, dob_line_idx, gender_line_idx)
    print(f"[DEBUG] Name: {obj['name']}")

    # =========================================================================
    # STEP 6: Extract Father/Mother/Husband Name
    # =========================================================================
    print("\n[DEBUG] === Extracting Relation Names ===")
    father, mother, husband = extract_relation_names(lines, obj['name'])
    obj['father_name'] = father
    obj['mother_name'] = mother
    obj['husband_name'] = husband
    print(f"[DEBUG] Father: {obj['father_name']}")
    print(f"[DEBUG] Mother: {obj['mother_name']}")
    print(f"[DEBUG] Husband: {obj['husband_name']}")

    # =========================================================================
    # STEP 7: Extract Address
    # =========================================================================
    print("\n[DEBUG] === Extracting Address ===")
    obj['address'] = extract_address(lines, obj['name'])
    print(f"[DEBUG] Address: {obj['address'][:100]}..." if len(obj['address']) > 100 else f"[DEBUG] Address: {obj['address']}")

    # =========================================================================
    # STEP 8: Extract Nationality
    # =========================================================================
    if re.search(r'(GOVERNMENT OF INDIA|भारत सरकार|REPUBLIC OF INDIA|Unique Identification|UIDAI|Aadhar|Aadhaar)', text, re.I):
        obj['nationality'] = 'INDIAN'

    print("\n[DEBUG] ===== Final Extraction Results =====")
    for key, val in obj.items():
        print(f"[DEBUG] {key}: '{val}'")
    print("[DEBUG] =====================================\n")

    return obj


def extract_aadhaar_number(text: str, lines: List[str]) -> str:
    """Extract 12-digit Aadhaar number with Verhoeff validation."""

    candidates = []

    # Pattern 1: Spaced format "1234 5678 9012"
    # Find all occurrences to handle duplicates or merged OCR noise
    spaced_matches = re.finditer(r'\b(\d{4}\s+\d{4}\s+\d{4})\b', text)
    for m in spaced_matches:
        num = re.sub(r'\s+', '', m.group(1))
        if validate_verhoeff(num):
            candidates.append(num)

    # Pattern 2: After "Aadhaar No" label
    label_matches = re.finditer(r'Aadhaar\s*No\.?\s*:?\s*(\d{4}\s*\d{4}\s*\d{4})', text, re.I)
    for m in label_matches:
        num = re.sub(r'\s+', '', m.group(1))
        if validate_verhoeff(num):
            candidates.append(num)

    # Pattern 3: Continuous 12 digits
    for ln in lines:
        for m in re.finditer(r'\b(\d{12})\b', ln):
            num = m.group(1)
            # Check context - shouldn't have more digits around
            full_match = re.search(r'\d{12,}', ln)
            if full_match and len(full_match.group(0)) == 12:
                if validate_verhoeff(num):
                    candidates.append(num)

    # Pattern 4: 4+4+4 with any small separator (fallback for messy OCR)
    # Search for local sequences only, NOT global concatenation
    merged_matches = re.finditer(r'\b(\d{4})[^\d\n]{0,2}(\d{4})[^\d\n]{0,2}(\d{4})\b', text)
    for m in merged_matches:
        num = m.group(1) + m.group(2) + m.group(3)
        if validate_verhoeff(num):
            candidates.append(num)

    # Pattern 5: Continuous 12 digits regardless of boundaries (last resort)
    for m in re.finditer(r'\d{12}', text):
        num = m.group(0)
        if validate_verhoeff(num):
            candidates.append(num)

    # Return the most frequent valid candidate or the last one found
    if candidates:
        from collections import Counter
        most_common = Counter(candidates).most_common(1)
        return most_common[0][0]

    # Fallback to Pattern 4 original regex if no Verhoeff match found (to keep behavior consistent)
    match = re.search(r'(\d{4})[^\d\n]{0,3}(\d{4})[^\d\n]{0,3}(\d{4})(?!\d)', text)
    if match:
        candidate = match.group(1) + match.group(2) + match.group(3)
        return candidate

    return ""


def extract_vid(text: str, lines: List[str]) -> str:
    """Extract 16-digit Virtual ID - only if explicitly labeled."""

    # VID should only be extracted if explicitly labeled with "VID" keyword
    # to avoid mistakenly capturing random 16-digit numbers
    
    # Pattern 1: With VID label (most reliable and only reliable pattern)
    match = re.search(r'VID\s*[:\s]*(\d{4}\s*\d{4}\s*\d{4}\s*\d{4})', text, re.I)
    if match:
        return re.sub(r'\s+', '', match.group(1))

    # Do NOT capture generic 16-digit patterns - they're likely not VID
    return ""


def extract_dob(lines: List[str], records: List[Dict], text: str) -> Tuple[str, int]:
    """
    Extract Date of Birth with improved accuracy.

    Key insight: DOB appears as "जन्म तिथि/DOB: DD/MM/YYYY" or "DOB: DD/MM/YYYY"
    Must exclude: Issue Date, Download Date, Print Date (usually on card edges)
    """

    # Keywords that indicate this is NOT the DOB
    exclude_keywords = ['issue', 'download', 'print', 'enrolment', 'enrollment', 'valid']

    # Priority 1: Look for explicit DOB label pattern
    # Pattern: "DOB: DD/MM/YYYY" or "जन्म तिथि/DOB: DD/MM/YYYY"
    for i, ln in enumerate(lines):
        ln_lower = ln.lower()

        # Skip if line contains exclude keywords
        if any(kw in ln_lower for kw in exclude_keywords):
            print(f"[DEBUG] Skipping line {i} (contains exclude keyword): '{ln}'")
            continue

        # Check for DOB pattern
        if re.search(r'\bDOB\b', ln, re.I) or 'तिथि' in ln or 'तारी' in ln:
            # Extract date from this line
            date_match = re.search(r'(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})', ln)
            if date_match:
                print(f"[DEBUG] Found DOB with label at line {i}: '{ln}'")
                return date_match.group(1).replace('-', '/').replace('.', '/'), i

    # Priority 2: Look for date on same line as gender (common in PVC cards)
    for i, ln in enumerate(lines):
        ln_lower = ln.lower()
        if any(kw in ln_lower for kw in exclude_keywords):
            continue

        # Check if line has both date and gender indicators
        if re.search(r'\b(MALE|FEMALE|पुरुष|महिला)\b', ln, re.I):
            # Check previous line for DOB
            if i > 0:
                prev_ln = lines[i-1]
                if not any(kw in prev_ln.lower() for kw in exclude_keywords):
                    date_match = re.search(r'(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})', prev_ln)
                    if date_match:
                        print(f"[DEBUG] Found DOB above gender line {i-1}: '{prev_ln}'")
                        return date_match.group(1).replace('-', '/').replace('.', '/'), i-1

    # Priority 3: Look for date in records with y-position context
    # DOB typically appears in the middle section, not at edges
    date_candidates = []
    for i, rec in enumerate(records):
        ln = rec.get('text', '')
        ln_lower = ln.lower()

        # Skip exclude keywords
        if any(kw in ln_lower for kw in exclude_keywords):
            continue

        date_match = re.search(r'(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})', ln)
        if date_match:
            y_pos = rec.get('y', 0)
            date_candidates.append({
                'date': date_match.group(1).replace('-', '/').replace('.', '/'),
                'line_idx': i,
                'y': y_pos,
                'has_dob_label': bool(re.search(r'\bDOB\b', ln, re.I)),
                'text': ln
            })

    # Prefer candidates with DOB label
    for cand in date_candidates:
        if cand['has_dob_label']:
            print(f"[DEBUG] Found DOB candidate with label: '{cand['text']}'")
            return cand['date'], cand['line_idx']

    # Otherwise take first candidate that's not an edge date
    if date_candidates:
        # Sort by whether it looks like a birth date (year between 1940-2020)
        for cand in date_candidates:
            year_match = re.search(r'(\d{4})$', cand['date'])
            if year_match:
                year = int(year_match.group(1))
                if 1940 <= year <= 2020:
                    print(f"[DEBUG] Found DOB by year validation: '{cand['text']}'")
                    return cand['date'], cand['line_idx']

    # Priority 4: Full text search with context validation
    # Find all dates in text and pick the one most likely to be DOB
    all_dates = re.findall(r'(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})', text)
    for date in all_dates:
        # Check context around the date
        idx = text.find(date)
        context = text[max(0, idx-30):idx+len(date)+10].lower()

        if any(kw in context for kw in exclude_keywords):
            continue

        # Validate year
        year = int(date[-4:])
        if 1940 <= year <= 2020:
            print(f"[DEBUG] Found DOB from full text search: '{date}'")
            return date.replace('-', '/').replace('.', '/'), -1

    print("[DEBUG] No DOB found!")
    return "", -1


def extract_gender(lines: List[str], records: List[Dict]) -> Tuple[str, int]:
    """Extract gender with line index."""

    for i, ln in enumerate(lines):
        # Check for gender keywords
        match = re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', ln, re.I)
        if match:
            return match.group(1).upper(), i

        # Check Hindi gender words
        if 'पुरुष' in ln:
            return 'MALE', i
        if 'महिला' in ln:
            return 'FEMALE', i

    return "", -1


def extract_name(lines: List[str], records: List[Dict], dob_idx: int, gender_idx: int) -> str:
    """
    Extract name using multiple strategies optimized for Indian Aadhaar formats.

    Key patterns:
    1. Hindi name (Devanagari) followed by English name on next line
    2. Name appears after "To" in e-Aadhaar
    3. Name appears before DOB/Gender lines
    4. ALL CAPS names like "NIKKY LAXMAN BISEN"
    """

    # Build list of boilerplate/institutional text to skip
    skip_patterns = [
        'government', 'india', 'aadhaar', 'uidai', 'unique', 'identification',
        'authority', 'enrolment', 'enrollment', 'issue', 'date', 'download',
        'print', 'proof', 'identity', 'citizenship', 'address', 'help@',
        'www.', 'mera', 'meri', 'pahchan', 'signature', 'valid',
        'mobile', 'phone', 'tel', 'contact', 'email', 'website', 'fax'
    ]

    def is_valid_name_candidate(text: str) -> bool:
        """Check if text could be a valid name."""
        if not text or len(text) < 3:
            return False

        text_lower = text.lower().replace(' ', '')

        # Skip if contains boilerplate
        if any(p in text_lower for p in skip_patterns):
            return False

        # Skip if contains digits
        if re.search(r'\d', text):
            return False

        # Skip if too short after cleaning
        words = text.split()
        if len(words) < 2:
            return False

        # Skip if looks like an address (contains locality keywords)
        address_keywords = ['nagar', 'village', 'road', 'street', 'lane', 'sector', 
                           'colony', 'house', 'flat', 'building', 'block']
        if any(kw in text_lower for kw in address_keywords):
            return False

        # Check vowel ratio (names have vowels)
        if not has_reasonable_vowel_ratio(text):
            return False

        # Skip garbage patterns
        if is_likely_garbage(text):
            return False

        return True

    def clean_name(text: str) -> str:
        """Clean and format name."""
        # Extract English only
        text = extract_english_only(text)

        # Remove common prefixes
        text = re.sub(r'^(To|TO)\s+', '', text)

        # Clean OCR garbage
        text = clean_ocr_garbage(text)

        # Remove duplicate consecutive words
        words = text.split()
        if words:
            cleaned = [words[0]]
            for w in words[1:]:
                if w.lower() != cleaned[-1].lower():
                    cleaned.append(w)
            text = ' '.join(cleaned)

        return text.strip()

    # -------------------------------------------------------------------------
    # Strategy 1: Find English name after Hindi (Devanagari) name
    # This is the most reliable pattern for Aadhaar cards
    # -------------------------------------------------------------------------
    print("[DEBUG] Strategy 1: Looking for English name after Hindi name...")
    for i in range(len(lines) - 1):
        if contains_devanagari(lines[i]):
            # Check next few lines for English name
            for j in range(i + 1, min(i + 3, len(lines))):
                next_ln = lines[j].strip()

                # Skip if this line also has Devanagari
                if contains_devanagari(next_ln):
                    continue

                # Skip if it's a DOB or gender line
                if re.search(r'\bDOB\b|जन्म|तिथि|\d{2}/\d{2}/\d{4}', next_ln, re.I):
                    continue
                if re.search(r'\b(MALE|FEMALE)\b', next_ln, re.I):
                    continue

                # Check if it looks like a name
                english_text = extract_english_only(next_ln)
                if is_valid_name_candidate(english_text):
                    name = clean_name(english_text)
                    if name and len(name) >= 5:
                        print(f"[DEBUG] Found name via Devanagari anchor: '{name}'")
                        return name

    # -------------------------------------------------------------------------
    # Strategy 2: Find name in "To" block (e-Aadhaar format)
    # Pattern: "To\n[Hindi name]\n[English name]\n[Address lines...]"
    # Key: Take FIRST valid name candidate after "To" label
    # -------------------------------------------------------------------------
    print("[DEBUG] Strategy 2: Looking for name in 'To' block...")
    for i, ln in enumerate(lines):
        if re.match(r'^\s*To\s*$', ln, re.I) or re.search(r'^To\s+[A-Z]', ln):
            # Check next few lines - take FIRST valid name, skip address lines
            for j in range(i + 1, min(i + 5, len(lines))):
                candidate = lines[j]

                # Remove "To" prefix if on same line
                candidate = re.sub(r'^To\s+', '', candidate, flags=re.I)

                # Skip Devanagari lines (often the Hindi name)
                if contains_devanagari(candidate):
                    continue

                # Extract English
                english_text = extract_english_only(candidate)
                
                # Skip if looks like address (has locality/area keywords)
                address_keywords = ['nagar', 'village', 'road', 'street', 'lane', 'sector',
                                   'colony', 'house', 'flat', 'building', 'block', 'side',
                                   'bapera', 'deosarra', 'taperi']
                if any(kw in english_text.lower() for kw in address_keywords):
                    print(f"[DEBUG] Skipping address-like line: '{english_text}'")
                    continue
                
                if is_valid_name_candidate(english_text):
                    name = clean_name(english_text)
                    if name and len(name) >= 5:
                        print(f"[DEBUG] Found name in 'To' block: '{name}'")
                        return name

    # -------------------------------------------------------------------------
    # Strategy 3: Find name before DOB line
    # -------------------------------------------------------------------------
    if dob_idx > 0:
        print(f"[DEBUG] Strategy 3: Looking for name before DOB (line {dob_idx})...")
        # Check lines above DOB
        for i in range(dob_idx - 1, max(0, dob_idx - 5), -1):
            candidate = lines[i]

            # Skip Devanagari lines
            if contains_devanagari(candidate) and not extract_english_only(candidate):
                continue

            english_text = extract_english_only(candidate)
            if is_valid_name_candidate(english_text):
                name = clean_name(english_text)
                if name and len(name) >= 5:
                    print(f"[DEBUG] Found name before DOB: '{name}'")
                    return name

    # -------------------------------------------------------------------------
    # Strategy 4: Find ALL CAPS name pattern
    # Pattern: "FIRSTNAME MIDDLENAME LASTNAME" (2-4 words, all caps)
    # -------------------------------------------------------------------------
    print("[DEBUG] Strategy 4: Looking for ALL CAPS name pattern...")
    all_caps_candidates = []
    for i, ln in enumerate(lines):
        # Look for ALL CAPS pattern with 2-4 words
        match = re.search(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b', ln)
        if match:
            candidate = match.group(1)
            if is_valid_name_candidate(candidate):
                name = clean_name(candidate)
                if name and len(name) >= 5:
                    all_caps_candidates.append((name, candidate))
    
    # Only accept all-caps if it looks really solid (length check)
    for name, original in all_caps_candidates:
        # Require at least 12 characters for all-caps names to avoid short garbage
        if len(name) >= 12:
            print(f"[DEBUG] Found solid ALL CAPS name: '{name}'")
            return name

    # -------------------------------------------------------------------------
    # Strategy 5: Find Title Case name pattern
    # Pattern: "Firstname Middlename Lastname"
    # -------------------------------------------------------------------------
    print("[DEBUG] Strategy 5: Looking for Title Case name pattern...")
    for i, ln in enumerate(lines):
        # Look for Title Case pattern
        match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', ln)
        if match:
            candidate = match.group(1)
            if is_valid_name_candidate(candidate):
                name = clean_name(candidate)
                if name and len(name) >= 5:
                    print(f"[DEBUG] Found Title Case name: '{name}'")
                    return name

    # -------------------------------------------------------------------------
    # Strategy 6: Use relation marker as anchor
    # Look for line above S/O, D/O, W/O, C/O
    # -------------------------------------------------------------------------
    print("[DEBUG] Strategy 6: Looking for name above relation markers...")
    for i, ln in enumerate(lines):
        if re.search(r'\b(S/O|D/O|W/O|C/O|SON OF|DAUGHTER OF|WIFE OF)\b', ln, re.I):
            if i > 0:
                candidate = lines[i - 1]
                english_text = extract_english_only(candidate)
                if is_valid_name_candidate(english_text):
                    name = clean_name(english_text)
                    if name and len(name) >= 5:
                        print(f"[DEBUG] Found name above relation marker: '{name}'")
                        return name

    # -------------------------------------------------------------------------
    # Strategy 7: Fallback - scan all records for best name candidate
    # -------------------------------------------------------------------------
    print("[DEBUG] Strategy 7: Scanning all records for name candidates...")
    candidates = []
    for i, rec in enumerate(records):
        text = rec.get('text', '')
        english_text = extract_english_only(text)

        if is_valid_name_candidate(english_text):
            name = clean_name(english_text)
            if name and len(name) >= 5:
                # Score based on position (prefer middle of document)
                y = rec.get('y', 0)
                # Names typically in upper-middle section
                candidates.append({
                    'name': name,
                    'y': y,
                    'conf': rec.get('conf', 0)
                })

    # Sort by confidence and pick best
    if candidates:
        candidates.sort(key=lambda x: x['conf'], reverse=True)
        name = candidates[0]['name']
        print(f"[DEBUG] Found name via fallback: '{name}'")
        return name

    print("[DEBUG] No name found!")
    return ""


def extract_relation_names(lines: List[str], person_name: str) -> Tuple[str, str, str]:
    """Extract father, mother, and husband names from relation markers."""

    father = ""
    mother = ""
    husband = ""

    def extract_name_after_marker(line: str, marker_pattern: str) -> str:
        """Extract name after a relation marker."""
        # More flexible marker matching for S/O, S:O, S.O etc.
        # Handle cases where S/O is separated by spaces
        flexible_marker = marker_pattern.replace('/', r'\s*[\/:]\s*')
        match = re.search(flexible_marker + r'[:\-]?\s*([A-Za-z\.\-\s]+)', line, re.I)
        if match:
            name = match.group(1).strip()
            # If name followed by address without comma, check for common address keywords
            # Use case-insensitive regex split for robust extraction
            addr_keywords = r'\b(House|Flat|Ward|Nagar|Village|Road|Street|Sector|Plot|Behind|Opposite|Near)\b'
            parts = re.split(addr_keywords, name, flags=re.I)
            name = parts[0].strip()
            
            name = extract_english_only(name)
            name = clean_ocr_garbage(name)

            if name and len(name) > 2:
                # Allow dots and hyphens in names
                if re.match(r'^[A-Za-z\.\-\s]+$', name):
                    return name
        return ""

    def looks_like_address(text: str) -> bool:
        """Check if text looks like address line."""
        address_keywords = ['nagar', 'village', 'road', 'street', 'lane', 'sector',
                           'colony', 'house', 'flat', 'building', 'block', 'side',
                           'behind', 'opposite', 'junction', 'district', 'state']
        text_lower = text.lower()
        return any(kw in text_lower for kw in address_keywords)

    for i, ln in enumerate(lines):
        # Son of / S/O -> Father
        if re.search(r'\b(S/O|SON\s*OF)\b', ln, re.I) and not father:
            father = extract_name_after_marker(ln, r'\b(S/O|SON\s*OF)\b')
            if not father and i + 1 < len(lines):
                # Check next line, but skip if it looks like address
                next_ln = extract_english_only(lines[i + 1])
                if next_ln and re.match(r'^[A-Za-z\s]+$', next_ln) and len(next_ln) > 3:
                    if not looks_like_address(next_ln):
                        father = clean_ocr_garbage(next_ln.split(',')[0])

        # Daughter of / D/O -> Father
        elif re.search(r'\b(D/O|DAUGHTER\s*OF)\b', ln, re.I) and not father:
            father = extract_name_after_marker(ln, r'\b(D/O|DAUGHTER\s*OF)\b')
            if not father and i + 1 < len(lines):
                next_ln = extract_english_only(lines[i + 1])
                if next_ln and re.match(r'^[A-Za-z\s]+$', next_ln) and len(next_ln) > 3:
                    if not looks_like_address(next_ln):
                        father = clean_ocr_garbage(next_ln.split(',')[0])

        # Wife of / W/O -> Husband
        elif re.search(r'\b(W/O|WIFE\s*OF)\b', ln, re.I) and not husband:
            husband = extract_name_after_marker(ln, r'\b(W/O|WIFE\s*OF)\b')
            if not husband and i + 1 < len(lines):
                next_ln = extract_english_only(lines[i + 1])
                if next_ln and re.match(r'^[A-Za-z\s]+$', next_ln) and len(next_ln) > 3:
                    if not looks_like_address(next_ln):
                        husband = clean_ocr_garbage(next_ln.split(',')[0])

        # Care of / C/O -> Father (usually)
        elif re.search(r'\b(C/O|CARE\s*OF)\b', ln, re.I) and not father:
            father = extract_name_after_marker(ln, r'\b(C/O|CARE\s*OF)\b')
            if not father and i + 1 < len(lines):
                next_ln = extract_english_only(lines[i + 1])
                if next_ln and re.match(r'^[A-Za-z\s]+$', next_ln) and len(next_ln) > 3:
                    if not looks_like_address(next_ln):
                        father = clean_ocr_garbage(next_ln.split(',')[0])

    # Fallback: Extract father from person's name (if 3+ words)
    if not father and person_name:
        name_parts = person_name.split()
        if len(name_parts) >= 3:
            # Middle name is often father's first name
            potential_father = ' '.join(name_parts[1:])
            if len(potential_father) > 4:
                father = potential_father

    return father, mother, husband


def _clean_and_deduplicate_address(address_parts: List[str], person_name: str = "") -> str:
    """Helper to deduplicate and clean address parts, handling merged column artifacts."""
    if not address_parts:
        return ""
        
    unique_parts = []
    seen = set()
    for p in address_parts:
        # Skip if part of person name
        if person_name and p.upper() == person_name.upper():
            continue
            
        # Skip if it contains a relation marker (already extracted separately)
        if re.search(r'\b(S/O|D/O|W/O|C/O)\b', p, re.I):
            continue
        
        p_clean = p.lower().strip()
        # Remove non-alphanumeric for comparison
        comp_key = re.sub(r'[^a-z0-9]', '', p_clean)
        
        if comp_key and comp_key not in seen:
            # Check for near-duplicates (one part contained in another)
            # Common when horizontal OCR merges English and Marathi columns
            is_duplicate = False
            for seen_key in seen:
                if len(comp_key) > 5 and len(seen_key) > 5:
                    if comp_key in seen_key or seen_key in comp_key:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_parts.append(p)
                seen.add(comp_key)
    
    addr = ', '.join(unique_parts)
    # Final cleanup
    addr = re.sub(r',\s*,', ',', addr)
    addr = re.sub(r'\s+', ' ', addr)
    return addr.strip().strip(',')


def extract_address(lines: List[str], person_name: str) -> str:
    """Extract address from Aadhaar card."""

    address_parts = []
    collecting = False

    # Address indicators
    # Address indicators - ensure D/O doesn't match D/O/B
    address_start = r'\b(S/O|D/O(?!\/B)|W/O|C/O|Address|पता|HOUSE|FLAT|VILLAGE|VTC|PO:|POST)\b'
    address_end_indicators = [
        r'\b\d{6}\b',  # PIN code
    ]
    skip_patterns = [
        r'(Issue|Download|Print)\s*Date',
        r'\b\d{10,12}\b',  # Phone or Aadhaar numbers
        r'VID\s*:',
        r'Aadhaar\s*No',
        r'signature',
        r'help@',
        r'www\.',
        r'\bDOB\b',
        r'Date of Birth',
        r'\b(MALE|FEMALE|TRANSGENDER)\b',
        r'(/|\b)(MALE|FEMALE)\b',
    ]

    # Strategy 1: Find "Address:" label
    for i, ln in enumerate(lines):
        if re.search(r'\bAddress\b|पता\s*:', ln, re.I):
            collecting = True
            # Start collecting from this line
            addr_text = re.sub(r'^.*Address\s*:?\s*', '', ln, flags=re.I)
            if addr_text.strip():
                address_parts.append(addr_text.strip())
            continue

        if collecting:
            # 1. Skip lines that are purely boilerplate
            pure_skip = [r'signature', r'help@', r'www\.', r'Aadhaar\s*No', r'VID\s*:']
            if any(re.search(p, ln, re.I) for p in pure_skip):
                continue

            # 2. Clean merged boilerplate from line instead of skipping
            # (Essential for e-Aadhaar where columns merge)
            clean_ln = ln
            for p in [r'Date of Birth', r'\bDOB\b', r'\d{2}/\d{2}/\d{4}', r'\b(MALE|FEMALE|TRANSGENDER)\b', r'(/|\b)(MALE|FEMALE)\b']:
                clean_ln = re.sub(p, '', clean_ln, flags=re.I).strip()
            
            # 3. Final cleaning of individual line
            clean_ln = extract_english_only(clean_ln).strip()
            # Remove leading/trailing symbols that might remain after cleaning
            clean_ln = re.sub(r'^[:\-,\.\s]+|[:\-,\.\s]+$', '', clean_ln).strip()
            
            if clean_ln:
                address_parts.append(clean_ln)

            # Stop at PIN code
            if re.search(r'\b\d{6}\b', ln):
                break

    if address_parts:
        return _clean_and_deduplicate_address(address_parts, person_name)

    # Strategy 2: Start from "To" block (e-Aadhaar format)
    # Pattern: "To\n[Hindi name]\n[English name]\n[Address lines...]\n[PIN]"
    for i, ln in enumerate(lines):
        if re.match(r'^\s*To\s*$', ln, re.I) or re.search(r'^To\s+[A-Z]', ln):
            # Skip name lines (Hindi + English name), then collect address
            name_found = False
            for j in range(i + 1, min(len(lines), i + 15)):
                line_j = lines[j].strip()
                if not line_j:
                    continue

                # Skip any line that matches skip_patterns
                if any(re.search(p, line_j, re.I) for p in skip_patterns):
                    continue

                # Skip Hindi-only lines (Devanagari name)
                if contains_devanagari(line_j) and not extract_english_only(line_j).strip():
                    continue

                # Initial cleaning of individual line
                clean_line = extract_english_only(line_j).strip()
                if not clean_line:
                    continue

                # Skip the person's name if seen right after "To"
                if person_name and clean_line.upper() == person_name.upper():
                    name_found = True
                    continue
                
                # Check for likely name pattern if name_found is False
                if not name_found:
                    words = clean_line.split()
                    if 2 <= len(words) <= 4 and all(w.isalpha() for w in words) and not looks_like_address(clean_line) and has_reasonable_vowel_ratio(clean_line):
                        name_found = True
                        continue

                # Clean merged boilerplate from line
                # (Essential for e-Aadhaar where columns merge)
                for p in [r'Date of Birth', r'\bDOB\b', r'\d{2}/\d{2}/\d{4}', r'\b(MALE|FEMALE|TRANSGENDER)\b', r'(/|\b)(MALE|FEMALE)\b']:
                    clean_line = re.sub(p, '', clean_line, flags=re.I).strip()
                
                # Final strip of symbols
                clean_line = re.sub(r'^[:\-,\.\s]+|[:\-,\.\s]+$', '', clean_line).strip()

                if clean_line:
                    address_parts.append(clean_line)

                # Stop at PIN code
                if re.search(r'\b\d{6}\b', line_j):
                    break

            if address_parts:
                return _clean_and_deduplicate_address(address_parts, person_name)

    # Strategy 3: Start from relation marker (S/O, D/O, etc.)
    for i, ln in enumerate(lines):
        if re.search(r'\b(S/O|D/O|W/O|C/O)\b', ln, re.I):
            collecting = True
            continue

        if collecting:
            # Skip boilerplate-only lines
            if any(re.search(p, ln, re.I) for p in [r'signature', r'help@', r'www\.', r'Aadhaar\s*No', r'VID\s*:']):
                continue

            # Clean merged boilerplate from line
            clean_line = ln
            for p in [r'Date of Birth', r'\bDOB\b', r'\d{2}/\d{2}/\d{4}', r'\b(MALE|FEMALE|TRANSGENDER)\b', r'(/|\b)(MALE|FEMALE)\b']:
                clean_line = re.sub(p, '', clean_line, flags=re.I).strip()
            
            clean_line = extract_english_only(clean_line).strip()
            clean_line = re.sub(r'^[:\-,\.\s]+|[:\-,\.\s]+$', '', clean_line).strip()
            
            if clean_line and not (person_name and clean_line.upper() == person_name.upper()):
                address_parts.append(clean_line)

            if re.search(r'\b\d{6}\b', ln):
                break

    if address_parts:
        return _clean_and_deduplicate_address(address_parts, person_name)

    # Strategy 4: Find lines with address keywords
    for i, ln in enumerate(lines):
        if re.search(address_start, ln, re.I):
            for j in range(i, min(len(lines), i + 8)):
                if any(re.search(p, lines[j], re.I) for p in skip_patterns):
                    continue

                clean_ln = extract_english_only(lines[j]).strip()
                if clean_ln:
                    address_parts.append(clean_ln)

                if re.search(r'\b\d{6}\b', lines[j]):
                    break
            break

    if address_parts:
        return _clean_and_deduplicate_address(address_parts, person_name)

    return ""
