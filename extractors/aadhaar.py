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
import logging
logger = logging.getLogger("aadhaar")


def _contains_indic_script(text: str) -> bool:
    """Check if text contains any Indic script character.

    Covers all major Indian scripts found on Aadhaar cards:
    Devanagari (Hindi/Marathi), Telugu, Tamil, Odia, Bengali,
    Gurmukhi, Gujarati, Kannada, Malayalam.
    """
    if not text:
        return False
    for char in text:
        cp = ord(char)
        if (0x0900 <= cp <= 0x097F or  # Devanagari
                0x0980 <= cp <= 0x09FF or  # Bengali/Assamese
                0x0A00 <= cp <= 0x0A7F or  # Gurmukhi (Punjabi)
                0x0A80 <= cp <= 0x0AFF or  # Gujarati
                0x0B00 <= cp <= 0x0B7F or  # Odia
                0x0B80 <= cp <= 0x0BFF or  # Tamil
                0x0C00 <= cp <= 0x0C7F or  # Telugu
                0x0C80 <= cp <= 0x0CFF or  # Kannada
                0x0D00 <= cp <= 0x0D7F):   # Malayalam
            return True
    return False


def is_aadhaar_back_side(lines: List[str], text: str) -> bool:
    """Detect if the image is the back side of an Aadhaar card.

    Back side characteristics:
    - Has "Address:" label or Hindi "पत्ता" label
    - Does NOT have DOB or gender (front side only fields)
    - Typically has bilingual text (English + Hindi/regional language) in two columns
    """
    # Back side indicators (handle Hindi पता, Marathi पत्ता, and English Address)
    has_address_label = bool(
        re.search(r'\bAddress\s*:', text, re.I) or 'पत्ता' in text or 'पता' in text
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
        logger.debug("Detected Aadhaar BACK side")
        return True

    return False


def is_full_eaadhaar_letter(lines: List[str], text: str) -> bool:
    """Detect the full downloadable e-Aadhaar LETTER scanned/saved as an image.

    This is the multi-column UIDAI letter (4-up): front + dense bilingual
    INFORMATION panel on top, back card + address + QR on the bottom. When OCR
    runs over the whole page, the INFORMATION panel and the right-hand columns
    merge into the data columns at the same y-position, producing garbage.
    The caller should crop to the LEFT column (which carries name, DOB, gender,
    address and the Aadhaar number) and re-run OCR for clean text.

    Distinguished from PVC cards, cropped front/back photos and KYC reports by
    the long INFORMATION-panel boilerplate paragraphs, which appear ONLY on the
    full letter. Requiring two or more distinct phrases avoids false positives
    on single-column documents (cropping those would cut off half the text).
    """
    t = text.lower()
    info_markers = [
        'obligated to seek consent',
        'lock/unlock',
        'qr code reader',
        'documents to support identity',
        'aadhaar is unique and secure',
        'proof of identity, not of citizenship',
        'authentication agency or qr code',
        'avail of various government',
        'keep your mobile number and email',
    ]
    hits = sum(1 for m in info_markers if m in t)
    return hits >= 2


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

    logger.debug("===== Aadhaar BACK Side Extraction Started =====")
    logger.debug(f"Total OCR lines: {len(lines)}")
    logger.debug("OCR Lines:")
    for i, ln in enumerate(lines[:20]):
        logger.debug(f"  Line {i}: '{ln}'")

    # Extract Aadhaar number (same logic as front side)
    obj['aadhaar_number'] = extract_aadhaar_number(text, lines)
    logger.debug(f"Aadhaar Number: {obj['aadhaar_number']}")

    # Extract VID
    obj['vid'] = extract_vid(text, lines)
    logger.debug(f"VID: {obj['vid']}")

    # Extract address (use existing logic - works well with clean left-column text)
    obj['address'] = extract_address(lines, "")
    logger.debug(f"Address: {obj['address'][:100]}..." if len(obj['address']) > 100 else f"Address: {obj['address']}")

    # Back side: relation names are part of the address (S/O, D/O etc.)
    # so don't populate separate father/mother/husband fields

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
    print(f"[DEBUG] Address: {obj['address'][:100]}..." if obj['address'] and len(obj['address']) > 100 else f"[DEBUG] Address: {obj['address']}")

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

    # Pattern 0: Masked e-Aadhaar format "XXXX XXXX 9420" (last 4 digits visible)
    # This is the standard format for e-Aadhaar PDFs downloaded from UIDAI
    masked_match = re.search(r'\bXXXX\s+XXXX\s+(\d{4})\b', text)
    if masked_match:
        return f"XXXX XXXX {masked_match.group(1)}"

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

        # Skip if contains relation markers (S/O, C/O, D/O, W/O)
        # These lines contain father/husband names, not the person's name
        if re.search(r'\b(S/O|C/O|D/O|W/O|SON\s*OF|DAUGHTER\s*OF|WIFE\s*OF|CARE\s*OF)\b', text, re.I):
            return False

        # Skip lines that begin with an address/locality label (e.g. "State Uttar
        # Pradesh", "District Mainpuri", "VTC Tikuri", "PO Mota"). These are address
        # lines, never the person's name, and would otherwise slip past the
        # state-name check below because of the leading label word.
        if re.match(r'^\s*(?:state|sub\s*-?\s*dist(?:rict)?|dist(?:rict)?|vtc|p\.?\s*o\.?|post|pin)\b',
                    text, re.I):
            return False

        # Skip if contains digits
        if re.search(r'\d', text):
            return False

        # Skip if too short after cleaning
        words = text.split()
        if len(words) < 2:
            return False

        # Skip if looks like an address (contains locality/admin keywords)
        address_keywords = ['nagar', 'village', 'road', 'street', 'lane', 'sector',
                           'colony', 'house', 'flat', 'building', 'block',
                           'district', 'tehsil', 'taluk', 'mandal', 'taluka',
                           'sub district', 'subdistrict', 'subdivision',
                           'vtc', ' po ', 'post office', 'pin code', 'pincode']
        if any(kw in text_lower for kw in address_keywords):
            return False

        # Skip Indian state names (appear in addresses, not person names)
        # NOTE: text_lower has spaces stripped, so compare against text.lower().strip()
        indian_states = {
            'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
            'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
            'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
            'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu',
            'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal',
            'delhi', 'jammu', 'kashmir', 'ladakh', 'chandigarh', 'puducherry'
        }
        if text.lower().strip() in indian_states:
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

        # Drop a leading regional-script-bleed junk token (see helper below)
        text = _consensus_trim_leading(text)

        return text.strip()

    def _consensus_trim_leading(t: str) -> str:
        """Drop a leading OCR-junk token that the regional script bled in front
        of the English name.

        On e-Aadhaar the holder's name appears in two places (the address/"To"
        block and beside the photo). When the regional script bleeds a junk word
        in front of the English name, the two readings AGREE on the real name but
        DISAGREE on that leading word, e.g. "OL Inoh Bibinaz Tabrez Pathan" vs
        "Inleh Bibinaz Tabrez Pathan". If the shared trailing name also appears
        elsewhere behind a different leading token AND one of those leading tokens
        is itself clear garbage (<=2 chars or digit-bearing), the leading word is
        junk -> drop it. A consistently-read real first name is never trimmed, so
        this is high precision.
        """
        words = t.split()
        if len(words) < 3:
            return t
        suffix = [w.lower() for w in words[1:]]
        n = len(suffix)

        def _is_junk_tok(tok: str) -> bool:
            return len(tok) <= 2 or bool(re.search(r'\d', tok))

        leading_variants = {words[0].lower()}
        junk_seen = _is_junk_tok(words[0].lower())
        for ln in lines:
            # Skip relation lines so a father's name after S/O can't masquerade
            # as a repeated suffix and trim the holder's real first name.
            if re.search(r'\b(S/O|D/O|W/O|C/O|SON OF|DAUGHTER OF|WIFE OF|CARE OF)\b', ln, re.I):
                continue
            toks = [w.strip('.,:;-').lower() for w in extract_english_only(ln).split()]
            toks = [w for w in toks if w]
            for s in range(1, len(toks) - n + 1):
                if toks[s:s + n] == suffix:
                    leading_variants.add(toks[s - 1])
                    if any(_is_junk_tok(p) for p in toks[:s]):
                        junk_seen = True
        if len(leading_variants) >= 2 and junk_seen:
            return ' '.join(words[1:])
        return t

    # Address-token keywords seen on Aadhaar address lines. Used to reject
    # address fragments (e.g. "VTC: Ladalla", "State: Maharashtra") that the
    # lighter name checks would otherwise accept. Supplements looks_like_address,
    # which misses Aadhaar-specific tokens like VTC / PO / mandal.
    _addr_kw = ('nagar', 'village', 'road', 'street', 'lane', 'sector', 'colony',
                'house', 'flat', 'building', 'block', 'district', 'tehsil', 'taluk',
                'mandal', 'taluka', 'subdistrict', 'sub district', 'subdivision',
                'vtc', 'post office', 'pin code', 'pincode', 'state', 'room', 'floor',
                'marg', 'apartment', 'near ', ' po ', 'po:', 'post', 'gav', 'gaon')

    def looks_addressy(t: str) -> bool:
        """True if the text is an address line rather than a name."""
        tl = t.lower()
        if any(kw in tl for kw in _addr_kw):
            return True
        return looks_like_address(t)

    def strip_leading_garbage(t: str) -> str:
        """Drop leading OCR-garbage tokens from a name candidate.

        On full e-Aadhaar letters the regional-script name often bleeds a short
        junk token in front of the English name (e.g. "S e Pinnoju Vijaya
        Lakshmi", "3o Parasa Naga ..."). Strip leading tokens that are single
        characters, digit-bearing, or short and vowel-less, until a plausible
        name word is reached. (May drop a leading single-letter initial — an
        acceptable trade-off for removing clear garbage.)
        """
        words = t.split()
        while words:
            w = words[0].strip('.,:;-')
            if (not w or len(w) <= 1 or re.search(r'\d', w)
                    or (len(w) <= 3 and not has_reasonable_vowel_ratio(w))):
                words.pop(0)
                continue
            break
        return ' '.join(words)

    # -------------------------------------------------------------------------
    # Strategy 1: Find English name after regional script name
    # This is the most reliable pattern for Aadhaar cards
    # Also handles merged lines where regional+English name are on same line
    # Covers all Indian scripts: Devanagari, Telugu, Tamil, Odia, etc.
    # -------------------------------------------------------------------------
    print("[DEBUG] Strategy 1: Looking for English name after regional script name...")
    for i in range(len(lines)):
        if _contains_indic_script(lines[i]):
            # First: check if this line has BOTH regional script AND English name
            # (OCR may merge "आशा लक्ष्मन बिसाने ASHA LAXMAN BISANE" into one line)
            # Exclude gender lines like "స్త్రీ/ FEMALE" which have Indic + English gender word
            english_part = extract_english_only(lines[i])
            if english_part and not re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', english_part, re.I):
                # Case A: merged line has S/O inside e.g. "Sachin S/O Chandra Shekhar"
                # Extract name from BEFORE the relation marker
                pre_so_m = re.match(r'^([A-Za-z][A-Za-z\s]{1,30}?)\s+\b(?:S/O|D/O|W/O|C/O|SON OF|DAUGHTER OF|WIFE OF)\b', english_part, re.I)
                if pre_so_m:
                    pre_name = pre_so_m.group(1).strip()
                    pre_words = pre_name.split()
                    if (1 <= len(pre_words) <= 4
                            and not re.search(r'\d', pre_name)
                            and not any(p in pre_name.lower() for p in skip_patterns)
                            and has_reasonable_vowel_ratio(pre_name)
                            and all(len(w) >= 2 and (w[0].isupper() or w.isupper()) for w in pre_words)):
                        name = clean_name(pre_name)
                        if name and len(name) >= 3:
                            print(f"[DEBUG] Found name before S/O in merged Indic+English line: '{name}'")
                            return name
                # Case B: no S/O - validate normally
                elif is_valid_name_candidate(english_part):
                    name = clean_name(english_part)
                    if name and len(name) >= 5:
                        print(f"[DEBUG] Found name in merged regional+English line: '{name}'")
                        return name

            # Then: check next few lines for English name
            for j in range(i + 1, min(i + 3, len(lines))):
                next_ln = lines[j].strip()

                # Skip if this line also has any Indic script
                if _contains_indic_script(next_ln):
                    continue

                # Skip if it's a DOB or gender line
                if re.search(r'\bDOB\b|जन्म|तिथि|\d{2}/\d{2}/\d{4}', next_ln, re.I):
                    continue
                if re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', next_ln, re.I):
                    continue

                # Check if it looks like a name
                english_text = extract_english_only(next_ln)
                if is_valid_name_candidate(english_text):
                    name = clean_name(english_text)
                    if name and len(name) >= 5:
                        print(f"[DEBUG] Found name via regional anchor: '{name}'")
                        return name
                # Lighter fallback: Indian names like "Lakshmi" have consonant clusters
                # (KSHM) that is_likely_garbage incorrectly flags. Use direct check here
                # since we have strong anchor (Indic script on previous line).
                # Also handles single-word names like "Sachin" with strong positional anchor.
                elif english_text and not re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', english_text, re.I):
                    s1_words = english_text.split()
                    if (1 <= len(s1_words) <= 5
                            and not re.search(r'\d', english_text)
                            and not re.search(r'\b(S/O|D/O|W/O|C/O)\b', english_text, re.I)
                            and not any(p in english_text.lower() for p in skip_patterns)
                            and has_reasonable_vowel_ratio(english_text)
                            and not looks_like_address(english_text)
                            and all(len(w) >= 2 and (w[0].isupper() or w.isupper()) for w in s1_words)):
                        name = clean_name(english_text)
                        min_s1_len = 3 if len(s1_words) == 1 else 5
                        if name and len(name) >= min_s1_len:
                            print(f"[DEBUG] Found name via regional anchor (lighter check): '{name}'")
                            return name

    # -------------------------------------------------------------------------
    # Strategy 2: Find name in "To" block (e-Aadhaar format)
    # Pattern: "To\n[Hindi name]\n[English name]\n[Address lines...]"
    # Key: Take FIRST valid name candidate after "To" label
    # -------------------------------------------------------------------------
    print("[DEBUG] Strategy 2: Looking for name in 'To' block...")
    relation_re = r'\b(S/O|D/O|W/O|C/O|SON\s*OF|DAUGHTER\s*OF|WIFE\s*OF|CARE\s*OF)\b'
    for i, ln in enumerate(lines):
        if re.match(r'^\s*To\s*$', ln, re.I) or re.search(r'^To\s+[A-Z]', ln):
            # Case A: the name is on the SAME line as "To" (e.g. "To Mohd Junaid").
            # This happens when OCR groups the label and name into one box (common
            # once the noisy right column is removed). Read it directly.
            inline = re.match(r'^\s*To\s+(.+)$', ln, re.I)
            if inline:
                inline_text = strip_leading_garbage(extract_english_only(inline.group(1)).strip())
                inline_words = inline_text.split()
                if (inline_text
                        and not re.search(relation_re, inline_text, re.I)
                        and 1 <= len(inline_words) <= 5
                        and not re.search(r'\d', inline_text)
                        and not re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', inline_text, re.I)
                        and not any(p in inline_text.lower() for p in skip_patterns)
                        and has_reasonable_vowel_ratio(inline_text)
                        and not looks_addressy(inline_text)
                        and all(len(w) >= 2 and (w[0].isupper() or w.isupper()) for w in inline_words)):
                    name = clean_name(inline_text)
                    min_len = 3 if len(inline_words) == 1 else 5
                    if name and len(name) >= min_len:
                        print(f"[DEBUG] Found name on 'To' line: '{name}'")
                        return name

            # Case B: check next few lines - take FIRST valid name, skip address lines
            for j in range(i + 1, min(i + 5, len(lines))):
                candidate = lines[j]

                # Remove "To" prefix if on same line
                candidate = re.sub(r'^To\s+', '', candidate, flags=re.I)

                # Skip regional script lines (Devanagari, Telugu, Tamil, Odia, etc.)
                if _contains_indic_script(candidate):
                    continue

                # Extract English, then strip any leading regional-script garbage
                # token (e.g. "3o Parasa ...", "S e Pinnoju ...").
                english_text = strip_leading_garbage(extract_english_only(candidate))

                # Skip if it looks like an address line (locality/area keywords)
                if looks_addressy(english_text):
                    print(f"[DEBUG] Skipping address-like line: '{english_text}'")
                    continue

                # In the To-block we have a strong positional anchor (right after "To"
                # + skipped all Indic script lines). Use a lighter check here instead of
                # is_valid_name_candidate, which calls is_likely_garbage and incorrectly
                # rejects common Indian names with consonant clusters (e.g., Lakshmi→KSHM).
                to_words = english_text.split()
                if (1 <= len(to_words) <= 5
                        and not re.search(r'\d', english_text)
                        and not re.search(relation_re, english_text, re.I)
                        and not re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', english_text, re.I)
                        and not any(p in english_text.lower() for p in skip_patterns)
                        and has_reasonable_vowel_ratio(english_text)
                        and all(len(w) >= 2 and (w[0].isupper() or w.isupper()) for w in to_words)):
                    name = clean_name(english_text)
                    min_len = 3 if len(to_words) == 1 else 5
                    if name and len(name) >= min_len:
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

            # Strip leading OCR garbage so a name like "22 1 Roshan Vilas Yaday"
            # (number prefix bleeding in from the card) is still recognised.
            english_text = strip_leading_garbage(extract_english_only(candidate))
            if looks_addressy(english_text):
                continue
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
    # Strategy 5: Use relation marker as anchor
    # Look for name BEFORE S/O on same line OR on line above S/O.
    # This is more reliable than generic Title Case scan, so run it first.
    # Also handles single-word names like "Sachin".
    # -------------------------------------------------------------------------
    print("[DEBUG] Strategy 5: Looking for name above/before relation markers...")
    for i, ln in enumerate(lines):
        if re.search(r'\b(S/O|D/O|W/O|C/O|SON OF|DAUGHTER OF|WIFE OF)\b', ln, re.I):
            # Case A: Name is BEFORE S/O on the same line e.g. "Sachin S/O Chandra Shekhar"
            pre_so = re.match(r'^([A-Za-z][A-Za-z\s]{1,30}?)\s+\b(?:S/O|D/O|W/O|C/O|SON OF|DAUGHTER OF|WIFE OF)\b', ln, re.I)
            if pre_so:
                pre_text = pre_so.group(1).strip()
                pre_words = pre_text.split()
                if (1 <= len(pre_words) <= 4
                        and not re.search(r'\d', pre_text)
                        and not any(p in pre_text.lower() for p in skip_patterns)
                        and has_reasonable_vowel_ratio(pre_text)
                        and all(len(w) >= 2 and (w[0].isupper() or w.isupper()) for w in pre_words)):
                    name = clean_name(pre_text)
                    if name and len(name) >= 3:
                        print(f"[DEBUG] Found name before S/O on same line: '{name}'")
                        return name

            # Case B: Name is on the line ABOVE S/O
            if i > 0:
                candidate = lines[i - 1]
                # Strip leading OCR garbage so "$2}$ Sachin" -> "2 Sachin" -> "Sachin"
                # is recognised instead of failing the digit/word checks below.
                english_text = strip_leading_garbage(extract_english_only(candidate).strip())
                # First try full validation (multi-word names)
                if is_valid_name_candidate(english_text):
                    name = clean_name(english_text)
                    if name and len(name) >= 5:
                        print(f"[DEBUG] Found name above relation marker: '{name}'")
                        return name
                # Fallback: allow single-word names (e.g. "Sachin") when anchored by S/O
                words = english_text.split()
                if (len(words) == 1
                        and len(english_text) >= 3
                        and not re.search(r'\d', english_text)
                        and not any(p in english_text.lower() for p in skip_patterns)
                        and has_reasonable_vowel_ratio(english_text)
                        and english_text[0].isupper()):
                    name = clean_name(english_text)
                    if name:
                        print(f"[DEBUG] Found single-word name above relation marker: '{name}'")
                        return name

    # -------------------------------------------------------------------------
    # Strategy 6: Find Title Case name pattern
    # Pattern: "Firstname Middlename Lastname"
    # Skip lines that contain relation markers (S/O etc.) to avoid extracting
    # the father's name from "S/O Chandra Shekhar" as the person's name.
    # -------------------------------------------------------------------------
    print("[DEBUG] Strategy 6: Looking for Title Case name pattern...")
    for i, ln in enumerate(lines):
        # Skip lines with relation markers
        if re.search(r'\b(S/O|D/O|W/O|C/O|SON\s*OF|DAUGHTER\s*OF|WIFE\s*OF|CARE\s*OF)\b', ln, re.I):
            continue
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
        # Make marker groups non-capturing so group(1) captures the name
        nc_marker = marker_pattern.replace('(', '(?:')
        flexible_marker = nc_marker.replace('/', r'\s*[\/:]\s*')
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


def _clean_address_token(token: str) -> str:
    """Clean a single address token, removing OCR garbage from bilingual column merges.

    When OCR reads Aadhaar back side, English and Hindi columns merge per line.
    After stripping Devanagari, fragments like '3ITETT', 'g', 'a' remain.
    This function removes such garbage while keeping valid address tokens.
    """
    token = token.strip().strip(',-.:;')
    if not token:
        return ""

    # Keep tokens that are clearly valid: house numbers (A-202), PIN codes (401203),
    # known patterns like (West), (East), etc.
    # House number pattern: letter(s)-digits or digits-letter(s)
    if re.match(r'^[A-Za-z]-?\d+$', token) or re.match(r'^\d+-?[A-Za-z]$', token):
        return token
    # Pure digits (PIN code, house number)
    if token.isdigit():
        return token
    # Parenthetical like (West), (East)
    if re.match(r'^\([A-Za-z]+\)$', token):
        return token

    # Remove tokens that are single lowercase letter (garbage from Hindi strip)
    # e.g. "g", "a" but NOT "A" (could be part of house number like A-202)
    if len(token) == 1 and token.islower():
        return ""

    # Remove tokens that mix digits and uppercase letters in garbage patterns
    # e.g. "3ITETT", "5HRTT" but NOT "A-202" or "401203"
    if re.match(r'^\d+[A-Z]{2,}$', token) or re.match(r'^[A-Z]+\d+[A-Z]+$', token):
        return ""

    return token


def _clean_address_line(line: str) -> str:
    """Clean a full address line by removing bilingual merge garbage tokens.

    Processes the line token by token, removing OCR artifacts while preserving
    the meaningful English address components.
    """
    if not line:
        return ""

    # Split by comma first to handle comma-separated parts
    parts = [p.strip() for p in line.split(',')]
    cleaned_parts = []

    for part in parts:
        if not part:
            continue
        # Preserve "State - PIN" patterns (e.g. "Maharashtra - 401203")
        if re.match(r'^[A-Za-z]+\s*-\s*\d{6}$', part.strip()):
            cleaned_parts.append(part.strip())
            continue
        # Split part into words and clean each
        words = part.split()
        cleaned_words = []
        for word in words:
            cleaned = _clean_address_token(word)
            if cleaned:
                cleaned_words.append(cleaned)

        if cleaned_words:
            reassembled = ' '.join(cleaned_words)
            # Skip parts that are just a single short word (likely garbage remnant)
            if len(reassembled) > 1:
                cleaned_parts.append(reassembled)

    return ', '.join(cleaned_parts)


def _clean_and_deduplicate_address(address_parts: List[str], person_name: str = "") -> str:
    """Helper to deduplicate and clean address parts, handling merged column artifacts."""
    if not address_parts:
        return ""

    # First pass: clean each part of bilingual merge garbage
    cleaned_address_parts = []
    for p in address_parts:
        cleaned = _clean_address_line(p)
        if cleaned:
            cleaned_address_parts.append(cleaned)

    # Filter out the person's own name line (keep S/O, D/O etc. as part of address).
    # Use normalized containment, not exact match, so a name line that picked up an
    # OCR garbage prefix (e.g. "3o Parasa Naga Venkata Sai Manikanta") is still removed.
    def _norm_name(s: str) -> str:
        return re.sub(r'[^a-z0-9]', '', s.lower())
    pn = _norm_name(person_name) if person_name else ""
    filtered = []
    for p in cleaned_address_parts:
        if (pn and len(pn) >= 6 and pn in _norm_name(p)
                and not re.search(r'\b(S/O|D/O|W/O|C/O)\b', p, re.I)):
            continue
        filtered.append(p)

    # Split all parts into individual comma-separated components for better dedup
    all_components = []
    for p in filtered:
        for comp in p.split(','):
            comp = comp.strip().strip(',-.:')
            if comp:
                all_components.append(comp)

    # Deduplicate at the component level
    def normalize(s):
        return re.sub(r'[^a-z0-9\s]', '', s.lower().strip())

    def norm_key(s):
        return re.sub(r'[^a-z0-9]', '', s.lower().strip())

    # Clean within-component word repetitions
    # e.g. "Giddoba Mandir Giddoba Mandir" → "Giddoba Mandir"
    # e.g. "440035 440035" → "440035"
    deduplicated_components = []
    for comp in all_components:
        words = comp.split()
        if len(words) >= 2:
            # Check if the second half repeats the first half
            for split_at in range(1, len(words)):
                first_half = [w.lower() for w in words[:split_at]]
                second_half = [w.lower() for w in words[split_at:split_at + len(first_half)]]
                if first_half == second_half:
                    # Keep only the first half + any remainder after the repeat
                    comp = ' '.join(words[:split_at] + words[split_at + len(first_half):])
                    break
        deduplicated_components.append(comp.strip())
    all_components = [c for c in deduplicated_components if c]

    # Remove short garbage fragments from Hindi/Marathi OCR leakage
    # e.g. "ON 3", "O/s Anor", single letters
    cleaned_components = []
    for comp in all_components:
        # Remove very short garbage fragments (e.g. "g", "a", "ON 3")
        # but keep legitimate short components like "No 22" when part of larger context
        alpha_only = re.sub(r'[^a-zA-Z]', '', comp)
        if len(alpha_only) < 2:
            continue
        # Remove standalone 2-letter fragments with digits that look like OCR garbage
        if len(alpha_only) == 2 and len(comp) <= 4 and re.search(r'\d', comp):
            continue
        # Remove fragments that look like garbled relation markers (O/s, S/0, etc.)
        if re.match(r'^[A-Za-z]/[a-z]\b', comp) and not re.match(r'^(S/O|D/O|W/O|C/O)\b', comp, re.I):
            continue
        cleaned_components.append(comp)
    all_components = cleaned_components

    # Remove exact duplicates while preserving order
    seen_norm = set()
    unique_components = []
    for comp in all_components:
        nk = norm_key(comp)
        if nk and nk not in seen_norm:
            unique_components.append(comp)
            seen_norm.add(nk)

    # Merge overlapping fragments: if tail words of one match head words of another,
    # merge them into one longer component.
    # e.g. "Lodha Shopping" + "Shopping Center" -> "Lodha Shopping Center"
    # Only merge if result is strictly longer than both inputs (avoids absorbing subsets).
    did_merge = True
    while did_merge:
        did_merge = False
        for i in range(len(unique_components)):
            if not unique_components[i]:
                continue
            words_i = normalize(unique_components[i]).split()
            for j in range(len(unique_components)):
                if i == j or not unique_components[j]:
                    continue
                words_j = normalize(unique_components[j]).split()
                # Check if tail of i overlaps with head of j
                for overlap_len in range(1, min(len(words_i), len(words_j)) + 1):
                    if words_i[-overlap_len:] == words_j[:overlap_len]:
                        # Only merge if result has MORE words than either input
                        new_word_count = len(words_i) + len(words_j) - overlap_len
                        if new_word_count <= max(len(words_i), len(words_j)):
                            continue  # Skip: result isn't longer, it's a subset
                        orig_words_j = unique_components[j].split()
                        merged_text = unique_components[i] + ' ' + ' '.join(orig_words_j[overlap_len:])
                        merged_text = merged_text.strip()
                        if merged_text:
                            # Place result at the earlier position to preserve order
                            target = min(i, j)
                            other = max(i, j)
                            unique_components[target] = merged_text
                            unique_components[other] = ""
                            if target != i:
                                unique_components[i] = ""
                            did_merge = True
                            break
                if did_merge:
                    break
            if did_merge:
                break
        unique_components = [c for c in unique_components if c]

    # Remove fragment components that are clearly redundant.
    # A component is removed if ALL its words appear in another longer component.
    # e.g. "Center" removed when "Lodha Shopping Center" exists.
    # e.g. "S/O Shrirang Shende" removed when "S/O Shrirang Shende Plot No 22" exists.
    final_components = []
    for i, comp in enumerate(unique_components):
        words_i = normalize(comp).split()
        if not words_i:
            continue
        is_fragment = False
        for j, other in enumerate(unique_components):
            if i == j:
                continue
            words_j = normalize(other).split()
            # Only remove if the other component is strictly longer
            if len(words_j) > len(words_i):
                # Check if all words of comp appear in other
                if all(w in words_j for w in words_i):
                    is_fragment = True
                    break
        if is_fragment:
            continue
        final_components.append(comp)

    addr = ', '.join(final_components)
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

    # Strategy 1: Find English "Address:" label (preferred) or Hindi पता/पत्ता label
    # For stacked layouts (Hindi block + English block), prefer the English "Address:" section
    # For column layouts (side-by-side), extract_english_only filters out Devanagari

    # First pass: find the English "Address:" label line index
    english_addr_idx = -1
    hindi_addr_idx = -1
    for i, ln in enumerate(lines):
        if re.search(r'\bAddress\s*:', ln, re.I) and english_addr_idx == -1:
            english_addr_idx = i
        if re.search(r'पत्ता|पता', ln) and hindi_addr_idx == -1:
            hindi_addr_idx = i

    # Prefer English "Address:" label; fall back to Hindi label only if no English found
    start_idx = english_addr_idx if english_addr_idx >= 0 else hindi_addr_idx

    if start_idx >= 0:
        for i in range(start_idx, len(lines)):
            ln = lines[i]

            if not collecting:
                collecting = True
                # Start collecting from this line (extract text after "Address:")
                addr_text = re.sub(r'^.*Address\s*:?\s*', '', ln, flags=re.I)
                # Also strip Hindi "पत्ता/पता :" prefix if present
                addr_text = re.sub(r'^.*पत्ता\s*:?\s*', '', addr_text)
                addr_text = re.sub(r'^.*पता\s*:?\s*', '', addr_text).strip()
                addr_text = extract_english_only(addr_text).strip()
                if addr_text:
                    address_parts.append(addr_text)
                continue

            # 1. Skip lines that are purely boilerplate
            pure_skip = [r'signature', r'help@', r'www\.', r'Aadhaar\s*No', r'VID\s*:',
                         r'\b1947\b', r'uidai\.gov']
            if any(re.search(p, ln, re.I) for p in pure_skip):
                continue

            # 1b. Skip lines with Aadhaar number (12 digits spaced or continuous)
            if re.search(r'\b\d{4}\s+\d{4}\s+\d{4}\b', ln) or re.search(r'\b\d{12}\b', ln):
                break

            # 1c. Skip purely regional script lines (Hindi/Marathi/Telugu/Tamil/Odia etc.)
            english_content = extract_english_only(ln).strip()
            # Remove just digits and punctuation to check if any real English words remain
            english_words_only = re.sub(r'[^a-zA-Z\s]', '', english_content).strip()
            if _contains_indic_script(ln) and len(english_words_only) < 3:
                continue

            # 2. Clean merged boilerplate from line instead of skipping
            # (Essential for e-Aadhaar where columns merge)
            clean_ln = ln
            for p in [r'Date of Birth', r'\bDOB\b', r'\d{2}/\d{2}/\d{4}', r'\b(MALE|FEMALE|TRANSGENDER)\b', r'(/|\b)(MALE|FEMALE)\b']:
                clean_ln = re.sub(p, '', clean_ln, flags=re.I).strip()

            # 3. Extract only English text, strip Devanagari and other scripts
            clean_ln = extract_english_only(clean_ln).strip()
            # Strip "Address" label if it appears in subsequent lines (stacked cards)
            clean_ln = re.sub(r'\bAddress\s*:?\s*', '', clean_ln, flags=re.I).strip()
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

    # Strategy 4: Find lines with address key