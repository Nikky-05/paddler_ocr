"""
KYC Report Extraction Logic

Detects and extracts data from KYC verification reports (e.g., SPEEL FINANCE)
that contain Aadhaar verification data in a tabular format.

These reports have masked Aadhaar numbers like *********3699 and a table
with labels: Id number, Name, Dob, Father Name, Gender, Pincode, Address.

Extraction uses a search-based approach (regex on full text + per-line scans)
rather than sequential label matching, to handle unpredictable OCR line splits
from table structures.
"""

import re
from typing import Dict, List, Optional


# Labels we look for in the table (case-insensitive matching)
_TABLE_LABELS = ['id number', 'name', 'dob', 'father name', 'gender', 'pincode', 'address']

# Month name -> number mapping
_MONTHS = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
    'january': '01', 'february': '02', 'march': '03', 'april': '04',
    'june': '06', 'july': '07', 'august': '08', 'september': '09',
    'october': '10', 'november': '11', 'december': '12',
}

# All known table label patterns — used to check if a line is "just a label"
_ALL_LABEL_RE = re.compile(
    r'^(?:id\s*number|name|dob|father\s*name|gender|pincode|address)\s*:?\s*$',
    re.I,
)


def is_kyc_report(lines: List[str], text: str) -> bool:
    """Detect whether the OCR output is from a KYC verification report.

    Two-tier detection:
    - Primary (need 1+): 'KYC REPORT', 'KYC STATUS', 'DETAILED REPORT' in text
    - Secondary (need 1+): 'XML VERIFIED', 'VERIFICATION FROM', masked Aadhaar
      pattern (****1234), or 3+ table labels present
    """
    text_upper = text.upper()

    # Primary signals
    primary_keywords = ['KYC REPORT', 'KYC STATUS', 'DETAILED REPORT']
    primary_hit = any(kw in text_upper for kw in primary_keywords)

    if not primary_hit:
        return False

    # Secondary signals
    secondary_count = 0

    if 'XML VERIFIED' in text_upper:
        secondary_count += 1
    if 'VERIFICATION FROM' in text_upper:
        secondary_count += 1
    # Masked Aadhaar pattern: 4+ asterisks followed by 4 digits
    if re.search(r'\*{4,}\d{4}', text):
        secondary_count += 1
    # Count how many table labels are present
    text_lower = text.lower()
    label_count = sum(1 for lbl in _TABLE_LABELS if lbl in text_lower)
    if label_count >= 3:
        secondary_count += 1

    return secondary_count >= 1


def _normalize_ordinal_spaces(s: str) -> str:
    """Fix OCR spaces in ordinal suffixes: '1 st' -> '1st', '23 rd' -> '23rd'."""
    return re.sub(r'(\d)\s+(st|nd|rd|th)\b', r'\1\2', s, flags=re.I)


def _convert_kyc_dob(dob_str: str) -> str:
    """Convert KYC report DOB format to DD/MM/YYYY.

    Handles:
    - '1st Jan 1995'  -> '01/01/1995'
    - '1 st Jan 1995' -> '01/01/1995'  (OCR space in ordinal)
    - '23rd Feb 2000'  -> '23/02/2000'
    - Already DD/MM/YYYY -> pass through
    """
    if not dob_str:
        return ""

    dob_str = dob_str.strip()

    # Fix OCR-introduced spaces in ordinal suffixes
    dob_str = _normalize_ordinal_spaces(dob_str)

    # Already in DD/MM/YYYY or DD-MM-YYYY format
    if re.match(r'^\d{2}[/\-]\d{2}[/\-]\d{4}$', dob_str):
        return dob_str.replace('-', '/')

    # Pattern: "1st Jan 1995" or "23rd February 2000"
    match = re.match(
        r'(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})',
        dob_str, re.I
    )
    if match:
        day = match.group(1).zfill(2)
        month_str = match.group(2).lower()
        year = match.group(3)
        month = _MONTHS.get(month_str, _MONTHS.get(month_str[:3], ''))
        if month:
            return f"{day}/{month}/{year}"

    # Fallback: return as-is
    return dob_str


def _find_value_after_label(lines: List[str], label_idx: int, label_end_col: int,
                            label_line_text: str) -> Optional[str]:
    """Get value for a label: same-line remainder first, else next non-label line."""
    remainder = label_line_text[label_end_col:].strip().lstrip(':').strip()
    if remainder:
        return remainder

    # Try next lines, skipping lines that are themselves bare labels
    for j in range(label_idx + 1, min(label_idx + 3, len(lines))):
        candidate = lines[j].strip()
        if not candidate:
            continue
        if _ALL_LABEL_RE.match(candidate):
            # Next line is just another label — value is missing
            return None
        return candidate

    return None


def extract_kyc_report(lines: List[str], text: str, records: List[Dict]) -> Dict:
    """Extract fields from a KYC verification report.

    Uses a multi-strategy approach resilient to unpredictable OCR line splits:
    1. Regex search on full text for same-line "Label Value" patterns
    2. Per-line label scan with next-line lookahead (skipping bare labels)
    3. Header fallbacks (e.g. "Customer Name: Md Furqan")
    4. Pattern-based value detection (masked Aadhaar, MALE/FEMALE, dates)

    Returns dict with same keys as extract_aadhaar.
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
        "nationality": "INDIAN",
    }

    print("\n[DEBUG] ===== KYC Report Extraction Started =====")
    print(f"[DEBUG] Total OCR lines: {len(lines)}")
    for i, ln in enumerate(lines):
        print(f"[DEBUG]   Line {i}: '{ln}'")

    # Normalize text: fix "1 st" -> "1st" etc. for searching
    text_norm = _normalize_ordinal_spaces(text)

    # ===================================================================
    # 1. AADHAAR NUMBER — masked pattern like *********3699
    # ===================================================================
    m = re.search(r'(\*{4,}\d{4})', text)
    if m:
        obj['aadhaar_number'] = m.group(1)
        print(f"[DEBUG] Aadhaar (masked): {obj['aadhaar_number']}")

    # ===================================================================
    # 2. NAME — try multiple strategies
    # ===================================================================
    # Strategy A: "Customer Name: XXX" in header (very reliable)
    m = re.search(r'Customer\s*Name\s*:\s*(.+)', text, re.I)
    if m:
        obj['name'] = m.group(1).strip()
        print(f"[DEBUG] Name (from header): {obj['name']}")

    # Strategy B: table "Name <value>" on same line (but NOT "Father Name")
    if not obj['name']:
        for ln in lines:
            # Match "Name Md Furqan" but not "Father Name Md Irfan"
            m = re.match(r'^(?<!Father\s)Name\s+(.+)', ln.strip(), re.I)
            if m:
                val = m.group(1).strip()
                # Reject if value starts with another label keyword
                if not re.match(r'^(?:dob|father|gender|pincode|address|id)\b', val, re.I):
                    obj['name'] = val
                    print(f"[DEBUG] Name (same-line): {obj['name']}")
                    break
        # Also try: line starts with "Name" exactly, not preceded by "Father" on the
        # same line — need to check the full line doesn't contain "Father"
        if not obj['name']:
            for i, ln in enumerate(lines):
                ln_s = ln.strip()
                if re.match(r'^Name\s*$', ln_s, re.I) and 'father' not in ln_s.lower():
                    val = _find_value_after_label(lines, i, len(ln_s), ln_s)
                    if val and not re.match(r'^(?:dob|father|gender|pincode|address|id)\b', val, re.I):
                        obj['name'] = val
                        print(f"[DEBUG] Name (next-line): {obj['name']}")
                        break

    # ===================================================================
    # 3. DOB — date patterns
    # ===================================================================
    # Strategy A: "Dob <value>" on same line
    for ln in lines:
        m = re.match(r'^Dob\s+(.+)', ln.strip(), re.I)
        if m:
            obj['dob'] = _convert_kyc_dob(m.group(1).strip())
            print(f"[DEBUG] DOB (same-line): {obj['dob']}")
            break

    # Strategy B: "Dob" alone, value on next line
    if not obj['dob']:
        for i, ln in enumerate(lines):
            if re.match(r'^Dob\s*$', ln.strip(), re.I):
                val = _find_value_after_label(lines, i, len(ln.strip()), ln.strip())
                if val:
                    obj['dob'] = _convert_kyc_dob(val)
                    print(f"[DEBUG] DOB (next-line): {obj['dob']}")
                    break

    # Strategy C: search full text for date-like pattern near "Dob" keyword
    if not obj['dob']:
        m = re.search(r'\bDob\s*[:\s]*(\d{1,2}\s*(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})', text_norm, re.I)
        if m:
            obj['dob'] = _convert_kyc_dob(m.group(1).strip())
            print(f"[DEBUG] DOB (text search): {obj['dob']}")

    # ===================================================================
    # 4. FATHER NAME
    # ===================================================================
    # Strategy A: "Father Name <value>" on same line
    for ln in lines:
        m = re.match(r'^Father\s*Name\s+(.+)', ln.strip(), re.I)
        if m:
            val = m.group(1).strip()
            if not re.match(r'^(?:dob|gender|pincode|address|id)\b', val, re.I):
                obj['father_name'] = val
                print(f"[DEBUG] Father (same-line): {obj['father_name']}")
                break

    # Strategy B: "Father Name" alone (possibly split across lines), value on next
    if not obj['father_name']:
        for i, ln in enumerate(lines):
            ln_s = ln.strip()
            # "Father Name" on one line
            if re.match(r'^Father\s*Name\s*$', ln_s, re.I):
                val = _find_value_after_label(lines, i, len(ln_s), ln_s)
                if val and not re.match(r'^(?:dob|gender|pincode|address|id|name)\b', val, re.I):
                    obj['father_name'] = val
                    print(f"[DEBUG] Father (next-line): {obj['father_name']}")
                    break
            # "Father" alone — next line might be "Name" (OCR split), value after that
            if re.match(r'^Father\s*$', ln_s, re.I):
                # Check if next line is "Name" or "Name <value>"
                if i + 1 < len(lines):
                    next_ln = lines[i + 1].strip()
                    m2 = re.match(r'^Name\s+(.*)', next_ln, re.I)
                    if m2 and m2.group(1).strip():
                        obj['father_name'] = m2.group(1).strip()
                        print(f"[DEBUG] Father (split label, same-line value): {obj['father_name']}")
                        break
                    elif re.match(r'^Name\s*$', next_ln, re.I):
                        # "Name" alone — value on line after that
                        if i + 2 < len(lines):
                            val = lines[i + 2].strip()
                            if val and not re.match(r'^(?:dob|gender|pincode|address|id)\b', val, re.I):
                                obj['father_name'] = val
                                print(f"[DEBUG] Father (split label, next-next-line): {obj['father_name']}")
                                break

    # Strategy C: per-line search for "Father Name <value>" with flexible spacing
    if not obj['father_name']:
        for ln in lines:
            m = re.search(
                r'Father\s*Name\s*[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                ln.strip(), re.I
            )
            if m:
                val = m.group(1).strip()
                # Reject if it's a label keyword
                if not re.match(r'^(?:dob|gender|pincode|address|id)\b', val, re.I):
                    obj['father_name'] = val
                    print(f"[DEBUG] Father (line search): {obj['father_name']}")
                    break

    # Strategy D: find a name-like value that isn't the person's name
    # Only search in the table/values section (after "XML Verified" marker)
    if not obj['father_name'] and obj['name']:
        person_name_lower = obj['name'].strip().lower()
        # Find table start marker
        table_start = 0
        for i, ln in enumerate(lines):
            if re.search(r'XML\s*Verified', ln, re.I):
                table_start = i + 1
                break
        for ln in lines[table_start:]:
            s = ln.strip()
            # Must look like a name: 2-3 title-case words, all alpha
            if not re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}$', s):
                continue
            # Must not be the person's own name
            if s.lower() == person_name_lower:
                continue
            # Must not be a label or keyword
            if re.match(r'^(?:Name|Dob|Gender|Pincode|Address|Father|Male|Female|'
                        r'Transgender|Detailed|Report|Verified|Approved|Aadhaar)$', s, re.I):
                continue
            obj['father_name'] = s
            print(f"[DEBUG] Father (name-like fallback): {obj['father_name']}")
            break

    # ===================================================================
    # 5. GENDER — MALE / FEMALE keyword search
    # ===================================================================
    # Strategy A: "Gender <value>" on same line
    for ln in lines:
        m = re.match(r'^Gender\s+(.+)', ln.strip(), re.I)
        if m:
            val = m.group(1).strip()
            if re.match(r'^(?:male|female|transgender)\b', val, re.I):
                obj['gender'] = val.upper()
                print(f"[DEBUG] Gender (same-line): {obj['gender']}")
                break

    # Strategy B: "Gender" alone, value on next line
    if not obj['gender']:
        for i, ln in enumerate(lines):
            if re.match(r'^Gender\s*$', ln.strip(), re.I):
                val = _find_value_after_label(lines, i, len(ln.strip()), ln.strip())
                if val and re.match(r'^(?:male|female|transgender)\b', val, re.I):
                    obj['gender'] = val.strip().upper()
                    print(f"[DEBUG] Gender (next-line): {obj['gender']}")
                    break

    # Strategy C: standalone MALE/FEMALE anywhere in lines (fallback)
    if not obj['gender']:
        for ln in lines:
            m = re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', ln, re.I)
            if m:
                obj['gender'] = m.group(1).upper()
                print(f"[DEBUG] Gender (keyword search): {obj['gender']}")
                break

    # ===================================================================
    # 6. PINCODE — 6-digit number
    # ===================================================================
    pincode = ""
    for ln in lines:
        m = re.match(r'^Pincode\s+(\d{6})\b', ln.strip(), re.I)
        if m:
            pincode = m.group(1)
            break
    if not pincode:
        for i, ln in enumerate(lines):
            if re.match(r'^Pincode\s*$', ln.strip(), re.I):
                val = _find_value_after_label(lines, i, len(ln.strip()), ln.strip())
                if val:
                    m = re.match(r'^(\d{6})\b', val)
                    if m:
                        pincode = m.group(1)
                        break

    # ===================================================================
    # 7. ADDRESS — collect from "Address" label through end / next section
    # ===================================================================
    addr_parts = []

    def _is_address_line(line: str) -> bool:
        """Check if a line looks like address content (not a stray value)."""
        s = line.strip()
        if not s:
            return False
        # Reject: bare labels
        if _ALL_LABEL_RE.match(s):
            return False
        # Reject: masked Aadhaar
        if re.match(r'^\*+\d{4}$', s):
            return False
        # Reject: bare number (e.g. pincode alone, aadhaar digits)
        if re.match(r'^\d+$', s):
            return False
        # Reject: looks like a date (ordinal + month + year)
        if re.match(r'^\d{1,2}\s*(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}$', s, re.I):
            return False
        # Reject: single gender keyword
        if re.match(r'^(?:Male|Female|Transgender)$', s, re.I):
            return False
        # Reject: single short name-like word (no commas, no digits, <= 3 words, all alpha)
        words = s.replace(',', '').split()
        if len(words) <= 3 and all(w.isalpha() for w in words) and ',' not in s:
            # Could be a name value — reject unless it has address indicators
            if not re.search(r'\b(?:nagar|colony|ward|village|bihar|delhi|pradesh|punjab|'
                             r'maharashtra|rajasthan|gujarat|tamil|karnataka|kerala|bengal)\b',
                             s, re.I):
                return False
        return True

    # Strategy A: "Address <value>" on same line, then continue collecting
    for i, ln in enumerate(lines):
        m = re.match(r'^Address\s+(.+)', ln.strip(), re.I)
        if m:
            addr_parts.append(m.group(1).strip())
            for j in range(i + 1, len(lines)):
                next_ln = lines[j].strip()
                if re.match(r'^(?:id\s*number|name|dob|father|gender|pincode)\b', next_ln, re.I):
                    break
                if _is_address_line(next_ln):
                    addr_parts.append(next_ln)
            break

    # Strategy B: "Address" alone, value on next lines (filter non-address)
    if not addr_parts:
        for i, ln in enumerate(lines):
            if re.match(r'^Address\s*$', ln.strip(), re.I):
                for j in range(i + 1, len(lines)):
                    next_ln = lines[j].strip()
                    if re.match(r'^(?:id\s*number|name|dob|father|gender|pincode)\b', next_ln, re.I):
                        break
                    if _is_address_line(next_ln):
                        addr_parts.append(next_ln)
                break

    # Strategy C: find address by content pattern (lines with commas + locations)
    if not addr_parts:
        for ln in lines:
            s = ln.strip()
            if ',' in s and re.search(r'[A-Za-z]', s):
                # Has commas and letters — likely address
                if not re.match(r'^(?:Customer|KYC|KID|Date)\b', s, re.I):
                    addr_parts.append(s)

    if addr_parts:
        combined = ', '.join(addr_parts)
        combined = re.sub(r',\s*,', ',', combined)
        combined = re.sub(r'\s+', ' ', combined).strip()
        obj['address'] = combined
        if pincode and pincode not in obj['address']:
            obj['address'] = f"{obj['address']}, {pincode}"
    elif pincode:
        obj['address'] = pincode

    print("\n[DEBUG] ===== KYC Report Extraction Results =====")
    for key, val in obj.items():
        print(f"[DEBUG]   {key}: '{val}'")
    print("[DEBUG] ==========================================\n")

    return obj
