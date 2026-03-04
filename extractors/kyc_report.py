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

# All known table label patterns — used to check if a line is or starts with a label
_ALL_LABEL_RE = re.compile(
    r'^(?:id\s*number|name|dob|father\s*name|gender|pincode|address)\b',
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
        return "".replace('-', '/')

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


def _clean_val(s: Optional[str]) -> str:
    """Clean OCR value: strip colons, pipes, and whitespace."""
    if not s:
        return ""
    # Strip leading/trailing pipes, colons, dots, and spaces
    s = s.strip().strip('|').strip(':').strip('.').strip()
    # Normalize internal spaces
    s = re.sub(r'\s+', ' ', s)
    return s


def _find_value_in_lines(lines: List[str], label_pattern: str, skip_father_for_name: bool = False) -> str:
    """Search for a label and return its value (same line or next)."""
    for i, ln in enumerate(lines):
        ln_s = ln.strip()
        # Strategy A: Label and Value on same line
        # Use non-greedy match for label to handle merged borders
        m = re.search(f'[\\s|]*({label_pattern})[\\s:|]+(.+)', ln_s, re.I)
        if not m:
             # Try without mandatory space for merged OCR (e.g. Dob1st)
             m = re.search(f'[\\s|]*({label_pattern})[\\s:|]*(.+)', ln_s, re.I)
        
        if m:
            label_part = m.group(1).lower()
            val = m.group(2).strip()
            if skip_father_for_name and 'father' in label_part:
                continue
            # If value is just another label keyword, it's not the value
            if re.match(r'^(?:name|dob|father|gender|pincode|address|id)\b', val, re.I):
                continue
            return _clean_val(val)

        # Strategy B: Label alone, value on next line (Strategy B) or previous line (Strategy C)
        if re.search(f'^[\s|]*{label_pattern}[\s:|]*$', ln_s, re.I):
            if skip_father_for_name and 'father' in ln_s.lower():
                continue
            # Try next line (Strategy B)
            if i + 1 < len(lines):
                val = lines[i + 1].strip()
                if not re.match(r'^(?:name|dob|father|gender|pincode|address|id|xml|aadhaar)\b', val, re.I):
                    return _clean_val(val)
            # Try previous line (Strategy C - common in some table OCRs)
            if i - 1 >= 0:
                val = lines[i - 1].strip()
                # Must look like data, not another label or section header
                if not re.match(r'^(?:name|dob|father|gender|pincode|address|id|xml|aadhaar|kyc|customer|kid)\b', val, re.I):
                    return _clean_val(val)
    return ""


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
    # For Aadhaar, prioritize the masked pattern search to avoid cross-field errors
    m = re.search(r'(\*{4,}\d{4})', text)
    if m:
        obj['aadhaar_number'] = m.group(1)
    
    if not obj['aadhaar_number']:
        obj['aadhaar_number'] = _find_value_in_lines(lines, r'Id\s*number')

    # ===================================================================
    # 2. NAME
    # ===================================================================
    # Strategy A: "Customer Name: XXX" in header 
    m = re.search(r'Customer\s*Name\s*[:\s|]+(.*?)(?:\s+(?:Date|Customer|KID|Identification):|$)', text, re.I)
    if m:
        obj['name'] = _clean_val(m.group(1))

    # Strategy B: table scan
    if not obj['name']:
        obj['name'] = _find_value_in_lines(lines, r'(?<!Father\s)Name', skip_father_for_name=True)

    # Strategy C: merged Name line fallback (e.g. Line 14: "number Md Furqan")
    if not obj['name']:
        for ln in lines:
            if 'number' in ln.lower() and re.search(r'[A-Z][a-z]+', ln):
                m = re.search(r'number\s+(.*)', ln, re.I)
                if m:
                    candidate = _clean_val(m.group(1))
                    if not re.match(r'^(?:dob|father|gender|pincode|address|id)\b', candidate, re.I):
                        obj['name'] = candidate
                        break

    # ===================================================================
    # 3. DOB
    # ===================================================================
    # Strategy A: Find in table
    dob_val = _find_value_in_lines(lines, r'Dob')
    if not dob_val:
        # Strategy B: Flexible search on lines since Dob can be merged with labels
        for ln in lines:
            m = re.search(r'\bDob\s*[:\s|]*(\d{1,2}\s*(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})', ln, re.I)
            if m:
                dob_val = m.group(1).strip()
                break
    
    if dob_val:
        obj['dob'] = _convert_kyc_dob(dob_val)
    
    # Strategy C: Global text search
    if not obj['dob']:
        m = re.search(r'\bDob\s*[:\s]*(\d{1,2}\s*(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})', text_norm, re.I)
        if m:
            obj['dob'] = _convert_kyc_dob(m.group(1).strip())

    # ===================================================================
    # 4. FATHER NAME
    # ===================================================================
    # Strategy A: Table scan (same line, next line, or previous line)
    obj['father_name'] = _find_value_in_lines(lines, r'Father(?:\s*Name)?')
    if obj['father_name']:
        # Cleanup: sometimes OCR merges 'Name' into the value (e.g. 'NameHUSENSAB')
        obj['father_name'] = re.sub(r'^Name\s*[:\s]*', '', obj['father_name'], flags=re.I).strip()

    # Strategy B: Merged line regex (handles "Dob Father NameHUSENSAB")
    if not obj['father_name']:
        for ln in lines:
            m = re.search(r'Father\s*Name\s*[:\s]*([A-Z]+(?:\s+[A-Z]+)*)', ln, re.I)
            if m:
                obj['father_name'] = _clean_val(m.group(1))
                break

    # Strategy B: split labels "Father" ... "Name" (merged with Gender or something)
    if not obj['father_name']:
         for i, ln in enumerate(lines):
             if re.search(r'^[\s|]*Father[\s|]*$', ln.strip(), re.I):
                 # Look at next line, might be "Name Gender Male" or "Name Md Irfan"
                 if i + 1 < len(lines):
                     next_ln = lines[i+1].strip()
                     m = re.search(r'^Name\s+(.*?)(?:\s+Gender|$)', next_ln, re.I)
                     if m:
                         obj['father_name'] = _clean_val(m.group(1))
                         break
                     # If next line is just "Name", try the line after that
                     if re.match(r'^Name\s*$', next_ln, re.I) and i + 2 < len(lines):
                         obj['father_name'] = _clean_val(lines[i+2])
                         break

    # Strategy C: name-like fallback
    if not obj['father_name'] and obj['name']:
        person_name_lower = obj['name'].lower()
        table_started = False
        for ln in lines:
            if 'XML Verified' in ln: table_started = True
            if not table_started: continue
            s = _clean_val(ln)
            if re.match(r'^[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){1,2}$', s):
                if s.lower() != person_name_lower and not re.match(r'^(?:Name|Dob|Gender|Pincode|Address|Father|Male|Female)$', s, re.I):
                    obj['father_name'] = s
                    break

    # ===================================================================
    # 5. GENDER
    # ===================================================================
    gender_val = _find_value_in_lines(lines, r'Gender')
    if gender_val and re.match(r'^(?:male|female|transgender)', gender_val, re.I):
        obj['gender'] = gender_val.upper()
    
    if not obj['gender']:
        for ln in lines:
            m = re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', ln, re.I)
            if m:
                obj['gender'] = m.group(1).upper()
                break

    # ===================================================================
    # 6. PINCODE & ADDRESS
    # ===================================================================
    pincode = _find_value_in_lines(lines, r'Pincode')
    if not pincode:
        m = re.search(r'\b(\d{6})\b', text)
        if m: pincode = m.group(1)

    # Address collection: gather ALL lines between "XML Verified" and End 
    # that look like address content, filtering out known labels/values.
    addr_parts = []
    in_table = False
    for i, ln in enumerate(lines):
        if 'XML Verified' in ln:
            in_table = True
            continue
        if not in_table: continue
        
        s = _clean_val(ln)
        if not s: continue
        
        # EXCLUSIONS: Skip lines that are clearly other fields
        if re.search(r'\*{4,}\d{4}', s): continue # Masked Aadhaar
        if re.search(r'\d{1,2}\s*(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}', s, re.I): continue # Date
        if re.match(r'^(?:Male|Female|Transgender)$', s, re.I): continue # Standalone gender
        if re.search(r'^Pincode$', s, re.I): continue # Standalone label

        # Stop if we hit a signature/footer section (optional)
        if re.search(r'^(?:Note|Disclaimer|Signature)', s, re.I): break

        # Check if line looks like address component
        is_candidate = False
        if ',' in s or re.search(r'\b(?:Bihar|Karnataka|State|District|Village|Ward|Nagar|Colony)\b', s, re.I):
            is_candidate = True
        elif re.search(r'\d{6}', s): # pincode line
            is_candidate = True
        elif len(s.split()) >= 3 and not re.match(r'^(?:Name|Dob|Father|Gender|Pincode|Address|Id|number)', s, re.I):
            # 3+ words and not a label
            is_candidate = True
            
        if is_candidate:
            # Final check: make sure it's not the Name or Father Name
            if s.lower() != obj['name'].lower() and s.lower() != obj['father_name'].lower():
                addr_parts.append(s)

    if addr_parts:
        # Deduplicate and join
        seen = set()
        unique_parts = []
        for p in addr_parts:
            if p.lower() not in seen:
                unique_parts.append(p)
                seen.add(p.lower())
        obj['address'] = ', '.join(unique_parts)
        if pincode and pincode not in obj['address']:
            obj['address'] += f", {pincode}"
    elif pincode:
        obj['address'] = pincode

    # Final cleanup of address: multiple commas, etc.
    obj['address'] = re.sub(r',\s*,', ',', obj['address'])
    obj['address'] = re.sub(r'\s+', ' ', obj['address']).strip()

    print("\n[DEBUG] ===== KYC Report Extraction Results =====")
    for key, val in obj.items():
        print(f"[DEBUG]   {key}: '{val}'")
    print("[DEBUG] ==========================================\n")

    return obj
