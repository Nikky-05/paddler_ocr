import re
from typing import Dict, List, Optional


_TABLE_LABELS = [
    'id number', 'name', 'dob',
    'father name', 'gender',
    'pincode', 'address'
]


_MONTHS = {
    'jan': '01','feb': '02','mar': '03','apr': '04',
    'may': '05','jun': '06','jul': '07','aug': '08',
    'sep': '09','oct': '10','nov': '11','dec': '12'
}


DATE_PATTERN = r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}'


def _clean(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace("|", " ")
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _clean_label(s: Optional[str]) -> str:
    """Clean for label matching — also strips colons."""
    s = _clean(s)
    s = s.replace(":", " ")
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _normalize_ordinal(s: str) -> str:
    # Handle fully split: OCR may read "31st" as "3 1 st"
    s = re.sub(r'(\d)\s+(\d)\s+(st|nd|rd|th)', r'\1\2\3', s, flags=re.I)
    # Handle split digits: OCR may read "31st" as "3 1st"
    s = re.sub(r'(\d)\s+(\d)(st|nd|rd|th)', r'\1\2\3', s, flags=re.I)
    # Handle normal split: "31 st" → "31st"
    s = re.sub(r'(\d)\s+(st|nd|rd|th)', r'\1\2', s, flags=re.I)
    return s


def _fix_split_ordinal(s: str) -> str:
    """Rejoin OCR-split two-digit ordinal day numbers.

    OCR sometimes reads '31st' as '3 1 st' or '3 1st' due to spacing artefacts.
    This collapses them back before any date parsing:
      '3 1 st'  -> '31st'
      '3 1st'   -> '31st'
      '2 2 nd'  -> '22nd'
    """
    # Step 1: "3 1 st" -> "31st"  (digit SPACE digit SPACE ordinal)
    s = re.sub(r'(\d)\s+(\d)\s+(st|nd|rd|th)\b', r'\1\2\3', s, flags=re.I)
    # Step 2: "3 1st" -> "31st"  (digit SPACE digit+ordinal already joined)
    s = re.sub(r'(\d)\s+(\d(?:st|nd|rd|th))\b', r'\1\2', s, flags=re.I)
    return s


def _convert_dob(dob: str) -> str:

    dob = dob.strip()

    # Fix OCR-split ordinals like '3 1 st' -> '31st' BEFORE any other processing
    dob = _fix_split_ordinal(dob)

    # Passthrough: already in DD/MM/YYYY or DD-MM-YYYY format
    m_num = re.match(r'^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$', dob)
    if m_num:
        day = m_num.group(1).zfill(2)
        month = m_num.group(2).zfill(2)
        year = m_num.group(3)
        return f"{day}/{month}/{year}"

    dob = _normalize_ordinal(dob)

    # Use \b boundary and \d{1,2} anchored so it doesn't partially match
    # e.g. "31st" -> after _normalize_ordinal -> "31st" (no space) so group(1)=31
    m = re.search(
        r'(?<![\d])(\d{1,2})(?:st|nd|rd|th)?[\s]+([A-Za-z]+)[\s]+(\d{4})',
        dob
    )

    if not m:
        return ""

    day = m.group(1).zfill(2)
    month = m.group(2).lower()[:3]
    year = m.group(3)

    if month not in _MONTHS:
        return ""

    return f"{day}/{_MONTHS[month]}/{year}"


def _find_label_value(lines: List[str], label: str) -> str:

    label_re = re.compile(label, re.I)
    # Build regex to split at table label boundaries.
    # OCR may crop first 1-2 chars of bold labels (e.g. "Father Name" → "ather Name"),
    # so we match both full and truncated forms.
    _table_label_pattern = (
        r'\b(?:[Ii]?d\s*number|[Nn]?ame|[Dd]?ob|[Ff]?ather\s*[Nn]?ame'
        r'|[Gg]?ender|[Pp]?incode|[Aa]?ddress|[Mm]?obile)\b'
    )

    for i, line in enumerate(lines):

        cleaned = _clean_label(line)

        if label_re.search(cleaned):

            # same line
            m = re.search(label + r'[:\s|]*(.+)', cleaned, re.I)
            if m:
                val = _clean(m.group(1))
                if val:
                    # If value has a table label merged in, take portion before it
                    parts = re.split(_table_label_pattern, val, flags=re.I)
                    val = parts[0].strip(' ,|')
                    if val:
                        return val

            # next line
            if i + 1 < len(lines):
                val = _clean(lines[i+1])
                if val:
                    # If next line contains a table label, extract portion before it
                    if re.search(_table_label_pattern, val, re.I):
                        parts = re.split(_table_label_pattern, val, flags=re.I)
                        val = parts[0].strip(' ,|')
                    if val:
                        return val

            # second next line
            if i + 2 < len(lines):
                val = _clean(lines[i+2])
                if val:
                    if re.search(_table_label_pattern, val, re.I):
                        parts = re.split(_table_label_pattern, val, flags=re.I)
                        val = parts[0].strip(' ,|')
                    if val:
                        return val

    return ""


def _extract_gender(text: str) -> str:

    m = re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', text, re.I)

    if m:
        return m.group(1).upper()

    return ""


def _extract_aadhaar(text: str) -> str:

    m = re.search(r'\*{4,}\d{4}', text)

    if m:
        return m.group()

    return ""


# Numeric date patterns: DD/MM/YYYY, DD-MM-YYYY, YYYY/MM/DD, YYYY-MM-DD
_NUMERIC_DATE_PATTERN = r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b|\b(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\b'


def _convert_numeric_date(m: re.Match) -> str:
    """Convert a numeric date regex match to DD/MM/YYYY."""
    if m.group(1):  # DD/MM/YYYY
        day = m.group(1).zfill(2)
        month = m.group(2).zfill(2)
        year = m.group(3)
    else:  # YYYY/MM/DD
        year = m.group(4)
        month = m.group(5).zfill(2)
        day = m.group(6).zfill(2)
    return f"{day}/{month}/{year}"


def _extract_dob(lines: List[str], text: str) -> str:

    # OCR may crop first char: "Dob" → "ob"
    dob_val = _find_label_value(lines, r'\b[Dd]?ob\b')

    print(f"[DEBUG DOB] raw dob_val from label = {repr(dob_val)}")

    if dob_val:
        # First try text-based month name (e.g. "31st Jan 2008")
        dob = _convert_dob(dob_val)
        print(f"[DEBUG DOB] _convert_dob({repr(dob_val)}) = {repr(dob)}")
        if dob:
            return dob
        # Then try numeric format directly in the fetched value
        m_num = re.search(_NUMERIC_DATE_PATTERN, dob_val)
        if m_num:
            result = _convert_numeric_date(m_num)
            print(f"[DEBUG DOB] numeric match in dob_val => {repr(result)}")
            return result

    # Fallback: text-based date in full text (e.g. "31st Jan 2008")
    m = re.search(DATE_PATTERN, text)
    if m:
        print(f"[DEBUG DOB] DATE_PATTERN matched in full text: {repr(m.group())}")
        dob = _convert_dob(m.group())
        if dob:
            return dob

    # Fallback: numeric date in full text
    m_num = re.search(_NUMERIC_DATE_PATTERN, text)
    if m_num:
        result = _convert_numeric_date(m_num)
        print(f"[DEBUG DOB] numeric fallback in full text => {repr(result)}")
        return result

    return ""


def _extract_name(lines: List[str], text: str) -> str:

    # OCR may crop first char: "Customer" → "ustomer"
    m = re.search(
        r'[Cc]?ustomer\s*[Nn]?ame\s*[:\s]+(.+)',
        text,
        re.I
    )

    if m:
        return _clean(m.group(1))

    # OCR may crop first char: "Name" → "ame"
    name = _find_label_value(lines, r'\b[Nn]?ame\b')

    if name and not re.search(r'[Ff]?ather', name, re.I):
        return name

    return ""


def _extract_father(lines: List[str], text: str) -> str:

    # OCR often crops first 1-2 chars of bold table labels:
    #   "Father Name" → "ather Name" or "ther Name"
    # So we use flexible patterns that match both full and truncated labels.
    _father_label = r'[Ff]?ather\s*[Nn]?ame'
    _trailing_labels = r'\b(?:[Gg]?ender|[Mm]ale|[Ff]emale|[Tt]ransgender|[Pp]?incode|[Aa]?ddress|[Dd]?ob|[Ii]?d\s*number|[Mm]?obile)\b'

    father = _find_label_value(lines, _father_label)

    if not father:
        father = _find_label_value(lines, r'[Ff]?ather\b')

    # Strategy 2: scan lines directly for father label + value
    # Handles cases where _find_label_value misses due to merged table rows
    if not father:
        for i, line in enumerate(lines):
            m = re.search(
                _father_label + r'\s*[:\s|]*([A-Za-z][A-Za-z .\-]{2,})',
                line, re.I
            )
            if m:
                val = m.group(1).strip()
                # Stop at table labels that may be appended (e.g. "Singh Gender")
                val = re.split(_trailing_labels, val, flags=re.I)[0].strip()
                if len(val) >= 3:
                    father = val
                    break

    # Strategy 3: look at next line after father label line
    if not father:
        for i, line in enumerate(lines):
            if re.search(_father_label, line, re.I):
                if i + 1 < len(lines):
                    val = _clean(lines[i + 1])
                    # Strip any trailing table label that got merged
                    val = re.split(_trailing_labels, val, flags=re.I)[0].strip()
                    if val and len(val) >= 3 and re.match(r'^[A-Za-z .\-]+$', val):
                        father = val
                        break

    if father:
        # Reject if it's just a gender word
        if father.upper() in ["MALE", "FEMALE", "TRANSGENDER"]:
            father = ""
        else:
            # Clean: keep only letters, spaces, dots, hyphens
            father = re.sub(r'[^A-Za-z .\-]', '', father).strip()
            # Remove trailing noise words
            father = re.sub(r'\s+(Male|Female|Transgender)\s*$', '', father, flags=re.I).strip()
            if len(father) >= 3:
                return father

    # Fallback: extract from S/O, C/O, D/O, W/O in address text
    m = re.search(r'(?:C/O|S/O|D/O|W/O)\s*[:\-]?\s*([A-Za-z][A-Za-z ]{2,})', text, re.I)

    if m:
        val = m.group(1).strip()
        # Stop at address parts (comma, digits, etc.)
        val = re.split(r'[,\d]', val)[0].strip()
        if len(val) >= 3:
            return val

    return ""


def _extract_pincode(text: str) -> str:

    m = re.search(r'\b\d{6}\b', text)

    if m:
        return m.group()

    return ""


# ============ Address Extraction ============

def _is_noise(s: str) -> bool:
    """Check if a cleaned line is noise (not address content).
    Uses generic patterns — not tied to any specific KYC provider.
    """
    if not s:
        return True
    # masked aadhaar numbers
    if re.search(r'\*{4,}', s):
        return True
    # standalone gender
    if re.match(r'^(MALE|FEMALE|TRANSGENDER)$', s, re.I):
        return True
    # KYC report chrome (generic patterns)
    if re.match(r'^(XML\s*Verified|AADHAAR)$', s, re.I):
        return True
    if re.match(r'^KYC\b', s, re.I):
        return True
    if re.match(r'^Detailed\s+Report$', s, re.I):
        return True
    # header lines: "Customer Name:", "Customer Identification:", "KYC Status:"
    if re.match(r'^(Customer\s+(Name|Identification)|KYC\s+Status|Verification\s+From)\s*', s, re.I):
        return True
    # date header line like "Date: 05 Mar, 2026 09:05:02 AM"
    if re.match(r'^Date\s*\d', s, re.I):
        return True
    # KID line
    if re.match(r'^KID\s*', s, re.I):
        return True
    # standalone UUID fragments
    if re.match(r'^[0-9a-f]{8,}[\-0-9a-f]*$', s, re.I) and '-' in s:
        return True
    # Mobile number line: "Mobile 1234567890", "obile 1234567890", "Iobile ..."
    if re.match(r'^[MmIi]?obile\s*\d', s, re.I):
        return True
    # Standalone phone number (10+ digits)
    if re.match(r'^\d{10,}$', s.replace(' ', '')):
        return True
    return False


def _is_table_label(s: str) -> bool:
    """Check if line is a table row label (not address content).
    Handles OCR-truncated labels (e.g. 'ather Name' for 'Father Name').
    """
    return bool(re.match(
        r'^([Ii]?d\s*number|[Nn]?ame|[Dd]?ob|[Ff]?ather\s*[Nn]?ame|[Gg]?ender|[Pp]?incode|[Mm]?obile)\b',
        s, re.I
    ))


def _clean_address(addr: str) -> str:
    """Clean up final address string."""
    # remove leaked "Address" label word (full or partial) anywhere
    addr = re.sub(r'\b[Aa]ddres\w*\b', '', addr)
    addr = re.sub(r'\bddress\b', '', addr, flags=re.I)
    # remove leaked Mobile number (e.g. "Mobile 8376887112", "obile 123...", "Iobile ...")
    addr = re.sub(r',?\s*[MmIi]?obile\s*\d[\d\s]*', '', addr, flags=re.I)
    addr = re.sub(r',\s*,', ',', addr)
    addr = re.sub(r'\s+', ' ', addr)
    return addr.strip(" ,")


def _extract_address(lines: List[str], name: str, father: str,
                     records: List[Dict] = None) -> str:
    """Extract address using multiple strategies for robustness.

    KYC table structure:  Address is always the LAST row.
    OCR may split multi-line addresses or reorder label vs value.
    """

    print("\n[DEBUG] ===== Address Extraction =====")
    print(f"[DEBUG] Lines ({len(lines)}):")
    for i, ln in enumerate(lines):
        print(f"[DEBUG]   {i}: '{ln}'")

    # Strategy 1 (after Pincode) is most reliable: Address is always last
    # row in KYC table, and Pincode is always just above it.
    addr = _strategy_after_pincode(lines)
    if addr and len(addr) > 10:
        print(f"[DEBUG] Strategy 1 (after Pincode) => '{addr}'")
        return addr

    # Fallback: find "Address" label in lines
    addr = _strategy_address_label(lines)
    if addr and len(addr) > 10:
        print(f"[DEBUG] Strategy 2 (Address label) => '{addr}'")
        return addr

    # Fallback: regex S/O, C/O, D/O, W/O in full text
    addr = _strategy_care_of_regex("\n".join(lines))
    if addr and len(addr) > 10:
        print(f"[DEBUG] Strategy 3 (C/O regex) => '{addr}'")
        return addr

    # Last resort: collect comma-heavy lines after table
    addr = _strategy_comma_lines(lines)
    if addr and len(addr) > 10:
        print(f"[DEBUG] Strategy 4 (comma lines) => '{addr}'")
        return addr

    print("[DEBUG] All strategies failed for address")
    return ""


def _strategy_after_pincode(lines: List[str]) -> str:
    """Address is always the last row — collect everything after the Pincode row.

    In narrow images, OCR may read the address value lines BEFORE the 'Address'
    label because the label is vertically centered in the cell. But Pincode is
    always the row above, so everything after Pincode (excluding noise and
    the 'Address' label word itself) is the address.
    """

    pincode_idx = -1

    for i, line in enumerate(lines):
        s = _clean_label(line)
        # "Pincode 202138", "incode 755026", or "Pincode" alone
        # OCR may crop first char: "Pincode" → "incode"
        if re.search(r'\b[Pp]?incode\b', s, re.I):
            pincode_idx = i
            break

    if pincode_idx < 0:
        # fallback: find a standalone 6-digit line after Gender
        for i, line in enumerate(lines):
            s = line.strip()
            if re.match(r'^\d{6}$', s):
                if i > 0:
                    prev = _clean_label(lines[i-1])
                    if re.search(r'([Gg]?ender|Male|Female)', prev, re.I):
                        pincode_idx = i
                        break

    if pincode_idx < 0:
        # last fallback: find line containing both "Pincode" word fragment
        # OR a 6-digit number right after Gender/Male/Female line
        in_table = False
        for i, line in enumerate(lines):
            if 'XML Verified' in line or 'Xml Verified' in line:
                in_table = True
                continue
            if not in_table:
                continue
            s = _clean_label(line)
            # line has pincode label word anywhere (handles "incode" too)
            if re.search(r'[Pp]?in\s*code', s, re.I):
                pincode_idx = i
                break
            # standalone 6-digit after Gender
            if i > 0 and re.match(r'^\d{6}$', line.strip()):
                prev = _clean_label(lines[i-1])
                if re.search(r'([Gg]?ender|Male|Female)', prev, re.I):
                    pincode_idx = i
                    break

    if pincode_idx < 0:
        return ""

    address_parts = []
    for line in lines[pincode_idx + 1:]:
        s = _clean(line)
        if not s:
            continue

        if _is_noise(s):
            continue

        if _is_table_label(s):
            continue

        # strip the "Address" label word if present
        s = re.sub(r'\bAddres\w*\b', '', s, flags=re.I).strip()
        s = re.sub(r'\bddress\b', '', s, flags=re.I).strip()
        if not s:
            continue

        address_parts.append(s)

    if not address_parts:
        return ""

    return _clean_address(", ".join(dict.fromkeys(address_parts)))


def _strategy_address_label(lines: List[str]) -> str:
    """Find 'Address' label and collect value from same line + subsequent lines."""

    address_parts = []
    capture = False

    for i, line in enumerate(lines):
        s = _clean(line)
        if not s:
            continue

        # flexible match: "Address", "AddressS/O", "Addres" etc.
        if not capture and re.search(r'Addres', s, re.I):
            capture = True
            # remove the label portion
            addr = re.sub(r'^Addres\w*\s*', '', s, flags=re.I).strip()
            if addr:
                address_parts.append(addr)
            continue

        if capture and _is_table_label(s):
            break

        if not capture:
            continue

        if _is_noise(s):
            continue

        address_parts.append(s)

    if not address_parts:
        return ""

    return _clean_address(", ".join(dict.fromkeys(address_parts)))


def _strategy_care_of_regex(text: str) -> str:
    """Extract address by finding S/O, C/O, D/O, W/O pattern in full text."""

    # Match: S/O: Name, addr parts, ..., pincode, ..., state
    m = re.search(
        r'([SCDW]/O\s*[:\-]?\s*[A-Za-z][\w\s,./\-]+(?:\d{6})[\w\s,./\-]*)',
        text, re.I
    )
    if m:
        addr = m.group(1).strip()
        # clean up: remove trailing noise
        addr = re.split(r'\n(?=\s*(?:Id\s*number|Name|Dob|Father|Gender|Pincode))', addr)[0]
        return _clean_address(addr)

    return ""


def _strategy_comma_lines(lines: List[str]) -> str:
    """Last resort: find comma-heavy lines after 'XML Verified' section."""

    capture = False
    address_parts = []

    for line in lines:
        s = _clean(line)
        if not s:
            continue

        if 'XML Verified' in line or 'Xml Verified' in line:
            capture = True
            continue

        if not capture:
            continue

        if _is_noise(s) or _is_table_label(s):
            continue

        if re.search(r'\*{4,}', s):
            continue

        # address-like: has commas and looks like location text
        if ',' in s and len(s) > 15:
            address_parts.append(s)

    if not address_parts:
        return ""

    return _clean_address(", ".join(dict.fromkeys(address_parts)))


# ============ KYC Report Detection ============

def is_kyc_report(lines: List[str], text: str) -> bool:
    """Detect if document is a KYC report (not raw Aadhaar card).

    KYC Report markers:
    - "KYC" keyword in text
    - Table with labels: "id number", "name", "dob", "father name", etc.
    - "KYC Status:" field
    - Company/report header pattern
    """
    text_lower = text.lower()

    # KYC Report keyword indicators
    # OCR may crop first char: "Customer" → "ustomer", "KYC" → "YC"
    kyc_markers = [
        'kyc report',
        'kyc status',
        'customer name',
        'ustomer name',
        'customer identification',
        'ustomer identification',
        'verification from'
    ]

    # Check for KYC markers
    has_kyc_marker = any(marker in text_lower for marker in kyc_markers)

    # Check for KYC report table structure
    # OCR may truncate labels: "id number" → "l number", "gender" → "ender"
    has_table_labels = all(
        re.search(pattern, text_lower)
        for pattern in [r'(?:id\s*)?number', r'(?:n)?ame', r'(?:d)?ob', r'(?:g)?ender']
    )

    # If both KYC indicators found, it's likely a KYC report
    if has_kyc_marker and has_table_labels:
        return True

    # Additional check: if "KYC Report" explicitly in first few lines
    first_lines = "\n".join(lines[:10]).lower()
    if 'kyc' in first_lines and ('report' in first_lines or 'status' in first_lines):
        return True

    # Otherwise treat as raw Aadhaar card
    return False


# ============ Main Extractor ============

def extract_kyc_report(lines: List[str], text: str, records: List[Dict]) -> Dict:

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
        "nationality": "INDIAN"
    }

    text = _normalize_ordinal(text)

    obj["aadhaar_number"] = _extract_aadhaar(text)

    obj["name"] = _extract_name(lines, text)

    obj["dob"] = _extract_dob(lines, text)

    obj["father_name"] = _extract_father(lines, text)

    obj["gender"] = _extract_gender(text)

    pincode = _extract_pincode(text)

    obj["address"] = _extract_address(
        lines,
        obj["name"],
        obj["father_name"],
        records
    )

    if pincode and pincode not in obj["address"]:
        obj["address"] += f", {pincode}"

    obj["address"] = re.sub(r'\s+', ' ', obj["address"]).strip()

    return obj