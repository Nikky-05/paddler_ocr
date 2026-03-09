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
    return re.sub(r'(\d)\s+(st|nd|rd|th)', r'\1\2', s, flags=re.I)


def _convert_dob(dob: str) -> str:

    dob = _normalize_ordinal(dob)

    m = re.search(
        r'(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})',
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

    for i, line in enumerate(lines):

        cleaned = _clean_label(line)

        if label_re.search(cleaned):

            # same line
            m = re.search(label + r'[:\s|]*(.+)', cleaned, re.I)
            if m:
                val = _clean(m.group(1))
                if val:
                    return val

            # next line
            if i + 1 < len(lines):
                val = _clean(lines[i+1])
                if val and not re.search('|'.join(_TABLE_LABELS), val, re.I):
                    return val

            # second next line
            if i + 2 < len(lines):
                val = _clean(lines[i+2])
                if val and not re.search('|'.join(_TABLE_LABELS), val, re.I):
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


def _extract_dob(lines: List[str], text: str) -> str:

    dob_val = _find_label_value(lines, r'\bDob\b')

    if dob_val:
        dob = _convert_dob(dob_val)
        if dob:
            return dob

    m = re.search(DATE_PATTERN, text)

    if m:
        return _convert_dob(m.group())

    return ""


def _extract_name(lines: List[str], text: str) -> str:

    m = re.search(
        r'Customer\s*Name\s*[:\s]+(.+)',
        text,
        re.I
    )

    if m:
        return _clean(m.group(1))

    name = _find_label_value(lines, r'\bName\b')

    if name and not re.search(r'Father', name, re.I):
        return name

    return ""


def _extract_father(lines: List[str], text: str) -> str:

    father = _find_label_value(lines, r'Father\s*Name')

    if not father:
        father = _find_label_value(lines, r'\bFather\b')

    if father:

        if father.upper() in ["MALE","FEMALE","TRANSGENDER"]:
            father = ""

        if re.match(r'^[A-Za-z ]{3,}$', father):
            return father.strip()

    # fallback extraction from address text
    m = re.search(r'(?:C/O|S/O|D/O|W/O)\s*[:\-]?\s*([A-Za-z ]{3,})', text, re.I)

    if m:
        return m.group(1).strip()

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
    return False


def _is_table_label(s: str) -> bool:
    """Check if line is a table row label (not address content)."""
    return bool(re.match(
        r'^(Id\s*number|Name|Dob|Father\s*Name|Gender|Pincode)\b', s, re.I
    ))


def _clean_address(addr: str) -> str:
    """Clean up final address string."""
    # remove leaked "Address" label word (full or partial) anywhere
    addr = re.sub(r'\b[Aa]ddres\w*\b', '', addr)
    addr = re.sub(r'\bddress\b', '', addr, flags=re.I)
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
        # "Pincode 202138", "483440 Pincode", or "Pincode" alone
        if re.search(r'\bPincode\b', s, re.I):
            pincode_idx = i
            break

    if pincode_idx < 0:
        # fallback: find a standalone 6-digit line after Gender
        for i, line in enumerate(lines):
            s = line.strip()
            if re.match(r'^\d{6}$', s):
                if i > 0:
                    prev = _clean_label(lines[i-1])
                    if re.search(r'(Gender|Male|Female)', prev, re.I):
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
            # line has pincode label word anywhere
            if re.search(r'pin\s*code', s, re.I):
                pincode_idx = i
                break
            # standalone 6-digit after Gender
            if i > 0 and re.match(r'^\d{6}$', line.strip()):
                prev = _clean_label(lines[i-1])
                if re.search(r'(Gender|Male|Female)', prev, re.I):
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
    kyc_markers = [
        'kyc report',
        'kyc status',
        'customer name',
        'customer identification',
        'verification from'
    ]

    # Check for KYC markers
    has_kyc_marker = any(marker in text_lower for marker in kyc_markers)

    # Check for KYC report table structure
    has_table_labels = all(
        label in text_lower
        for label in ['id number', 'name', 'dob', 'gender']
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