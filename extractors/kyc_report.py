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

        line = line.strip()

        if label_re.search(line):

            # same line
            m = re.search(label + r'[:\s|]*(.+)', line, re.I)
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


def _extract_address(lines: List[str], name: str, father: str) -> str:

    address_parts = []
    start = False

    for line in lines:

        if "XML Verified" in line:
            start = True
            continue

        if not start:
            continue

        s = _clean(line)

        if not s:
            continue

        if re.search('|'.join(_TABLE_LABELS), s, re.I):
            continue

        if re.search(r'\*{4,}', s):
            continue

        if re.match(r'\b(MALE|FEMALE)\b', s, re.I):
            continue

        if re.search(DATE_PATTERN, s):
            continue

        if name and name.lower() in s.lower():
            continue

        if father and father.lower() in s.lower():
            continue

        if "," in s or len(s.split()) >= 3:
            address_parts.append(s)

    address = ", ".join(dict.fromkeys(address_parts))

    return address


def is_kyc_report(lines: List[str], text: str) -> bool:
    return True


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
        obj["father_name"]
    )

    if pincode and pincode not in obj["address"]:
        obj["address"] += f", {pincode}"

    obj["address"] = re.sub(r'\s+', ' ', obj["address"]).strip()

    return obj