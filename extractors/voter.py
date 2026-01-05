"""
Voter ID Extraction Logic
Extracts information from Voter ID cards (B).
"""

import re
from typing import Dict, List
from .utils import nearest_line


def extract_voter(lines: List[str], text: str) -> Dict:
    """Extract data from Voter ID."""
    obj = {
        "name": "",
        "father_name": "",
        "epic_number": "",
        "dob": "",
        "gender": "",
        "address": ""
    }

    # EPIC Number
    epic = re.search(r'\b([A-Z]{3}\d{7})\b', text)
    if epic:
        obj['epic_number'] = epic.group(1)

    # Name
    for i, ln in enumerate(lines):
        if re.search(r"Elector'?s?\s*Name", ln, re.I):
            val = re.sub(r".*Elector'?s?\s*Name\s*[:\.]?", '', ln, flags=re.I).strip()
            if val and len(val) > 2:
                obj['name'] = val
            else:
                val = nearest_line(lines, i, 1)
                if val and not re.search(r'Father|Address|Sex|Date|Birth', val, re.I):
                    obj['name'] = val
            break

    if not obj['name']:
        for i, ln in enumerate(lines):
            if re.search(r'^Name\s*[:\.]?', ln, re.I):
                val = re.sub(r'^Name\s*[:\.]?', '', ln, flags=re.I).strip()
                if val and len(val) > 2:
                    obj['name'] = val
                else:
                    val = nearest_line(lines, i, 1)
                    if val and not re.search(r'Father|Address|Sex|Date|Birth', val, re.I):
                        obj['name'] = val
                break

    # Father's Name
    for i, ln in enumerate(lines):
        if re.search(r"Father'?s?\s*Name", ln, re.I):
            val = re.sub(r".*Father'?s?\s*Name\s*[:\.]?", '', ln, flags=re.I).strip()
            if val and len(val) > 2:
                obj['father_name'] = val
            else:
                val = nearest_line(lines, i, 1)
                if val and not re.search(r'Name|Address|Sex|Date|Birth', val, re.I):
                    obj['father_name'] = val
            break

    if not obj['father_name']:
        for i, ln in enumerate(lines):
            if re.search(r'S/O|SON OF', ln, re.I):
                val = re.sub(r'.*S/O\s*[:\.]?', '', ln, flags=re.I).strip()
                val = re.sub(r'.*SON OF\s*[:\.]?', '', val, flags=re.I).strip()
                if val and len(val) > 2:
                    obj['father_name'] = val
                break

    # DOB
    for i, ln in enumerate(lines):
        if re.search(r'Date\s*of\s*Birth|DOB', ln, re.I):
            m = re.search(r'(\d{2}[\-\/]\d{2}[\-\/]\d{4})', ln)
            if m:
                obj['dob'] = m.group(1)
            else:
                val = nearest_line(lines, i, 1)
                m = re.search(r'(\d{2}[\-\/]\d{2}[\-\/]\d{4})', val)
                if m:
                    obj['dob'] = m.group(1)
            break

    if not obj['dob']:
        m = re.search(r'(\d{2}[\-\/]\d{2}[\-\/]\d{4})', text)
        if m:
            obj['dob'] = m.group(1)

    # Gender
    for i, ln in enumerate(lines):
        if re.search(r'Sex|Gender', ln, re.I):
            if re.search(r'MALE|FEMALE', ln, re.I):
                m = re.search(r'(MALE|FEMALE)', ln, re.I)
                if m:
                    obj['gender'] = m.group(1).upper()
            else:
                val = nearest_line(lines, i, 1)
                if re.search(r'MALE|FEMALE', val, re.I):
                    m = re.search(r'(MALE|FEMALE)', val, re.I)
                    if m:
                        obj['gender'] = m.group(1).upper()
            break

    # Address
    for i, ln in enumerate(lines):
        if re.search(r'Address', ln, re.I):
            addr = []
            for j in range(i + 1, min(len(lines), i + 8)):
                next_ln = lines[j].strip()
                if not next_ln:
                    continue
                if re.search(r'(Name|Father|Sex|Date|Birth|EPIC)', next_ln, re.I):
                    break
                addr.append(next_ln)
            if addr:
                obj['address'] = ", ".join(addr)
            break

    return obj

