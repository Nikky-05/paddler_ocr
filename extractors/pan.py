"""
PAN Card Extraction Logic
Extracts information from PAN cards (C).
"""

import re
from typing import Dict, List
from .utils import nearest_line


def extract_pan(lines: List[str], text: str) -> Dict:
    """Extract data from PAN card."""
    obj = {
        "name": "",
        "father_name": "",
        "pan_number": "",
        "dob": ""
    }

    pan_match = re.search(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b', text)
    if pan_match:
        obj['pan_number'] = pan_match.group(1)

    for i, ln in enumerate(lines):
        up = ln.upper()

        if 'FATHER' in up or 'S/O' in up or 'SON OF' in up:
            below = nearest_line(lines, i, 1)
            if below and len(below) > 2 and not re.search(r'FATHER|NAME|MOTHER|DATE|BIRTH', below, re.I):
                obj['father_name'] = below
            else:
                above = nearest_line(lines, i, -1)
                if above and len(above) > 2:
                    obj['father_name'] = above

        if 'NAME' in up and 'FATHER' not in up and 'MOTHER' not in up and 'FILE' not in up:
            if re.match(r'^(NUMBER\s+)?NAME[:\s]*$', up.strip()):
                val = nearest_line(lines, i, 1)
                if val and not re.search(r'FATHER|DATE|BIRTH|PAN|NUMBER', val, re.I):
                    obj['name'] = val
            elif ':' in ln:
                val = ln.split(':')[-1].strip()
                if val:
                    obj['name'] = val

            if not obj['name']:
                val = nearest_line(lines, i, 1)
                if val and not re.search(r'FATHER|DATE|BIRTH|PAN|NUMBER', val, re.I):
                    obj['name'] = val

    if not obj['name']:
        for ln in lines:
            if re.match(r'^[A-Z\s\.]{5,50}$', ln) and not re.search(r'\b(INCOME|GOVERNMENT|INDIA|INCOME TAX|NAME|NUMBER|ACCOUNT|CARD|PERMANENT)\b', ln.upper()):
                obj['name'] = ln.strip()
                break

    dob_match = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})', text)
    if dob_match:
        obj['dob'] = dob_match.group(1)

    return obj

