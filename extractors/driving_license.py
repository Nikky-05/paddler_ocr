"""
Driving License Extraction Logic
Extracts information from Driving Licenses (D).
"""

import re
from typing import Dict, List
from .utils import nearest_line


def extract_driving_license(lines: List[str], text: str) -> Dict:
    """Extract data from Driving License."""
    obj = {
        "dl_number": "",
        "name": "",
        "dob": "",
        "address": "",
        "validity": "",
        "issue_date": "",
        "blood_group": "",
        "cov": "",
        "father_name": ""
    }

    # DL Number
    dl_match = re.search(r'DL\s*No\s*[:\.]?\s*([A-Z0-9\s]+)', text, re.I)
    if dl_match:
        raw_dl = dl_match.group(1).strip()
        parts = re.split(r'\n|\s{2,}', raw_dl)
        if parts:
            obj['dl_number'] = parts[0].strip()
    else:
        for ln in lines:
            mh_match = re.search(r'\b([A-Z]{2}\d{2}\s?[A-Z0-9]{5,})\b', ln)
            if mh_match:
                obj['dl_number'] = mh_match.group(1).strip()
                break

    # Dates - DOI (Issue Date) - also match "DO/" (OCR misread)
    doi_match = re.search(r'(DOI|DO/)\s*[:\.]?\s*(\d{2}[\-\/]\d{2}[\-\/]\d{4})', text, re.I)
    if doi_match:
        obj['issue_date'] = doi_match.group(2)
    
    # Fallback: Look for date after "DOI" or "DO/" label
    if not obj['issue_date']:
        for i, ln in enumerate(lines):
            if re.search(r'\b(DOI|DO/)\b', ln, re.I):
                # Check same line
                date_match = re.search(r'(\d{2}[\-\/]\d{2}[\-\/]\d{4})', ln)
                if date_match:
                    obj['issue_date'] = date_match.group(1)
                    break

    # Valid Till
    valid_match = re.search(r'Valid\s*Till\s*[:\.]?\s*(\d{2}[\-\/]\d{2}[\-\/]\d{4})', text, re.I)
    if valid_match:
        obj['validity'] = valid_match.group(1)

    # DOB - Enhanced extraction with multiple patterns
    # Pattern 1: DOB with various separators
    dob_match = re.search(r'DOB\s*[:\.]?\s*(\d{2}[-\/]\d{2}[-\/]\d{4})', text, re.I)
    if dob_match:
        obj['dob'] = dob_match.group(1)
    
    # Pattern 2: D.O.B or D0B (OCR misread)
    if not obj['dob']:
        dob_match = re.search(r'D[\.\s]?O[\.\s]?B\s*[:\.]?\s*(\d{2}[-\/]\d{2}[-\/]\d{4})', text, re.I)
        if dob_match:
            obj['dob'] = dob_match.group(1)
    
    # Pattern 3: Date of Birth
    if not obj['dob']:
        dob_match = re.search(r'Date\s*of\s*Birth\s*[:\.]?\s*(\d{2}[-\/]\d{2}[-\/]\d{4})', text, re.I)
        if dob_match:
            obj['dob'] = dob_match.group(1)
    
    # Fallback 1: Search in lines for DOB label
    if not obj['dob']:
        for i, ln in enumerate(lines):
            if re.search(r'\b(DOB|D\.O\.B|D\s*O\s*B|Date\s*of\s*Birth)\b', ln, re.I):
                # Check same line for date
                date_match = re.search(r'(\d{2}[-\/]\d{2}[-\/]\d{4})', ln)
                if date_match:
                    obj['dob'] = date_match.group(1)
                    break
                # Check next line
                if i + 1 < len(lines):
                    next_ln = lines[i + 1].strip()
                    date_match = re.search(r'(\d{2}[-\/]\d{2}[-\/]\d{4})', next_ln)
                    if date_match:
                        obj['dob'] = date_match.group(1)
                        break

    # Name - Enhanced extraction
    for i, ln in enumerate(lines):
        # Look for "Name" label
        if re.search(r'\bName\b\s*[:\.]?', ln, re.I):
            # Try to extract from same line
            val = re.sub(r'^.*\bName\b\s*[:\.]?', '', ln, flags=re.I).strip().lstrip(':').strip()
            # Remove trailing labels
            val = re.sub(r'\s*(BG|DOB|DOI|Add|S/D/W|of).*$', '', val, flags=re.I).strip()

            if val and len(val) > 2 and not re.match(r'^\d{2}[\-\/]\d{2}[\-\/]\d{4}$', val):
                obj['name'] = val
                break
            
            # Try next line
            val = nearest_line(lines, i, 1).lstrip(':').strip()
            if val and not re.search(r'S/D/W|Add|DOB|BG|DOI|Valid|S/DM|of', val, re.I):
                if not re.search(r'\b(LMV|MCWG|MCWOG|MCW|HGMV|HPMV|TRANS|LDRXCV)\b', val, re.I):
                    if not re.match(r'^\d{2}[\-\/]\d{2}[\-\/]\d{4}$', val):
                        obj['name'] = val
                        break

    # Fallback: Name is usually above S/D/W line
    if not obj['name']:
        for i, ln in enumerate(lines):
            # Match S/D/W, S/DM, S/D/M, etc.
            if re.search(r'\bS/D[/\s]?[WM]\b', ln, re.I):
                above = nearest_line(lines, i, -1)
                if above and not re.search(r'Name|DOB|BG|Add|Address', above, re.I):
                    # Clean up the name
                    above = re.sub(r'\s*(BG|DOB|DOI).*$', '', above, flags=re.I).strip()
                    if len(above) > 2:
                        obj['name'] = above
                        break

    # Father's Name - Extract from S/D/W, S/DM, etc. (Son/Daughter/Wife/Mother of)
    for i, ln in enumerate(lines):
        # Match S/D/W, S/DM, S/D/M and variations
        if re.search(r'\bS/D[/\s]?[WM]\b', ln, re.I):
            # Try to extract from same line - handle both "S/D/W" and "S/DM"
            val = re.sub(r'.*\bS/D[/\s]?[WM]\b\s*(?:of)?\s*[:\.]?', '', ln, flags=re.I).strip()
            if val and len(val) > 2 and not re.search(r'(Add|Address|DOB|BG|Name)', val, re.I):
                obj['father_name'] = val
                break
            
            # Try next line
            val = nearest_line(lines, i, 1)
            if val and len(val) > 2 and not re.search(r'(Add|Address|DOB|BG|Name|PIN|Signature)', val, re.I):
                obj['father_name'] = val
                break

    # Address - Enhanced extraction
    for i, ln in enumerate(lines):
        if re.match(r'^(Add|Address)\b\s*[:\.]?', ln, re.I):
            addr_parts = []
            same_line_content = re.sub(r'^(Add|Address)\s*[:\.]?', '', ln, flags=re.I).strip()
            if same_line_content and not re.search(r'^(S/D/W|Name|DOB)', same_line_content, re.I):
                addr_parts.append(same_line_content)

            for j in range(i + 1, min(len(lines), i + 8)):
                next_ln = lines[j].strip()
                if not next_ln:
                    continue
                
                # Stop conditions
                if re.search(r'(Signature|Issuing|Authority|MH\d{2}\s*\d{7})', next_ln, re.I):
                    break
                
                # Include PIN line and stop
                if re.search(r'\bPIN\b', next_ln, re.I):
                    pin_match = re.search(r'(PIN\s*[:\.]?\s*\d{6})', next_ln, re.I)
                    if pin_match:
                        addr_parts.append(pin_match.group(0))
                    break
                
                # Skip labels
                if re.search(r'^(S/D/W|Name|DOB|BG|DOI|Valid)', next_ln, re.I):
                    continue
                
                # Skip COV types
                if re.search(r'\b(LMV|MCWG|MCWOG|MCW|HGMV|HPMV|TRANS|LDRXCV)\b', next_ln, re.I):
                    continue
                
                addr_parts.append(next_ln)

            if addr_parts:
                obj['address'] = ", ".join(addr_parts)
                break

    # Blood Group
    bg_match = re.search(r'BG\s*[:\.]?\s*([A-Z]{1,2}[\+\-])', text, re.I)
    if bg_match:
        obj['blood_group'] = bg_match.group(1)

    # COV - Enhanced to include MCWOG variant
    cov_types = set()
    for ln in lines:
        # Look for COV label and extract from same or next line
        if re.search(r'\bCOV\b', ln, re.I):
            # Check same line and next few lines
            idx = lines.index(ln)
            for check_ln in [ln] + lines[idx+1:min(len(lines), idx+3)]:
                parts = check_ln.split()
                for p in parts:
                    p_clean = p.strip().upper()
                    # Match standard COV types + MCWOG variant
                    if p_clean in ['LMV', 'MCWG', 'MCWOG', 'MCW', 'HGMV', 'HPMV', 'TRANS', 'LDRXCV']:
                        cov_types.add(p_clean)
    
    # Fallback: Search entire text for COV types
    if not cov_types:
        for ln in lines:
            parts = ln.split()
            for p in parts:
                p_clean = p.strip().upper()
                if p_clean in ['LMV', 'MCWG', 'MCWOG', 'MCW', 'HGMV', 'HPMV', 'TRANS', 'LDRXCV']:
                    cov_types.add(p_clean)
    
    if cov_types:
        obj['cov'] = ", ".join(sorted(list(cov_types)))

    return obj

