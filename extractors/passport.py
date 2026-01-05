"""
Passport Extraction Logic
Extracts information from Passports (A).
"""

import re
from typing import Dict, List
from .utils import nearest_line


def extract_passport(lines: List[str], text: str) -> Dict:
    """Extract data from Passport."""
    obj = {
        "passport_number": "",
        "name": "",
        "nationality": "",
        "dob": "",
        "place_of_birth": "",
        "gender": "",
        "address": "",
        "father_name": "",
        "mother_name": "",
        "spouse_name": ""
    }

    # MRZ Strategy
    mrz_lines = [ln for ln in lines if '<<' in ln and len(ln) > 10]
    mrz_lines.sort(key=len, reverse=True)

    if len(mrz_lines) >= 1:
        line1 = None
        line2 = None

        for ln in mrz_lines:
            if ln.startswith('P') and '<' in ln:
                line1 = ln
            elif re.search(r'\d', ln) and ('IND' in ln or '<' in ln):
                line2 = ln

        if line1:
            try:
                content = line1[5:] if line1.startswith('P<IND') else line1[2:]
                parts = content.split('<<')
                surname = parts[0].replace('<', '').strip()
                given_name = parts[1].replace('<', ' ').strip() if len(parts) > 1 else ""
                full_name = f"{given_name} {surname}".strip()
                if full_name:
                    obj['name'] = full_name
            except Exception:
                pass

        if line2:
            try:
                pp_no = line2[:9].replace('<', '')
                if re.match(r'^[A-Z0-9]+$', pp_no):
                    obj['passport_number'] = pp_no

                ind_idx = line2.find('IND')
                if ind_idx != -1:
                    dob_str = line2[ind_idx+3:ind_idx+9]
                    if re.match(r'\d{6}', dob_str):
                        yy = int(dob_str[0:2])
                        mm = dob_str[2:4]
                        dd = dob_str[4:6]
                        year = f"19{yy}" if yy > 30 else f"20{yy}"
                        obj['dob'] = f"{dd}/{mm}/{year}"

                    sex_char = line2[ind_idx+10] if len(line2) > ind_idx+10 else ''
                    if sex_char in ['M', 'F']:
                        obj['gender'] = 'MALE' if sex_char == 'M' else 'FEMALE'
            except Exception:
                pass

    # Passport Number fallback
    if not obj['passport_number']:
        for ln in lines:
            m = re.search(r'\b([A-Z][0-9]{7})\b', ln)
            if m:
                obj['passport_number'] = m.group(1)
                break

    # Name fallback
    if not obj['name']:
        surname = ""
        given_name = ""
        for i, ln in enumerate(lines):
            if 'SURNAME' in ln.upper():
                val = re.sub(r'.*SURNAME[:\s]*', '', ln, flags=re.I).strip()
                if val and len(val) > 2 and not re.search(r'GIVEN|NAME|SEX|DOB|INDIAN', val, re.I):
                    surname = val
                else:
                    val = nearest_line(lines, i, 1)
                    if val and not re.search(r'GIVEN|NAME|SEX|DOB|INDIAN', val, re.I):
                        surname = val
            if 'GIVEN NAME' in ln.upper() or 'GIVEN' in ln.upper():
                val = re.sub(r'.*GIVEN\s*NAME[:\s]*', '', ln, flags=re.I).strip()
                if val and len(val) > 2 and not re.search(r'SURNAME|SEX|DOB|INDIAN', val, re.I):
                    given_name = val
                else:
                    val = nearest_line(lines, i, 1)
                    if val and not re.search(r'SURNAME|SEX|DOB|INDIAN', val, re.I):
                        given_name = val

        if surname or given_name:
            obj['name'] = f"{given_name} {surname}".strip()

    # DOB fallback
    if not obj['dob']:
        for i, ln in enumerate(lines):
            if re.search(r'(DATE OF BIRTH|DOB|Date of Birth)', ln, re.I):
                m = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})', ln)
                if m:
                    obj['dob'] = m.group(1)
                    break
                val = nearest_line(lines, i, 1)
                m = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})', val)
                if m:
                    obj['dob'] = m.group(1)
                    break

    # Gender fallback
    if not obj['gender']:
        for i, ln in enumerate(lines):
            if re.search(r'\b(Sex|Gender)\b', ln, re.I):
                # Check on same line first
                if re.search(r'[/:\s]\s*M\b', ln, re.I) or re.search(r'\bMALE\b', ln, re.I):
                    obj['gender'] = 'MALE'
                    break
                elif re.search(r'[/:\s]\s*F\b', ln, re.I) or re.search(r'\bFEMALE\b', ln, re.I):
                    obj['gender'] = 'FEMALE'
                    break
                
                # Check nearest line
                val = nearest_line(lines, i, 1)
                if val:
                    if re.search(r'^\s*M\b', val, re.I) or re.search(r'\bMALE\b', val, re.I):
                        obj['gender'] = 'MALE'
                        break
                    elif re.search(r'^\s*F\b', val, re.I) or re.search(r'\bFEMALE\b', val, re.I):
                        obj['gender'] = 'FEMALE'
                        break

    # Place of Birth
    for i, ln in enumerate(lines):
        if re.search(r'(PLACE OF BIRTH|Place of Birth)', ln, re.I):
            val = re.sub(r'.*PLACE OF BIRTH[:\s]*', '', ln, flags=re.I).strip()
            # Clean junk characters
            val = re.sub(r'^[^\w\s]+', '', val).strip()
            
            if val and len(val) > 2 and not re.search(r'PLACE|ISSUE|DATE|FILE|EXPIRY|PASSPORT', val, re.I):
                obj['place_of_birth'] = val
                break
            
            val = nearest_line(lines, i, 1)
            if val:
                val = re.sub(r'^[^\w\s]+', '', val).strip()
                if val and not re.search(r'PLACE|ISSUE|DATE|FILE|SEX|GENDER|EXPIRY|PASSPORT|DETAILS', val, re.I):
                    obj['place_of_birth'] = val
                    break

    # Nationality
    for i, ln in enumerate(lines):
        if re.search(r'(Nationality|NATIONALITY)', ln):
            val = re.sub(r'.*Nationality[:\s]*', '', ln, flags=re.I).strip()
            if val and len(val) > 2:
                obj['nationality'] = val
                break
            val = nearest_line(lines, i, 1)
            if val and not re.search(r'DATE|PLACE|SEX', val, re.I):
                obj['nationality'] = val
                break

    if not obj['nationality'] and ('REPUBLIC OF INDIA' in text.upper() or 'IND' in text):
        obj['nationality'] = 'INDIAN'

    # Address
    for i, ln in enumerate(lines):
        if re.search(r'\b(Address|ADDRESS)\b', ln):
            addr_parts = []
            for j in range(i + 1, min(len(lines), i + 15)):
                next_ln = lines[j].strip()
                if not next_ln:
                    continue
                # Stop words
                if re.search(r'(FILE|OLD PASSPORT|Date of Issue|Place of Issue|Passport No)', next_ln, re.I):
                    break
                # If we encounter a new label block
                if re.search(r'^(Father|Mother|Spouse|Name of|Pin|P\.I\.N)', next_ln, re.I) and len(next_ln) < 20: 
                    if re.search(r'PIN', next_ln, re.I):
                         addr_parts.append(next_ln) # capture PIN line
                         break
                    continue
                
                addr_parts.append(next_ln)
                # If looks like PIN code at end
                if re.search(r'PIN\s*[:\-\s]*\d{6}', next_ln, re.I) or re.search(r'\b\d{6}\b', next_ln):
                    break

            if addr_parts:
                full_addr = ", ".join(addr_parts)
                # Cleanup
                full_addr = re.sub(r'\s*,\s*,\s*', ', ', full_addr)
                full_addr = re.sub(r'^,\s*|,\s*$', '', full_addr)
                # Filter out obvious non-address garbage if it slipped in
                if len(full_addr) > 5:
                    obj['address'] = full_addr
                break

    def is_valid_human_name(s):
        if not s: return False
        # Reject digits/symbols
        if re.search(r'[\d<>]', s): return False
        # Must have at least 2 chars
        if len(s) < 2: return False
        # If strictly uppercase (typical passport), allow. If Mixed, require Proper Case.
        # But mostly reject "Passport No" or "File No" masquerading as name
        if re.search(r'(Address|Passport|File|Mother|Spouse)', s, re.I): return False
        return True

    # Father's Name (back page)
    for i, ln in enumerate(lines):
        if re.search(r"(Father'?s?\s*Name|Name of Father)", ln, re.I):
            val = re.sub(r".*(?:Father'?s?\s*Name|Name of Father)\s*[:\.]?", '', ln, flags=re.I).strip()
            
            # Check validation
            if is_valid_human_name(val) and len(val) > 2:
                obj['father_name'] = val
            else:
                # Try next line
                val = nearest_line(lines, i, 1)
                if val and is_valid_human_name(val):
                     obj['father_name'] = val
            break

    # Mother's Name (back page)
    for i, ln in enumerate(lines):
        if re.search(r"(Mother'?s?\s*Name|Name of Mother)", ln, re.I):
            val = re.sub(r".*(?:Mother'?s?\s*Name|Name of Mother)\s*[:\.]?", '', ln, flags=re.I).strip()
            
            if is_valid_human_name(val) and len(val) > 2:
                obj['mother_name'] = val
            else:
                val = nearest_line(lines, i, 1)
                if val and is_valid_human_name(val):
                    obj['mother_name'] = val
            break

    # Spouse Name (back page)
    for i, ln in enumerate(lines):
        if re.search(r"(Spouse'?s?\s*Name|Name of Spouse)", ln, re.I):
            val = re.sub(r".*(?:Spouse'?s?\s*Name|Name of Spouse)\s*[:\.]?", '', ln, flags=re.I).strip()
            
            if is_valid_human_name(val) and len(val) > 2:
                obj['spouse_name'] = val
            else:
                val = nearest_line(lines, i, 1)
                if val and is_valid_human_name(val):
                    obj['spouse_name'] = val
            break

    return obj

