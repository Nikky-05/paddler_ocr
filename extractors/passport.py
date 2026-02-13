"""
Passport Extraction Logic
Extracts information from Indian Passports (A).

Handles:
- MRZ (Machine Readable Zone) parsing for both line 1 and line 2
- Visual field extraction with Hindi/English label support
- Front page: Name, DOB, Gender, Place of Birth, Nationality, Passport Number
- Back page: Address, Father/Mother/Spouse names
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

    print("\n[DEBUG] ===== Passport Extraction Started =====")
    print(f"[DEBUG] Total OCR lines: {len(lines)}")
    for i, ln in enumerate(lines[:30]):
        print(f"[DEBUG]   Line {i}: '{ln}'")

    # =========================================================================
    # MRZ Strategy - parse both MRZ lines
    # =========================================================================
    line1 = None
    line2 = None

    # Find MRZ line 1: starts with P and has << (name separator)
    for ln in lines:
        cleaned = ln.replace(' ', '')
        if cleaned.startswith('P') and '<<' in cleaned and len(cleaned) > 30:
            line1 = cleaned
            break

    # Find MRZ line 2: long alphanumeric+< string with digits, NOT starting with P<
    # Line 2 does NOT necessarily have '<<', only single '<' separators
    for ln in lines:
        cleaned = ln.replace(' ', '')
        if line1 and cleaned == line1:
            continue
        # MRZ line 2 is ~44 chars of [A-Z0-9<], contains digits and often 'IND'
        if len(cleaned) > 28 and re.match(r'^[A-Z0-9<]+$', cleaned):
            if re.search(r'\d', cleaned) and not cleaned.startswith('P<'):
                line2 = cleaned
                break

    print(f"[DEBUG] MRZ Line 1: {line1}")
    print(f"[DEBUG] MRZ Line 2: {line2}")

    # Parse MRZ line 1 -> name
    if line1:
        try:
            content = line1[5:] if line1.startswith('P<IND') else line1[2:]
            parts = content.split('<<')
            surname = parts[0].replace('<', '').strip()
            given_name = parts[1].replace('<', ' ').strip() if len(parts) > 1 else ""
            full_name = f"{given_name} {surname}".strip()
            if full_name:
                obj['name'] = full_name
                print(f"[DEBUG] Name from MRZ: {obj['name']}")
        except Exception:
            pass

    # Parse MRZ line 2 -> passport number, DOB, gender
    if line2:
        try:
            # Passport number: first 9 chars (may include < as filler)
            pp_no = line2[:9].replace('<', '')
            if re.match(r'^[A-Z]{1,2}[0-9]+$', pp_no) and len(pp_no) >= 7:
                obj['passport_number'] = pp_no
                print(f"[DEBUG] Passport number from MRZ: {obj['passport_number']}")

            # Find IND (nationality) to anchor DOB and gender positions
            ind_idx = line2.find('IND')
            if ind_idx != -1:
                # DOB is 6 digits immediately after IND: YYMMDD
                dob_str = line2[ind_idx+3:ind_idx+9]
                if re.match(r'\d{6}', dob_str):
                    yy = int(dob_str[0:2])
                    mm = dob_str[2:4]
                    dd = dob_str[4:6]
                    year = f"19{yy}" if yy > 30 else f"20{yy}"
                    obj['dob'] = f"{dd}/{mm}/{year}"
                    print(f"[DEBUG] DOB from MRZ: {obj['dob']}")

                # Check digit after DOB, then sex character
                # Position: IND(3) + DOB(6) + check(1) = 10 chars after IND start
                sex_char = line2[ind_idx+10] if len(line2) > ind_idx+10 else ''
                if sex_char in ['M', 'F']:
                    obj['gender'] = 'MALE' if sex_char == 'M' else 'FEMALE'
                    print(f"[DEBUG] Gender from MRZ: {obj['gender']}")
        except Exception as e:
            print(f"[DEBUG] MRZ line 2 parse error: {e}")

    # =========================================================================
    # Passport Number fallback
    # =========================================================================
    if not obj['passport_number']:
        # Indian passport: 1-2 uppercase letters + 6-7 digits = 8 chars total
        for ln in lines:
            m = re.search(r'\b([A-Z]{1,2}\d{6,7})\b', ln)
            if m:
                candidate = m.group(1)
                if len(candidate) == 8:
                    obj['passport_number'] = candidate
                    print(f"[DEBUG] Passport number from fallback: {obj['passport_number']}")
                    break

    # =========================================================================
    # Name fallback (label-based)
    # =========================================================================
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

    # =========================================================================
    # DOB fallback
    # =========================================================================
    if not obj['dob']:
        # Strategy A: English label
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

    if not obj['dob']:
        # Strategy B: Hindi label (जन्म तिथि / जन्म तारीख)
        for i, ln in enumerate(lines):
            if 'जन्म' in ln or 'तिथि' in ln or 'तारी' in ln:
                m = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})', ln)
                if m:
                    obj['dob'] = m.group(1)
                    break
                val = nearest_line(lines, i, 1)
                m = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})', val)
                if m:
                    obj['dob'] = m.group(1)
                    break

    if not obj['dob']:
        # Strategy C: Scan for any DD/MM/YYYY that looks like a birth date (year 1940-2015)
        exclude_keywords = ['issue', 'expiry', 'valid', 'download', 'print']
        for ln in lines:
            if any(kw in ln.lower() for kw in exclude_keywords):
                continue
            m = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})', ln)
            if m:
                date_str = m.group(1)
                year = int(date_str[-4:])
                if 1940 <= year <= 2015:
                    obj['dob'] = date_str
                    print(f"[DEBUG] DOB from date scan: {obj['dob']}")
                    break

    # =========================================================================
    # Gender fallback
    # =========================================================================
    if not obj['gender']:
        # Strategy A: English label
        for i, ln in enumerate(lines):
            if re.search(r'\b(Sex|Gender)\b', ln, re.I):
                if re.search(r'[/:\s]\s*M\b', ln, re.I) or re.search(r'\bMALE\b', ln, re.I):
                    obj['gender'] = 'MALE'
                    break
                elif re.search(r'[/:\s]\s*F\b', ln, re.I) or re.search(r'\bFEMALE\b', ln, re.I):
                    obj['gender'] = 'FEMALE'
                    break
                val = nearest_line(lines, i, 1)
                if val:
                    if re.search(r'^\s*M\b', val, re.I) or re.search(r'\bMALE\b', val, re.I):
                        obj['gender'] = 'MALE'
                        break
                    elif re.search(r'^\s*F\b', val, re.I) or re.search(r'\bFEMALE\b', val, re.I):
                        obj['gender'] = 'FEMALE'
                        break

    if not obj['gender']:
        # Strategy B: Hindi label (लिंग) or standalone M/F near date
        for i, ln in enumerate(lines):
            if 'लिंग' in ln or 'लिग' in ln:
                if re.search(r'\bM\b', ln) or 'पुरुष' in ln:
                    obj['gender'] = 'MALE'
                    break
                elif re.search(r'\bF\b', ln) or 'महिला' in ln:
                    obj['gender'] = 'FEMALE'
                    break
                val = nearest_line(lines, i, 1)
                if val:
                    if re.search(r'^\s*M\s*$', val) or 'पुरुष' in val:
                        obj['gender'] = 'MALE'
                        break
                    elif re.search(r'^\s*F\s*$', val) or 'महिला' in val:
                        obj['gender'] = 'FEMALE'
                        break

    if not obj['gender']:
        # Strategy C: Standalone M or F on a line (common in passport visual zone)
        for ln in lines:
            stripped = ln.strip()
            if stripped == 'M':
                obj['gender'] = 'MALE'
                break
            elif stripped == 'F':
                obj['gender'] = 'FEMALE'
                break

    # =========================================================================
    # Place of Birth
    # =========================================================================
    for i, ln in enumerate(lines):
        if re.search(r'(PLACE OF BIRTH|Place of Birth|जन्म\s*स्थान)', ln, re.I):
            val = re.sub(r'.*(?:PLACE OF BIRTH|Place of Birth|जन्म\s*स्थान)[:\s]*', '', ln, flags=re.I).strip()
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

    if not obj['place_of_birth']:
        # Fallback: look for place_of_birth pattern: CITY,STATE or CITY, STATE
        # on lines that contain a comma and Indian state names
        indian_states = [
            'MAHARASHTRA', 'KARNATAKA', 'TAMIL NADU', 'KERALA', 'TELANGANA',
            'ANDHRA PRADESH', 'GUJARAT', 'RAJASTHAN', 'MADHYA PRADESH',
            'UTTAR PRADESH', 'BIHAR', 'WEST BENGAL', 'PUNJAB', 'HARYANA',
            'ODISHA', 'JHARKHAND', 'CHHATTISGARH', 'ASSAM', 'GOA', 'DELHI',
        ]
        for ln in lines:
            ln_upper = ln.upper().strip()
            # Skip lines that are clearly labels or MRZ
            if '<' in ln or re.search(r'(DATE|ISSUE|EXPIRY|PASSPORT|REPUBLIC|INDIA|SURNAME|GIVEN)', ln, re.I):
                continue
            for state in indian_states:
                if state in ln_upper and ',' in ln:
                    obj['place_of_birth'] = ln.strip()
                    print(f"[DEBUG] Place of birth from state match: {obj['place_of_birth']}")
                    break
            if obj['place_of_birth']:
                break

    # =========================================================================
    # Nationality
    # =========================================================================
    for i, ln in enumerate(lines):
        if re.search(r'(Nationality|NATIONALITY|राष्ट्रीयता|नागरिकता)', ln):
            val = re.sub(r'.*(?:Nationality|राष्ट्रीयता|नागरिकता)[:\s]*', '', ln, flags=re.I).strip()
            if val and len(val) > 2:
                obj['nationality'] = val
                break
            val = nearest_line(lines, i, 1)
            if val and not re.search(r'DATE|PLACE|SEX', val, re.I):
                obj['nationality'] = val
                break

    if not obj['nationality']:
        # Check for INDIAN text or REPUBLIC OF INDIA
        if re.search(r'\bINDIAN\b', text):
            obj['nationality'] = 'INDIAN'
        elif 'REPUBLIC OF INDIA' in text.upper() or 'IND' in text:
            obj['nationality'] = 'INDIAN'

    # =========================================================================
    # Address (typically on back page)
    # =========================================================================
    for i, ln in enumerate(lines):
        if re.search(r'\b(Address|ADDRESS|पता)\b', ln):
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
                         addr_parts.append(next_ln)
                         break
                    continue

                addr_parts.append(next_ln)
                # If looks like PIN code at end
                if re.search(r'PIN\s*[:\-\s]*\d{6}', next_ln, re.I) or re.search(r'\b\d{6}\b', next_ln):
                    break

            if addr_parts:
                full_addr = ", ".join(addr_parts)
                full_addr = re.sub(r'\s*,\s*,\s*', ', ', full_addr)
                full_addr = re.sub(r'^,\s*|,\s*$', '', full_addr)
                if len(full_addr) > 5:
                    obj['address'] = full_addr
                break

    # =========================================================================
    # Family names (back page)
    # =========================================================================
    def is_valid_human_name(s):
        if not s: return False
        if re.search(r'[\d<>]', s): return False
        if len(s) < 2: return False
        if re.search(r'(Address|Passport|File|Mother|Spouse)', s, re.I): return False
        return True

    # Father's Name
    for i, ln in enumerate(lines):
        if re.search(r"(Father'?s?\s*Name|Name of Father)", ln, re.I):
            val = re.sub(r".*(?:Father'?s?\s*Name|Name of Father)\s*[:\.]?", '', ln, flags=re.I).strip()
            if is_valid_human_name(val) and len(val) > 2:
                obj['father_name'] = val
            else:
                val = nearest_line(lines, i, 1)
                if val and is_valid_human_name(val):
                     obj['father_name'] = val
            break

    # Mother's Name
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

    # Spouse Name
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

    print("\n[DEBUG] ===== Passport Extraction Results =====")
    for key, val in obj.items():
        print(f"[DEBUG] {key}: '{val}'")
    print("[DEBUG] =========================================\n")

    return obj
