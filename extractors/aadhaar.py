

"""
Aadhaar Card Extraction Logic
Extracts information from Aadhaar cards (E).
"""

import re
from typing import Dict, List
from .utils import clean_ocr_garbage, is_valid_name, looks_like_address, contains_devanagari, is_title_case_name, is_uidai_boilerplate, looks_like_uidai_text, is_likely_garbage, has_reasonable_vowel_ratio


def extract_aadhaar(lines: List[str], text: str, records: List[Dict]) -> Dict:
    """Extract data from Aadhaar card."""
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
        "nationality": ""
    }
    
    print("\n[DEBUG] ===== Aadhaar Extraction Started =====")
    print(f"[DEBUG] Total OCR lines: {len(lines)}")
    print(f"[DEBUG] Total OCR records: {len(records)}")
    print("[DEBUG] OCR Lines:")
    for i, ln in enumerate(lines[:15]):  # Show first 15 lines
        print(f"[DEBUG]   Line {i}: '{ln}'")
    print()
    print("[DEBUG] Full OCR Text:")
    print(f"[DEBUG] {text[:1000]}...")  # Show first 1000 chars
    print()
    
    # Aadhaar Number
    print("[DEBUG] === Extracting Aadhaar Number ===")
    
    # Pattern 1: Spaced format "1234 5678 9012"
    aadhaar_spaced = re.search(r'\b(\d{4}\s+\d{4}\s+\d{4})\b', text)
    if aadhaar_spaced:
        obj['aadhaar_number'] = re.sub(r'\s+', '', aadhaar_spaced.group(1))
        print(f"[DEBUG] Found Aadhaar (spaced pattern): {obj['aadhaar_number']}")
    
    # Pattern 2: Continuous 12 digits
    if not obj['aadhaar_number']:
        print("[DEBUG] Trying continuous 12 digits...")
        for i, ln in enumerate(lines):
            m = re.search(r'\b(\d{12})\b', ln)
            if m:
                obj['aadhaar_number'] = m.group(1)
                print(f"[DEBUG] Found Aadhaar at line {i}: {obj['aadhaar_number']}")
                break
    
    # Pattern 3: 4+4+4 digits with any separator
    if not obj['aadhaar_number']:
        print("[DEBUG] Trying 4+4+4 pattern with any separator...")
        m = re.search(r'(\d{4})[^\d]*(\d{4})[^\d]*(\d{4})(?!\d)', text)
        if m:
            aadhaar_cand = m.group(1) + m.group(2) + m.group(3)
            # Verify it's not part of VID (16 digits)
            start_pos = m.start()
            end_pos = m.end()
            # Check if there's more digits around (would be VID)
            context_before = text[max(0, start_pos-10):start_pos]
            context_after = text[end_pos:min(len(text), end_pos+10)]
            if not re.search(r'\d{4}', context_after):  # Not VID
                obj['aadhaar_number'] = aadhaar_cand
                print(f"[DEBUG] Found Aadhaar (4+4+4): {obj['aadhaar_number']}")
    
    if not obj['aadhaar_number']:
        print("[DEBUG] No Aadhaar number found!")

    # VID
    print("\n[DEBUG] === Extracting VID ===")
    
    # Pattern 1: With VID label
    vid_match = re.search(r'VID\s*[:\s]*(\d{4}\s?\d{4}\s?\d{4}\s?\d{4})', text, re.I)
    if vid_match:
        obj['vid'] = re.sub(r'\s+', '', vid_match.group(1))
        print(f"[DEBUG] Found VID (with label): {obj['vid']}")
    
    # Pattern 2: 16 digit number (4+4+4+4 format)
    if not obj['vid']:
        print("[DEBUG] Trying 16 digit pattern...")
        m = re.search(r'(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})', text)
        if m:
            vid_cand = m.group(1) + m.group(2) + m.group(3) + m.group(4)
            # Make sure it's not the Aadhaar number (12 digits vs 16 digits)
            if len(vid_cand) == 16:
                obj['vid'] = vid_cand
                print(f"[DEBUG] Found VID (16 digits): {obj['vid']}")
    
    # Pattern 3: Continuous 16 digits
    if not obj['vid']:
        m = re.search(r'\b(\d{16})\b', text)
        if m:
            obj['vid'] = m.group(1)
            print(f"[DEBUG] Found VID (continuous 16): {obj['vid']}")
    
    if not obj['vid']:
        print("[DEBUG] No VID found!")

    # DOB
    dob_idx = -1
    
    print("\n[DEBUG] === Extracting DOB ===")
    
    # Priority 1: Search for explicit labels
    print("[DEBUG] Priority 1: Looking for DOB labels...")
    for i, r in enumerate(records):
        if re.search(r'\b(DOB|Date of Birth|D0B|D\.0\.B|D\.O\.B)\b', r['text'], re.I):
            print(f"[DEBUG] Found DOB label at line {i}: '{r['text']}'")
            if re.search(r'(Download|Issue|Print)\s*[:\-\.]?\s*Date', r['text'], re.I):
                print(f"[DEBUG] Skipping (Download/Issue date)")
                continue
            m = re.search(r'(\d{2}[/\-]\d{2}[/\-]\d{4})', r['text'])
            if m:
                obj['dob'] = m.group(1)
                dob_idx = i
                print(f"[DEBUG] Found DOB: {obj['dob']}")
                break
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', r['text'])
            if year_match:
                obj['dob'] = year_match.group(1)
                dob_idx = i
                print(f"[DEBUG] Found DOB (year only): {obj['dob']}")
                break
    
    # Priority 2: Fallback search for date patterns
    if not obj['dob']:
        print("[DEBUG] Priority 2: Looking for date patterns without labels...")
        for i, r in enumerate(records):
            prev_text = records[i-1]['text'] if i > 0 else ""
            if re.search(r'(\d{2}[/\-]\d{2}[/\-]\d{4})', r['text']):
                m = re.search(r'(\d{2}[/\-]\d{2}[/\-]\d{4})', r['text'])
                print(f"[DEBUG] Found date at line {i}: '{r['text']}'")
                if re.search(r'(Download|Issue|Print)\s*[:\-\.]?\s*Date', r['text'], re.I):
                    print(f"[DEBUG] Skipping (Download/Issue date in current line)")
                    continue
                if re.search(r'(Download|Issue|Print)\s*[:\-\.]?\s*Date', prev_text, re.I):
                    print(f"[DEBUG] Skipping (Download/Issue date in previous line)")
                    continue
                obj['dob'] = m.group(1)
                dob_idx = i
                print(f"[DEBUG] Found DOB: {obj['dob']}")
                break
    
    # Priority 3: Search in full text for any date pattern
    if not obj['dob']:
        print("[DEBUG] Priority 3: Looking for date in full text...")
        # Try various date patterns
        date_patterns = [
            r'DOB[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})',  # DOB: 10/01/2004
            r'(\d{2}[/\-]\d{2}[/\-]\d{4})',  # Simple date pattern
            r'(\d{4}[/\-]\d{2}[/\-]\d{2})',  # YYYY-MM-DD format
        ]
        for pattern in date_patterns:
            m = re.search(pattern, text)
            if m:
                date_str = m.group(1)
                # Verify it's not an issue/download date
                context = text[max(0, m.start()-20):m.end()+5]
                if not re.search(r'(Download|Issue|Print)', context, re.I):
                    obj['dob'] = date_str
                    print(f"[DEBUG] Found DOB in full text: {obj['dob']}")
                    break
    
    if not obj['dob']:
        print("[DEBUG] No DOB found!")

    # Gender
    print("\n[DEBUG] === Extracting Gender ===")
    for i, r in enumerate(records):
        if re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', r['text'], re.I):
            obj['gender'] = re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', r['text'], re.I).group(1).upper()
            print(f"[DEBUG] Found Gender at line {i}: {obj['gender']} (text: '{r['text']}')")
            break
    if not obj['gender']:
        print("[DEBUG] No Gender found!")

    # Name Extraction
    print("\n[DEBUG] === Extracting Name ===")
    
    # Page Metrics for Layout Rules
    dob_y = records[dob_idx]['y'] if dob_idx != -1 else -1

    def clean_duplicate_words(text):
        if not text: return ""
        words = text.split()
        if not words: return ""
        deduped = [words[0]]
        for w in words[1:]:
            if w.lower() != deduped[-1].lower():
                deduped.append(w)
        return " ".join(deduped)

    def is_strongly_valid_aadhaar_name(val, y_pos):
        if not val: return False
        
        val_lower = val.lower()
        val_clean = val_lower.replace(" ", "")
        
        # 1. HARD BLOCK (Absolute Rejection List)
        # Matches partial overlap for critical keywords
        hard_block_tokens = [
            "unique", "identif", "authorit", "uidai", "government", "india", 
            "aadhaar", "andhaar", "adhar", "informat", "enrolment", "enrollment",
            "issue", "date", "proof", "identity", "citizenship", "download",
            "father", "mother", "husband", "address", "poi", "fet", "year",
            "goyem", "goverm", "govem", "iindin", "indin"
        ]
        
        # Check strict substring presence
        if any(t in val_clean for t in hard_block_tokens):
            return False
            
        # Word-based checks for short common words
        words_lower = val_lower.split()
        if any(w in words_lower for w in ["no", "is", "of", "to", "a", "the", "dob", "yob"]):
            return False

        # 2. LOCATION RULE (Strict Zone: ±120px of DOB)
        if dob_y != -1 and abs(y_pos - dob_y) > 120:
            return False

        # 3. TEXT RULE
        # Alphabet + Space only (allow dot)
        if not re.match(r'^[A-Za-z\s\.]+$', val):
            return False
            
        words = val.strip().split()
        # Minimum 2 words
        if len(words) < 2:
            return False

        # 4. OCR GARBAGE & VOWEL CHECK
        if is_likely_garbage(val):
            return False
        if not has_reasonable_vowel_ratio(val):
            return False

        # 5. HUMAN-NAME REQUIREMENT
        # Must have at least one Proper-case word OR be valid ALL-CAPS
        has_proper_case = re.search(r'\b[A-Z][a-z]{2,}\b', val)
        
        is_valid_all_caps = False
        if val.isupper():
            # Ensure words are not single letter garbage
            valid_words = [w for w in words if len(w) >= 3 and w.isalpha()]
            if len(valid_words) >= 1:
                is_valid_all_caps = True
        
        if not (has_proper_case or is_valid_all_caps):
             return False
                
        return True

    # Strategy 0: Devanagari line anchor
    print("[DEBUG] Strategy 0: Looking for Devanagari + English pattern...")
    for i in range(len(lines) - 1):
        if contains_devanagari(lines[i]):
            next_ln = lines[i+1].strip()
            next_ln_y = records[i+1]['y'] if i+1 < len(records) else -1
            if is_strongly_valid_aadhaar_name(next_ln, next_ln_y):
                final_name = clean_ocr_garbage(next_ln)
                obj['name'] = clean_duplicate_words(final_name)
                print(f"[DEBUG] ✓ Extracted name via Devanagari anchor: '{obj['name']}'")
                break
    
    # Strategy 1: "To" block strategy
    if not obj['name']:
        print("[DEBUG] Strategy 1: Looking for 'To' block...")
        for i, ln in enumerate(lines):
            if re.search(r'^\s*To\b', ln, re.I) or (len(ln.strip()) < 10 and re.search(r'\bTo\b', ln, re.I)):
                print(f"[DEBUG] Found 'To' anchor at line {i}: '{ln}'")
                name_candidate = re.sub(r'^.*?To\s*', '', ln, flags=re.I).strip()
                name_candidate = re.sub(r'^[^a-zA-Z]+', '', name_candidate).strip()
                curr_y = records[i]['y'] if i < len(records) else -1
                
                if name_candidate and is_strongly_valid_aadhaar_name(name_candidate, curr_y):
                    final_name = clean_ocr_garbage(name_candidate)
                    obj['name'] = clean_duplicate_words(final_name)
                    print(f"[DEBUG] Valid name found on 'To' line: '{obj['name']}'")
                else:
                    print(f"[DEBUG] Skipping boilerplate, searching next...")
                    for k in range(1, 4):
                        if i + k < len(lines):
                            next_ln = lines[i+k].strip()
                            next_ln_clean = next_ln.lower().replace(" ","")
                            skip_tokens = ["unique", "identif", "authorit", "uidai", "government", "india", "aadhaar", "andhaar", "proof", "issue", "date", "enrolment", "address"]
                            if any(t in next_ln_clean for t in skip_tokens):
                                print(f"[DEBUG] Skipping institutional line: '{next_ln}'")
                                continue
                            
                            next_y = records[i+k]['y'] if i+k < len(records) else -1
                            if is_strongly_valid_aadhaar_name(next_ln, next_y):
                                final_name = clean_ocr_garbage(next_ln)
                                obj['name'] = clean_duplicate_words(final_name)
                                print(f"[DEBUG] Found valid name in line {i+k}: '{obj['name']}'")
                                break
                if obj['name']: break

    # Strategy 2: Extract name from lines near Gender labels
    if not obj['name']:
        print("[DEBUG] Strategy 2: Looking for name near /MALE or /FEMALE...")
        for i, ln in enumerate(lines):
            # Regex for name candidate
            m = re.search(r'/?(MALE|FEMALE)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})', ln, re.I)
            if m:
                name_cand = m.group(2).strip()
                curr_y = records[i]['y'] if i < len(records) else -1
                if is_strongly_valid_aadhaar_name(name_cand, curr_y):
                    final_name = clean_ocr_garbage(name_cand)
                    obj['name'] = clean_duplicate_words(final_name)
                    print(f"[DEBUG] Extracted name via Gender anchor: '{obj['name']}'")
                    break

    # Strategy 3: General pattern search
    if not obj['name']:
        print("[DEBUG] Strategy 3: Looking for general name pattern...")
        for i, ln in enumerate(lines):
            # Search for pattern that allows Proper case or All caps (min 2 words)
            m = re.search(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\b', ln)
            if m:
                name_cand = m.group(1).strip()
                curr_y = records[i]['y'] if i < len(records) else -1
                if is_strongly_valid_aadhaar_name(name_cand, curr_y):
                    final_name = clean_ocr_garbage(name_cand)
                    obj['name'] = clean_duplicate_words(final_name)
                    print(f"[DEBUG] Extracted name via pattern search: '{obj['name']}'")
                    break

    # Strategy 4: Merged Line Repeated Pattern
    if not obj['name']:
        print("[DEBUG] Strategy 4: Looking for repeated name patterns...")
        from collections import Counter
        for i, ln in enumerate(lines):
            candidates = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\b', ln)
            if candidates:
                counts = Counter(candidates)
                for cand, count in counts.items():
                    if count >= 2:
                        curr_y = records[i]['y'] if i < len(records) else -1
                        if is_strongly_valid_aadhaar_name(cand, curr_y):
                            final_name = clean_ocr_garbage(cand)
                            obj['name'] = clean_duplicate_words(final_name)
                            print(f"[DEBUG] ✓ Extracted repeated name: '{obj['name']}'")
                            break
                if obj['name']: break

    # Strategy 5: Fallback - Line directly above DOB
    if not obj['name'] and dob_idx != -1:
        print(f"[DEBUG] Strategy 5: Fallback searching above DOB...")
        curr_y = records[dob_idx]['y']
        above_lines = [r for r in records if r['y'] < curr_y - 10]
        if above_lines:
            above_lines.sort(key=lambda r: r['y'], reverse=True)
            for r in above_lines[:4]:
                txt = r['text'].strip()
                if is_strongly_valid_aadhaar_name(txt, r['y']):
                    final_name = clean_ocr_garbage(txt)
                    obj['name'] = clean_duplicate_words(final_name)
                    print(f"[DEBUG] Final name from fallback: '{obj['name']}'")
                    break
    elif not obj['name']:
        print("[DEBUG] Cannot use fallback strategy (no DOB found)")


    # -------------------------------------------------------------------------
    # FINAL ADD-ON: LAST RESORT CORRECTION (NON-DESTRUCTIVE FALLBACK)
    # -------------------------------------------------------------------------
    def is_invalid_aadhaar_name_candidate(val):
        """Check if name is invalid (garbage, institutional, or empty)."""
        if not val: return True
        v_low = val.lower()
        bad_keys = [
            "government", "india", "aadhaar", "uidai", "enrolment", "enrollment", 
            "issue", "date", "proof", "identity", "citizenship", "poi", "fet", 
            "download", "information", "nteu", "wonte", "dow", "fabl", "ibj", "father", "address"
        ]
        if any(k in v_low for k in bad_keys): return True
        # Consonant heavy garbage
        if re.search(r'\b[BCDFGHJKLMNPQRSTVWXYZ]{4,}\b', val, re.I): return True
        return False

    # Activate only if current name is missing or invalid
    if is_invalid_aadhaar_name_candidate(obj['name']):
        print(f"[DEBUG] ADD-ON: Current name '{obj['name']}' is invalid/garbage. Attempting strict fallback...")
        obj['name'] = "" # Reset
        
        # 1. DOB-NEIGHBOUR RULE (Scan lines above DOB)
        if dob_idx != -1:
            dob_y = records[dob_idx]['y']
            # Get lines strictly above DOB (within 200px)
            candidates = []
            for r in records:
                if dob_y - 200 < r['y'] < dob_y - 5: # Strictly above
                    candidates.append(r)
            # Sort bottom-up (closest to DOB first)
            candidates.sort(key=lambda x: x['y'], reverse=True)
            
            for cand in candidates:
                txt = cand['text'].strip()
                # STRICT HUMAN NAME FILTER for fallback
                # Must be Proper Case (Abc Xyz), Min 2 words, No digits, No All-caps (to allow header safety)
                # Regex: Start with Capital, then lower, 2+ words
                if (re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$', txt) and 
                    len(txt.split()) >= 2 and 
                    not is_invalid_aadhaar_name_candidate(txt)):
                    
                    obj['name'] = clean_ocr_garbage(txt)
                    obj['name'] = clean_duplicate_words(obj['name'])
                    print(f"[DEBUG] ADD-ON: Found name via DOB-Neighbor: '{obj['name']}'")
                    break
        
        # 2. RELATION FALLBACK (Scan lines above S/O, D/O, W/O)
        if not obj['name']:
            for i, ln in enumerate(lines):
                 # Look for relation/address start
                 if re.search(r'\b(S/O|D/O|W/O|C/O)\b', ln, re.I):
                     # Check line immediately above
                     if i > 0:
                         prev = lines[i-1].strip()
                         # Same strict filter
                         if (re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$', prev) and 
                             len(prev.split()) >= 2 and
                             not is_invalid_aadhaar_name_candidate(prev)):
                             
                             obj['name'] = clean_ocr_garbage(prev)
                             obj['name'] = clean_duplicate_words(obj['name'])
                             print(f"[DEBUG] ADD-ON: Found name above Relation: '{obj['name']}'")
                             break

    # Father/Mother/Husband - Look for S/O, D/O, W/O, C/O patterns
    for i, ln in enumerate(lines):
        if re.search(r'S/O|SON OF', ln, re.I):
            val = re.sub(r'.*S/O\s*[:\.]?\s*', '', ln, flags=re.I).strip()
            val = re.sub(r'.*SON OF\s*[:\.]?\s*', '', val, flags=re.I).strip()
            if ',' in val:
                val = val.split(',')[0].strip()
            val = re.sub(r'[,\-\.]+$', '', val).strip()
            
            if not val or len(val) < 3:
                if i + 1 < len(lines):
                    next_ln = lines[i + 1].strip()
                    if (not re.search(r'\d{6}|\b(PIN|VTC|PO|District|State|Mobile|Address)\b', next_ln, re.I) and
                        re.match(r'^[A-Za-z\s\.]+$', next_ln) and
                        len(next_ln) > 3):
                        val = next_ln
                        if ',' in val:
                            val = val.split(',')[0].strip()
            
            if val and len(val) > 2:
                obj['father_name'] = val
                
        elif re.search(r'D/O|DAUGHTER OF', ln, re.I):
            val = re.sub(r'.*D/O\s*[:\.]?\s*', '', ln, flags=re.I).strip()
            val = re.sub(r'.*DAUGHTER OF\s*[:\.]?\s*', '', val, flags=re.I).strip()
            if ',' in val:
                val = val.split(',')[0].strip()
            val = re.sub(r'[,\-\.]+$', '', val).strip()
            
            if not val or len(val) < 3:
                if i + 1 < len(lines):
                    next_ln = lines[i + 1].strip()
                    if (not re.search(r'\d{6}|\b(PIN|VTC|PO|District|State|Mobile|Address)\b', next_ln, re.I) and
                        re.match(r'^[A-Za-z\s\.]+$', next_ln) and
                        len(next_ln) > 3):
                        val = next_ln
                        if ',' in val:
                            val = val.split(',')[0].strip()
            
            if val and len(val) > 2:
                obj['father_name'] = val
                
        elif re.search(r'W/O|WIFE OF', ln, re.I):
            val = re.sub(r'.*W/O\s*[:\.]?\s*', '', ln, flags=re.I).strip()
            val = re.sub(r'.*WIFE OF\s*[:\.]?\s*', '', val, flags=re.I).strip()
            if ',' in val:
                val = val.split(',')[0].strip()
            val = re.sub(r'[,\-\.]+$', '', val).strip()
            
            if not val or len(val) < 3:
                if i + 1 < len(lines):
                    next_ln = lines[i + 1].strip()
                    if (not re.search(r'\d{6}|\b(PIN|VTC|PO|District|State|Mobile|Address)\b', next_ln, re.I) and
                        re.match(r'^[A-Za-z\s\.]+$', next_ln) and
                        len(next_ln) > 3):
                        val = next_ln
                        if ',' in val:
                            val = val.split(',')[0].strip()
            
            if val and len(val) > 2:
                obj['husband_name'] = val
                
        elif re.search(r'C/O|CARE OF', ln, re.I):
            val = re.sub(r'.*C/O\s*[:\.]?\s*', '', ln, flags=re.I).strip()
            val = re.sub(r'.*CARE OF\s*[:\.]?\s*', '', val, flags=re.I).strip()
            if ',' in val:
                val = val.split(',')[0].strip()
            val = re.sub(r'[,\-\.]+$', '', val).strip()
            
            if not val or len(val) < 3:
                if i + 1 < len(lines):
                    next_ln = lines[i + 1].strip()
                    if (not re.search(r'\d{6}|\b(PIN|VTC|PO|District|State|Mobile|Address)\b', next_ln, re.I) and
                        re.match(r'^[A-Za-z\s\.]+$', next_ln) and
                        len(next_ln) > 3):
                        val = next_ln
                        if ',' in val:
                            val = val.split(',')[0].strip()
            
            if val and len(val) > 2:
                obj['father_name'] = val

    # Father name fallback
    if not obj['father_name'] and obj['name']:
        name_parts = obj['name'].split()
        if len(name_parts) >= 3:
            father_candidate = ' '.join(name_parts[1:])
            if len(father_candidate) > 5 and len(father_candidate.split()) >= 2:
                father_words = father_candidate.split()
                if all(len(w) >= 2 for w in father_words):
                    obj['father_name'] = father_candidate

    # Address
    address_indicators = r'\b(S/O|W/O|C/O|D/O|HOUSE|FLAT|SECTOR|ROAD|LANE|STREET|VILLAGE|MANDAL|DISTRICT|STATE|PIN|PINCODE|PO:|VTC:)\b'

    for i, ln in enumerate(lines):
        if re.match(r'^To\b', ln, re.I):
            addr_lines = []
            start_collecting = False
            for j in range(i, min(len(lines), i + 10)):
                if j == i and obj['name'] in lines[j]:
                    start_collecting = True
                    continue
                if obj['name'] and lines[j] == obj['name']:
                    start_collecting = True
                    continue
                if start_collecting or re.search(address_indicators, lines[j], re.I):
                    start_collecting = True
                    if re.search(r'(Issue|Download|Print)\s*Date', lines[j], re.I):
                        continue
                    if re.search(r'\b\d{10,12}\b', lines[j]):
                        continue
                    addr_lines.append(lines[j])
                    if re.search(r'\b\d{6}\b', lines[j]):
                        break
            if addr_lines:
                cand = ", ".join(addr_lines)
                if re.search(r'\b\d{6}\b', cand) or re.search(address_indicators, cand, re.I):
                    obj['address'] = cand
            break

    if not obj['address']:
        for i, ln in enumerate(lines):
            if 'Address' in ln or 'ADDRESS' in ln:
                addr_lines = []
                for j in range(i+1, min(len(lines), i+10)):
                    if re.search(r'(Issue|Download|Print)\s*Date', lines[j], re.I):
                        continue
                    if re.search(r'\b\d{10,12}\b', lines[j]):
                        continue
                    addr_lines.append(lines[j])
                    if re.search(r'\b\d{6}\b', lines[j]):
                        break
                if addr_lines:
                    cand = ", ".join(addr_lines)
                    if re.search(r'\b\d{6}\b', cand):
                        obj['address'] = cand
                break

    if not obj['address']:
        addr_lines = []
        for i, ln in enumerate(lines):
            if re.search(address_indicators, ln, re.I):
                for j in range(i, min(len(lines), i+8)):
                    if re.search(r'(Issue|Download|Print)\s*Date', lines[j], re.I):
                        continue
                    if re.search(r'\b\d{10,12}\b', lines[j]):
                        continue
                    addr_lines.append(lines[j])
                    if re.search(r'\b\d{6}\b', lines[j]):
                        break
                break
        if addr_lines:
            cand = ", ".join(addr_lines)
            if re.search(r'\b\d{6}\b', cand):
                obj['address'] = cand

    # Nationality
    print("\n[DEBUG] === Extracting Nationality ===")
    if re.search(r'(GOVERNMENT OF INDIA|भारत सरकार)', text, re.I):
        obj['nationality'] = 'INDIAN'
        print(f"[DEBUG] Found nationality: {obj['nationality']}")
    else:
        print("[DEBUG] No nationality indicators found")

    print("\n[DEBUG] ===== Final Extraction Results =====")
    print(f"[DEBUG] Name: '{obj['name']}'")
    print(f"[DEBUG] Gender: '{obj['gender']}'")
    print(f"[DEBUG] DOB: '{obj['dob']}'")
    print(f"[DEBUG] Aadhaar: '{obj['aadhaar_number']}'")
    print(f"[DEBUG] VID: '{obj['vid']}'")
    print(f"[DEBUG] Father: '{obj['father_name']}'")
    print(f"[DEBUG] Address: '{obj['address']}'")
    print(f"[DEBUG] Nationality: '{obj['nationality']}'")
    print("[DEBUG] =====================================\n")

    return obj
