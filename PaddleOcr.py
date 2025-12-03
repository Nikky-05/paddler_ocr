# ...existing code...
r"""
Lightweight OCR pipeline using PaddleOCR to extract structured JSON
from Indian documents: Aadhaar, PAN, Driving Licence, Voter ID, Passport.

Required package versions:
  paddlepaddle==3.2.2
  paddleocr==3.3.2

Usage (Windows):
  pip install paddlepaddle==3.2.2 paddleocr==3.3.2 pillow
  python c:\Users\hp\paddle\paddleOcr.py --input "C:\path\to\images" --output results.json

Outputs a JSON per image (or a combined JSON file) with keys/values for the detected document.
"""
from pathlib import Path
import re
import json
import argparse

# version checks
try:
    import importlib.metadata as _m
except Exception:
    import pkg_resources as _m

try:
    PADDLE_VER = None
    PADDLEOCR_VER = None
    try:
        # importlib.metadata on py3.8+
        PADDLE_VER = _m.version("paddlepaddle")
    except Exception:
        try:
            PADDLE_VER = _m.version("paddle")
        except Exception:
            PADDLE_VER = None
    try:
        PADDLEOCR_VER = _m.version("paddleocr")
    except Exception:
        PADDLEOCR_VER = None
except Exception:
    PADDLE_VER = None
    PADDLEOCR_VER = None

# enforce required versions
EXPECTED_PADDLE = "3.2.2"
EXPECTED_PADDLEOCR = "3.3.2"
if PADDLE_VER and PADDLE_VER != EXPECTED_PADDLE:
    raise RuntimeError(f"Installed paddlepaddle version is {PADDLE_VER}; required {EXPECTED_PADDLE}. Install with: pip install paddlepaddle=={EXPECTED_PADDLE}")
if PADDLEOCR_VER and PADDLEOCR_VER != EXPECTED_PADDLEOCR:
    raise RuntimeError(f"Installed paddleocr version is {PADDLEOCR_VER}; required {EXPECTED_PADDLEOCR}. Install with: pip install paddleocr=={EXPECTED_PADDLEOCR}")

OCR = None


def get_ocr():
    """Lazy-initialize PaddleOCR to avoid heavy work at import time and provide friendly errors."""
    global OCR
    if OCR is not None:
        return OCR
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        raise RuntimeError(
            "Failed to import paddleocr. Ensure `paddleocr` and `paddlepaddle` are installed and internet access is available for first-run model downloads. Original error: "
            + str(e)
        )
    # initialize OCR with orientation detection
    # initialize OCR with optimized settings
    try:
        # use_angle_cls=True enables orientation classification (standard)
        # enable_mkldnn=False to avoid 'cat: not found' errors on minimal Linux
        OCR = PaddleOCR(
            use_angle_cls=True, 
            lang='en', 
            enable_mkldnn=False
        )
    except TypeError:
        # fallback for older versions - absolute minimum
        OCR = PaddleOCR(lang='en')
    return OCR
from PIL import Image
import numpy as np
import os
from typing import List, Dict

# Initialize PaddleOCR once (downloads models on first run)
OCR = None

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp'}


def ocr_image(path: str) -> List[str]:
    """Run PaddleOCR on original + preprocessed images and return list of recognized text lines (reading order).

    This runs OCR twice (raw image and an enhanced/preprocessed image) and merges results by line vertical position,
    preferring higher-confidence reads. Returns a list of text lines ordered top->bottom.
    """
    ocr = get_ocr()

    def run_ocr_on(arg):
        try:
            res = ocr.ocr(arg)
        except Exception as e:
            if os.environ.get('OCR_DEBUG'):
                print(f"DEBUG: OCR failed: {e}")
            # fallback to predict if ocr() signature changed
            try:
                res = ocr.predict(arg)
            except Exception as e2:
                if os.environ.get('OCR_DEBUG'):
                    print(f"DEBUG: Fallback predict failed: {e2}")
                return []
        return res

    def parse_result(res):
        recs = []
        # Newer PaddleOCR / Paddlex may return a single-element list containing a dict with 'rec_texts' and 'rec_polys'
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict) and 'rec_texts' in res[0]:
            page = res[0]
            texts = page.get('rec_texts', [])
            scores = page.get('rec_scores', [])
            polys = page.get('rec_polys', [])
            for idx, txt in enumerate(texts):
                try:
                    if not txt or not str(txt).strip():
                        continue
                    conf = float(scores[idx]) if idx < len(scores) else 0.0
                    poly = polys[idx] if idx < len(polys) else None
                    if poly is not None:
                        y = int(min([int(p[1]) for p in poly]))
                    else:
                        y = 0
                    recs.append({"text": str(txt).strip(), "conf": conf, "y": y})
                except Exception:
                    continue
            return recs

        for page in res:
            for item in page:
                try:
                    box = item[0]
                    recog = item[1]
                    # recog may be tuple (text, confidence)
                    if isinstance(recog, (list, tuple)):
                        text = recog[0]
                        conf = float(recog[1]) if len(recog) > 1 else 0.0
                    else:
                        text = str(recog)
                        conf = 0.0
                    if not text or not text.strip():
                        continue
                    # y position (top) used for ordering
                    y = min([p[1] for p in box])
                    recs.append({"text": text.strip(), "conf": conf, "y": y})
                except Exception:
                    continue
        return recs

    def preprocess_image(path: str):
        img = Image.open(path).convert('RGB')
        # upscale for small images
        w, h = img.size
        target_w = max(1200, w)
        if w < target_w:
            new_h = int(h * (target_w / w))
            img = img.resize((target_w, new_h), Image.LANCZOS)
        # enhance contrast and sharpen
        try:
            from PIL import ImageEnhance, ImageFilter
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.6)
            img = img.filter(ImageFilter.SHARPEN)
        except Exception:
            pass
        return np.array(img)

    # Use ocr_records to build ordered merged lines (keeps positional info)
    recs = ocr_records(path)
    if not recs:
        return []
    return [r['text'] for r in recs]


def ocr_records(path: str) -> List[Dict]:
    """Run PaddleOCR (raw + preprocessed) and return ordered list of records:
    [{'text':..., 'conf':..., 'y':...}, ...] ordered top->bottom.

    This mirrors the previous internal logic but preserves confidences and y
    positions so downstream extractors can use spatial heuristics.
    """
    ocr = get_ocr()

    def run_ocr_on(arg):
        try:
            res = ocr.ocr(arg)
        except Exception as e:
            if os.environ.get('OCR_DEBUG'):
                print(f"DEBUG: OCR records failed: {e}")
            try:
                res = ocr.predict(arg)
            except Exception as e2:
                if os.environ.get('OCR_DEBUG'):
                    print(f"DEBUG: Fallback predict records failed: {e2}")
                return []
        return res

    def is_english_text(text):
        """Check if text contains only English characters (Latin script, numbers, punctuation).
        Filters out Marathi/Devanagari and other non-English scripts."""
        if not text:
            return False
        # Remove common punctuation and whitespace
        cleaned = re.sub(r'[\s\.,\-\/:;()\[\]{}!?@#$%^&*+=_~`\'"<>|\\]', '', text)
        if not cleaned:
            return True  # Allow pure punctuation/whitespace
        # Check if remaining characters are Latin alphabet (A-Z, a-z) or digits
        # This will filter out Devanagari (0900-097F), Arabic, Chinese, etc.
        return bool(re.match(r'^[A-Za-z0-9]+$', cleaned))

    def parse_result(res):
        recs = []
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict) and 'rec_texts' in res[0]:
            page = res[0]
            texts = page.get('rec_texts', [])
            scores = page.get('rec_scores', [])
            polys = page.get('rec_polys', [])
            for idx, txt in enumerate(texts):
                try:
                    if not txt or not str(txt).strip():
                        continue
                    # Filter out non-English text (Marathi/Devanagari, etc.)
                    if not is_english_text(str(txt)):
                        continue
                    conf = float(scores[idx]) if idx < len(scores) else 0.0
                    poly = polys[idx] if idx < len(polys) else None
                    if poly is not None:
                        y = int(min([int(p[1]) for p in poly]))
                    else:
                        y = 0
                    recs.append({"text": str(txt).strip(), "conf": conf, "y": y})
                except Exception:
                    continue
            return recs

        for page in res:
            for item in page:
                try:
                    box = item[0]
                    recog = item[1]
                    if isinstance(recog, (list, tuple)):
                        text = recog[0]
                        conf = float(recog[1]) if len(recog) > 1 else 0.0
                    else:
                        text = str(recog)
                        conf = 0.0
                    if not text or not text.strip():
                        continue
                    # Filter out non-English text (Marathi/Devanagari, etc.)
                    if not is_english_text(text):
                        continue
                    y = min([p[1] for p in box])
                    recs.append({"text": text.strip(), "conf": conf, "y": y})
                except Exception:
                    continue
        return recs

    ext = Path(path).suffix.lower()
    results = []
    
    # Pass 1: Original Image (Resized if too large)
    try:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        
        # Optimization: Resize if too large to speed up OCR
        if w > 1600:
            new_h = int(h * (1600 / w))
            img = img.resize((1600, new_h), Image.LANCZOS)
            if os.environ.get('OCR_DEBUG'):
                print(f"DEBUG: Resized image to 1600x{new_h}")
        
        orig_arg = np.array(img)
        res1 = run_ocr_on(orig_arg)
        if os.environ.get('OCR_DEBUG'):
            print("DEBUG: raw res1 length:", len(res1))
        
        parsed1 = parse_result(res1)
        results.extend(parsed1)
        
        # Optimization: Check if Pass 1 is good enough
        # Criteria: Found significant text with high confidence AND detected a document keyword
        if parsed1:
            avg_conf = sum(r['conf'] for r in parsed1) / len(parsed1)
            full_text = " ".join([r['text'] for r in parsed1]).upper()
            
            # Keywords that suggest we found a valid document
            strong_keywords = [
                'INCOME TAX', 'PERMANENT ACCOUNT', 'AADHAAR', 'GOVERNMENT OF INDIA', 
                'DRIVING LICENCE', 'ELECTION COMMISSION', 'PASSPORT', 'REPUBLIC OF INDIA',
                'MAHARASHTRA', 'UNIQUE IDENTIFICATION'
            ]
            
            has_keyword = any(k in full_text for k in strong_keywords)
            
            if os.environ.get('OCR_DEBUG'):
                print(f"DEBUG: Pass 1 Avg Conf: {avg_conf:.4f}")
                print(f"DEBUG: Pass 1 Keywords Found: {[k for k in strong_keywords if k in full_text]}")
            
            # If we have good confidence and found a document, SKIP Pass 2
            # Threshold: 0.80 confidence is usually sufficient
            if has_keyword and avg_conf > 0.80:
                if os.environ.get('OCR_DEBUG'):
                    print(f"DEBUG: Early exit. Avg Conf: {avg_conf:.2f}, Keyword Found.")
                return results

    except Exception:
        pass

    # Pass 2: Preprocessed Image (Only if Pass 1 wasn't good enough)
    try:
        # reuse same preprocess as ocr_image
        img = Image.open(path).convert('RGB')
        w, h = img.size
        target_w = max(1200, w)
        if w < target_w:
            new_h = int(h * (target_w / w))
            img = img.resize((target_w, new_h), Image.LANCZOS)
        try:
            from PIL import ImageEnhance, ImageFilter
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.6)
            img = img.filter(ImageFilter.SHARPEN)
        except Exception:
            pass
        preproc = np.array(img)
        res2 = run_ocr_on(preproc)
        if os.environ.get('OCR_DEBUG'):
            print("DEBUG: raw res2 length:", len(res2))
        
        results.extend(parse_result(res2))
    except Exception:
        pass

    if not results:
        return []

    # bucket by rounded y position and pick highest confidence text per bucket
    buckets = {}
    for r in results:
        b = int(round(r['y'] / 5.0) * 5)
        if b not in buckets or r['conf'] > buckets[b]['conf']:
            buckets[b] = r

    ordered = [buckets[k] for k in sorted(buckets.keys())]

    # merge tiny fragments (join very short lines into next)
    merged = []
    buffer = None
    for rec in ordered:
        ln = rec['text']
        if buffer and len(buffer['text']) < 8 and re.match(r'^[A-Za-z]+$', ln.replace(' ', '')):
            buffer['text'] = buffer['text'] + ' ' + ln
            buffer['conf'] = max(buffer.get('conf', 0.0), rec.get('conf', 0.0))
        else:
            if buffer:
                merged.append(buffer)
            buffer = dict(text=ln, conf=rec.get('conf', 0.0), y=rec.get('y', 0))
    if buffer:
        merged.append(buffer)
    return merged


def text_blob(lines: List[str]) -> str:
    return "\n".join(lines)


def detect_doc_type(text: str) -> str:
    """Heuristic doc type detection."""
    t = text.upper()
    # Passport
    if re.search(r'\bP(?!AN)[A-Z0-9]{7}\b', text) and 'PASSPORT' in t:
        return 'passport'
    if 'PASSPORT' in t or re.search(r'\b[P][0-9]{7}\b', t):
        return 'passport'
    # PAN
    if re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', text):
        return 'pan'
    if 'INCOME TAX' in t and 'PERMANENT ACCOUNT' in t or 'INCOME' in t and 'PAN' in t:
        return 'pan'
    # Aadhaar
    if re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text) or 'AADHAAR' in t or 'AADHAR' in t:
        return 'aadhaar'
    # Driving Licence
    if 'DRIVING LICEN' in t or 'DRIVING LICENCE' in t or re.search(r'\bDL\d', t):
        return 'driving_license'
    if 'FORM 7' in t and 'RULE 16' in t:
        return 'driving_license'
    if 'MCWG' in t or 'LMV' in t:
        return 'driving_license'
    if 'UNION OF INDIA' in t and 'DRIVING' in t:
        return 'driving_license'
    if 'MAHARASHTRA STATE' in t and 'LICENCE' in t:
        return 'driving_license'
    # Voter ID / EPIC
    if 'ELECTOR' in t or 'ELECTION' in t or 'EPIC' in t or re.search(r'\b[A-Z]{3}\d{7}\b', text):
        return 'voter_id'
    # Fallback: try passport pattern
    if re.search(r'\b[A-Z]\d{7}\b', text):
        return 'passport'
    return 'unknown'


def nearest_line(lines: List[str], index: int, direction=-1, skip_empty=True):
    """Get nearest non-empty line from index in given direction (-1 above, +1 below)."""
    i = index + direction
    while 0 <= i < len(lines):
        if not skip_empty or lines[i].strip():
            return lines[i].strip()
        i += direction
    return ""


def extract_pan(lines: List[str], text: str) -> Dict:
    obj = {"doc_type": "pan", "name": "", "father_name": "", "pan_number": "", "dob": ""}
    pan_match = re.search(r'\b([A-Z]{5}[0-9]{4}[A-Z])\b', text)
    if pan_match:
        obj['pan_number'] = pan_match.group(1)
    
    # heuristics: look for lines with "Father" or "S/O" and near that the name
    for i, ln in enumerate(lines):
        up = ln.upper()
        
        # Father's Name: Standard PAN layout has "Father's Name" label ABOVE the value.
        # So we should look BELOW (i+1) first.
        if 'FATHER' in up or 'S/O' in up or 'SON OF' in up:
            # Check line below first
            below = nearest_line(lines, i, 1)
            if below and len(below) > 2 and not re.search(r'FATHER|NAME|MOTHER|DATE|BIRTH', below, re.I):
                obj['father_name'] = below
            else:
                # Fallback to above if below is empty or looks like a label
                above = nearest_line(lines, i, -1)
                if above and len(above) > 2:
                    obj['father_name'] = above

        # Name: Standard PAN layout has "Name" label ABOVE the value.
        # Avoid capturing "Number Name" or "Name" itself.
        if 'NAME' in up and 'FATHER' not in up and 'MOTHER' not in up and 'FILE' not in up:
            # If line is just "Name" or "Number Name", look below
            if re.match(r'^(NUMBER\s+)?NAME[:\s]*$', up.strip()):
                val = nearest_line(lines, i, 1)
                if val and not re.search(r'FATHER|DATE|BIRTH|PAN|NUMBER', val, re.I):
                    obj['name'] = val
            elif ':' in ln:
                # "Name: Value" format
                val = ln.split(':')[-1].strip()
                if val:
                    obj['name'] = val
            
            # If we haven't found name yet, and this line looks like "Name", try next line
            if not obj['name']:
                val = nearest_line(lines, i, 1)
                if val and not re.search(r'FATHER|DATE|BIRTH|PAN|NUMBER', val, re.I):
                    obj['name'] = val

    # fallback: find a good uppercase name line (long, alphabetic)
    if not obj['name']:
        for ln in lines:
            # Ignore lines that contain "NAME" or "NUMBER" when searching for the value itself
            if re.match(r'^[A-Z\s\.]{5,50}$', ln) and not re.search(r'\b(INCOME|GOVERNMENT|INDIA|INCOME TAX|NAME|NUMBER|ACCOUNT|CARD|PERMANENT)\b', ln.upper()):
                obj['name'] = ln.strip()
                break
    
    # DOB: PAN sometimes has dob in dd/mm/yyyy
    dob_match = re.search(r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})', text)
    if dob_match:
        obj['dob'] = dob_match.group(1)
    return obj


def extract_aadhaar(lines: List[str], text: str, records: List[Dict] = None) -> Dict:
    obj = {"doc_type": "aadhaar", "name": "", "gender": "", "dob": "", "aadhaar_number": "", "address": ""}
    
    # 1. Aadhaar Number Strategy
    # Prefer strong pattern with spaces: 1234 5678 9012
    aadhaar_spaced = re.search(r'\b(\d{4}\s\d{4}\s\d{4})\b', text)
    if aadhaar_spaced:
        obj['aadhaar_number'] = re.sub(r'\s+', '', aadhaar_spaced.group(1))
    else:
        # Fallback to 12 digits
        for ln in lines:
            m = re.search(r'\b(\d{12})\b', ln)
            if m:
                obj['aadhaar_number'] = m.group(1)
                break

    # 2. DOB Strategy
    # Exclude "Download Date", "Issue Date", "Date of Issue"
    dob_idx = -1
    dob_regex = r'(\d{2}[\/\-]\d{2}[\/\-]\d{4})'
    yob_regex = r'(?:YEAR|BIRTH|YOB).*?(\d{4})'

    def is_valid_dob_line(line_text, prev_line_text=""):
        # Check current line
        if re.search(r'(Download|Issue|Print)\s*Date', line_text, re.I):
            return False
        # Check previous line if current line is just a date
        if re.match(r'^\s*\d{2}[\/\-]\d{2}[\/\-]\d{4}\s*$', line_text):
            if re.search(r'(Download|Issue|Print)\s*Date', prev_line_text, re.I):
                return False
        return True

    # First pass: Look for explicit DOB labels
    if records:
        for i, r in enumerate(records):
            if re.search(r'(DOB|Date of Birth|Year of Birth)', r['text'], re.I):
                # Check if this line itself is an Issue Date line (unlikely if it has DOB label, but possible if mixed)
                if not is_valid_dob_line(r['text']):
                    continue
                
                m = re.search(dob_regex, r['text'])
                if m:
                    obj['dob'] = m.group(1)
                    dob_idx = i
                    break
                m_yob = re.search(yob_regex, r['text'], re.I)
                if m_yob:
                    obj['dob'] = m_yob.group(1)
                    dob_idx = i
                    break
    
    # Second pass: Look for any date pattern if not found
    if not obj['dob']:
        if records:
            for i, r in enumerate(records):
                prev_text = records[i-1]['text'] if i > 0 else ""
                if re.search(dob_regex, r['text']) and is_valid_dob_line(r['text'], prev_text):
                    obj['dob'] = re.search(dob_regex, r['text']).group(1)
                    dob_idx = i
                    break
        else:
            for i, ln in enumerate(lines):
                prev_text = lines[i-1] if i > 0 else ""
                if re.search(dob_regex, ln) and is_valid_dob_line(ln, prev_text):
                    obj['dob'] = re.search(dob_regex, ln).group(1)
                    dob_idx = i
                    break

    # 3. Gender Strategy
    if records:
        for r in records:
            if re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', r['text'], re.I):
                obj['gender'] = re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', r['text'], re.I).group(1).title()
                break
    else:
        for ln in lines:
            if re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', ln, re.I):
                obj['gender'] = re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', ln, re.I).group(1).title()
                break

    # 4. Name Strategy
    # Strategy A: "To" Block (High Confidence)
    for i, ln in enumerate(lines):
        if re.match(r'^To\b', ln, re.I):
            # Check same line
            name_candidate = re.sub(r'^To\s+', '', ln, flags=re.I).strip()
            
            # Helper to check if string looks like a valid name
            def is_good_name(s):
                if len(s) < 3: return False
                if re.search(r'\d', s): return False
                # Must have at least one uppercase letter (reject "un uer i")
                if not re.search(r'[A-Z]', s): return False
                # Reject if it looks like a sentence or random chars
                if re.search(r'[^a-zA-Z\s\.]', s): return False
                return True

            # If "To" line is just "To" or contains garbage, look at next lines
            if not is_good_name(name_candidate):
                 # Look at the next few lines for a valid name
                 for k in range(1, 4):
                     if i + k < len(lines):
                         next_ln = lines[i+k].strip()
                         # Skip empty lines or lines with digits (likely address/phone)
                         if not next_ln or re.search(r'\d', next_ln):
                             continue
                         # Skip lines that look like address parts or common headers
                         if re.search(r'Address|C/O|S/O|W/O|Near|Behind|Plot|Sector|Road|Nagar|Enrolment|Download|Date', next_ln, re.I):
                             continue
                         # If it looks like a name (mostly alpha, reasonable length), take it
                         if is_good_name(next_ln):
                             obj['name'] = next_ln
                             break
            else:
                 # "To Name" on same line
                 obj['name'] = name_candidate
            
            if obj['name']:
                break
    
    # Strategy B: Above DOB (Medium Confidence)
    if not obj['name'] and dob_idx != -1 and records:
        curr_y = records[dob_idx]['y']
        above_lines = [r for r in records if r['y'] < curr_y - 10]
        if above_lines:
            above_lines.sort(key=lambda r: r['y'], reverse=True)
            for r in above_lines:
                txt = r['text'].strip()
                if re.search(r'GOVERNMENT|INDIA|AADHAAR|VID|Father|Mother|Address|Issue|Download|Date|Year|Birth|Male|Female|Valid|Throughout|Country|Help|Avail|Benefits|Citizenship|Identity', txt, re.I):
                    continue
                if len(txt) < 3:
                    continue
                if re.match(r'^[A-Za-z\s\.]+$', txt):
                     # Avoid single words if possible, unless they look like a full name
                    if ' ' in txt or len(txt) > 4:
                        obj['name'] = txt
                        break

    # Strategy C: Title Case Scan (Fallback)
    if not obj['name']:
        try:
            raw = text
            name_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", raw)
            if name_matches:
                valid_names = []
                for name in name_matches:
                    if not re.search(r'\b(Government Of India|Unique Identification Authority|State Of Maharashtra|Father|Mother|Address|Issue Date|Download Date|Help You|Avail Various|Valid Throughout|Proof Of Identity|Not Of Citizenship)\b', name, re.I):
                        valid_names.append(name)
                if valid_names:
                    obj['name'] = max(valid_names, key=len)
        except:
            pass

    # Strategy D: English Text Block (Last Resort)
    if not obj['name']:
        for ln in lines:
            if re.match(r'^[A-Za-z\s\.]+$', ln) and len(ln) > 3:
                 if not re.search(r'GOVERNMENT|INDIA|AADHAAR|VID|DOWNLOAD|ISSUE|MALE|FEMALE|Date|Address|Father|Mother|Valid|Throughout|Country|Help|Avail|Benefits|Citizenship|Identity', ln, re.I):
                     if ln.isupper() or ln[0].isupper():
                         obj['name'] = ln.strip()
                         break
    
    # Cleanup Name
    if obj['name']:
        # Remove any newline chars if they crept in
        obj['name'] = obj['name'].replace('\n', ' ').strip()
        # If name contains "Nikky" twice (e.g. "Nikky Laxman Bisen Nikky"), clean it
        # This happens if multiple lines were merged or captured
        parts = obj['name'].split()
        if len(parts) > 2 and parts[0] == parts[-1]:
             obj['name'] = " ".join(parts[:-1])


    # 5. Address Strategy
    # Only extract if we see strong address indicators
    address_indicators = r'\b(S/O|W/O|C/O|D/O|HOUSE|FLAT|SECTOR|ROAD|LANE|STREET|VILLAGE|MANDAL|DISTRICT|STATE|PIN|PINCODE|PO:|VTC:)\b'
    
    # Strategy A: "To" Block Address
    # If we found "To", the lines following name are likely address
    if not obj['address']:
        for i, ln in enumerate(lines):
            if re.match(r'^To\b', ln, re.I):
                addr_lines = []
                start_collecting = False
                for j in range(i, min(len(lines), i + 10)):
                    # Skip the "To" line itself if it has name
                    if j == i and obj['name'] in lines[j]:
                        start_collecting = True
                        continue
                    # Skip the name line if it was found on next line
                    if obj['name'] and lines[j] == obj['name']:
                        start_collecting = True
                        continue
                    
                    if start_collecting or re.search(address_indicators, lines[j], re.I):
                        start_collecting = True
                        if re.search(r'(Issue|Download|Print)\s*Date', lines[j], re.I):
                            continue
                        if re.search(r'\b\d{10,12}\b', lines[j]): # Skip phone/aadhaar
                            continue
                        addr_lines.append(lines[j])
                        if re.search(r'\b\d{6}\b', lines[j]): # Pincode ends address usually
                            break
                
                if addr_lines:
                    cand = ", ".join(addr_lines)
                    if re.search(r'\b\d{6}\b', cand) or re.search(address_indicators, cand, re.I):
                        obj['address'] = cand
                break

    # Strategy B: "Address" Label
    if not obj['address']:
        for i, ln in enumerate(lines):
            if 'Address' in ln or 'ADDRESS' in ln:
                addr_lines = []
                for j in range(i + 1, min(len(lines), i + 8)):
                    if re.search(r'(Issue|Download|Print)\s*Date', lines[j], re.I):
                        continue
                    if re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', lines[j], re.I):
                        continue
                    if re.search(r'DOB|Year of Birth', lines[j], re.I):
                        continue
                    addr_lines.append(lines[j])
                    if re.search(r'\b\d{6}\b', lines[j]):
                        break
                cand = ", ".join(addr_lines)
                if re.search(r'\b\d{6}\b', cand) or re.search(address_indicators, cand, re.I):
                    obj['address'] = cand
                break

    # Cleanup
    if obj['address']:
        # Remove Aadhaar number
        if obj['aadhaar_number']:
             obj['address'] = obj['address'].replace(obj['aadhaar_number'], '')
             spaced = " ".join(list(obj['aadhaar_number'])) # 1 2 3 4 ...
             # Simple heuristic for spaced aadhaar removal might be needed but usually regex handles it
        
        # Remove Name from address if it leaked in
        if obj['name']:
            obj['address'] = obj['address'].replace(obj['name'], '')

        obj['address'] = re.sub(r'^[\s,]+|[\s,]+$', '', obj['address'])
        
        # Final check: if address is very short or just numbers, clear it
        if len(obj['address']) < 10 or re.match(r'^[\d\s,]+$', obj['address']):
            obj['address'] = ""

    return obj


def extract_voter(lines: List[str], text: str) -> Dict:
    obj = {
        "doc_type": "voter_id",
        "name": "",
        "father_name": "",
        "epic_number": "",
        "dob": "",
        "gender": "",
        "address": ""
    }
    
    # 1. EPIC Number - Pattern: ABC1234567 (3 letters + 7 digits)
    epic = re.search(r'\b([A-Z]{3}\d{7})\b', text)
    if epic:
        obj['epic_number'] = epic.group(1)
    
    # 2. Name - Look for "Elector's Name" or "Name"
    for i, ln in enumerate(lines):
        if re.search(r"Elector'?s?\s*Name", ln, re.I):
            # Check same line after colon
            val = re.sub(r".*Elector'?s?\s*Name\s*[:\.]?", '', ln, flags=re.I).strip()
            if val and len(val) > 2:
                obj['name'] = val
            else:
                # Check next line
                val = nearest_line(lines, i, 1)
                if val and not re.search(r'Father|Address|Sex|Date|Birth', val, re.I):
                    obj['name'] = val
            break
    
    # Fallback: Look for "Name" if Elector's Name not found
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
    
    # 3. Father's Name - Look for "Father's Name" or "S/O"
    for i, ln in enumerate(lines):
        if re.search(r"Father'?s?\s*Name", ln, re.I):
            # Check same line after colon
            val = re.sub(r".*Father'?s?\s*Name\s*[:\.]?", '', ln, flags=re.I).strip()
            if val and len(val) > 2:
                obj['father_name'] = val
            else:
                # Check next line
                val = nearest_line(lines, i, 1)
                if val and not re.search(r'Name|Address|Sex|Date|Birth', val, re.I):
                    obj['father_name'] = val
            break
    
    # Fallback: S/O pattern
    if not obj['father_name']:
        for i, ln in enumerate(lines):
            if re.search(r'S/O|SON OF', ln, re.I):
                val = re.sub(r'.*S/O\s*[:\.]?', '', ln, flags=re.I).strip()
                val = re.sub(r'.*SON OF\s*[:\.]?', '', val, flags=re.I).strip()
                if val and len(val) > 2:
                    obj['father_name'] = val
                break
    
    # 4. DOB - Look for "Date of Birth" or date pattern (English only)
    for i, ln in enumerate(lines):
        if re.search(r'Date\s*of\s*Birth|DOB', ln, re.I):
            # Extract date from same line
            m = re.search(r'(\d{2}[\-\/]\d{2}[\-\/]\d{4})', ln)
            if m:
                obj['dob'] = m.group(1)
            else:
                # Check next line
                val = nearest_line(lines, i, 1)
                m = re.search(r'(\d{2}[\-\/]\d{2}[\-\/]\d{4})', val)
                if m:
                    obj['dob'] = m.group(1)
            break
    
    # Fallback: any date pattern
    if not obj['dob']:
        m = re.search(r'(\d{2}[\-\/]\d{2}[\-\/]\d{4})', text)
        if m:
            obj['dob'] = m.group(1)
    
    # 5. Gender - Look for "Sex" or "Gender" or direct MALE/FEMALE (English only)
    for i, ln in enumerate(lines):
        if re.search(r'Sex|Gender', ln, re.I):
            # Check same line
            if re.search(r'MALE|FEMALE', ln, re.I):
                m = re.search(r'(MALE|FEMALE)', ln, re.I)
                if m:
                    gender_val = m.group(1).upper()
                    if gender_val == 'MALE':
                        obj['gender'] = 'Male'
                    elif gender_val == 'FEMALE':
                        obj['gender'] = 'Female'
            else:
                # Check next line
                val = nearest_line(lines, i, 1)
                if re.search(r'MALE|FEMALE', val, re.I):
                    m = re.search(r'(MALE|FEMALE)', val, re.I)
                    if m:
                        gender_val = m.group(1).upper()
                        if gender_val == 'MALE':
                            obj['gender'] = 'Male'
                        elif gender_val == 'FEMALE':
                            obj['gender'] = 'Female'
            break
    
    # 6. Address - Look for "Address" label (English only)
    for i, ln in enumerate(lines):
        if re.search(r'Address', ln, re.I):
            addr = []
            for j in range(i + 1, min(len(lines), i + 8)):
                next_ln = lines[j].strip()
                if not next_ln:
                    continue
                # Stop at other labels
                if re.search(r'(Name|Father|Sex|Date|Birth|EPIC)', next_ln, re.I):
                    break
                addr.append(next_ln)
            if addr:
                obj['address'] = ", ".join(addr)
            break
    
    return obj


def extract_passport(lines: List[str], text: str) -> Dict:
    obj = {"doc_type": "passport", "passport_number": "", "name": "", "nationality": "", "dob": "", "place_of_birth": ""}
    p_m = re.search(r'\b([A-Z]\d{7})\b', text)
    if p_m:
        obj['passport_number'] = p_m.group(1)
    for i, ln in enumerate(lines):
        u = ln.upper()
        if 'NATIONALITY' in u:
            obj['nationality'] = ln.split(':')[-1].strip() or nearest_line(lines, i, 1)
        if 'SURNAME' in u or 'GIVEN NAME' in u or 'NAME' in u:
            # passport MRZ style often has surname/given name
            val = ln.split(':')[-1].strip()
            if val:
                obj['name'] = val
            else:
                obj['name'] = nearest_line(lines, i, 1)
        if 'DATE OF BIRTH' in u or 'DOB' in u:
            m = re.search(r'(\d{2}[\-\/]\d{2}[\-\/]\d{4}|\d{2}\s?[A-Z]{3}\s?\d{4})', ln)
            if m:
                obj['dob'] = m.group(1)
            else:
                obj['dob'] = nearest_line(lines, i, 1)
        if 'PLACE OF BIRTH' in u or 'POB' in u:
            obj['place_of_birth'] = ln.split(':')[-1].strip() or nearest_line(lines, i, 1)
    # MRZ fallback: try to parse MRZ lines for name and passport number
    mrz = [ln for ln in lines if '<<' in ln]
    if mrz:
        mrz_text = mrz[0]
        # passport number is first 9 chars in MRZ usually
        m = re.search(r'([A-Z0-9<]{9})', mrz_text)
        if m and not obj['passport_number']:
            obj['passport_number'] = m.group(1).replace('<', '')
        # name part
        parts = mrz_text.split('<<')
        if len(parts) >= 2 and not obj['name']:
            nm = parts[1].replace('<', ' ').strip()
            obj['name'] = nm
    return obj


def extract_driving_license(lines: List[str], text: str) -> Dict:
    obj = {
        "doc_type": "driving_license",
        "dl_number": "",
        "name": "",
        "dob": "",
        "address": "",
        "validity": "",
        "issue_date": "",
        "blood_group": "",
        "cov": ""
    }
    
    # 1. DL Number
    # Pattern: DL No : MH36 2C220004543
    # Allow alphanumeric in the second part
    dl_match = re.search(r'DL\s*No\s*[:\.]?\s*([A-Z0-9\s]+)', text, re.I)
    if dl_match:
        # Clean up the result: take only the first valid block if it contains newlines or extra junk
        raw_dl = dl_match.group(1).strip()
        # Split by newline or double space to avoid capturing next field
        parts = re.split(r'\n|\s{2,}', raw_dl)
        if parts:
            obj['dl_number'] = parts[0].strip()
    else:
        # Fallback: Look for MH pattern (State Code + RTO Code + Alphanumeric)
        # MH36 2C220004543
        # Use lines to avoid capturing across lines
        for ln in lines:
             mh_match = re.search(r'\b([A-Z]{2}\d{2}\s?[A-Z0-9]{5,})\b', ln)
             if mh_match:
                 obj['dl_number'] = mh_match.group(1).strip()
                 break

    # 2. Dates (DOI, Valid Till, DOB)
    # DOI :17-06-2022
    doi_match = re.search(r'DOI\s*[:\.]?\s*(\d{2}[\-\/]\d{2}[\-\/]\d{4})', text, re.I)
    if doi_match:
        obj['issue_date'] = doi_match.group(1)
    
    # Valid Till :04-04-2039
    valid_match = re.search(r'Valid\s*Till\s*[:\.]?\s*(\d{2}[\-\/]\d{2}[\-\/]\d{4})', text, re.I)
    if valid_match:
        obj['validity'] = valid_match.group(1)

    # DOB : 05-04-1999
    dob_match = re.search(r'DOB\s*[:\.]?\s*(\d{2}[\-\/]\d{2}[\-\/]\d{4})', text, re.I)
    if dob_match:
        obj['dob'] = dob_match.group(1)

    # 3. Name
    # Name : NIKKY L BISEN
    # Sometimes "Name" is on one line and value on next
    # Be strict: only match "Name" at start of line or after BG/DOB labels
    for i, ln in enumerate(lines):
        # Match "Name" at start of line, or "BG Name", "DOI Name", etc.
        if re.search(r'^Name\b\s*[:\.]?', ln, re.I) or re.search(r'(BG|DOB|DOI)\s+Name\b\s*[:\.]?', ln, re.I):
            # Check same line - extract everything after "Name"
            val = re.sub(r'^.*Name\b\s*[:\.]?', '', ln, flags=re.I).strip()
            # If value starts with :, remove it
            val = val.lstrip(':').strip()
            
            # Remove any trailing "BG" or other labels that might be on same line
            val = re.sub(r'\s*(BG|DOB|DOI|Add|S/D/W).*$', '', val, flags=re.I).strip()
            
            if val and len(val) > 2:
                # Exclude if it looks like a date
                if not re.match(r'^\d{2}[\-\/]\d{2}[\-\/]\d{4}$', val):
                    obj['name'] = val
            else:
                # Check next line
                val = nearest_line(lines, i, 1)
                # If next line starts with :, remove it
                val = val.lstrip(':').strip()
                
                if val and not re.search(r'S/D/W|Add|DOB|BG|DOI|Valid|S/DM', val, re.I):
                    # Also exclude COV codes if they appear as "Name"
                    if not re.search(r'\b(LMV|MCWG|MCW|HGMV|HPMV|TRANS|LDRXCV)\b', val, re.I):
                        # Exclude if it looks like a date
                        if not re.match(r'^\d{2}[\-\/]\d{2}[\-\/]\d{4}$', val):
                            obj['name'] = val
            
            if obj['name']:
                break
    
    # Fallback for Name: Look for "S/D/W of" and take line above
    if not obj['name']:
        for i, ln in enumerate(lines):
            if 'S/D/W' in ln or 'S/O' in ln or 'D/O' in ln:
                # Look above
                above = nearest_line(lines, i, -1)
                if above and not re.search(r'Name|DOB|BG', above, re.I):
                    obj['name'] = above
                break

    # 4. Address
    # Add : AT DEOSARRA...
    for i, ln in enumerate(lines):
        if re.match(r'^(Add|Address)\s*[:\.]?', ln, re.I):
            addr_parts = []
            # Check if content is on the same line
            same_line_content = re.sub(r'^(Add|Address)\s*[:\.]?', '', ln, flags=re.I).strip()
            if same_line_content:
                addr_parts.append(same_line_content)
            
            # Capture next few lines
            for j in range(i + 1, min(len(lines), i + 6)):
                next_ln = lines[j].strip()
                if not next_ln: continue
                # Stop if we hit other labels
                if re.search(r'(PIN|Signature|Issuing|DOI|Valid|DOB|BG|MH36)', next_ln, re.I):
                    # If PIN is on this line, capture it and stop
                    if 'PIN' in next_ln.upper():
                        pin_match = re.search(r'PIN\s*[:\.]?\s*(\d{6})', next_ln, re.I)
                        if pin_match:
                            addr_parts.append(pin_match.group(0))
                    break
                # Exclude S/D/W lines
                if re.search(r'S/D/W|S/O|D/O|W/O|S/DM', next_ln, re.I):
                    continue
                # Exclude COV lines (LMV, MCWG, etc.)
                if re.search(r'\b(LMV|MCWG|MCW|HGMV|HPMV|TRANS|LDRXCV)\b', next_ln, re.I):
                    continue
                addr_parts.append(next_ln)
            
            obj['address'] = ", ".join(addr_parts)
            break
    
    # 5. Blood Group
    # BG :
    bg_match = re.search(r'BG\s*[:\.]?\s*([A-Z]{1,2}[\+\-])', text, re.I)
    if bg_match:
        obj['blood_group'] = bg_match.group(1)

    # 6. Class of Vehicle (COV)
    # Scan for known vehicle classes directly
    cov_types = set()
    for ln in lines:
        parts = ln.split()
        for p in parts:
            p_clean = p.strip().upper()
            if p_clean in ['LMV', 'MCWG', 'MCW', 'HGMV', 'HPMV', 'TRANS', 'LDRXCV']:
                cov_types.add(p_clean)
    
    if cov_types:
        obj['cov'] = ", ".join(sorted(list(cov_types)))

    return obj


def extract_structured(lines: List[str], records: List[Dict] = None) -> Dict:
    text = text_blob(lines)
    print("text",text)
    doc = detect_doc_type(text)
    if doc == 'pan':
        return extract_pan(lines, text)
    if doc == 'aadhaar':
        return extract_aadhaar(lines, text, records)
    if doc == 'driving_license':
        return extract_driving_license(lines, text)
    if doc == 'voter_id':
        return extract_voter(lines, text)
    if doc == 'passport':
        return extract_passport(lines, text)
    return {"doc_type": "unknown", "raw_text": text}


def canonicalize(structured: Dict, raw_text: str = "") -> Dict:
    """Map document-specific extraction into a canonical JSON schema.
    Only includes relevant fields based on document type.
    """
    dt = structured.get("doc_type", "unknown")
    
    # Common fields for all documents
    out = {
        "name": structured.get('name', ''),
        "dob": structured.get('dob', ''),
        "address": structured.get('address', '')
    }
    
    # Add document-specific fields
    if dt == 'aadhaar':
        out['adhar_no'] = structured.get('aadhaar_number', '')
        out['gender'] = structured.get('gender', '')
    
    elif dt == 'pan':
        out['pan_no'] = structured.get('pan_number', '')
        out['father_name'] = structured.get('father_name', '')
    
    elif dt == 'driving_license':
        out['dl_number'] = structured.get('dl_number', '')
        out['validity'] = structured.get('validity', '')
        out['issue_date'] = structured.get('issue_date', '')
        out['blood_group'] = structured.get('blood_group', '')
        out['cov'] = structured.get('cov', '')
    
    elif dt == 'voter_id':
        out['epic_no'] = structured.get('epic_number', '')
        out['father_name'] = structured.get('father_name', '')
        out['gender'] = structured.get('gender', '')
    
    elif dt == 'passport':
        out['passport_no'] = structured.get('passport_number', '')
        out['nationality'] = structured.get('nationality', '')
        out['place_of_birth'] = structured.get('place_of_birth', '')

    return out


def process_path(path: Path) -> Dict:
    records = ocr_records(str(path))
    lines = [r['text'] for r in records]
    structured = extract_structured(lines, records)
    # keep raw_text for canonicalization
    structured['raw_text'] = text_blob(lines)
    canonical = canonicalize(structured, structured['raw_text'])

    return canonical


def gather_image_paths(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_dir():
        return [x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS]
    if p.is_file():
        return [p] if p.suffix.lower() in IMAGE_EXTS else []
    # if wildcard or pattern
    return []


def main():
    parser = argparse.ArgumentParser(description="Extract structured JSON from Indian ID documents using PaddleOCR")
    parser.add_argument('--input', '-i', required=True, help="Image file or folder containing images")
    parser.add_argument('--output', '-o', default='ocr_results.json', help="Output JSON file path")
    args = parser.parse_args()

    paths = gather_image_paths(args.input)
    if not paths:
        print("No images found in input. Supported extensions:", ",".join(IMAGE_EXTS))
        return

    results = []
    for p in paths:
        print("Processing:", p)
        try:
            res = process_path(p)
            results.append(res)
            # print structured result to terminal for quick inspection
            try:
                print(json.dumps(res, ensure_ascii=False, indent=2))
            except Exception:
                print(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"file_name": p.name, "error": str(e)})

    outp = Path(args.output)
    with outp.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved results to", outp)


if __name__ == "__main__":
    main()
