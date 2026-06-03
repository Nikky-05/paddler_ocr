"""
Vision-LLM document extraction via Ollama.

This is the sole OCR/extraction engine: every uploaded document image is sent
directly to a local vision-language model (served by Ollama, default
``qwen3-vl:2b-instruct`` — a small, fast Qwen3-VL model) which reads the printed
fields and returns them as JSON. There is no PaddleOCR / regex pipeline behind
this — the model both "reads" the pixels and structures the result.

The model is asked for the exact response field names used by ``main.py`` (name,
dob, aadhaar, pan, ...). We then translate those into the ``extracted`` dict that
``create_uniform_response`` consumes, so the API response shape is unchanged.

Configuration (environment variables):
    OLLAMA_HOST           default "http://localhost:11434"
    OLLAMA_VISION_MODEL   default "qwen3-vl:2b-instruct"
                          NOTE: use the "-instruct" (non-thinking) variant. The
                          plain "thinking" tags (e.g. qwen3-vl:2b / :8b) emit a
                          long <think> block and ignore think=False; on the small
                          2b model the reasoning alone exhausts num_predict, so
                          the JSON answer is never produced and extraction returns
                          nothing (HTTP 400 "does not match document type").
    LLM_TIMEOUT           default "150" (seconds)
    LLM_IMAGE_WIDTH       default "1100"
"""

import base64
import json
import os
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration (environment variables)
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_VISION_MODEL = _env("OLLAMA_VISION_MODEL", "qwen3-vl:2b-instruct")
LLM_TIMEOUT = float(_env("LLM_TIMEOUT", "150"))
# KV-cache context size. With flash attention disabled the GPU may fail to
# allocate large contexts, so we retry smaller (see _call_ollama).
LLM_NUM_CTX = int(_env("LLM_NUM_CTX", "4096"))
# Keep the model resident between requests so calls stay fast.
LLM_KEEP_ALIVE = _env("LLM_KEEP_ALIVE", "10m")
# Max image width sent to the model. Smaller -> fewer vision tokens. ~1100px is
# plenty to read printed ID fields.
LLM_IMAGE_WIDTH = int(_env("LLM_IMAGE_WIDTH", "1100"))
# Max tokens to generate. Must cover qwen3-vl's hidden "thinking" tokens PLUS the
# JSON answer (thinking alone can be ~800 tokens on a busy card).
LLM_NUM_PREDICT = int(_env("LLM_NUM_PREDICT", "2048"))


# ---------------------------------------------------------------------------
# Low-level Ollama plumbing
# ---------------------------------------------------------------------------

def _image_to_b64(img: Image.Image, max_width: int = LLM_IMAGE_WIDTH) -> str:
    """Encode a PIL image as base64 JPEG, downscaling very large images."""
    work = img if img.mode == "RGB" else img.convert("RGB")
    w, h = work.size
    if w > max_width:
        work = work.resize((max_width, int(h * max_width / w)), Image.LANCZOS)
    buf = BytesIO()
    work.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_content(content: str) -> Optional[Dict]:
    """Parse the model's reply into a dict, tolerating prose/markdown wrapping."""
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


def _post_chat(img_b64: str, prompt: str, num_ctx: int) -> Optional[Dict]:
    payload = {
        "model": OLLAMA_VISION_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "format": "json",
        "think": False,
        "keep_alive": LLM_KEEP_ALIVE,
        # qwen3-vl is a reasoning model: it emits hidden "thinking" tokens before
        # the JSON answer and ignores think=False. Those thinking tokens count
        # toward num_predict, so the cap must be high enough to fit BOTH the
        # reasoning and the answer — otherwise generation stops mid-think and
        # `content` comes back empty. 2048 leaves comfortable headroom.
        "options": {"temperature": 0, "num_predict": LLM_NUM_PREDICT, "num_ctx": num_ctx},
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return _parse_content((data.get("message") or {}).get("content", ""))


def _call_ollama(img_b64: str, prompt: str) -> Optional[Dict]:
    """Call Ollama, retrying once at a smaller context if the GPU can't allocate
    the KV-cache layout (e.g. flash attention disabled / tight VRAM)."""
    try:
        return _post_chat(img_b64, prompt, LLM_NUM_CTX)
    except requests.HTTPError as e:
        body = (e.response.text if e.response is not None else "").lower()
        if e.response is not None and e.response.status_code == 500 and (
            "memory" in body or "allocat" in body or "resource" in body
        ):
            print(f"[DEBUG] Ollama KV-cache alloc failed at num_ctx={LLM_NUM_CTX}; "
                  "retrying at 2048. Set OLLAMA_FLASH_ATTENTION=1 to avoid this.")
            return _post_chat(img_b64, prompt, 2048)
        raise


# ---------------------------------------------------------------------------
# Per-document field definitions
# ---------------------------------------------------------------------------
# The model returns these response-level field names. Order matters only for
# prompt readability.

_FIELD_HINTS = {
    "name": "full name of the document holder, exactly as printed (English/Latin letters)",
    "dob": "date of birth in DD/MM/YYYY format",
    "gender": "gender: MALE, FEMALE or TRANSGENDER",
    "father": "father's name",
    "mother": "mother's name",
    "husband": "husband's / spouse's name",
    "address": "full postal address exactly as printed, as ONE single line with no line breaks (join the printed lines with ', ')",
    "aadhaar": "12-digit Aadhaar number (digits only, may keep grouping spaces)",
    "vid": "16-digit Virtual ID (VID) number",
    "pan": "10-character PAN (5 letters, 4 digits, 1 letter)",
    "dl_number": "driving licence number",
    "epic": "voter EPIC number",
    "passport": "passport number",
    "place_of_birth": "place of birth",
    "nationality": "nationality",
    "validity": "validity / expiry date in DD/MM/YYYY format",
    "issue_date": "date of issue in DD/MM/YYYY format",
    "blood_group": "blood group",
    "cov": "class of vehicle (COV)",
}

# All fields to extract per document type.
_DOC_FIELDS = {
    "aadhaar": ["name", "dob", "aadhaar", "vid", "address", "father"],
    "pan": ["name", "pan", "dob", "father"],
    "driving_license": ["name", "dl_number", "dob", "address", "issue_date",
                        "validity", "blood_group", "cov", "father"],
    "voter_id": ["name", "epic", "gender", "dob", "father", "address"],
    "passport": ["name", "passport", "dob", "gender", "place_of_birth",
                 "nationality", "father", "mother", "husband", "issue_date",
                 "validity"],
}

_DOC_LABELS = {
    "aadhaar": "Indian Aadhaar card",
    "pan": "Indian PAN card",
    "driving_license": "Indian driving licence",
    "voter_id": "Indian Voter ID (EPIC) card",
    "passport": "Indian passport",
}

# Document-specific guidance that helps the model read the layout/format of each
# Indian ID correctly. Appended to the prompt for that document type.
_DOC_NOTES = {
    "aadhaar": [
        "The card is bilingual: a regional/Hindi (Devanagari etc.) line is usually",
        "followed by the same text in English. Always return the ENGLISH spelling of",
        "name and address.",
        "Aadhaar number = 12 digits printed as 'XXXX XXXX XXXX'. If it is masked it",
        "shows as 'XXXX XXXX 1234' — return it exactly as shown.",
        "VID = 16 digits printed as four groups of four.",
        "DOB may be labelled 'DOB' or 'Year of Birth' (then only a year is present).",
        "Gender appears as MALE / FEMALE / पुरुष / महिला.",
    ],
    "pan": [
        "PAN is EXACTLY 10 characters: 5 letters, 4 digits, then 1 letter",
        "(e.g. ABCDE1234F). Read it character-by-character; never output spaces.",
        "Two names are printed: the cardholder's name and the father's name (often",
        "labelled). The cardholder's name is the primary 'name'.",
    ],
    "driving_license": [
        "The DL number format varies by state, e.g. 'MH12 20110012345' or",
        "'DL-0420110149646'. Return it as printed.",
        "issue_date is labelled 'DOI' or 'Date of Issue'; validity is 'Valid Till' /",
        "'Validity (NT)'. blood_group is like 'B+', 'O+'. cov = Class Of Vehicle",
        "(e.g. LMV, MCWG, MCWOG, TRANS).",
    ],
    "voter_id": [
        "EPIC number is usually 3 letters followed by 7 digits (e.g. ABC1234567),",
        "sometimes with a different state prefix — return it as printed.",
        "The relative's name may be labelled Father's Name / Husband's Name.",
    ],
    "passport": [
        "Passport number = 1 letter followed by 7 digits (e.g. M1234567).",
        "If printed fields are unclear, cross-check the MRZ (the two long lines of",
        "'<<<' characters at the bottom of the data page).",
        "place_of_birth, and the parents'/spouse names, appear on the back/address",
        "page — only fill them if visible.",
    ],
}

# Map response field name -> the key create_uniform_response() expects in the
# `extracted` dict. (husband is special-cased for passport -> spouse_name.)
_EXTRACTED_KEY = {
    "name": "name",
    "dob": "dob",
    "gender": "gender",
    "address": "address",
    "aadhaar": "aadhaar_number",
    "vid": "vid",
    "pan": "pan_number",
    "dl_number": "dl_number",
    "epic": "epic_number",
    "passport": "passport_number",
    "nationality": "nationality",
    "place_of_birth": "place_of_birth",
    "validity": "validity",
    "issue_date": "issue_date",
    "blood_group": "blood_group",
    "cov": "cov",
    "father": "father_name",
    "mother": "mother_name",
    "husband": "husband_name",
}

# The five document types the API understands; used to validate that the image
# actually matches the requested type.
_KNOWN_TYPES = {"aadhaar", "pan", "driving_license", "voter_id", "passport"}

# Address tokens that mark where a relation name ends (e.g. "S/O Laxman Bisane at
# deosara ..." — the father's name stops before "at").
_ADDR_STOP = re.compile(
    r'\b(at|near|post|vill(age)?|vtc|dist(rict)?|po|ps|teh(sil)?|taluka|mandal|'
    r'house|h\.?no|road|street|colony|nagar|gali|sector|pin|state|landmark)\b',
    re.I,
)


# ---------------------------------------------------------------------------
# Father's-name derivation (Indian IDs rarely print it as its own field)
# ---------------------------------------------------------------------------

def _father_from_address(address: str) -> str:
    """Pull a father's name from an Aadhaar-style relation marker at the start of
    the address: "S/O <name>", "C/O <name>", "D/O <name>" (son/care/daughter of).
    Returns the name up to the first address keyword/comma, or "" if not found."""
    if not address:
        return ""
    m = re.search(r'\b([SCD])\s*[/.]?\s*[O0]\b[:\-\s]+(.+)', address, re.I)
    if not m:
        return ""
    rest = m.group(2)
    # Cut at the first comma or address keyword.
    rest = re.split(r',', rest, 1)[0]
    stop = _ADDR_STOP.search(rest)
    if stop:
        rest = rest[:stop.start()]
    return " ".join(rest.split()).strip(" ,-")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _build_prompt(doc_type: str) -> str:
    label = _DOC_LABELS.get(doc_type, "Indian identity document")
    fields = _DOC_FIELDS.get(doc_type, [])

    lines = [
        f"You are an OCR engine reading a {label} from the attached image.",
        "Read the printed text directly from the image and extract the fields below.",
        "Return STRICTLY one JSON object with exactly these keys (no extra keys, no commentary):",
        "",
    ]
    for f in fields:
        lines.append(f'  "{f}": {_FIELD_HINTS.get(f, f)}')
    lines += [
        '  "document_type": one of "aadhaar", "pan", "driving_license", '
        '"voter_id", "passport", or "other" — what this image actually is',
    ]

    notes = _DOC_NOTES.get(doc_type)
    if notes:
        lines += ["", f"Notes for reading this {label}:", " ".join(notes)]

    lines += [
        "",
        "Rules:",
        "- Transcribe Latin characters exactly as printed. Do not translate.",
        '- If a field is genuinely not visible/readable, return an empty string "".',
        "- Do not invent or guess values.",
        "- Numbers (Aadhaar, VID, PAN, EPIC, passport, DL) must be transcribed digit-for-digit.",
        "- Output JSON only.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _query_model(img: Image.Image, doc_type: str, max_width: int) -> Optional[Dict]:
    """One vision-model call. Returns the parsed JSON dict, or None on any
    failure (HTTP error, timeout, or a reply that wasn't usable JSON — which is
    what happens when the model overthinks and exhausts the token budget)."""
    try:
        img_b64 = _image_to_b64(img, max_width=max_width)
        result = _call_ollama(img_b64, _build_prompt(doc_type))
        return result if isinstance(result, dict) else None
    except Exception as e:  # noqa: BLE001 - never break the request
        print(f"[DEBUG] Ollama OCR error: {e}")
        return None


def _result_to_extracted(result: Dict, fields: List[str], doc_type: str) -> Dict[str, str]:
    """Map the model's JSON onto the keys create_uniform_response expects."""
    extracted: Dict[str, str] = {}
    for f in fields:
        val = result.get(f)
        if isinstance(val, (int, float)):
            val = str(val)
        if isinstance(val, str):
            # Collapse newlines/tabs/multiple spaces into single spaces so multi-
            # line fields (esp. the Aadhaar address, which is printed across
            # several lines) come back as one clean exact line — no "\n" artifacts.
            val = " ".join(val.split())
            if val:
                key = _EXTRACTED_KEY.get(f, f)
                # Passport stores spouse name under spouse_name, not husband_name.
                if doc_type == "passport" and f == "husband":
                    key = "spouse_name"
                extracted[key] = val
    return extracted


# Neutral classification prompt. Deliberately does NOT mention which document the
# caller expects — telling a small, suggestible model "this is a Voter ID" makes
# it just echo that back. Judging only from the pixels keeps validation honest.
_CLASSIFY_PROMPT = (
    "You are shown an image of an identity document. Identify which ONE Indian "
    "identity document it is, based ONLY on what is printed and the layout. Do "
    "not assume any particular type.\n"
    'Return STRICTLY this JSON: {"document_type": X} where X is exactly one of:\n'
    '"aadhaar", "pan", "driving_license", "voter_id", "passport", "other".\n'
    "Clues: Aadhaar has a 12-digit number (XXXX XXXX XXXX), often a VID, and the "
    "UIDAI logo. PAN is an Income-Tax 'Permanent Account Number' card with a "
    "10-character PAN. Voter ID (EPIC) is an Election Commission card with an EPIC "
    "number. Driving Licence shows a DL number, validity dates and vehicle class "
    "(COV). Passport is a booklet data page with an MRZ (two '<<<' lines). "
    "Output JSON only."
)


def classify_document(img: Image.Image) -> Optional[str]:
    """Independently classify the document type from the image, WITHOUT being told
    which type the caller expects.

    Returns one of the five known types ("aadhaar", "pan", "driving_license",
    "voter_id", "passport"), "other", or None when the model gave no usable
    answer. Used for STRICT type validation: because the prompt never reveals the
    expected type, the model can't simply agree with it — it must actually read
    the document. Never raises.
    """
    try:
        img_b64 = _image_to_b64(img)
        result = _call_ollama(img_b64, _CLASSIFY_PROMPT)
    except Exception as e:  # noqa: BLE001 - never break the request
        print(f"[DEBUG] Ollama classify error: {e}")
        return None
    if not isinstance(result, dict):
        return None
    detected = result.get("document_type")
    if isinstance(detected, str):
        detected = detected.strip().lower()
        if detected in _KNOWN_TYPES or detected == "other":
            return detected
    return None


def extract_document(img: Image.Image, doc_type: str) -> Tuple[Dict, Optional[str]]:
    """Run the vision model on `img` and return (extracted, detected_type).

    `extracted` uses the same keys ``create_uniform_response`` expects (e.g.
    ``aadhaar_number``, ``father_name``). `detected_type` is the model's guess of
    what the document actually is (one of the five known types, "other", or
    None when the model didn't say) — used by the caller for validation.

    Never raises: on any failure returns ({}, None) so the caller can decide.
    """
    fields = _DOC_FIELDS.get(doc_type)
    if not fields:
        return {}, None

    # Attempt order. A full e-Aadhaar "letter" is a portrait A4 page whose right
    # half is a dense bilingual INFORMATION panel; feeding the whole page makes
    # the model overthink and return nothing. For such portrait Aadhaar pages,
    # read the LEFT column (where all the real data lives) at higher zoom first,
    # then fall back to the full image for normal single-card scans.
    attempts = [(img, LLM_IMAGE_WIDTH)]
    if doc_type == "aadhaar" and img.height > img.width * 1.15:
        w, h = img.size
        left = img.crop((0, 0, int(w * 0.55), h))
        attempts = [(left, 1400), (img, LLM_IMAGE_WIDTH)]

    result: Optional[Dict] = None
    extracted: Dict[str, str] = {}
    for im, width in attempts:
        result = _query_model(im, doc_type, width)
        if result is not None:
            extracted = _result_to_extracted(result, fields, doc_type)
            if extracted:
                break

    if not extracted:
        return {}, None

    # Father's name. Indian IDs (esp. Aadhaar) do NOT print a dedicated "Father's
    # Name" field, so the vision model often leaves it empty or — worse — just
    # echoes the holder's OWN name back. Resolve it deterministically:
    #   1) Drop a model value that merely repeats the holder's name (a misread).
    #   2) Otherwise take it from the "S/O <name>" (son of) / "C/O" / "D/O" marker
    #      printed at the start of the address, e.g. "S/O: Umashankar" -> "Umashankar".
    #   3) If neither yields a real name, leave the field EMPTY — never guess it
    #      from the holder's own name.
    if "father" in fields:
        name_norm = (extracted.get("name") or "").strip().lower()
        father = (extracted.get("father_name") or "").strip()
        if father.lower() == name_norm:
            father = ""
        if not father:
            father = _father_from_address(extracted.get("address", ""))
        if father:
            extracted["father_name"] = father
        else:
            extracted.pop("father_name", None)

    if doc_type in ("aadhaar", "passport"):
        extracted.setdefault("nationality", "INDIAN")

    detected = result.get("document_type")
    if isinstance(detected, str):
        detected = detected.strip().lower()
        detected = detected if detected in _KNOWN_TYPES or detected == "other" else None
    else:
        detected = None

    print(f"[DEBUG] Ollama OCR extracted {list(extracted.keys())}; detected={detected}")
    return extracted, detected
