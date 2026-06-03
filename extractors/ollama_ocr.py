"""
Vision-LLM document extraction via Ollama.

This is the sole OCR/extraction engine: every uploaded document image is sent
directly to a local vision-language model (served by Ollama, default
``qwen2.5vl:7b``) which reads the printed fields and returns them as JSON. There
is no PaddleOCR / regex pipeline behind this — the model both "reads" the pixels
and structures the result.

Hallucination control (production): besides the prompt (which tells the model to
return "" rather than guess, and not to confuse the Enrolment No. with the
Aadhaar number), every format-checkable field is validated after extraction
(see _FIELD_VALIDATORS) and any value that fails its format is DROPPED, so a
wrong Aadhaar/VID/DOB/PAN is never written into the response.

The model is asked for the exact response field names used by ``main.py`` (name,
dob, aadhaar, pan, ...). We then translate those into the ``extracted`` dict that
``create_uniform_response`` consumes, so the API response shape is unchanged.

Configuration (environment variables):
    OLLAMA_HOST           default "http://localhost:11434"
    OLLAMA_VISION_MODEL   default "qwen2.5vl:7b"
                          NOTE: use a non-thinking model. The "thinking" Qwen3-VL
                          tags (e.g. qwen3-vl:2b / :8b) emit a long <think> block
                          and ignore think=False; the reasoning can exhaust
                          num_predict so the JSON answer is never produced and
                          extraction returns nothing (HTTP 400 "does not match
                          document type"). qwen2.5vl:7b and qwen3-vl:2b-instruct
                          are both safe (non-thinking).
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

from .utils import validate_verhoeff, extract_english_only


# ---------------------------------------------------------------------------
# Configuration (environment variables)
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_VISION_MODEL = _env("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
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

# --- Sampling / randomness controls ----------------------------------------
# For OCR/KYC we want DETERMINISTIC, repeatable reads, so the defaults are the
# most conservative possible: temperature 0 with a fixed seed => the model picks
# the single most-likely token every time and the same image yields the same
# JSON on every call. These are exposed as env vars so randomness CAN be dialled
# up when wanted (e.g. LLM_TEMPERATURE=0.7 to let the model vary its output, or a
# different LLM_SEED to sample a different deterministic path). Raising the
# temperature increases hallucination risk, so keep it at 0 in production.
#   LLM_TEMPERATURE  0.0  -> greedy/deterministic; higher = more random
#   LLM_TOP_P        0.1  -> nucleus sampling cutoff (1.0 = consider all tokens)
#   LLM_TOP_K        1    -> only the single most-likely token (0 = disabled)
#   LLM_SEED         0    -> fixed RNG seed for reproducibility
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0"))
LLM_TOP_P = float(_env("LLM_TOP_P", "0.1"))
LLM_TOP_K = int(_env("LLM_TOP_K", "1"))
LLM_SEED = int(_env("LLM_SEED", "0"))


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
        # Sampling is env-controlled (see config block). With the defaults
        # (temperature 0 + top_p 0.1 + top_k 1 + fixed seed) decoding is greedy and
        # deterministic — the decoding-level guard against hallucination: the model
        # picks the single most-likely token instead of "creatively" inventing a
        # plausible digit, and repeats the exact same JSON on every call.
        "options": {
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "top_k": LLM_TOP_K,
            "seed": LLM_SEED,
            "repeat_penalty": 1.0,
            "num_predict": LLM_NUM_PREDICT,
            "num_ctx": num_ctx,
        },
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
    "aadhaar": ["name", "dob", "gender", "aadhaar", "vid", "address", "father"],
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
        "CRITICAL — do NOT confuse these two different numbers:",
        "  * 'Enrolment No.' / 'Enrolment Number' is printed near the TOP as",
        "    'NNNN/NNNNN/NNNNN' (it contains slashes). This is NOT the Aadhaar",
        "    number. NEVER put it in the 'aadhaar' field — leave 'aadhaar' empty",
        "    rather than use the Enrolment No.",
        "  * The 'aadhaar' field is the 12-digit Aadhaar number near 'Your Aadhaar",
        "    No.' / the holder's photo. It is USUALLY printed in FULL as",
        "    'NNNN NNNN NNNN' — read and return all 12 digits exactly. ONLY if the",
        "    first 8 digits are actually hidden behind X/* masking, return the masked",
        "    form 'XXXX XXXX 1234' as shown. Never output 'X' for digits that ARE",
        "    printed, and never leave this empty when a 12-digit number is visible.",
        "VID = exactly 16 digits printed as four groups of four (labelled 'VID').",
        "  Read all 16 digits and return them; only return '' if no VID is printed.",
        "name = the person's name in English near the photo. father = the name",
        "  after an 'S/O' / 'C/O' / 'D/O' marker in the address; if there is no such",
        "  marker, return '' for father (do NOT repeat the holder's own name).",
        "DOB may be labelled 'DOB' or 'Year of Birth' (then only a year is present).",
        "Gender appears as MALE / FEMALE / पुरुष / महिला.",
        "address — THIS IS MANDATORY. Read the COMPLETE postal address block and",
        "  return it as ONE single line (join the printed lines with ', '). It",
        "  almost ALWAYS begins with a relationship line — 'S/O' (son of), 'D/O'",
        "  (daughter of), 'W/O' (wife of) or 'C/O' (care of) followed by a name —",
        "  which you MUST include as the start of the address. Then read every",
        "  following line: house/area, VTC or village, PO, sub-district, district,",
        "  state, and the 6-digit PIN code, EXACTLY as printed. Do NOT skip the",
        "  first lines and do NOT start in the middle of the address.",
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

def _relation_from_address(address: str) -> Tuple[str, str]:
    """Parse an Aadhaar relation marker at the start of the address.

    Returns (kind, name):
      * kind == "father"  for S/O (son of), D/O (daughter of), C/O (care of)
      * kind == "husband" for W/O (wife of)
      * ("", "") when no relation marker is present.
    The name is taken up to the first comma or address keyword.
    """
    if not address:
        return "", ""
    m = re.search(r'\b([SCDW])\s*[/.]?\s*[O0]\b[:\-\s]+(.+)', address, re.I)
    if not m:
        return "", ""
    kind = "husband" if m.group(1).upper() == "W" else "father"
    rest = re.split(r',', m.group(2), 1)[0]
    stop = _ADDR_STOP.search(rest)
    if stop:
        rest = rest[:stop.start()]
    name = " ".join(rest.split()).strip(" ,-")
    return (kind, name) if name else ("", "")


def _father_from_address(address: str) -> str:
    """Back-compat shim: return the relation name only when it is a father-type
    marker (S/O, D/O, C/O). Used by the non-Aadhaar document fallbacks."""
    kind, name = _relation_from_address(address)
    return name if kind == "father" else ""


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
        "- It is BETTER to return an empty string than a guessed or partial value.",
        "  Never fill a field with a value taken from a DIFFERENT field/label.",
        "- Do not invent, complete, or 'fix' values. Return only what you can read.",
        "- Numbers (Aadhaar, VID, PAN, EPIC, passport, DL) must be transcribed digit-for-digit.",
        "  Do not guess digits you cannot see. For a MASKED number, return the visible",
        "  digits together with the printed mask exactly (e.g. 'XXXX XXXX 1234') — that",
        "  is a complete, valid answer, not an uncertain one.",
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


# ---------------------------------------------------------------------------
# Per-field validators (anti-hallucination guard)
# ---------------------------------------------------------------------------
# A vision model on a low-res ID will sometimes return a confident-but-WRONG
# value — most dangerously it grabs the "Enrolment No." (NNNN/NNNNN/NNNNN) and
# returns it as the Aadhaar number. Every field below has a deterministic,
# checkable format, so we VALIDATE the model's answer and DROP anything that
# doesn't fit. Dropping a field (-> empty in the response) is always safer than
# emitting a wrong identifier. Each validator returns the cleaned/normalised
# string, or "" to signal "reject — do not append".

def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _clean_aadhaar_number(val: str) -> str:
    """Accept ONLY a real Aadhaar number; reject Enrolment No. / wrong lengths.

    Valid forms:
      * Fully printed 12 digits that PASS the Verhoeff checksum -> "NNNN NNNN NNNN"
      * Masked card form (8 hidden + last 4 visible)            -> "XXXX XXXX NNNN"
    Everything else (Enrolment 'NNNN/NNNNN/NNNNN', 14-digit blobs, partials,
    checksum failures) returns "" so it is never written to the response.
    """
    if not val:
        return ""
    s = val.strip()
    # Enrolment numbers contain a slash and are never the Aadhaar number.
    if "/" in s:
        return ""
    digs = _digits(s)
    mask_count = len(re.findall(r"[Xx*•]", s))  # X, x, *, • used for masking
    # Fully visible 12-digit Aadhaar -> must satisfy the Verhoeff check digit.
    if mask_count == 0 and len(digs) == 12:
        return f"{digs[0:4]} {digs[4:8]} {digs[8:12]}" if validate_verhoeff(digs) else ""
    # Masked print: exactly the last 4 digits visible, the first 8 hidden.
    if len(digs) == 4 and mask_count >= 6:
        return f"XXXX XXXX {digs}"
    return ""


def _clean_vid(val: str) -> str:
    """VID is EXACTLY 16 digits. Anything else is a misread -> reject."""
    digs = _digits(val)
    if len(digs) == 16:
        return f"{digs[0:4]} {digs[4:8]} {digs[8:12]} {digs[12:16]}"
    return ""


def _clean_dob(val: str) -> str:
    """Accept DD/MM/YYYY (with sane ranges) or a bare 4-digit year of birth."""
    if not val:
        return ""
    s = val.strip()
    m = re.search(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12 and 1900 <= y <= 2100:
            return f"{d:02d}/{mo:02d}/{y:04d}"
        return ""
    m2 = re.fullmatch(r"(\d{4})", s)
    if m2 and 1900 <= int(m2.group(1)) <= 2100:
        return m2.group(1)
    return ""


def _clean_pan(val: str) -> str:
    """PAN = 5 letters + 4 digits + 1 letter (ABCDE1234F). Else reject."""
    s = re.sub(r"\s", "", val or "").upper()
    return s if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", s) else ""


def _clean_gender(val: str) -> str:
    """Normalise gender to MALE / FEMALE / TRANSGENDER (handles Hindi + 'M'/'F').
    Anything unrecognised is rejected so no garbage lands in the field."""
    if not val:
        return ""
    s = val.strip().lower()
    if "female" in s or "महिला" in val or s in ("f", "स्त्री"):
        return "FEMALE"
    if "transgender" in s or "trans" in s or "किन्नर" in val:
        return "TRANSGENDER"
    if "male" in s or "पुरुष" in val or s == "m":
        return "MALE"
    return ""


def _clean_person_name(val: str) -> str:
    """Keep a plausible Latin-script person name; strip Devanagari/digits/noise.

    Does NOT hard-reject misreads (a name has no checksum) — it only removes
    obvious non-name characters so we never append digits or Hindi glyphs into
    a name field. Returns "" only when nothing name-like remains.
    """
    if not val:
        return ""
    s = extract_english_only(val)              # drop non-ASCII (Devanagari) glyphs
    # Strip relationship markers (S/O, C/O, D/O, W/O) — they are not part of the
    # name. Without this, a model that returns "S/O" leaks "S O" into the field.
    s = re.sub(r"(?i)\b[scdw]\s*[/.]?\s*o\b", " ", s)
    s = re.sub(r"[^A-Za-z .'\-]", " ", s)       # keep letters + name punctuation
    words = [w for w in s.split() if w]
    # A real name has at least one word of 2+ letters; this rejects stray
    # single-letter noise like "S O" while still allowing initials ("A Kumar").
    if not any(len(w) >= 2 for w in words):
        return ""
    return " ".join(words)


# Map response key -> the validator that must approve its value.
_FIELD_VALIDATORS = {
    "aadhaar_number": _clean_aadhaar_number,
    "vid": _clean_vid,
    "dob": _clean_dob,
    "pan_number": _clean_pan,
    "gender": _clean_gender,
}
_NAME_KEYS = {"name", "father_name", "mother_name", "husband_name", "spouse_name"}


# A focused, single-purpose prompt. The big multi-field Aadhaar prompt sometimes
# makes the model skip the large bold Aadhaar number (it reads the smaller VID
# right below it but not the number itself). Asking for ONLY the number/VID, with
# no other fields competing for attention, reliably recovers it.
_AADHAAR_NUM_PROMPT = (
    "This is an Indian Aadhaar card. Near the label 'Your Aadhaar No.' a large "
    "12-digit Aadhaar number is printed as 'NNNN NNNN NNNN' (occasionally masked "
    "as 'XXXX XXXX 1234'). Just below it a 16-digit 'VID' may be printed as four "
    "groups of four. Read them EXACTLY, digit for digit. Return ONLY this JSON: "
    '{"aadhaar": "<the 12-digit number>", "vid": "<the 16-digit VID>"}. '
    "If a value is genuinely not visible, return an empty string for it."
)


def _recover_aadhaar_numbers(img: Image.Image) -> Dict[str, str]:
    """Focused second call to read the Aadhaar number / VID that the big
    multi-field prompt sometimes skips. Returns a dict with validated
    'aadhaar_number' and/or 'vid' only (same anti-hallucination gates), or {}."""
    try:
        b64 = _image_to_b64(img, max_width=1400)
        r = _call_ollama(b64, _AADHAAR_NUM_PROMPT)
    except Exception as e:  # noqa: BLE001 - never break the request
        print(f"[DEBUG] Aadhaar number recovery error: {e}")
        return {}
    out: Dict[str, str] = {}
    if isinstance(r, dict):
        a = _clean_aadhaar_number(str(r.get("aadhaar") or ""))
        if a:
            out["aadhaar_number"] = a
        v = _clean_vid(str(r.get("vid") or ""))
        if v:
            out["vid"] = v
    return out


# Focused address recovery. The full postal address (incl. the leading S/O|D/O|
# W/O|C/O relation line) is occasionally skipped or truncated by the big prompt.
_ADDRESS_PROMPT = (
    "This is an Indian Aadhaar card. Read the holder's FULL postal address EXACTLY "
    "as printed, as ONE single line (join the printed lines with ', '). It almost "
    "always STARTS with a relationship line — 'S/O' (son of), 'D/O' (daughter of), "
    "'W/O' (wife of) or 'C/O' (care of) followed by a name — then house/area, VTC "
    "or village, PO, sub-district, district, state and the 6-digit PIN. Include "
    "EVERY part, especially that first relationship line; never start in the "
    'middle. Return ONLY this JSON: {"address": "<the full address>"}. If no '
    'address is printed at all, return "".'
)

# Focused relation/father recovery. Asks ONLY about the relation line, and is
# explicitly allowed to return empty — so a card with no S/O|D/O|W/O|C/O (where
# the holder genuinely has no parent/spouse printed) is not forced to fabricate.
_RELATION_PROMPT = (
    "This is an Indian Aadhaar card. Look ONLY for a relationship line in the "
    "address that starts with 'S/O', 'D/O', 'W/O' or 'C/O' followed by a person's "
    "name (a parent's or husband's name — NOT the cardholder's own name). Return "
    'ONLY this JSON: {"relation": "<S/O|D/O|W/O|C/O or empty>", "name": "<the name '
    'after it, or empty>"}. If there is genuinely NO such relationship line, return '
    "empty strings for both."
)


def _recover_address(img: Image.Image) -> str:
    """Focused call to read the full postal address. Returns a single-line address
    or "" — never raises."""
    try:
        b64 = _image_to_b64(img, max_width=1400)
        r = _call_ollama(b64, _ADDRESS_PROMPT)
    except Exception as e:  # noqa: BLE001 - never break the request
        print(f"[DEBUG] Aadhaar address recovery error: {e}")
        return ""
    if isinstance(r, dict) and isinstance(r.get("address"), str):
        return " ".join(r["address"].split())
    return ""


def _recover_relation(img: Image.Image) -> Tuple[str, str]:
    """Focused call to read the relation marker + name. Returns (kind, name) where
    kind is 'father' (S/O|D/O|C/O), 'husband' (W/O), or '' when none is printed."""
    try:
        b64 = _image_to_b64(img, max_width=1400)
        r = _call_ollama(b64, _RELATION_PROMPT)
    except Exception as e:  # noqa: BLE001 - never break the request
        print(f"[DEBUG] Aadhaar relation recovery error: {e}")
        return "", ""
    if not isinstance(r, dict):
        return "", ""
    rel = str(r.get("relation") or "").strip().upper()
    name = _clean_person_name(str(r.get("name") or ""))
    if not name:
        return "", ""
    kind = "husband" if rel.startswith("W") else "father" if rel[:1] in ("S", "D", "C") else ""
    return (kind, name) if kind else ("", "")


def _result_to_extracted(result: Dict, fields: List[str], doc_type: str) -> Dict[str, str]:
    """Map the model's JSON onto the keys create_uniform_response expects,
    validating each field's format and DROPPING anything that doesn't fit so no
    wrong/hallucinated value is ever written into the response."""
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
            if not val:
                continue
            key = _EXTRACTED_KEY.get(f, f)
            # Passport stores spouse name under spouse_name, not husband_name.
            if doc_type == "passport" and f == "husband":
                key = "spouse_name"

            # Format-checked fields: reject the whole value if it fails.
            validator = _FIELD_VALIDATORS.get(key)
            if validator is not None:
                val = validator(val)
            elif key in _NAME_KEYS:
                val = _clean_person_name(val)

            if val:
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

    # Attempt strategy + MERGE. Two e-Aadhaar layouts need different framing:
    #   * Two-column "letter": the right half is a dense bilingual INFORMATION
    #     panel — reading the LEFT column avoids the model drowning in that text.
    #   * Single-column "letter": the Aadhaar number, VID and full address are
    #     printed CENTERED, so a left-column crop slices straight through them.
    # We can't tell the two apart reliably from the aspect ratio alone, so for a
    # portrait Aadhaar we read BOTH the full image and the left column and MERGE:
    # for each field we keep the first VALIDATED, non-empty value (full image
    # first, because it sees the centered number band and the complete address).
    # The left-column pass then only fills fields the full pass missed. This is
    # why a single blind crop used to lose the Aadhaar number / VID / DOB.
    attempts = [(img, LLM_IMAGE_WIDTH)]
    if doc_type == "aadhaar" and img.height > img.width * 1.15:
        w, h = img.size
        left = img.crop((0, 0, int(w * 0.55), h))
        attempts = [(img, 1400), (left, 1400)]

    # All high-value fields we'd like before we can stop early.
    _core = ("name", "dob", "aadhaar_number", "vid", "address")

    result: Optional[Dict] = None
    extracted: Dict[str, str] = {}
    for im, width in attempts:
        r = _query_model(im, doc_type, width)
        if r is None:
            continue
        if result is None:
            result = r  # keep the first usable raw dict for document_type detection
        part = _result_to_extracted(r, fields, doc_type)
        for k, v in part.items():
            # First validated, non-empty value wins (earlier attempt = higher trust).
            if v and not extracted.get(k):
                extracted[k] = v
        # Stop once every core field is in hand — no need to run the extra pass.
        if all(extracted.get(k) for k in _core):
            break

    if not extracted:
        return {}, None

    # The large bold Aadhaar number is the one field the multi-field prompt most
    # often skips. If it's still missing, make ONE focused call that asks only for
    # the number (and VID) — cheap, fires only on a miss, and validated the same way.
    if doc_type == "aadhaar" and not extracted.get("aadhaar_number"):
        for k, v in _recover_aadhaar_numbers(img).items():
            extracted.setdefault(k, v)

    # Address is MANDATORY on an Aadhaar. If the main passes missed it (e.g. the
    # model started in the middle or skipped the block), re-read it with a focused
    # address-only prompt before we derive the father from it.
    if doc_type == "aadhaar" and not extracted.get("address"):
        addr = _recover_address(img)
        if addr:
            extracted["address"] = addr

    # Father's name. Indian IDs (esp. Aadhaar) do NOT print a dedicated "Father's
    # Name" field, so the vision model often leaves it empty or — worse — just
    # echoes the holder's OWN name back. Resolve it deterministically:
    #   1) Drop a model value that merely repeats the holder's name (a misread).
    #   2) Otherwise take it from the "S/O <name>" (son of) / "C/O" / "D/O" / "W/O"
    #      marker in the address (W/O routes to husband, not father).
    #   3) If still nothing, make ONE focused relation-only call (which is allowed
    #      to return empty, so a card with no parent/spouse is not forced to guess).
    if "father" in fields:
        if doc_type == "aadhaar":
            # An Aadhaar card has NO dedicated father field, so the only reliable
            # source is the relation marker in the address. Ignore whatever the
            # model put in "father" (it tends to echo the holder's own surname).
            kind, rel_name = _relation_from_address(extracted.get("address", ""))
            if not kind:
                # Focused fallback — the address may have been truncated before the
                # relation line, or it's a W/O the inline read dropped.
                kind, rel_name = _recover_relation(img)
            if kind == "husband":
                extracted["husband_name"] = rel_name
                father = ""
            else:
                father = rel_name if kind == "father" else ""
        else:
            name_norm = (extracted.get("name") or "").strip().lower()
            father = (extracted.get("father_name") or "").strip()
            # Drop a value that merely repeats (or is contained in) the holder's
            # own name — that's a misread, not the father.
            if father and (father.lower() == name_norm or father.lower() in name_norm):
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
