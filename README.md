# Indian KYC Document OCR API

A FastAPI service that extracts structured data from Indian identity documents
using a **local Ollama vision-language model**, with optional face verification.

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Supported Documents](#supported-documents)
- [Supported Formats](#supported-formats)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Document Validation](#document-validation)
- [Anti-Hallucination Guarantees](#anti-hallucination-guarantees)
- [Response Schema](#response-schema)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)

---

## ⚙️ How It Works

There is **no on-device OCR engine** (no PaddleOCR / Tesseract). Every uploaded
image is sent directly to a local **vision-language model served by Ollama**
(default `qwen2.5vl:7b`), which both *reads the pixels* and *returns the fields
as JSON*. The pipeline for one request:

1. **Validate type** — the image is independently classified with a *neutral*
   prompt (not told which type you requested) so the model can't just echo the
   expected type. If the detected type ≠ the requested type, the request is
   rejected early.
2. **Extract fields** — a per-document-type prompt asks the model for exactly the
   fields that document carries (name, dob, number, address, …).
3. **Validate & recover** — every format-checkable field is validated (and dropped
   if malformed); missing high-value fields (Aadhaar number, address, relation)
   trigger focused recovery calls. See [Anti-Hallucination Guarantees](#anti-hallucination-guarantees).
4. **Enrich** — face crop (MTCNN), structured address split, uniform JSON response.

---

## 📄 Supported Documents

| Code | Document Type   | Extracted Fields |
|------|-----------------|------------------|
| **A** | Passport        | Name, DOB, Passport No., Gender, Place of Birth, Nationality, Father/Mother/Spouse, Issue/Validity |
| **B** | Voter ID (EPIC) | Name, DOB, EPIC No., Gender, Father, Address |
| **C** | PAN Card        | Name, DOB, PAN No., Father |
| **D** | Driving License | Name, DOB, DL No., Address, Issue Date, Validity, Blood Group, COV, Father |
| **E** | Aadhaar Card    | Name, DOB, Gender, Aadhaar No., VID, Address, Father (S/O‑D/O‑W/O‑C/O), Nationality |

---

## 🖼️ Supported Formats

- **Images**: PNG, JPG, JPEG, JFIF, WEBP
- **Documents**: PDF (first page, rendered at 300 DPI)

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com) running locally
- The vision model pulled:
  ```bash
  ollama pull qwen2.5vl:7b
  ```

### Setup

```bash
# 1. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the API
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API available at `http://localhost:8000` (Swagger UI at `/docs`).

---

## 🔧 Configuration

All settings are environment variables with safe defaults (no code change needed):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Vision model used for extraction + classification |
| `LLM_KEEP_ALIVE` | `10m` | How long the model stays loaded in VRAM (set `24h` to avoid cold-start reloads) |
| `LLM_IMAGE_WIDTH` | `1100` | Max image width sent to the model (smaller = faster) |
| `LLM_NUM_PREDICT` | `2048` | Max tokens generated |
| `LLM_NUM_CTX` | `4096` | KV-cache context size |
| `LLM_TIMEOUT` | `150` | Per-request timeout (seconds) |
| `LLM_TEMPERATURE` | `0` | Randomness — `0` = deterministic (recommended for KYC) |
| `LLM_TOP_P` / `LLM_TOP_K` | `0.1` / `1` | Sampling cutoffs (near-greedy by default) |
| `LLM_SEED` | `0` | Fixed RNG seed for reproducible output |

> **Tip:** for production keep `LLM_TEMPERATURE=0` (lowest hallucination) and set
> `LLM_KEEP_ALIVE=24h` so the model never unloads.

---

## 📖 Usage

```bash
# OCR extraction
curl -X POST "http://localhost:8000/api/ocr" \
  -F "file=@aadhaar.jpg" \
  -F "doc_type=E"

# Face verification (selfie vs. document)
curl -X POST "http://localhost:8000/verify" \
  -F "selfie=@selfie.jpg" \
  -F "document=@aadhaar.jpg" \
  -F "threshold=70.0"
```

---

## 🔌 API Endpoints

### `GET /api/health`
Reports whether the Ollama vision model is reachable/available.
```json
{ "status": "healthy", "ocr_status": "ready", "ocr_error": null }
```
`ocr_status`: `not_loaded` → `loading` → `ready` / `failed`.

### `POST /api/ocr`
- `file` (required): image or PDF
- `doc_type` (required): one of `A`–`E` (see table above)

Returns the uniform response schema (below). Rejects with **400** if the image
doesn't match the requested document type.

### `POST /verify`
- `selfie` (required), `document` (required), `threshold` (optional, default 70)
```json
{ "success": true, "match": true, "threshold": 70.0, "percent_similarity": 85.5 }
```

---

## 🔍 Document Validation

Before extraction, the document is **independently classified** by the vision
model using a neutral prompt that is *not* told which type you selected — so the
model must actually read the document rather than agree with you. The request is
accepted only when the detected type exactly matches the requested one; otherwise:

```json
{ "success": false, "message": "Please upload a valid <type>. ...", "error_code": "Service_Error" }
```

---

## 🛡️ Anti-Hallucination Guarantees

For KYC, a *wrong* value is worse than a *missing* one. The pipeline enforces:

- **Format validation** — `aadhaar` (12-digit Verhoeff checksum **or** masked
  `XXXX XXXX NNNN`), `vid` (16 digits), `dob` (`DD/MM/YYYY`), `pan` (`ABCDE1234F`),
  `gender` — any value failing its format is **dropped**, never returned.
- **No field cross-contamination** — e.g. the Aadhaar *Enrolment No.* can never be
  returned as the Aadhaar number.
- **Deterministic decoding** — `temperature=0` + fixed seed, so the same image
  yields the same JSON every time.
- **Focused recovery** — when a high-value field (Aadhaar number / address /
  relation) is missed by the main read, a single focused follow-up call recovers
  it, still passing the same validation.

---

## 📊 Response Schema

Uniform across all document types — fields not present for a type come back empty:

```jsonc
{
  "doc_type": "aadhaar",
  "name":   { "value": "John Doe", "confidence": 1.0 },
  "dob":    { "value": "01/01/1990", "confidence": 1.0 },
  "gender": { "value": "MALE", "confidence": 1.0 },
  "address":{ "value": "S/O ..., City, State - 123456", "confidence": 1.0 },
  "aadhaar":{ "value": "XXXX XXXX 9012", "confidence": 1.0 },
  "vid":    { "value": "9101 4185 9371 8555", "confidence": 1.0 },
  "father": { "value": "Richard Doe", "confidence": 1.0 },
  // pan, dl_number, epic, passport, mother, husband, nationality,
  // place_of_birth, validity, issue_date, blood_group, cov ... (empty if N/A)
  "additionalDetails": {
    "faceDetected": true,
    "faceImage": "<base64 jpeg>",
    "addressSplit": { "building": "", "city": "", "district": "", "pin": "...", "state": "...", "...": "" }
  }
}
```

Each field is `{ "value": str, "confidence": float }` (empty string + `0` when not found).

---

## 📁 Project Structure

```
paddler_ocr/
├── main.py                      # FastAPI app: routes, validation, face verify, response
├── requirements.txt
├── extractors/
│   ├── __init__.py              # exposes split_address
│   ├── ollama_ocr.py            # the OCR/extraction engine (prompts, validators, recovery)
│   └── utils.py                 # validate_verhoeff, extract_english_only, split_address
└── README.md
```

---

## 🛠️ Technology Stack

- **API**: FastAPI + Uvicorn
- **OCR / extraction**: Ollama vision-language model (`qwen2.5vl:7b`) via HTTP
- **Face verification**: facenet-pytorch (MTCNN + InceptionResnetV1)
- **PDF**: PyMuPDF (fitz)
- **Imaging**: Pillow, NumPy

---

## 🔐 Security Notes

- Uploaded files are processed **in memory** (no persistent storage of documents).
- Configure CORS `allow_origins` for production.
- Consider file-size limits and rate limiting before exposing publicly.
