"""
FastAPI OCR Application for Indian Identity Documents
Single-file implementation with uniform response schema

Supported Formats:
  - Images: PNG, JPG, JPEG, JFIF, WEBP
  - Documents: PDF (first page only)

Document Types:
  - A: Passport
  - B: Voter ID
  - C: PAN Card
  - D: Driving License
  - E: Aadhaar

Usage:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import os
import re
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from PIL import Image, ImageEnhance, ImageFilter
from werkzeug.utils import secure_filename
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import fitz  # PyMuPDF for PDF processing

# OCR Engine State
OCR = None
OCR_STATUS = "not_loaded"  # "not_loaded", "loading", "ready", "failed"
OCR_ERROR = None

# ============ Face Verification Setup ============
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "jfif", "webp", "pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Face detection device and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


async def load_ocr_model():
    """Load OCR model asynchronously at startup."""
    global OCR, OCR_STATUS, OCR_ERROR

    OCR_STATUS = "loading"
    OCR_ERROR = None

    try:
        # Run blocking OCR initialization in thread pool
        loop = asyncio.get_event_loop()
        OCR = await loop.run_in_executor(None, _init_ocr)
        OCR_STATUS = "ready"
        print("OCR model loaded successfully")
    except Exception as e:
        OCR_STATUS = "failed"
        OCR_ERROR = str(e)
        print(f"Failed to load OCR model: {e}")


def _init_ocr():
    """
    Initialize PaddleOCR with optimized parameters for small text detection.
    Enhanced for detecting DOB and other small text in PVC/Short Aadhaar cards.
    """
    from paddleocr import PaddleOCR
    
    # Using only core parameters that are universally supported
    return PaddleOCR(
        lang='en',
        det_db_thresh=0.2,                 # Lower threshold = more sensitive text detection
        det_db_box_thresh=0.4,             # Accept lower confidence text boxes
        use_angle_cls=True                 # Detect rotated text
    )


def get_ocr():
    """Get OCR instance. Raises if not ready."""
    global OCR, OCR_STATUS, OCR_ERROR

    if OCR_STATUS == "ready" and OCR is not None:
        return OCR
    elif OCR_STATUS == "loading":
        raise HTTPException(status_code=503, detail="OCR model is still loading. Please wait.")
    elif OCR_STATUS == "failed":
        raise HTTPException(status_code=503, detail=f"OCR model failed to load: {OCR_ERROR}")
    else:
        raise HTTPException(status_code=503, detail="OCR model not initialized.")


# ============ Response Schema ============

class FieldValue(BaseModel):
    value: str
    confidence: float


class OCRResponse(BaseModel):
    doc_type: str
    aadhaar: FieldValue
    dob: FieldValue
    father: FieldValue
    gender: FieldValue
    husband: FieldValue
    mother: FieldValue
    name: FieldValue
    vid: FieldValue
    pan: FieldValue
    address: FieldValue
    dl_number: FieldValue
    epic: FieldValue
    passport: FieldValue
    nationality: FieldValue
    place_of_birth: FieldValue
    validity: FieldValue
    issue_date: FieldValue
    blood_group: FieldValue
    cov: FieldValue


def empty_field() -> Dict:
    return {"value": "", "confidence": 0}


def field_with_value(value: str, confidence: float = 1.0) -> Dict:
    return {"value": value, "confidence": confidence}


def create_uniform_response(doc_type: str, extracted: Dict, records: List[Dict]) -> Dict:
    """Create uniform response with all fields, populating based on doc_type."""

    # Build confidence map from OCR records
    conf_map = {}
    for rec in records:
        text = rec.get('text', '').strip()
        conf = rec.get('conf', 0.0)
        if text:
            conf_map[text.lower()] = conf

    def get_confidence(value: str) -> float:
        if not value:
            return 0
        # Try exact match first
        if value.lower() in conf_map:
            return round(conf_map[value.lower()], 2)
        # Try partial match
        for key, conf in conf_map.items():
            if value.lower() in key or key in value.lower():
                return round(conf, 2)
        return 1.0  # Default confidence for regex-extracted values

    response = {
        "doc_type": doc_type,
        "aadhaar": empty_field(),
        "dob": empty_field(),
        "father": empty_field(),
        "gender": empty_field(),
        "husband": empty_field(),
        "mother": empty_field(),
        "name": empty_field(),
        "vid": empty_field(),
        "pan": empty_field(),
        "address": empty_field(),
        "dl_number": empty_field(),
        "epic": empty_field(),
        "passport": empty_field(),
        "nationality": empty_field(),
        "place_of_birth": empty_field(),
        "validity": empty_field(),
        "issue_date": empty_field(),
        "blood_group": empty_field(),
        "cov": empty_field()
    }

    # Common fields
    if extracted.get('name'):
        response['name'] = field_with_value(extracted['name'], get_confidence(extracted['name']))
    if extracted.get('dob'):
        response['dob'] = field_with_value(extracted['dob'], get_confidence(extracted['dob']))
    if extracted.get('address'):
        response['address'] = field_with_value(extracted['address'], get_confidence(extracted['address']))
    if extracted.get('gender'):
        response['gender'] = field_with_value(extracted['gender'], get_confidence(extracted['gender']))

    # Document-specific fields
    if doc_type == 'aadhaar':
        if extracted.get('aadhaar_number'):
            response['aadhaar'] = field_with_value(extracted['aadhaar_number'], get_confidence(extracted['aadhaar_number']))
        if extracted.get('vid'):
            response['vid'] = field_with_value(extracted['vid'], get_confidence(extracted['vid']))
        if extracted.get('father_name'):
            response['father'] = field_with_value(extracted['father_name'], get_confidence(extracted['father_name']))
        if extracted.get('mother_name'):
            response['mother'] = field_with_value(extracted['mother_name'], get_confidence(extracted['mother_name']))
        if extracted.get('husband_name'):
            response['husband'] = field_with_value(extracted['husband_name'], get_confidence(extracted['husband_name']))
        if extracted.get('nationality'):
            response['nationality'] = field_with_value(extracted['nationality'], get_confidence(extracted['nationality']))

    elif doc_type == 'pan':
        if extracted.get('pan_number'):
            response['pan'] = field_with_value(extracted['pan_number'], get_confidence(extracted['pan_number']))
        if extracted.get('father_name'):
            response['father'] = field_with_value(extracted['father_name'], get_confidence(extracted['father_name']))

    elif doc_type == 'driving_license':
        if extracted.get('dl_number'):
            response['dl_number'] = field_with_value(extracted['dl_number'], get_confidence(extracted['dl_number']))
        if extracted.get('validity'):
            response['validity'] = field_with_value(extracted['validity'], get_confidence(extracted['validity']))
        if extracted.get('issue_date'):
            response['issue_date'] = field_with_value(extracted['issue_date'], get_confidence(extracted['issue_date']))
        if extracted.get('blood_group'):
            response['blood_group'] = field_with_value(extracted['blood_group'], get_confidence(extracted['blood_group']))
        if extracted.get('cov'):
            response['cov'] = field_with_value(extracted['cov'], get_confidence(extracted['cov']))
        if extracted.get('father_name'):
            response['father'] = field_with_value(extracted['father_name'], get_confidence(extracted['father_name']))

    elif doc_type == 'voter_id':
        if extracted.get('epic_number'):
            response['epic'] = field_with_value(extracted['epic_number'], get_confidence(extracted['epic_number']))
        if extracted.get('father_name'):
            response['father'] = field_with_value(extracted['father_name'], get_confidence(extracted['father_name']))

    elif doc_type == 'passport':
        if extracted.get('passport_number'):
            response['passport'] = field_with_value(extracted['passport_number'], get_confidence(extracted['passport_number']))
        if extracted.get('nationality'):
            response['nationality'] = field_with_value(extracted['nationality'], get_confidence(extracted['nationality']))
        if extracted.get('place_of_birth'):
            response['place_of_birth'] = field_with_value(extracted['place_of_birth'], get_confidence(extracted['place_of_birth']))
        if extracted.get('father_name'):
            response['father'] = field_with_value(extracted['father_name'], get_confidence(extracted['father_name']))
        if extracted.get('mother_name'):
            response['mother'] = field_with_value(extracted['mother_name'], get_confidence(extracted['mother_name']))
        if extracted.get('spouse_name'):
            response['husband'] = field_with_value(extracted['spouse_name'], get_confidence(extracted['spouse_name']))

    return response


# ============ OCR Processing ============

def is_english_text(text: str) -> bool:
    """Check if text contains useful ASCII content (letters or numbers)."""
    if not text:
        return False
    
    # Remove spaces for accurate check
    t_clean = text.replace(' ', '')
    if not t_clean:
        return True
    
    # Count ASCII letters AND digits (both are useful for extraction)
    ascii_alphanum_count = sum(1 for c in t_clean if c.isascii() and (c.isalpha() or c.isdigit()))
    
    # Allow if it has ANY ASCII alphanumeric content
    # This keeps: numbers (Aadhaar/VID), English text, mixed Hindi+English lines
    if ascii_alphanum_count >= 3:  # At least 3 ASCII alphanumeric chars
        return True
    
    # Also allow if ratio of ASCII to total is reasonable (>20%)
    ratio = ascii_alphanum_count / len(t_clean) if t_clean else 0
    return ratio >= 0.2


def ocr_records_from_image(img: Image.Image, doc_type: str = None) -> List[Dict]:
    """Run OCR on PIL Image and return list of records with text, confidence, and y-position.
    
    Args:
        img: PIL Image to process
        doc_type: Document type ('aadhaar', 'pan', 'driving_license', 'voter_id', 'passport')
                  If 'aadhaar', preserves Devanagari text. Otherwise, filters to English only.
    """
    ocr = get_ocr()

    def parse_result(res) -> List[Dict]:
        """Parse OCR result from PaddleOCR predict() method."""
        recs = []
        if not res or not isinstance(res, list) or len(res) == 0:
            return recs

        # PaddleOCR 3.3.2 returns list of OCRResult objects
        # Each has: rec_texts, rec_scores, rec_polys
        page = res[0]

        # Handle OCRResult object (has dict-like access)
        try:
            texts = page.get('rec_texts', []) if hasattr(page, 'get') else getattr(page, 'rec_texts', [])
            scores = page.get('rec_scores', []) if hasattr(page, 'get') else getattr(page, 'rec_scores', [])
            polys = page.get('rec_polys', []) if hasattr(page, 'get') else getattr(page, 'rec_polys', [])
        except Exception:
            return recs

        for idx, txt in enumerate(texts):
            if not txt or not str(txt).strip():
                continue
            # For Aadhaar: Keep ALL text (including Devanagari) for name extraction
            # For other docs: Filter to English only (existing behavior)
            if doc_type != 'aadhaar' and not is_english_text(str(txt)):
                continue
            conf = float(scores[idx]) if idx < len(scores) else 0.0
            # Get y-position and x-position from polygon
            poly = polys[idx] if idx < len(polys) else None
            y = 0
            x = 0
            if poly is not None:
                try:
                    y = int(min([p[1] for p in poly]))
                    x = int(min([p[0] for p in poly]))
                except Exception:
                    y = 0
                    x = 0
            recs.append({"text": str(txt).strip(), "conf": conf, "y": y, "x": x})

        return recs

    results = []

    # Pass 1: Original image (resized if too large)
    try:
        img_pass1 = img.copy()
        w, h = img_pass1.size
        if w > 1600:
            new_h = int(h * (1600 / w))
            img_pass1 = img_pass1.resize((1600, new_h), Image.LANCZOS)

        # Use predict() method (new API)
        res1 = ocr.predict(np.array(img_pass1))
        parsed1 = parse_result(res1)
        results.extend(parsed1)

        # Early exit if good enough
        if parsed1:
            avg_conf = sum(r['conf'] for r in parsed1) / len(parsed1)
            full_text = " ".join([r['text'] for r in parsed1]).upper()
            keywords = ['INCOME TAX', 'PERMANENT ACCOUNT', 'AADHAAR', 'GOVERNMENT OF INDIA',
                       'DRIVING LICEN', 'ELECTION COMMISSION', 'PASSPORT', 'REPUBLIC OF INDIA']
            if any(k in full_text for k in keywords) and avg_conf > 0.80:
                return merge_results(results)
    except Exception:
        pass

    # Pass 2: Preprocessed image
    try:
        img_pass2 = img.copy()
        w, h = img_pass2.size
        target_w = max(1200, w)
        if w < target_w:
            new_h = int(h * (target_w / w))
            img_pass2 = img_pass2.resize((target_w, new_h), Image.LANCZOS)

        enhancer = ImageEnhance.Contrast(img_pass2)
        img_pass2 = enhancer.enhance(1.6)
        img_pass2 = img_pass2.filter(ImageFilter.SHARPEN)

        # Use predict() method (new API)
        res2 = ocr.predict(np.array(img_pass2))
        results.extend(parse_result(res2))
    except Exception:
        pass

    return merge_results(results)


def merge_results(results: List[Dict]) -> List[Dict]:
    """Merge OCR results by y-position, keeping all texts on same line."""
    if not results:
        return []

    # Group by y-position bucket
    buckets = {}
    for r in results:
        b = int(round(r['y'] / 5.0) * 5)
        if b not in buckets:
            buckets[b] = []
        buckets[b].append(r)

    # For each bucket, sort by x-position and merge texts on same line
    ordered = []
    for b in sorted(buckets.keys()):
        items = buckets[b]
        # Sort by x-position (left to right)
        items_sorted = sorted(items, key=lambda x: x.get('x', 0))
        
        # Merge items on same line
        merged_text = ' '.join([item['text'] for item in items_sorted])
        avg_conf = sum(item['conf'] for item in items_sorted) / len(items_sorted)
        
        ordered.append({
            'text': merged_text,
            'conf': avg_conf,
            'y': b
        })

    # Merge tiny fragments
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


# ============ Face Verification Functions ============

def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def get_face_embedding(img_path: str) -> np.ndarray:
    """
    Extract normalized face embedding from image.
    
    Args:
        img_path: Path to image file
        
    Returns:
        Normalized embedding as 1D numpy array
        
    Raises:
        ValueError: If no face detected in image
    """
    img = Image.open(img_path).convert('RGB')
    # MTCNN returns a tensor crop (3x160x160) or None
    face = mtcnn(img)
    if face is None:
        raise ValueError("No face detected")
    
    # Move to device and add batch dimension
    face = face.unsqueeze(0).to(device)  # shape (1,3,160,160)
    
    with torch.no_grad():
        emb = resnet(face)  # (1,512)
    
    emb = emb.squeeze(0).cpu().numpy()
    # L2-normalize embedding (important for cosine similarity)
    emb = emb / np.linalg.norm(emb)
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    """
    Convert PDF bytes to list of PIL Images (one per page).
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        List of PIL Image objects, one for each page
        
    Raises:
        ValueError: If PDF is invalid or cannot be processed
    """
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        images = []
        
        # Convert each page to image
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # Render page to image at 300 DPI for better OCR quality
            mat = fitz.Matrix(300/72, 300/72)  # 300 DPI scaling
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_bytes = pix.tobytes("png")
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
            images.append(img)
        
        pdf_document.close()
        
        if not images:
            raise ValueError("PDF contains no pages")
            
        return images
        
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {str(e)}")


# ============ Document Extractors ============
# Import extractors from separate modules

from extractors import (
    extract_aadhaar,
    extract_driving_license,
    extract_pan,
    extract_voter,
    extract_passport
)

# Import document validator
from extractors.document_validator import validate_document_type

# Legacy function for compatibility (now imported from extractors.utils)
from extractors.utils import nearest_line


# ============ Main Processing ============

DOC_TYPE_MAP = {
    'A': 'passport',
    'B': 'voter_id',
    'C': 'pan',
    'D': 'driving_license',
    'E': 'aadhaar'
}


async def process_document(file: UploadFile, doc_type_code: str) -> Dict:
    """Process uploaded document and return uniform response."""
    doc_type = DOC_TYPE_MAP.get(doc_type_code.upper())
    if not doc_type:
        raise HTTPException(status_code=400, detail=f"Invalid document type code: {doc_type_code}. Valid codes: A (Passport), B (Voter ID), C (PAN), D (Driving License), E (Aadhaar)")

    # Read file into memory buffer (no disk storage)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    # Check if file is PDF or image
    file_ext = file.filename.lower().split('.')[-1] if file.filename else ''
    
    if file_ext == 'pdf':
        # Process PDF: convert to images and process first page
        try:
            images = pdf_to_images(content)
            # Process first page (most PDFs of ID documents are single page)
            img = images[0]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    else:
        # Process as image
        try:
            img = Image.open(BytesIO(content)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file or format")
    
    # Run OCR on original image (no preprocessing)
    records = ocr_records_from_image(img, doc_type)
    lines = [r['text'] for r in records]
    text = "\n".join(lines)
    
    # Validate document type
    is_valid_document = validate_document_type(doc_type, lines, text)
    
    if not is_valid_document:
        raise HTTPException(
            status_code=400, 
            detail="Invalid Document"
        )
    
    # Extract based on document type
    if doc_type == 'aadhaar':
        extracted = extract_aadhaar(lines, text, records)
    elif doc_type == 'pan':
        extracted = extract_pan(lines, text)
    elif doc_type == 'driving_license':
        extracted = extract_driving_license(lines, text)
    elif doc_type == 'voter_id':
        extracted = extract_voter(lines, text)
    elif doc_type == 'passport':
        extracted = extract_passport(lines, text)
    else:
        extracted = {}

    # Create uniform response
    response = create_uniform_response(doc_type, extracted, records)
    return response


# ============ FastAPI App ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Load OCR model
    print("Starting OCR model loading...")
    await load_ocr_model()
    yield
    # Shutdown: Cleanup if needed
    print("Shutting down...")


app = FastAPI(
    title="OCR & Face Verification API",
    description="OCR API for Indian Identity Documents with Face Verification",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    ocr_status: str
    ocr_error: Optional[str] = None


class VerifyResponse(BaseModel):
    success: bool
    match: bool
    threshold: float
    percent_similarity: float


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_code: str
    data: Optional[Dict] = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Override default 422 validation error."""
    # Build a friendly message from the errors
    errors = exc.errors()
    if errors:
        err = errors[0]
        field = err.get("loc", ["Unknown"])[-1]
        msg = f"Field '{field}' is {err.get('msg', 'invalid')}"
        # Special case for "field required"
        if err.get("type") == "missing":
            msg = f"{field.replace('_', ' ').capitalize()} is required"
    else:
        msg = "Validation error"

    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            success=False,
            message=msg,
            error_code="Validation_Error"
        ).model_dump()
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Override default HTTPException handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            message=str(exc.detail),
            error_code="Service_Error"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all for internal server errors."""
    print(f"INTERNAL ERROR: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            message="Internal Server Error",
            error_code="SERVER_ERROR"
        ).model_dump()
    )


@app.get("/")
async def root():
    """
    API Information and Documentation.
    """
    return {
        "message": "OCR & Face Verification API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "health": "/api/health",
            "ocr": "/api/ocr (POST)",
            "verify": "/verify (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "document_types": {
            "A": "Passport",
            "B": "Voter ID",
            "C": "PAN Card",
            "D": "Driving License",
            "E": "Aadhaar"
        },
        "supported_formats": ["png", "jpg", "jpeg", "jfif", "webp", "pdf"]
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with OCR status.

    OCR Status:
    - not_loaded: OCR model not yet initialized
    - loading: OCR model is currently loading
    - ready: OCR model loaded and ready to process
    - failed: OCR model failed to load (check ocr_error)
    """
    return {
        "status": "healthy" if OCR_STATUS == "ready" else "degraded",
        "ocr_status": OCR_STATUS,
        "ocr_error": OCR_ERROR
    }


@app.post("/api/ocr", response_model=OCRResponse)
async def ocr_extract(
    file: UploadFile = File(..., description="Document image or PDF file (png, jpg, jpeg, jfif, webp, pdf)"),
    doc_type: str = Form(..., description="Document type code: A (Passport), B (Voter ID), C (PAN), D (Driving License), E (Aadhaar)")
):
    """
    Extract data from identity document.

    Supported Formats: PNG, JPG, JPEG, JFIF, WEBP, PDF
    
    Document Type Codes:
    - A: Passport
    - B: Voter ID
    - C: PAN Card
    - D: Driving License
    - E: Aadhaar
    
    Note: For PDF files, only the first page will be processed.
    """
    result = await process_document(file, doc_type)
    return result


@app.post("/verify", response_model=VerifyResponse)
async def verify_faces(
    selfie: UploadFile = File(..., description="Selfie image file"),
    document: UploadFile = File(..., description="Document image with face"),
    threshold: float = Form(70.0, description="Similarity threshold (0.0 to 100.0)")
):
    """
    Verify if the face in selfie matches the face in document.
    
    Args:
        selfie: Selfie image file
        document: Document image containing a face (e.g., Aadhaar, Passport)
        threshold: Similarity threshold for match (default: 70.0)
        
    Returns:
        VerifyResponse with match result and similarity scores
    """
    # Validate files
    if not selfie or not document:
        raise HTTPException(status_code=400, detail="Two files required: selfie and document.")
    
    if not (allowed_file(selfie.filename) and allowed_file(document.filename)):
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: png, jpg, jpeg, jfif, webp")
    
    # Save uploaded files
    selfie_fn = secure_filename(selfie.filename)
    doc_fn = secure_filename(document.filename)
    selfie_path = os.path.join(UPLOAD_FOLDER, f"selfie_{selfie_fn}")
    doc_path = os.path.join(UPLOAD_FOLDER, f"doc_{doc_fn}")
    
    # Read and save files
    selfie_content = await selfie.read()
    doc_content = await document.read()
    
    with open(selfie_path, "wb") as f:
        f.write(selfie_content)
    with open(doc_path, "wb") as f:
        f.write(doc_content)
    
    # Extract embeddings
    try:
        emb1 = get_face_embedding(selfie_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Selfie error: {str(e)}")
    
    try:
        emb2 = get_face_embedding(doc_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Document error: {str(e)}")
    
    # Calculate similarity
    cos = cosine_similarity(emb1, emb2)  # -1 to 1
    # Map to percentage: (-1..1) -> (0..100)
    percent = (cos + 1.0) / 2.0 * 100.0
    
    # Compare percentage against threshold
    match = percent >= threshold
    
    return VerifyResponse(
        success=True,
        match=bool(match),
        threshold=threshold,
        percent_similarity=percent
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
