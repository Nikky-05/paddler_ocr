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
import base64
import os
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from PIL import Image
from werkzeug.utils import secure_filename
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import fitz  # PyMuPDF for PDF processing

# OCR Engine State — the "OCR engine" is now the Ollama vision model. OCR_STATUS
# reflects whether Ollama is reachable and the configured vision model is pulled.
OCR_STATUS = "not_loaded"  # "not_loaded", "loading", "ready", "failed"
OCR_ERROR = None

OLLAMA_HOST = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL") or "qwen2.5vl:7b"

# ============ Face Verification Setup ============
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "jfif", "webp", "pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Face detection device and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


async def load_ocr_model():
    """Verify the Ollama vision model is available at startup.

    There is no local OCR model to load anymore; extraction happens by calling
    the Ollama vision model per request. We only check connectivity here so the
    health endpoint can report a meaningful status. A failed check does not stop
    the server — Ollama may come up after the API.
    """
    global OCR_STATUS, OCR_ERROR

    OCR_STATUS = "loading"
    OCR_ERROR = None

    try:
        loop = asyncio.get_event_loop()
        models = await loop.run_in_executor(None, _list_ollama_models)
        base = OLLAMA_VISION_MODEL.split(":")[0]
        if any(m == OLLAMA_VISION_MODEL or m.split(":")[0] == base for m in models):
            OCR_STATUS = "ready"
            print(f"OCR ready: Ollama vision model '{OLLAMA_VISION_MODEL}' available")
        else:
            OCR_STATUS = "failed"
            OCR_ERROR = (f"Vision model '{OLLAMA_VISION_MODEL}' not found in Ollama. "
                         f"Pull it with: ollama pull {OLLAMA_VISION_MODEL}")
            print(f"[WARN] {OCR_ERROR}")
    except Exception as e:
        OCR_STATUS = "failed"
        OCR_ERROR = f"Cannot reach Ollama at {OLLAMA_HOST}: {e}"
        print(f"[WARN] {OCR_ERROR}")


def _list_ollama_models() -> List[str]:
    """Return the list of model names installed in Ollama (raises on failure)."""
    resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m.get("name", "") for m in resp.json().get("models", [])]


# ============ Response Schema ============

class FieldValue(BaseModel):
    value: str
    confidence: float


class AddressSplit(BaseModel):
    building: str = ""
    city: str = ""
    district: str = ""
    pin: str = ""
    floor: str = ""
    house: str = ""
    locality: str = ""
    state: str = ""
    street: str = ""
    complex: str = ""
    landmark: str = ""
    untagged: str = ""


class AdditionalDetails(BaseModel):
    faceDetected: bool = False
    faceImage: str = ""
    addressSplit: AddressSplit = AddressSplit()
    llmFallbackUsed: bool = False
    llmFields: List[str] = []


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
    additionalDetails: Optional[AdditionalDetails] = None


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

# split_address turns the extracted address string into structured components
# for additionalDetails.addressSplit (unchanged response shape).
from extractors import split_address

# Ollama vision model is the sole OCR/extraction engine. extract_document builds
# a per-document-type prompt, sends the image to qwen, and returns the fields.
# classify_document independently identifies the document type (neutral prompt)
# for strict type validation.
from extractors.ollama_ocr import extract_document, classify_document


def detect_face_in_document(img: Image.Image) -> dict:
    """Detect a face in a document image and return base64-encoded face crop.

    Uses the global ``mtcnn`` instance for detection only (no embedding).
    Returns {"faceDetected": bool, "faceImage": str} where faceImage is a
    base64-encoded JPEG of the cropped face region, or "" if no face found.
    Never raises - failures are treated as no-face-detected.
    """
    try:
        boxes, _ = mtcnn.detect(img)
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]  # first face
            x1, y1, x2, y2 = [int(coord) for coord in box]
            # Add 10px margin, clamped to image bounds
            w, h = img.size
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(w, x2 + 10)
            y2 = min(h, y2 + 10)
            face_crop = img.crop((x1, y1, x2, y2))
            buf = BytesIO()
            face_crop.save(buf, format="JPEG")
            face_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {"faceDetected": True, "faceImage": face_b64}
    except Exception as e:
        print(f"[DEBUG] Face detection error: {e}")
    return {"faceDetected": False, "faceImage": ""}


# ============ Main Processing ============

DOC_TYPE_MAP = {
    'A': 'passport',
    'B': 'voter_id',
    'C': 'pan',
    'D': 'driving_license',
    'E': 'aadhaar'
}

# Human-friendly names for error messages, keyed by internal doc_type.
DOC_TYPE_NAME = {
    'passport': 'Passport',
    'voter_id': 'Voter ID',
    'pan': 'PAN Card',
    'driving_license': 'Driving License',
    'aadhaar': 'Aadhaar',
}


async def process_document(file: UploadFile, doc_type_code: str) -> Dict:
    """Process uploaded document via the Ollama vision model and return the
    uniform response.

    The vision model (qwen) is the sole OCR/extraction engine. PDFs are rendered
    to an image (first page) and sent to the model exactly like image uploads.
    Works for all five document types: Aadhaar, PAN, Passport, Voter ID, DL.
    """
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
        # Render the first page of the PDF to an image and feed it to the model.
        try:
            images = pdf_to_images(content)
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

    loop = asyncio.get_event_loop()

    # STRICT type validation FIRST. Independently classify the document with a
    # neutral prompt that is NOT told which type was requested — otherwise the
    # small vision model just echoes back the expected type. Reject early (before
    # the expensive field extraction) unless the detected type EXACTLY matches the
    # requested one:
    #   - detected == requested type   -> accept, continue to extraction
    #   - detected is a different type -> reject (wrong document)
    #   - detected is "other"          -> reject (not one of the 5 known types)
    #   - detected is None (unreadable
    #     / unclassifiable)            -> reject
    expected_name = DOC_TYPE_NAME.get(doc_type, doc_type)
    detected_type = await loop.run_in_executor(None, classify_document, img)
    if detected_type != doc_type:
        raise HTTPException(
            status_code=400,
            detail=f"Please upload a valid {expected_name}. The uploaded document does not match the selected document type ({expected_name})."
        )

    # Type confirmed — read all fields with the right per-document-type prompt.
    extracted, _ = await loop.run_in_executor(
        None, extract_document, img, doc_type
    )

    # Build the uniform response (records empty → default confidence of 1.0).
    response = create_uniform_response(doc_type, extracted, [])

    # additionalDetails: face crop, structured address, and which fields the
    # vision model filled (kept for response-shape compatibility).
    filled_fields = [k for k, v in response.items()
                     if isinstance(v, dict) and v.get("value")]
    face_info = detect_face_in_document(img)
    address_split = split_address(response["address"]["value"])
    response["additionalDetails"] = {
        "faceDetected": face_info["faceDetected"],
        "faceImage": face_info["faceImage"],
        "addressSplit": address_split,
        "llmFallbackUsed": bool(extracted),
        "llmFields": filled_fields,
    }

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
    - D: Driving License1
    - E: Aadhaar (auto-detects KYC Report vs raw card)
    
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