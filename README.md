# PaddleOCR - Indian KYC Document OCR API

A FastAPI-based OCR service for extracting information from Indian identity documents with face verification capabilities.

## 📋 Table of Contents

- [Features](#features)
- [Supported Documents](#supported-documents)
- [Supported Formats](#supported-formats)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Document Validation](#document-validation)
- [PDF Support](#pdf-support)
- [Performance Optimization](#performance-optimization)
- [Response Schema](#response-schema)
- [Error Handling](#error-handling)

---

## ✨ Features

- **Multi-Document OCR**: Extract data from 5 types of Indian identity documents
- **Document Validation**: Automatically validates if uploaded image matches selected document type
- **PDF Support**: Process PDF documents (first page only)
- **Face Verification**: Compare faces between selfie and document images
- **Uniform Response Schema**: Consistent JSON response across all document types
- **Async Processing**: Non-blocking OCR model loading at startup
- **CORS Enabled**: Ready for web application integration
- **Error Handling**: Comprehensive error messages and validation

---

## 📄 Supported Documents

| Code | Document Type      | Extracted Fields |
|------|--------------------|------------------|
| **A** | Passport           | Name, DOB, Passport Number, Nationality, Place of Birth, Father/Mother Name |
| **B** | Voter ID           | Name, DOB, EPIC Number, Father Name, Address, Gender |
| **C** | PAN Card           | Name, DOB, PAN Number, Father Name |
| **D** | Driving License    | Name, DOB, DL Number, Address, Blood Group, COV, Validity, Issue Date |
| **E** | Aadhaar Card       | Name, DOB, Aadhaar Number, VID, Gender, Address, Father/Mother/Husband Name, Nationality |

---

## 🖼️ Supported Formats

- **Images**: PNG, JPG, JPEG, JFIF, WEBP
- **Documents**: PDF (first page only, converted to image at 300 DPI)

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd paddler_ocr
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```
   
   Or using uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

The API will be available at: `http://localhost:8000`

---

## 📖 Usage

### Using cURL

```bash
# OCR Extraction
curl -X POST "http://localhost:8000/api/ocr" \
  -F "file=@aadhaar.jpg" \
  -F "doc_type=E"

# Face Verification
curl -X POST "http://localhost:8000/verify" \
  -F "selfie=@selfie.jpg" \
  -F "document=@aadhaar.jpg" \
  -F "threshold=70.0"
```

### Using Postman

1. **OCR Endpoint**
   - Method: `POST`
   - URL: `http://localhost:8000/api/ocr`
   - Body: `form-data`
     - `file`: Select image/PDF file
     - `doc_type`: Enter code (A/B/C/D/E)

2. **Face Verification Endpoint**
   - Method: `POST`
   - URL: `http://localhost:8000/verify`
   - Body: `form-data`
     - `selfie`: Select selfie image
     - `document`: Select document image
     - `threshold`: Enter threshold (0-100, default: 70)

---

## 🔌 API Endpoints

### 1. Health Check
```
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "ocr_status": "ready",
  "ocr_error": null
}
```

**OCR Status Values:**
- `not_loaded`: OCR model not initialized
- `loading`: OCR model is loading
- `ready`: OCR model ready to process
- `failed`: OCR model failed to load

---

### 2. OCR Extraction
```
POST /api/ocr
```

**Parameters:**
- `file` (required): Document image or PDF file
- `doc_type` (required): Document type code (A/B/C/D/E)

**Success Response (200):**
```json
{
  "doc_type": "aadhaar",
  "name": {"value": "John Doe", "confidence": 0.95},
  "dob": {"value": "01/01/1990", "confidence": 0.92},
  "aadhaar": {"value": "123456789012", "confidence": 0.98},
  "gender": {"value": "MALE", "confidence": 0.99},
  "address": {"value": "123 Street, City", "confidence": 0.85},
  "vid": {"value": "", "confidence": 0},
  "father": {"value": "Richard Doe", "confidence": 0.90},
  "mother": {"value": "", "confidence": 0},
  "husband": {"value": "", "confidence": 0},
  "pan": {"value": "", "confidence": 0},
  "dl_number": {"value": "", "confidence": 0},
  "epic": {"value": "", "confidence": 0},
  "passport": {"value": "", "confidence": 0},
  "nationality": {"value": "INDIAN", "confidence": 1.0},
  "place_of_birth": {"value": "", "confidence": 0},
  "validity": {"value": "", "confidence": 0},
  "issue_date": {"value": "", "confidence": 0},
  "blood_group": {"value": "", "confidence": 0},
  "cov": {"value": "", "confidence": 0}
}
```

**Error Response (400):**
```json
{
  "success": false,
  "message": "Invalid Document",
  "error_code": "Service_Error"
}
```

---

### 3. Face Verification
```
POST /verify
```

**Parameters:**
- `selfie` (required): Selfie image file
- `document` (required): Document image with face
- `threshold` (optional): Similarity threshold (0-100, default: 70)

**Success Response (200):**
```json
{
  "success": true,
  "match": true,
  "threshold": 70.0,
  "percent_similarity": 85.5
}
```

**Error Response (400):**
```json
{
  "success": false,
  "message": "Selfie error: No face detected",
  "error_code": "Service_Error"
}
```

---

## 🔍 Document Validation

The system validates whether uploaded images match the selected document type **before** processing OCR extraction.

### How Validation Works

Each document type is validated based on specific markers found in the OCR text:

#### **Aadhaar Card (E)**

✅ **Valid if:**
- Has keywords: `"AADHAAR"`, `"UIDAI"`, `"GOVERNMENT OF INDIA"`, `"भारत सरकार"`
- **AND** has 12-digit Aadhaar number OR 16-digit VID
- **OR** has Aadhaar number AND gender keyword

**Example Valid Text:**
```
GOVERNMENT OF INDIA
AADHAAR
1234 5678 9012
MALE
DOB: 01/01/1990
```

---

#### **PAN Card (C)**

✅ **Valid if:**
- Has PAN number format: `ABCDE1234F` (5 letters + 4 digits + 1 letter)
- **AND** has keywords: `"INCOME TAX"`, `"PERMANENT ACCOUNT NUMBER"`
- **OR** has PAN number AND `"GOVERNMENT OF INDIA"`

**Example Valid Text:**
```
INCOME TAX DEPARTMENT
PERMANENT ACCOUNT NUMBER
ABCDE1234F
```

---

#### **Driving License (D)**

✅ **Valid if:**
- Has keywords: `"DRIVING LICENCE"`, `"TRANSPORT"`, `"MOTOR VEHICLES"`
- **AND** has at least one of:
  - DL number pattern: `MH01 20110012345`
  - COV text: `"COV"`, `"CLASS OF VEHICLE"`
  - Validity text: `"VALIDITY"`, `"VALID TILL"`

**Example Valid Text:**
```
DRIVING LICENCE
DL Number: MH01 20110012345
COV: MCWG, LMV
Validity: 01/01/2030
```

---

#### **Voter ID (B)**

✅ **Valid if:**
- Has keywords: `"ELECTION COMMISSION"`, `"ELECTORAL"`, `"ELECTOR"`, `"VOTER"`, `"EPIC"`
- **OR** has EPIC number pattern (3 letters + 7 digits) AND `"ELECTION COMMISSION"`

**Example Valid Text:**
```
ELECTION COMMISSION OF INDIA
ELECTOR'S PHOTO IDENTITY CARD
EPIC No: ABC1234567
```

---

#### **Passport (A)**

✅ **Valid if:**
- Has keyword: `"PASSPORT"`
- **OR** has `"REPUBLIC OF INDIA"`
- **OR** has passport number (1 letter + 7 digits) AND `"NATIONALITY: INDIAN"`
- **OR** has MRZ pattern: `"P<IND"`

**Example Valid Text:**
```
REPUBLIC OF INDIA
PASSPORT
Passport No: P1234567
Nationality: INDIAN
```

---

### Validation Error

When validation fails, the API returns:

```json
{
  "success": false,
  "message": "Invalid Document",
  "error_code": "Service_Error"
}
```

**HTTP Status Code:** 400

**Common Scenarios:**
-  Random image uploaded
-  Wrong document type selected (e.g., PAN card uploaded with Aadhaar type)
-  Poor quality image with unreadable text
-  Non-KYC document uploaded

---

## 📑 PDF Support

The system supports PDF documents with the following specifications:

### Features
- Converts PDF to images at **300 DPI** for optimal OCR quality
- Processes **first page only** (most ID documents are single-page)
- Automatic format detection based on file extension
- Same validation and extraction logic as images

### Supported PDF Types
- Standard PDF documents
- Scanned PDF documents
- PDF exports from mobile apps

### Usage

Upload PDF files the same way as images:

```bash
curl -X POST "http://localhost:8000/api/ocr" \
  -F "file=@aadhaar.pdf" \
  -F "doc_type=E"
```

### Technical Details

**Conversion Process:**
1. PDF bytes read into memory
2. First page extracted using PyMuPDF (fitz)
3. Page rendered at 300 DPI (4.17x scaling from 72 DPI)
4. Converted to PIL Image (RGB format)
5. Processed through OCR pipeline

**Error Handling:**
- Invalid PDF: `"Failed to process PDF: <error>"`
- Empty PDF: `"PDF contains no pages"`
- Corrupted PDF: `"Failed to process PDF: <error>"`

---

## ⚡ Performance Optimization

### OCR Model Loading

**Asynchronous Startup:**
- OCR model loads in background during application startup
- Non-blocking initialization using `asyncio.run_in_executor()`
- Status tracking: `not_loaded` → `loading` → `ready` / `failed`

**Benefits:**
- Faster application startup
- API available immediately
- Graceful error handling if model fails to load

### OCR Processing

**Multi-Pass Strategy:**
1. **Pass 1**: Original image (resized if > 1600px width)
2. **Pass 2**: Preprocessed image (enhanced contrast + sharpening)
3. **Early Exit**: Stops if Pass 1 has good confidence (>80%) and keywords

**Optimizations:**
- Smart resizing (maintains aspect ratio)
- Confidence-based early termination
- Keyword detection for document type validation
- Result merging by y-position (line grouping)

### Image Preprocessing

**For Small Text (Aadhaar DOB, etc.):**
- Lower detection threshold: `det_db_thresh=0.2`
- Lower box threshold: `det_db_box_thresh=0.4`
- Angle classification enabled: `use_angle_cls=True`
- Upscaling to 1200px minimum width
- Contrast enhancement (1.6x)
- Sharpening filter

### PDF Processing

**High-Quality Conversion:**
- 300 DPI rendering (vs standard 72 DPI)
- Direct memory processing (no disk I/O)
- Single-page optimization

### Memory Management

- No disk storage for uploaded files (in-memory processing)
- Automatic cleanup of temporary objects
- Efficient numpy array handling

---

## 📊 Response Schema

All OCR responses follow a **uniform schema** regardless of document type:

```python
{
  "doc_type": str,           # Document type name
  "name": FieldValue,        # Person's name
  "dob": FieldValue,         # Date of birth
  "gender": FieldValue,      # Gender
  "address": FieldValue,     # Address
  
  # Document-specific fields
  "aadhaar": FieldValue,     # Aadhaar number (12 digits)
  "vid": FieldValue,         # Virtual ID (16 digits)
  "pan": FieldValue,         # PAN number
  "dl_number": FieldValue,   # Driving License number
  "epic": FieldValue,        # Voter ID EPIC number
  "passport": FieldValue,    # Passport number
  
  # Additional fields
  "father": FieldValue,      # Father's name
  "mother": FieldValue,      # Mother's name
  "husband": FieldValue,     # Husband/Spouse name
  "nationality": FieldValue, # Nationality
  "place_of_birth": FieldValue,
  "validity": FieldValue,    # Validity date (DL)
  "issue_date": FieldValue,  # Issue date (DL)
  "blood_group": FieldValue, # Blood group (DL)
  "cov": FieldValue          # Class of Vehicle (DL)
}
```

### FieldValue Schema

```python
{
  "value": str,       # Extracted value (empty string if not found)
  "confidence": float # OCR confidence (0.0 to 1.0, 0 if not found)
}
```

**Benefits:**
- Consistent structure across all document types
- Easy to parse and validate
- Empty fields instead of missing keys
- Confidence scores for quality assessment

---

##  Error Handling

### Error Response Format

```json
{
  "success": false,
  "message": "Error description",
  "error_code": "ERROR_TYPE"
}
```

### Common Errors

| Status Code | Error Message | Cause |
|-------------|---------------|-------|
| 400 | Invalid Document | Document validation failed |
| 400 | Invalid document type code | Wrong doc_type parameter |
| 400 | The uploaded file is empty | Empty file uploaded |
| 400 | Invalid image file or format | Corrupted or unsupported image |
| 400 | Failed to process PDF | PDF processing error |
| 400 | Selfie error: No face detected | No face found in selfie |
| 400 | Document error: No face detected | No face found in document |
| 503 | OCR model is still loading | Model not ready yet |
| 503 | OCR model failed to load | Model initialization failed |
| 500 | Internal Server Error | Unexpected server error |

### Validation Errors

**Field Validation:**
```json
{
  "success": false,
  "message": "Doc type is required",
  "error_code": "Validation_Error"
}
```

**File Validation:**
```json
{
  "success": false,
  "message": "Unsupported file type. Allowed: png, jpg, jpeg, jfif, webp, pdf",
  "error_code": "Service_Error"
}
```

---

## 🛠️ Technology Stack

- **Framework**: FastAPI
- **OCR Engine**: PaddleOCR 3.3.2
- **Face Recognition**: facenet-pytorch, MTCNN, InceptionResnetV1
- **PDF Processing**: PyMuPDF (fitz)
- **Image Processing**: Pillow (PIL)
- **Deep Learning**: PyTorch
- **Async**: asyncio

---

## 📁 Project Structure

```
paddler_ocr/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── extractors/            # Document extraction modules
│   ├── __init__.py
│   ├── aadhaar.py         # Aadhaar extraction logic
│   ├── pan.py             # PAN extraction logic
│   ├── driving_license.py # DL extraction logic
│   ├── voter.py           # Voter ID extraction logic
│   ├── passport.py        # Passport extraction logic
│   ├── utils.py           # Utility functions
│   └── document_validator.py # Validation logic
└── uploads/               # Temporary upload folder (for face verification)
```

---

## 🔐 Security Considerations

1. **No Persistent Storage**: Uploaded files are processed in-memory only
2. **CORS Configuration**: Configure `allow_origins` for production
3. **File Size Limits**: Consider adding file size restrictions
4. **Rate Limiting**: Implement rate limiting for production use
5. **Input Validation**: All inputs are validated before processing

---

## 🚦 API Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📝 License

[Add your license information here]

---

## 👥 Contributors

[Add contributor information here]

---

## 📞 Support

For issues, questions, or contributions, please [add contact/repository information here].

---

**Version**: 1.0.0  
**Last Updated**: February 2026
