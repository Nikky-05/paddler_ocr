# PDF Support Documentation

## Overview
The OCR API now supports **PDF documents** in addition to image formats. This allows you to upload scanned identity documents in PDF format for text extraction.

## Supported Formats
- **Images**: PNG, JPG, JPEG, JFIF, WEBP
- **Documents**: PDF ✨ (NEW)

## How It Works

### PDF Processing Pipeline
1. **Upload**: Upload a PDF file containing an identity document
2. **Conversion**: The PDF is automatically converted to high-quality images (300 DPI)
3. **OCR Processing**: Each page is processed through the PaddleOCR engine
4. **Extraction**: Document-specific data is extracted based on the document type

### Important Notes
- **Multi-page PDFs**: Only the **first page** is processed (most ID documents are single-page)
- **Quality**: PDFs are rendered at 300 DPI for optimal OCR accuracy
- **File Size**: No specific size limit, but larger files may take longer to process

## API Usage

### Endpoint
```
POST /api/ocr
```

### Parameters
- `file`: The PDF or image file (multipart/form-data)
- `doc_type`: Document type code (A-E)
  - A: Passport
  - B: Voter ID
  - C: PAN Card
  - D: Driving License
  - E: Aadhaar

### Example Request (cURL)
```bash
curl -X POST "http://localhost:8000/api/ocr" \
  -F "file=@aadhaar_card.pdf" \
  -F "doc_type=E"
```

### Example Request (Python)
```python
import requests

url = "http://localhost:8000/api/ocr"
files = {"file": open("aadhaar_card.pdf", "rb")}
data = {"doc_type": "E"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### Example Request (Postman)
1. Set method to **POST**
2. URL: `http://localhost:8000/api/ocr`
3. Go to **Body** tab
4. Select **form-data**
5. Add two keys:
   - `file` (type: File) - Select your PDF file
   - `doc_type` (type: Text) - Enter document code (e.g., "E" for Aadhaar)
6. Click **Send**

## Response Format
The response format is identical for both PDF and image uploads:

```json
{
  "doc_type": "aadhaar",
  "name": {
    "value": "John Doe",
    "confidence": 0.95
  },
  "aadhaar": {
    "value": "1234 5678 9012",
    "confidence": 0.98
  },
  "dob": {
    "value": "01/01/1990",
    "confidence": 0.92
  },
  // ... other fields
}
```

## Technical Details

### Dependencies
- **PyMuPDF (fitz)**: Used for PDF to image conversion
- Version: >= 1.23.0

### Installation
```bash
pip install PyMuPDF>=1.23.0
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Performance Considerations
- PDF processing adds minimal overhead (typically < 1 second for single-page documents)
- High DPI rendering ensures OCR accuracy comparable to direct image uploads
- Memory usage scales with PDF page count and resolution

## Error Handling

### Common Errors
1. **"PDF contains no pages"**: The uploaded PDF is empty or corrupted
2. **"Failed to process PDF"**: The file is not a valid PDF or is password-protected
3. **"Invalid image file or format"**: The file extension is incorrect or the file is corrupted

### Best Practices
- Ensure PDFs are not password-protected
- Use high-quality scans (300 DPI or higher)
- Keep PDFs to single-page for ID documents
- Verify file extension matches content type

## Testing

### Quick Test
1. Start the server:
   ```bash
   python main.py
   ```

2. Visit the API documentation:
   ```
   http://localhost:8000/docs
   ```

3. Use the interactive Swagger UI to test PDF uploads

### Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "ocr_status": "ready",
  "ocr_error": null
}
```

## Troubleshooting

### Issue: "No module named 'fitz'"
**Solution**: Install PyMuPDF
```bash
pip install PyMuPDF
```

### Issue: PDF processing is slow
**Possible causes**:
- Large file size
- Multiple pages (only first page is processed)
- Low system resources

**Solutions**:
- Compress PDF before uploading
- Ensure PDF is single-page
- Increase server resources

### Issue: Poor OCR accuracy on PDF
**Solutions**:
- Use higher quality PDF scans (300+ DPI)
- Ensure PDF is not a scanned image with low resolution
- Try converting PDF to PNG/JPG first with high quality settings

## Version History
- **v1.0.0**: Initial PDF support added
  - PyMuPDF integration
  - 300 DPI rendering
  - First-page processing for multi-page PDFs
