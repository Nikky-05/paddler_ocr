# API Documentation - Indian Document OCR API v1.0.0

## 🚀 Quick Start

### Base URL
```
Local: http://localhost:5000
Production: https://your-service-url.run.app
```

### Authentication
Most endpoints require an API key. Include it in the request header:
```
X-API-Key: your-api-key-here
```

---

## 📍 Endpoints

### Public Endpoints (No API Key Required)

#### 1. Web Interface
```http
GET /
```
Returns the HTML web interface for document upload.

#### 2. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "Indian Document OCR API",
  "version": "1.0.0",
  "uptime_seconds": 1234.56
}
```

#### 3. API Information
```http
GET /api/info
```
**Response:**
```json
{
  "service": "Indian Document OCR API",
  "version": "1.0.0",
  "supported_documents": ["Aadhaar Card", "PAN Card", "Driving License", "Voter ID", "Passport", "E-Aadhaar"],
  "supported_formats": ["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
  "max_file_size": "16MB",
  "endpoints": {...},
  "authentication": {...},
  "rate_limits": {...}
}
```

#### 4. Metrics
```http
GET /api/metrics
```
**Response:**
```json
{
  "uptime_seconds": 1234.56,
  "total_requests": 150,
  "successful_requests": 145,
  "failed_requests": 5,
  "success_rate": 96.67,
  "timestamp": "2025-12-02T17:00:00"
}
```

#### 5. Legacy Upload (Backward Compatible)
```http
POST /upload
Content-Type: multipart/form-data
```
**Request:**
- `file`: Image file (JPG, PNG, WEBP, BMP, TIFF)

**Rate Limit:** 10 requests per minute

**Response:**
```json
{
  "doc_type": "aadhaar",
  "name": "John Doe",
  "dob": "01/01/1990",
  "gender": "Male",
  "adhar_no": "1234 5678 9012",
  "address": "123 Main St, City - 123456",
  "request_id": "uuid-here"
}
```

---

### Authenticated Endpoints (API Key Required)

#### 6. Versioned Health Check
```http
GET /api/v1/health
X-API-Key: your-api-key
```
**Response:**
```json
{
  "status": "healthy",
  "service": "Indian Document OCR API",
  "version": "1.0.0",
  "api_version": "v1",
  "uptime_seconds": 1234.56,
  "request_id": "uuid-here"
}
```

#### 7. Versioned Upload
```http
POST /api/v1/upload
X-API-Key: your-api-key
Content-Type: multipart/form-data
```
**Request:**
- `file`: Image file (JPG, PNG, WEBP, BMP, TIFF)

**Rate Limit:** 10 requests per minute

**Response:**
```json
{
  "doc_type": "pan",
  "name": "John Doe",
  "father_name": "Richard Doe",
  "dob": "01/01/1990",
  "pan_no": "ABCDE1234F",
  "request_id": "uuid-here",
  "processing_time_seconds": 2.34,
  "api_version": "v1"
}
```

---

## 🔒 Authentication

### Getting an API Key

**Development:**
Default API key: `dev-key-123`

**Production:**
Set via environment variable:
```bash
export API_KEYS=key1,key2,key3
```

Generate secure keys:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Using API Key

Include in request headers:
```bash
curl -H "X-API-Key: your-api-key" \
     -X POST \
     -F "file=@document.jpg" \
     https://your-api.run.app/api/v1/upload
```

---

## ⚡ Rate Limits

| Endpoint | Limit |
|----------|-------|
| Global | 100 requests/hour per IP |
| `/upload` | 10 requests/minute per IP |
| `/api/v1/upload` | 10 requests/minute per IP |

**Rate Limit Response (429):**
```json
{
  "error": "Rate limit exceeded. Please try again later.",
  "request_id": "uuid-here"
}
```

### Disable Rate Limiting (Development)
```bash
export RATE_LIMIT_ENABLED=false
```

---

## 🌐 CORS

### Allowed Origins

**Default:** All origins (`*`)

**Production:** Set specific origins
```bash
export ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

---

## 📝 Request/Response Examples

### Example 1: Upload Aadhaar Card

**Request:**
```bash
curl -X POST \
  -H "X-API-Key: dev-key-123" \
  -F "file=@aadhaar.jpg" \
  http://localhost:5000/api/v1/upload
```

**Response:**
```json
{
  "doc_type": "aadhaar",
  "name": "Nikky Laxman Bisen",
  "dob": "05/04/1999",
  "gender": "Male",
  "adhar_no": "7084 5908 7860",
  "address": "S/O Laxman Bisane, at deosarra post bapera, Deosirra, PO: Bapera, Dist: Bhandara, Maharashtra-441915",
  "request_id": "abc-123-def",
  "processing_time_seconds": 3.45,
  "api_version": "v1"
}
```

### Example 2: Upload PAN Card

**Request:**
```bash
curl -X POST \
  -H "X-API-Key: dev-key-123" \
  -F "file=@pan.jpg" \
  http://localhost:5000/api/v1/upload
```

**Response:**
```json
{
  "doc_type": "pan",
  "name": "JOHN DOE",
  "father_name": "RICHARD DOE",
  "dob": "01/01/1990",
  "pan_no": "ABCDE1234F",
  "request_id": "xyz-456-abc",
  "processing_time_seconds": 2.12,
  "api_version": "v1"
}
```

### Example 3: Upload Driving License

**Request:**
```bash
curl -X POST \
  -H "X-API-Key: dev-key-123" \
  -F "file=@dl.jpg" \
  http://localhost:5000/api/v1/upload
```

**Response:**
```json
{
  "doc_type": "driving_license",
  "name": "NIKKY L BISEN",
  "dob": "05-04-1999",
  "dl_no": "MH36 2C220004543",
  "address": "AT DEOSARRA PO BAPERA TAH TUMSAR DIST BHANDARA,MH PIN:441915",
  "validity": "04-04-2039",
  "request_id": "def-789-ghi",
  "processing_time_seconds": 4.23,
  "api_version": "v1"
}
```

---

## ❌ Error Responses

### 400 Bad Request
```json
{
  "error": "No file uploaded",
  "request_id": "uuid-here"
}
```

### 401 Unauthorized
```json
{
  "error": "Invalid API key",
  "request_id": "uuid-here"
}
```

### 413 Payload Too Large
```json
{
  "error": "File too large. Maximum size is 16MB",
  "request_id": "uuid-here"
}
```

### 429 Too Many Requests
```json
{
  "error": "Rate limit exceeded. Please try again later.",
  "request_id": "uuid-here"
}
```

### 500 Internal Server Error
```json
{
  "error": "Error message here",
  "type": "ExceptionType",
  "request_id": "uuid-here"
}
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5000 | Server port |
| `FLASK_ENV` | development | Environment (development/production) |
| `API_KEYS` | dev-key-123 | Comma-separated API keys |
| `ALLOWED_ORIGINS` | * | CORS allowed origins |
| `RATE_LIMIT_ENABLED` | true | Enable rate limiting |
| `GLOBAL_RATE_LIMIT` | 100 per hour | Global rate limit |
| `UPLOAD_RATE_LIMIT` | 10 per minute | Upload endpoint rate limit |

### Example Configuration

**Development:**
```bash
export FLASK_ENV=development
export API_KEYS=dev-key-123
export RATE_LIMIT_ENABLED=false
python app.py
```

**Production:**
```bash
export FLASK_ENV=production
export API_KEYS=prod-key-abc,prod-key-xyz
export ALLOWED_ORIGINS=https://yourdomain.com
export RATE_LIMIT_ENABLED=true
gunicorn --bind :8080 --workers 2 --threads 4 --timeout 300 app:app
```

---

## 📊 Supported Documents

1. **Aadhaar Card** - Indian national ID
2. **PAN Card** - Permanent Account Number
3. **Driving License** - Indian driving license
4. **Voter ID** - Election ID card
5. **Passport** - Indian passport
6. **E-Aadhaar** - Electronic Aadhaar

---

## 🎯 Best Practices

1. **Always use API keys in production**
2. **Set specific CORS origins** (don't use `*`)
3. **Monitor rate limits** to avoid 429 errors
4. **Use versioned endpoints** (`/api/v1/*`) for stability
5. **Check request_id** in responses for debugging
6. **Handle errors gracefully** in your application
7. **Respect rate limits** to avoid service disruption

---

## 📞 Support

For issues or questions:
- Check logs for `request_id`
- Review `/api/metrics` for service health
- Verify API key is valid
- Ensure file format is supported
