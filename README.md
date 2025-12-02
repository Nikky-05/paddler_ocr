# OCR API - Indian Identity Documents

A Flask-based OCR application for extracting data from Indian identity documents using PaddleOCR.

## Supported Documents
- Aadhaar Card
- PAN Card
- Voter ID
- Driving License
- Passport
- E-Aadhaar

## Project Structure

```
paddle/
├── app.py                    # Flask web server
├── PaddleOcr.py             # OCR processing engine
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── .dockerignore            # Docker exclusions
├── .gitignore               # Git exclusions
└── templates/
    └── index.html           # Web interface
```

## Installation

### Local Development

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t ocr-api .
```

2. Run the container:
```bash
docker run -p 5000:5000 ocr-api
```

## API Usage

### Upload Endpoint

**POST** `/upload`

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Supported formats:** JPG, JPEG, PNG, WEBP, BMP, TIFF

**Response:**
```json
{
  "name": "John Doe",
  "dob": "01/01/1990",
  "gender": "Male",
  "adhar_no": "1234 5678 9012",
  "address": "123 Main St, City, State - 123456"
}
```

## Environment Variables

- `PORT` - Server port (default: 5000)
- `FLASK_ENV` - Set to `development` for debug mode

## Deployment

This application is ready for deployment on:
- Google Cloud Run
- AWS App Runner
- Azure Container Apps
- Any Docker-compatible platform

The application uses temporary file storage and is compatible with read-only filesystems.

## License

Proprietary
