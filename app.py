

from flask import Flask, request, render_template, jsonify
import os
import sys
import logging
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from functools import wraps
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from PaddleOcr import process_path
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load configuration
Config = get_config()
app.config.from_object(Config)

import tempfile

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logger.info(f"Upload folder configured: {UPLOAD_FOLDER}")
logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
logger.info(f"Rate limiting: {'enabled' if Config.RATE_LIMIT_ENABLED else 'disabled'}")

# Initialize CORS
CORS(app, origins=Config.get_allowed_origins())
logger.info(f"CORS configured for origins: {Config.get_allowed_origins()}")

# Ensure PaddleOCR can write to home directory in serverless env
if os.environ.get('GCP_PROJECT') or os.environ.get('GAE_APPLICATION'):
    os.environ['HOME'] = '/tmp'
    os.environ['OCR_DEBUG'] = '1'  # Enable debug logging


# Initialize Rate Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=Config.RATE_LIMIT_STORAGE_URL,
    default_limits=[Config.GLOBAL_RATE_LIMIT] if Config.RATE_LIMIT_ENABLED else []
)
logger.info("Rate limiter initialized")

# Request tracking
request_stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'start_time': datetime.now()
}


# Decorators
def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            logger.warning(f"Request {getattr(request, 'id', 'unknown')} - Missing API key")
            return jsonify({'error': 'Missing API key. Please provide X-API-Key header'}), 401
        
        if not Config.is_valid_api_key(api_key):
            logger.warning(f"Request {getattr(request, 'id', 'unknown')} - Invalid API key")
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function


# Request/Response Hooks
@app.before_request
def before_request():
    """Add request ID and track request"""
    request.id = str(uuid.uuid4())
    request.start_time = datetime.now()
    request_stats['total_requests'] += 1
    logger.info(f"Request {request.id} started: {request.method} {request.path}")


@app.after_request
def after_request(response):
    """Add request ID to response headers and log completion"""
    if hasattr(request, 'id'):
        response.headers['X-Request-ID'] = request.id
        
        # Calculate request duration
        if hasattr(request, 'start_time'):
            duration = (datetime.now() - request.start_time).total_seconds()
            logger.info(f"Request {request.id} completed: {response.status_code} ({duration:.2f}s)")
        
        # Update stats
        if 200 <= response.status_code < 400:
            request_stats['successful_requests'] += 1
        else:
            request_stats['failed_requests'] += 1
    
    return response


# Global error handlers
@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler to always return JSON"""
    logger.error(f"Unhandled exception in request {getattr(request, 'id', 'unknown')}: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Handle file size limit exceeded
    if isinstance(e, RequestEntityTooLarge):
        return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413
    
    # Return JSON error for all other exceptions
    return jsonify({
        'error': str(e),
        'type': type(e).__name__,
        'request_id': getattr(request, 'id', None)
    }), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors with JSON"""
    return jsonify({
        'error': 'Endpoint not found',
        'request_id': getattr(request, 'id', None)
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle 405 errors with JSON"""
    return jsonify({
        'error': 'Method not allowed',
        'request_id': getattr(request, 'id', None)
    }), 405


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit errors"""
    return jsonify({
        'error': 'Rate limit exceeded. Please try again later.',
        'request_id': getattr(request, 'id', None)
    }), 429


# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def validate_file_size(file):
    """Validate file size before processing"""
    # Get file size by seeking to end
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    max_size = Config.MAX_CONTENT_LENGTH
    if file_size > max_size:
        size_mb = file_size / (1024 * 1024)
        limit_mb = max_size / (1024 * 1024)
        raise ValueError(f"File size ({size_mb:.2f}MB) exceeds the maximum limit of {limit_mb}MB")
    
    return True


def convert_pdf_to_images(pdf_path):
    """Convert PDF file to images (one per page)
    Returns a list of image file paths
    """
    try:
        from pdf2image import convert_from_path
        import tempfile
        
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        # Convert PDF to images (one per page)
        # Use lower DPI for faster processing, but high enough for OCR
        images = convert_from_path(pdf_path, dpi=200)
        
        image_paths = []
        temp_dir = tempfile.gettempdir()
        
        for i, image in enumerate(images):
            # Save each page as a temporary image
            img_path = os.path.join(temp_dir, f"{os.path.basename(pdf_path)}_page_{i+1}.png")
            image.save(img_path, 'PNG')
            image_paths.append(img_path)
            logger.info(f"Saved PDF page {i+1} to: {img_path}")
        
        logger.info(f"Converted PDF to {len(image_paths)} image(s)")
        return image_paths
    
    except ImportError:
        raise ValueError(
            "PDF processing requires 'pdf2image' and 'poppler' to be installed. "
            "Please install: pip install pdf2image"
        )
    except Exception as e:
        logger.error(f"Error converting PDF: {str(e)}")
        raise ValueError(f"Failed to process PDF file: {str(e)}")


def validate_image_dimensions(filepath):
    """Validate image dimensions for optimal OCR processing
    Also handles PDF files by skipping dimension validation
    """
    try:
        # Skip validation for PDF files (will be converted to images later)
        if filepath.lower().endswith('.pdf'):
            logger.info("PDF file detected, skipping dimension validation")
            return True
        
        from PIL import Image
        
        with Image.open(filepath) as img:
            width, height = img.size
            max_dimension = 4096  # Maximum dimension for processing
            
            if width > max_dimension or height > max_dimension:
                raise ValueError(
                    f"Image dimensions ({width}x{height}) exceed maximum allowed size of {max_dimension}x{max_dimension} pixels. "
                    f"Please resize your image before uploading."
                )
            
            # Minimum dimension check for quality
            min_dimension = 100
            if width < min_dimension or height < min_dimension:
                raise ValueError(
                    f"Image dimensions ({width}x{height}) are too small. "
                    f"Minimum size is {min_dimension}x{min_dimension} pixels for accurate OCR."
                )
            
            logger.info(f"Image dimensions validated: {width}x{height}")
            return True
    except Exception as e:
        if "exceed maximum" in str(e) or "too small" in str(e):
            raise
        logger.error(f"Error validating image: {str(e)}")
        raise ValueError(f"Invalid or corrupted image file: {str(e)}")


# Public Routes (No authentication required)
@app.route('/')
def index():
    """Serve web interface"""
    logger.info("Serving index page")
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint for GCP"""
    uptime = (datetime.now() - request_stats['start_time']).total_seconds()
    return jsonify({
        'status': 'healthy',
        'service': Config.API_TITLE,
        'version': Config.API_VERSION,
        'uptime_seconds': round(uptime, 2)
    }), 200


@app.route('/api/info')
def api_info():
    """API information endpoint"""
    return jsonify({
        'service': Config.API_TITLE,
        'version': Config.API_VERSION,
        'supported_documents': Config.SUPPORTED_DOCUMENTS,
        'supported_formats': list(Config.ALLOWED_EXTENSIONS),
        'max_file_size': '16MB',
        'max_image_dimensions': '4096x4096 pixels',
        'min_image_dimensions': '100x100 pixels',
        'endpoints': {
            'web_interface': '/',
            'health': '/health',
            'api_info': '/api/info',
            'metrics': '/api/metrics',
            'upload_v1': '/api/v1/upload (requires API key)',
            'health_v1': '/api/v1/health (requires API key)',
            'legacy_upload': '/upload (public, for backward compatibility)'
        },
        'authentication': {
            'type': 'API Key',
            'header': 'X-API-Key',
            'required_for': ['/api/v1/*']
        },
        'rate_limits': {
            'enabled': Config.RATE_LIMIT_ENABLED,
            'global': Config.GLOBAL_RATE_LIMIT,
            'upload': Config.UPLOAD_RATE_LIMIT
        }
    }), 200


@app.route('/api/metrics')
def metrics():
    """Metrics endpoint for monitoring"""
    uptime = (datetime.now() - request_stats['start_time']).total_seconds()
    success_rate = (request_stats['successful_requests'] / request_stats['total_requests'] * 100) if request_stats['total_requests'] > 0 else 0
    
    return jsonify({
        'uptime_seconds': round(uptime, 2),
        'total_requests': request_stats['total_requests'],
        'successful_requests': request_stats['successful_requests'],
        'failed_requests': request_stats['failed_requests'],
        'success_rate': round(success_rate, 2),
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/upload', methods=['POST'])
@limiter.limit(Config.UPLOAD_RATE_LIMIT) if Config.RATE_LIMIT_ENABLED else lambda f: f
def upload_file():
    """Legacy upload endpoint (public, for backward compatibility)"""
    logger.info("Upload request received (legacy endpoint)")
    
    if 'file' not in request.files:
        logger.warning("No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(Config.ALLOWED_EXTENSIONS).upper()}'}), 400
    
    # Validate file size
    try:
        validate_file_size(file)
    except ValueError as e:
        logger.warning(f"File size validation failed: {str(e)}")
        return jsonify({'error': str(e)}), 413
    
    filepath = None
    pdf_image_paths = []
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{request.id}_{filename}")
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Validate image dimensions
        try:
            validate_image_dimensions(filepath)
        except ValueError as e:
            logger.warning(f"Image validation failed: {str(e)}")
            return jsonify({'error': str(e)}), 400
        
        # Handle PDF files
        if filepath.lower().endswith('.pdf'):
            logger.info("PDF file detected, converting to images")
            try:
                pdf_image_paths = convert_pdf_to_images(filepath)
                
                # Process all pages
                all_results = []
                for i, img_path in enumerate(pdf_image_paths):
                    logger.info(f"Processing PDF page {i+1}/{len(pdf_image_paths)}")
                    result = process_path(Path(img_path))
                    result['page_number'] = i + 1
                    all_results.append(result)
                
                # If single page, return just that result
                if len(all_results) == 1:
                    result = all_results[0]
                    result['request_id'] = request.id
                    result['source_file'] = filename
                    return jsonify(result)
                
                # For multi-page PDFs, return all results
                return jsonify({
                    'request_id': request.id,
                    'source_file': filename,
                    'total_pages': len(all_results),
                    'pages': all_results
                })
            
            except ValueError as e:
                logger.error(f"PDF processing failed: {str(e)}")
                return jsonify({'error': str(e)}), 400
        
        # Process regular image files
        logger.info(f"Processing OCR for: {filename}")
        result = process_path(Path(filepath))
        logger.info(f"OCR completed successfully for: {filename}")
        
        # Add request metadata
        result['request_id'] = request.id
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'request_id': request.id
        }), 500
    
    finally:
        # Clean up - delete the uploaded file and any PDF-generated images
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up temporary file: {filepath}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file {filepath}: {str(cleanup_error)}")
        
        # Clean up PDF-generated images
        for img_path in pdf_image_paths:
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                    logger.info(f"Cleaned up PDF image: {img_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup PDF image {img_path}: {str(cleanup_error)}")


# API v1 Routes (Require authentication)
@app.route('/api/v1/health')
@require_api_key
def health_v1():
    """Versioned health check endpoint"""
    uptime = (datetime.now() - request_stats['start_time']).total_seconds()
    return jsonify({
        'status': 'healthy',
        'service': Config.API_TITLE,
        'version': Config.API_VERSION,
        'api_version': 'v1',
        'uptime_seconds': round(uptime, 2),
        'request_id': request.id
    }), 200


@app.route('/api/v1/upload', methods=['POST'])
@require_api_key
@limiter.limit(Config.UPLOAD_RATE_LIMIT) if Config.RATE_LIMIT_ENABLED else lambda f: f
def upload_file_v1():
    """Versioned upload endpoint with authentication"""
    logger.info("Upload request received (v1 endpoint)")
    
    if 'file' not in request.files:
        logger.warning("No file in request")
        return jsonify({
            'error': 'No file uploaded',
            'request_id': request.id
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({
            'error': 'No file selected',
            'request_id': request.id
        }), 400
    
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(Config.ALLOWED_EXTENSIONS).upper()}',
            'request_id': request.id
        }), 400
    
    # Validate file size
    try:
        validate_file_size(file)
    except ValueError as e:
        logger.warning(f"File size validation failed: {str(e)}")
        return jsonify({
            'error': str(e),
            'request_id': request.id
        }), 413
    
    filepath = None
    pdf_image_paths = []
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{request.id}_{filename}")
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Validate image dimensions
        try:
            validate_image_dimensions(filepath)
        except ValueError as e:
            logger.warning(f"Image validation failed: {str(e)}")
            return jsonify({
                'error': str(e),
                'request_id': request.id
            }), 400
        
        # Handle PDF files
        if filepath.lower().endswith('.pdf'):
            logger.info("PDF file detected, converting to images")
            try:
                pdf_image_paths = convert_pdf_to_images(filepath)
                
                # Process all pages
                all_results = []
                for i, img_path in enumerate(pdf_image_paths):
                    logger.info(f"Processing PDF page {i+1}/{len(pdf_image_paths)}")
                    page_start_time = datetime.now()
                    result = process_path(Path(img_path))
                    page_processing_time = (datetime.now() - page_start_time).total_seconds()
                    result['page_number'] = i + 1
                    result['processing_time_seconds'] = round(page_processing_time, 2)
                    all_results.append(result)
                
                total_processing_time = sum(r['processing_time_seconds'] for r in all_results)
                
                # If single page, return just that result
                if len(all_results) == 1:
                    result = all_results[0]
                    result['request_id'] = request.id
                    result['source_file'] = filename
                    result['api_version'] = 'v1'
                    return jsonify(result)
                
                # For multi-page PDFs, return all results
                return jsonify({
                    'request_id': request.id,
                    'source_file': filename,
                    'total_pages': len(all_results),
                    'processing_time_seconds': round(total_processing_time, 2),
                    'api_version': 'v1',
                    'pages': all_results
                })
            
            except ValueError as e:
                logger.error(f"PDF processing failed: {str(e)}")
                return jsonify({
                    'error': str(e),
                    'request_id': request.id
                }), 400
        
        # Process regular image files
        logger.info(f"Processing OCR for: {filename}")
        start_time = datetime.now()
        result = process_path(Path(filepath))
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"OCR completed successfully for: {filename} in {processing_time:.2f}s")
        
        # Add metadata
        result['request_id'] = request.id
        result['processing_time_seconds'] = round(processing_time, 2)
        result['api_version'] = 'v1'
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'request_id': request.id
        }), 500
    
    finally:
        # Clean up - delete the uploaded file and any PDF-generated images
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up temporary file: {filepath}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file {filepath}: {str(cleanup_error)}")
        
        # Clean up PDF-generated images
        for img_path in pdf_image_paths:
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                    logger.info(f"Cleaned up PDF image: {img_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup PDF image {img_path}: {str(cleanup_error)}")


def warmup_ocr_models():
    """Pre-load PaddleOCR models on startup to avoid empty results on first requests.
    
    This function initializes the OCR engine and runs a dummy prediction to ensure
    all models are downloaded and loaded into memory before the first real request.
    """
    try:
        logger.info("Warming up PaddleOCR models...")
        from PaddleOcr import get_ocr
        import numpy as np
        
        # Initialize OCR (downloads models if needed)
        ocr = get_ocr()
        
        # Create a small dummy image (100x100 white image)
        dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Run a dummy OCR to load models into memory
        try:
            ocr.ocr(dummy_image)
        except:
            # Fallback for different PaddleOCR versions
            try:
                ocr.predict(dummy_image)
            except:
                pass
        
        logger.info("✓ PaddleOCR models loaded successfully")
    except Exception as e:
        logger.warning(f"Model warmup failed (will load on first request): {str(e)}")

# Run warmup on module import (so it runs in Gunicorn)
warmup_ocr_models()


if __name__ == '__main__':
    # Use PORT environment variable if available (required for Cloud Run/App Engine)
    port = int(os.environ.get('PORT', 5000))
    # Disable debug mode in production
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting {Config.API_TITLE} v{Config.API_VERSION}")
    logger.info(f"Server running on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
