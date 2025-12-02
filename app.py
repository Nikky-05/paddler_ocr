


from flask import Flask, request, render_template, jsonify
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from PaddleOcr import process_path

app = Flask(__name__)

import tempfile

# Configuration
# Use system temp directory for uploads (works on Cloud Run/App Engine which have read-only FS)
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'}

# Create uploads folder if it doesn't exist (not needed for tempdir usually, but good practice)
# os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: JPG, PNG, WEBP, BMP, TIFF'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image with OCR
        result = process_path(Path(filepath))
        
        # Clean up - delete the uploaded file after processing
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Use PORT environment variable if available (required for Cloud Run/App Engine)
    port = int(os.environ.get('PORT', 5000))
    # Disable debug mode in production
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
