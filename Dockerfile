# Use official lightweight Python image
FROM python:3.9-slim

# Install system dependencies required for PaddleOCR and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port
ENV PORT=5000
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
