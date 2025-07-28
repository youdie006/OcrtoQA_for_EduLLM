# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Tesseract OCR with language packs
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-kor \
    tesseract-ocr-equ \
    # PDF processing
    poppler-utils \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # General utilities
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download BERT model for offline use (optional)
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('roberta-large'); \
    AutoTokenizer.from_pretrained('roberta-large')"

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p data/raw_pdfs data/processed/text data/processed/latex data/processed/qa

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Default command
CMD ["python", "src/pipeline.py", "--help"]

# Example usage:
# Build: docker build -t ocr-qa-pipeline .
# Run: docker run -v $(pwd)/data:/app/data -e OPENAI_API_KEY=$OPENAI_API_KEY ocr-qa-pipeline python src/pipeline.py