FROM python:3.11-slim

# Install minimal system deps for OCR (faster, less memory)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python deps with no cache (saves memory)
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

EXPOSE $PORT

CMD ["./entrypoint.sh"]
