FROM python:3.11-slim

# Update and install minimal OCR deps (no deprecated GLX)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

EXPOSE $PORT

CMD ["./entrypoint.sh"]
