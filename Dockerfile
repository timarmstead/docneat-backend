FROM python:3.11-slim

# Update and install system deps for OCR (split for better error handling)
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT

CMD ["./entrypoint.sh"]
