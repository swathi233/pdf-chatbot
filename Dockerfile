FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including DNS tools
RUN apt-get update && apt-get install -y \
    gcc \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY templates/ templates/

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["gunicorn", "--workers=1", "--timeout=120", "--bind=0.0.0.0:8080", "app:app"]