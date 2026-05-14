# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY chatbot_core.py .
COPY app.py          .
COPY api.py          .
COPY main.py         .

# Create directories
RUN mkdir -p chroma_db pdfs .streamlit

# Copy streamlit config if exists
COPY .streamlit/ .streamlit/

EXPOSE 8000

# Default: run FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]