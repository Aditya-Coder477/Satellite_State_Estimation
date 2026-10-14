# Production Dockerfile for Satellite Orbit Lab
FROM python:3.10-slim

# Install system dependencies for scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications
COPY requirements.txt ./
COPY dashboard/requirements_dashboard.txt ./dashboard/

# Install Python scientific packages (PyTorch CPU version to keep image lightweight and fast)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r dashboard/requirements_dashboard.txt

# Copy source code, config, checkpoints, data, and built frontend
COPY config.yaml ./
COPY checkpoints/ ./checkpoints/
COPY data/ ./data/
COPY src/ ./src/
COPY dashboard/ ./dashboard/

# Expose port (default 8000 or Cloud Run $PORT)
ENV PORT=8000
EXPOSE 8000

# Start FastAPI backend (which statically serves frontend dist)
CMD ["sh", "-c", "uvicorn dashboard.backend.main:app --host 0.0.0.0 --port ${PORT}"]
