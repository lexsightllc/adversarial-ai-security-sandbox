# SPDX-License-Identifier: MPL-2.0

# Stage 1: Builder
FROM python:3.9-slim-bullseye as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt


# Stage 2: Runtime
FROM python:3.9-slim-bullseye

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Copy common modules and application code
COPY --chown=appuser:appuser ../database.py .
COPY --chown=appuser:appuser ../logging_config.py .
COPY --chown=appuser:appuser ../metrics_collector.py .
COPY --chown=appuser:appuser ../middleware.py .
COPY --chown=appuser:appuser ../config.py .
COPY --chown=appuser:appuser . .

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/', timeout=5)" || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
