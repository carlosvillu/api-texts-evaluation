FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

RUN mkdir -p /tmp/vllm_cache /workspace/transformers /workspace/hub

# Add non-root user for better security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app /tmp/vllm_cache /workspace
USER appuser

ENV HF_HOME=/workspace
ENV TRANSFORMERS_CACHE=/workspace/transformers
ENV HF_HUB_CACHE=/workspace/hub
ENV VLLM_USE_PRECOMPILED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]