FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code and scripts
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY .env.example .env

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/vllm_cache /workspace/transformers /workspace/hub

# Make scripts executable
RUN chmod +x scripts/start.sh scripts/test_client.py

# Environment variables optimized for RunPod.io L40
ENV HF_HOME=/workspace
ENV TRANSFORMERS_CACHE=/workspace/transformers
ENV HF_HUB_CACHE=/workspace/hub
ENV VLLM_USE_PRECOMPILED=1

# Default configuration for L40 GPU (24GB)
ENV MODEL_NAME=carlosvillu/gemma2-9b-teacher-eval-nota-feedback
ENV GPU_MEMORY_UTILIZATION=0.90
ENV MAX_MODEL_LEN=1024
ENV TENSOR_PARALLEL_SIZE=1
ENV DTYPE=bfloat16
ENV HOST=0.0.0.0
ENV PORT=8000
ENV BATCH_SIZE=10
ENV JOB_TTL_SECONDS=3600
ENV MAX_CONCURRENT_JOBS=5
ENV ENABLE_PREFIX_CACHING=true
ENV BLOCK_SIZE=16
ENV MAX_NUM_SEQS=128
ENV COMPILATION_LEVEL=2
ENV CLEANUP_INTERVAL_SECONDS=300
ENV TIME_ESTIMATION_PER_ITEM=2
ENV SSE_POLLING_INTERVAL=1
ENV RUNPOD_GPU_TYPE=L40
ENV RUNPOD_MEMORY_LIMIT=24GB

EXPOSE 8000

# Robust health check with longer startup time for model loading
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Use optimized startup script
CMD ["./scripts/start.sh"]