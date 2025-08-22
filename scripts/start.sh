#!/bin/bash

set -e  # Exit on any error

echo "🚀 Iniciando Text Evaluation Service para RunPod.io..."

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para logging
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Verificar GPU NVIDIA
log_info "Verificando disponibilidad de GPU NVIDIA..."
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi no encontrado. GPU NVIDIA requerida."
    exit 1
fi

# Mostrar información de GPU
log_info "Estado de GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits

# Verificar que hay al menos una GPU disponible
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
if [ "$GPU_COUNT" -eq 0 ]; then
    log_error "No se detectaron GPUs disponibles"
    exit 1
fi

log_info "✅ GPU detectada correctamente ($GPU_COUNT GPU(s))"

# 2. Verificar memoria GPU disponible (mínimo 20GB para el modelo)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
MIN_MEMORY=20000  # 20GB en MB

if [ "$GPU_MEMORY" -lt "$MIN_MEMORY" ]; then
    log_warn "⚠️  GPU tiene ${GPU_MEMORY}MB, se recomienda al menos ${MIN_MEMORY}MB"
    log_warn "El servicio puede experimentar problemas de memoria"
else
    log_info "✅ Memoria GPU suficiente: ${GPU_MEMORY}MB"
fi

# 3. Crear directorios necesarios para caché
log_info "Creando directorios de caché..."
mkdir -p /tmp/vllm_cache
mkdir -p /workspace/transformers
mkdir -p /workspace/hub

# 4. Configurar variables de entorno para optimización
log_info "Configurando variables de entorno..."
export HF_HOME=/workspace
export TRANSFORMERS_CACHE=/workspace/transformers
export HF_HUB_CACHE=/workspace/hub
export VLLM_USE_PRECOMPILED=1

# Variables específicas para RunPod.io
export RUNPOD_GPU_TYPE=${RUNPOD_GPU_TYPE:-"L40"}
export RUNPOD_MEMORY_LIMIT=${RUNPOD_MEMORY_LIMIT:-"24GB"}

# 5. Verificar conectividad de red (para descarga de modelo)
log_info "Verificando conectividad..."
if ! ping -c 1 huggingface.co &> /dev/null; then
    log_warn "⚠️  No se puede conectar a Hugging Face. Verificar conectividad de red."
fi

# 6. Limpiar caché GPU antes de iniciar
log_info "Limpiando caché GPU..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || log_warn "No se pudo limpiar caché GPU"

# 7. Verificar dependencias Python
log_info "Verificando dependencias Python..."
python3 -c "import fastapi, vllm, torch; print('✅ Dependencias OK')" || {
    log_error "Faltan dependencias Python críticas"
    exit 1
}

# 8. Verificar que CUDA está disponible
log_info "Verificando CUDA..."
python3 -c "import torch; print('✅ CUDA disponible:', torch.cuda.is_available()); print('Dispositivos CUDA:', torch.cuda.device_count())"

# 9. Configurar ulimits para mejor rendimiento
log_info "Configurando límites del sistema..."
ulimit -n 65536 2>/dev/null || log_warn "No se pudieron ajustar ulimits"

# 10. Mostrar configuración final
log_info "Configuración del servicio:"
echo "  - Modelo: ${MODEL_NAME:-carlosvillu/gemma2-9b-teacher-eval-nota-feedback}"
echo "  - GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION:-0.90}"
echo "  - Puerto: ${PORT:-8000}"
echo "  - Batch Size: ${BATCH_SIZE:-10}"
echo "  - Max Model Length: ${MAX_MODEL_LEN:-1024}"

# 11. Iniciar el servicio FastAPI
log_info "🔥 Iniciando FastAPI..."
log_info "Servicio estará disponible en: http://0.0.0.0:${PORT:-8000}"
log_info "Health check: http://0.0.0.0:${PORT:-8000}/health"

# Usar exec para que el proceso de Python sea PID 1 (importante para Docker)
exec python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --log-level info \
    --access-log \
    --no-server-header