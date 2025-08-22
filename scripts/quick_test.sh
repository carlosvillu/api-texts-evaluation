#!/bin/bash

# Quick test script para RunPod.io deployment
# Ejecuta tests básicos para validar el servicio

set -e

CONTAINER_NAME="text-eval-test"
SERVICE_URL="http://localhost:8000"
TIMEOUT=60

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

cleanup() {
    log_info "Limpiando recursos..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

trap cleanup EXIT

echo "🚀 Quick Test para Text Evaluation Service"
echo "========================================="

# 1. Build del container
log_info "1. Building container..."
if ! docker build -t text-eval-service .; then
    log_error "Docker build falló"
    exit 1
fi

# 2. Ejecutar container
log_info "2. Iniciando container..."
docker run --name $CONTAINER_NAME -d -p 8000:8000 text-eval-service

if [ $? -ne 0 ]; then
    log_error "No se pudo iniciar el container"
    exit 1
fi

# 3. Esperar a que el servicio esté listo
log_info "3. Esperando que el servicio esté listo..."
for i in $(seq 1 $TIMEOUT); do
    if curl -s $SERVICE_URL/health > /dev/null 2>&1; then
        log_info "✅ Servicio respondiendo después de ${i}s"
        break
    fi
    
    if [ $i -eq $TIMEOUT ]; then
        log_error "Timeout esperando el servicio"
        docker logs $CONTAINER_NAME
        exit 1
    fi
    
    echo -n "."
    sleep 1
done

# 4. Health check
log_info "4. Verificando health check..."
health_response=$(curl -s $SERVICE_URL/health)
echo "Health: $health_response"

# Verificar que el modelo esté cargado
if echo "$health_response" | grep -q '"model_loaded": *true'; then
    log_info "✅ Modelo cargado correctamente"
else
    log_error "❌ Modelo no está cargado"
    docker logs $CONTAINER_NAME
    exit 1
fi

# 5. Test de evaluación básico
log_info "5. Ejecutando test de evaluación..."
if [ -f scripts/test_client.py ]; then
    python3 scripts/test_client.py --test basic --url $SERVICE_URL
    if [ $? -eq 0 ]; then
        log_info "✅ Test básico completado exitosamente"
    else
        log_error "❌ Test básico falló"
        docker logs $CONTAINER_NAME
        exit 1
    fi
else
    log_warn "⚠️  test_client.py no encontrado, saltando test automático"
    
    # Test manual con curl
    log_info "Ejecutando test manual con curl..."
    
    test_data='{
      "items": [
        {
          "id_alumno": "QUICK_TEST",
          "curso": "3r ESO",
          "consigna": "Test básico",
          "respuesta": "Respuesta de prueba rápida"
        }
      ]
    }'
    
    job_response=$(curl -s -X POST $SERVICE_URL/evaluate \
        -H "Content-Type: application/json" \
        -d "$test_data")
    
    if echo "$job_response" | grep -q "job_id"; then
        log_info "✅ Evaluación iniciada correctamente"
        echo "Respuesta: $job_response"
    else
        log_error "❌ Error en evaluación"
        echo "Respuesta: $job_response"
        exit 1
    fi
fi

# 6. Mostrar logs del container
log_info "6. Últimas líneas de logs:"
docker logs --tail 10 $CONTAINER_NAME

log_info "🎉 Quick test completado exitosamente!"
log_info "El servicio está listo para deployment en RunPod.io"
log_info ""
log_info "Para usar el servicio:"
log_info "  Health check: curl $SERVICE_URL/health"
log_info "  API docs: $SERVICE_URL/docs"
log_info ""
log_info "Para deployment en RunPod.io:"
log_info "  docker run --gpus all -p 8000:8000 text-eval-service"