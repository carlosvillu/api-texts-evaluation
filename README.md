# Text Evaluation Service

Servicio de evaluación automatizada de textos educativos optimizado para deployment en **RunPod.io** usando FastAPI, vLLM y el modelo `carlosvillu/gemma2-9b-teacher-eval-nota-feedback`.

## 🚀 Quick Start para RunPod.io

### Comando de Deployment (Un solo paso)

```bash
docker run --gpus all -p 8000:8000 text-eval-service
```

Eso es todo. El servicio estará disponible en `http://localhost:8000` con:
- ✅ Modelo cargado automáticamente
- ✅ GPU L40 configurada optimizadamente 
- ✅ Variables de entorno preconfiguradas
- ✅ Health check en `/health`

## 📋 Requisitos RunPod.io

- **GPU**: NVIDIA L40 (24GB VRAM) recomendada
- **Memoria**: Mínimo 20GB VRAM disponible
- **Docker**: Con soporte `--gpus all`
- **Red**: Acceso a Hugging Face para descarga inicial del modelo

## 🔧 Build del Container

```bash
# Clonar repositorio
git clone https://github.com/carlosvillu/api-texts-evaluation.git
cd api-texts-evaluation

# Build del container
docker build -t text-eval-service .

# Ejecutar con GPU
docker run --gpus all -p 8000:8000 text-eval-service
```

## ⚡ Configuración GPU L40 Optimizada

El servicio incluye configuración preoptimizada para GPU L40:

```env
GPU_MEMORY_UTILIZATION=0.90    # 21.6GB de 24GB
BATCH_SIZE=10                  # Óptimo para L40
MAX_MODEL_LEN=1024            # Balance memoria/velocidad
COMPILATION_LEVEL=2           # Optimización CUDA
ENABLE_PREFIX_CACHING=true    # Cache para prompts similares
```

## 🧪 Testing del Servicio

### Health Check Rápido
```bash
# Verificar que todo funciona
curl http://localhost:8000/health

# Respuesta esperada:
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "active_jobs": 0
}
```

### Test de Evaluación Básico
```bash
# Usar cliente de prueba incluido
python scripts/test_client.py --test basic

# O test completo
python scripts/test_client.py --test full
```

### Test Manual con curl
```bash
# 1. Enviar evaluación
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id_alumno": "TEST_001",
        "curso": "3r ESO",
        "consigna": "Explica la fotosíntesi",
        "respuesta": "La fotosíntesi és quan les plantes fan menjar amb la llum del sol."
      }
    ]
  }'

# 2. Recibir job_id y hacer stream
curl http://localhost:8000/stream/{job_id}
```

## 📊 API Endpoints

### `POST /evaluate`
Envía batch de textos para evaluación asíncrona.

**Request:**
```json
{
  "items": [
    {
      "id_alumno": "A001",
      "curso": "3r ESO", 
      "consigna": "Explica què és la fotosíntesi",
      "respuesta": "La fotosíntesi és quan les plantes..."
    }
  ]
}
```

**Response:**
```json
{
  "job_id": "uuid-v4",
  "total_items": 1,
  "estimated_time_seconds": 2,
  "status": "queued",
  "stream_url": "/stream/uuid-v4"
}
```

### `GET /stream/{job_id}`
Stream SSE de resultados procesados en tiempo real.

**Event Format:**
```json
{
  "event": "batch_complete",
  "data": {
    "batch_number": 1,
    "results": [
      {
        "id_alumno": "A001",
        "nota": 4,
        "feedback": "Bon treball per a 3r ESO. La respuesta demostra..."
      }
    ],
    "progress": {
      "completed": 1,
      "total": 1,
      "percentage": 100.0,
      "elapsed_seconds": 2.1,
      "estimated_remaining_seconds": 0,
      "avg_time_per_item": 2.1
    }
  }
}
```

### `GET /health`
Health check con estado del modelo y GPU.

## 🐳 Docker Configuration

### Variables de Entorno Soportadas

```bash
# Modelo y GPU
MODEL_NAME=carlosvillu/gemma2-9b-teacher-eval-nota-feedback
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=1024
TENSOR_PARALLEL_SIZE=1
DTYPE=bfloat16

# API Configuration  
HOST=0.0.0.0
PORT=8000
BATCH_SIZE=10
JOB_TTL_SECONDS=3600
MAX_CONCURRENT_JOBS=5

# Performance Tuning
ENABLE_PREFIX_CACHING=true
BLOCK_SIZE=16
MAX_NUM_SEQS=128
COMPILATION_LEVEL=2
VLLM_USE_PRECOMPILED=1

# Service Configuration
CLEANUP_INTERVAL_SECONDS=300
TIME_ESTIMATION_PER_ITEM=2
SSE_POLLING_INTERVAL=1

# RunPod Specific
RUNPOD_GPU_TYPE=L40
RUNPOD_MEMORY_LIMIT=24GB
```

### Dockerfile Optimizado

El `Dockerfile` incluido está preoptimizado para RunPod.io:

- ✅ CUDA 12.1 runtime
- ✅ Python 3.11 
- ✅ Dependencias preinstaladas (vLLM, FastAPI, etc.)
- ✅ Caché de modelos configurado
- ✅ Health check integrado
- ✅ Script de inicio optimizado

## 📈 Métricas de Performance

### Objetivos en GPU L40
- **Throughput**: ≥5 inferencias/segundo
- **Latencia P95**: <3 segundos por inferencia
- **Memory Footprint**: <22GB para batches de 2000 textos
- **GPU Utilization**: >80% durante procesamiento
- **SSE Latency**: <100ms por evento

### Monitoreo
```bash
# Memoria GPU en tiempo real
watch -n 1 nvidia-smi

# Estado del servicio
curl -s http://localhost:8000/health | jq

# Jobs activos
curl -s http://localhost:8000/health | jq '.active_jobs'
```

## 🔍 Troubleshooting RunPod.io

### GPU no detectada
```bash
# Verificar NVIDIA runtime
nvidia-smi

# Verificar Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### Modelo no se carga
```bash
# Verificar logs del container
docker logs text-eval-service

# Verificar conectividad Hugging Face
curl -I https://huggingface.co

# Limpiar caché y reintentar
docker run --gpus all -p 8000:8000 \
  -e HF_HUB_CACHE=/tmp/cache \
  text-eval-service
```

### Out of Memory
```bash
# Reducir GPU memory utilization
docker run --gpus all -p 8000:8000 \
  -e GPU_MEMORY_UTILIZATION=0.80 \
  -e BATCH_SIZE=5 \
  text-eval-service
```

### Puertos ocupados
```bash
# Usar puerto alternativo
docker run --gpus all -p 8080:8000 text-eval-service

# O especificar puerto interno
docker run --gpus all -p 8080:8080 \
  -e PORT=8080 \
  text-eval-service
```

## 📁 Estructura del Proyecto

```
├── app/
│   ├── main.py              # FastAPI app principal
│   ├── config.py            # Settings con Pydantic
│   ├── models.py            # Esquemas de datos
│   ├── routes/
│   │   ├── evaluation.py    # POST /evaluate, GET /stream/{job_id}
│   │   └── health.py        # GET /health
│   ├── services/
│   │   ├── inference.py     # Motor vLLM (migrado de notebook)
│   │   └── state.py         # Gestión de jobs en memoria
│   └── utils/
│       └── timing.py        # Calculadora de progreso
├── scripts/
│   ├── start.sh            # Script inicio optimizado RunPod.io
│   └── test_client.py      # Cliente prueba con SSE
├── Dockerfile              # Optimizado para GPU L40
├── docker-compose.yml      # Para desarrollo local
├── requirements.txt        # Dependencias Python
├── .env.example           # Variables de entorno
├── prd.md                 # Product Requirements Document
└── vllm_inference_model.ipynb  # Notebook original
```

## 🎯 Flujo de Trabajo Típico

1. **Cliente** envía `POST /evaluate` con batch de textos
2. **API** crea `job_id` y responde inmediatamente  
3. **InferenceEngine** procesa en background por batches de 10
4. **Cliente** conecta a `/stream/{job_id}` para recibir resultados
5. **API** envía eventos SSE cada batch completado
6. **Evento final** "complete" indica terminación
7. **StateManager** limpia job después de TTL (1h)

## 🧬 Modelo de Evaluación

### Modelo Utilizado
`carlosvillu/gemma2-9b-teacher-eval-nota-feedback` - Modelo Gemma2 fine-tuned para evaluación de textos educativos en catalán.

### Formato de Evaluación
- **Nota**: Escala 0-5 (0=Molt per sota, 5=Excel·lent)
- **Feedback**: Comentario constructivo en catalán adaptado al nivel educativo
- **Contexto**: Considera curso del alumno y complejidad de la consigna

### Ejemplo de Respuesta
```json
{
  "id_alumno": "A001",
  "nota": 4,
  "feedback": "Bon treball per a 3r ESO. La respuesta demostra una comprensió sólida del concepte de fotosíntesi. Supera les expectatives en explicar la importància per a la vida. Per millorar, podries afegir més detalls sobre els components necessaris (clorofil·la, ATP, etc.)."
}
```

## 🚨 Limitaciones v1.0

- Sin autenticación/autorización
- Sin persistencia permanente (solo memoria con TTL 1h)  
- Sin gestión avanzada de errores
- Sin métricas detalladas (Prometheus)
- Una única versión del modelo
- Sin tests automatizados
- Servicio interno sin rate limiting

## 📞 Soporte

Para issues específicos del deployment:
- **RunPod.io**: Verificar configuración GPU y logs del container
- **Modelo**: Comprobar acceso a Hugging Face Hub
- **Performance**: Ajustar `GPU_MEMORY_UTILIZATION` y `BATCH_SIZE`

Ver logs detallados:
```bash
docker logs text-eval-service --follow
```

---

**¿Todo listo para evaluar textos?** 
```bash
docker run --gpus all -p 8000:8000 text-eval-service
```