# Text Evaluation Service - Claude Context

## Principios de Desarrollo para Claude Code

Claude Code debe comportarse como un **programador pragmático** que aplica los principios YAGNI (You Aren't Gonna Need It) y KISS (Keep It Simple, Stupid). Esto significa:

### Desarrollo Pragmático
- **Evitar código verboso**: No crear clases enormes con métodos innecesarios
- **Aplicar YAGNI**: Solo implementar lo que se necesita para la feature actual
- **Aplicar KISS**: Mantener el código simple y directo
- **Foco en la feature**: Todos los métodos de una clase deben contribuir directamente a la funcionalidad en desarrollo
- **Evitar sobre-ingeniería**: No anticipar requisitos futuros que no están definidos

### Reglas de Código
- Crear clases con muchos métodos solo si todos contribuyen a la feature actual
- Evitar abstracciones prematuras
- Priorizar legibilidad sobre patrones complejos
- Implementar solo lo mínimo viable para completar la tarea

## Descripción del Proyecto
Servicio de evaluación automatizada de textos educativos que transforma un notebook Jupyter en una API productiva usando FastAPI y vLLM. El modelo `carlosvillu/gemma2-9b-teacher-eval-nota-feedback` evalúa textos de estudiantes catalanes proporcionando notas (0-5) y feedback constructivo.

## Estructura del Proyecto
```
text-evaluation-service/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
├── prd.md                    # Product Requirements Document
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app principal
│   ├── config.py            # Configuración con Pydantic Settings
│   ├── models.py            # Modelos Pydantic (EvaluationItem, JobResponse, etc.)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── evaluation.py   # POST /evaluate, GET /stream/{job_id}
│   │   └── health.py       # GET /health
│   ├── services/
│   │   ├── __init__.py
│   │   ├── inference.py    # InferenceEngine (migrado del notebook)
│   │   ├── state.py        # StateManager (gestión de jobs en memoria)
│   │   └── formatter.py    # Utilidades de formato de prompts
│   └── utils/
│       ├── __init__.py
│       └── timing.py       # ProgressCalculator, métricas de rendimiento
├── scripts/
│   ├── start.sh            # Script de inicio para RunPod.io
│   └── test_client.py      # Cliente de prueba con SSE
└── vllm_inference_model.ipynb  # Notebook original (referencia)
```

## Tecnologías Utilizadas
- **FastAPI**: Framework web asíncrono de alto rendimiento
- **vLLM**: Motor de inferencia optimizado para LLMs
- **Pydantic**: Validación de datos y settings
- **Server-Sent Events (SSE)**: Streaming de resultados en tiempo real
- **Docker**: Containerización con soporte GPU
- **NVIDIA CUDA**: Aceleración GPU (optimizado para L40 24GB)

## APIs Principales

### POST /evaluate
Envía batch de textos para evaluación asíncrona.
```json
{
  "items": [
    {
      "id_alumno": "A001",
      "curso": "3r ESO", 
      "consigna": "Explica la fotosíntesi",
      "respuesta": "La fotosíntesi és quan les plantes..."
    }
  ]
}
```

### GET /stream/{job_id}
Stream SSE de resultados procesados en tiempo real.

### GET /health  
Health check con estado del modelo y GPU.

## Variables de Entorno Importantes
```bash
# Modelo y GPU
MODEL_NAME=carlosvillu/gemma2-9b-teacher-eval-nota-feedback
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=1024
DTYPE=bfloat16

# API
HOST=0.0.0.0
PORT=8000
BATCH_SIZE=10
JOB_TTL_SECONDS=3600

# Performance
ENABLE_PREFIX_CACHING=true
COMPILATION_LEVEL=2
VLLM_USE_PRECOMPILED=1
```

## Comandos Principales

### Desarrollo Local
```bash
# Setup inicial
cp .env.example .env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Ejecutar servicio
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Pruebas
python scripts/test_client.py
```

### Docker
```bash
# Build
docker build -t text-eval-service .

# Run con GPU
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  text-eval-service

# Con docker-compose
docker-compose up --build
```

### Deployment en RunPod.io
```bash
# Usar script de inicio optimizado
chmod +x scripts/start.sh
./scripts/start.sh

# Verificar estado
curl http://localhost:8000/health
```

## Patrones de Código

### Formato de Prompts (migrado del notebook)
```python
# Formato Gemma con instrucciones en catalán
prompt = f"""<start_of_turn>user
Ets una professora experimentada avaluant textos d'estudiants catalans.
Has d'avaluar amb una nota de 0 a 5 i proporcionar feedback constructiu.

Alumne de {curso} respon a "{consigna}":
{respuesta}<end_of_turn>
<start_of_turn>model"""
```

### Respuesta Esperada del Modelo
```json
{"nota": 4, "feedback": "Bon treball per a 3r ESO. Supera les expectatives..."}
```

### Gestión de Estado Asíncrono
```python
# StateManager con locks por job_id
async with self.locks[job_id]:
    self.jobs[job_id]["results"].extend(batch_results)
    self.jobs[job_id]["processed"] = len(self.jobs[job_id]["results"])
```

### Server-Sent Events
```python
# Generator para streaming
async def generate():
    while True:
        # Verificar progreso
        yield f"data: {json.dumps(event_data)}\n\n"
        await asyncio.sleep(1)
```

## Configuración GPU Optimizada (L40 24GB)
- **GPU Memory Utilization**: 0.90 (21.6GB)  
- **Batch Size**: 10 items por batch
- **Max Model Length**: 1024 tokens
- **Compilation Level**: 2 (balance velocidad/memoria)
- **Prefix Caching**: Habilitado para prompts similares
- **Block Size**: 16 (optimizado para L40)

## Métricas de Performance Objetivo
- **Throughput**: ≥5 inferencias/segundo
- **Latencia P95**: <3 segundos por inferencia  
- **Memory Footprint**: <20GB RAM para 2000 textos
- **GPU Utilization**: >80% durante procesamiento
- **SSE Latency**: <100ms por evento

## Flujo de Trabajo Típico
1. Cliente envía POST /evaluate con batch de textos
2. API crea job_id y responde inmediatamente  
3. InferenceEngine procesa en background por batches de 10
4. Cliente conecta a /stream/{job_id} para recibir resultados
5. API envía eventos SSE cada batch completado
6. Evento final "complete" indica terminación
7. StateManager limpia job después de TTL

## Debugging y Logs
```bash
# Logs del contenedor
docker logs text-eval-service

# Memoria GPU en tiempo real
watch -n 1 nvidia-smi

# Health check
curl -s http://localhost:8000/health | jq

# Test de carga
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

## Limitaciones del PoC
- Sin autenticación/autorización
- Sin persistencia permanente (solo memoria)
- Sin gestión avanzada de errores
- Sin métricas detalladas (Prometheus)
- Una única versión del modelo
- Sin tests automatizados

## Componentes Clave del Código

### InferenceEngine (services/inference.py)
Migra la lógica del notebook `vllm_inference_model.ipynb`:
- Inicialización optimizada de vLLM para L40
- Formato de prompts específico para Gemma
- Procesamiento por batches asíncrono
- Parsing de respuestas JSON del modelo

### StateManager (services/state.py) 
- Gestión de jobs en memoria con TTL
- Locks asíncronos por job_id
- Limpieza automática de jobs expirados
- Thread-safe para concurrencia

### FastAPI App (main.py)
- Lifespan events para inicialización
- Background tasks para procesamiento
- SSE streaming nativo
- Manejo de errores robusto

## Comandos de Mantenimiento
```bash
# Limpiar caché GPU
python -c "import torch; torch.cuda.empty_cache()"

# Monitorear jobs activos
curl -s http://localhost:8000/health | jq '.active_jobs'

# Reiniciar servicio
docker-compose restart text-eval-api

# Ver logs en tiempo real  
docker-compose logs -f text-eval-api
```