# Text Evaluation Service API

Servicio de evaluación automatizada de textos educativos usando FastAPI y vLLM con el modelo `carlosvillu/gemma2-9b-teacher-eval-nota-feedback`.

## Características

- **Procesamiento en batch**: Evalúa hasta 2000 textos por petición
- **Streaming en tiempo real**: Server-Sent Events (SSE) para feedback inmediato
- **Optimizado para GPU**: Configurado para NVIDIA L40 (24GB VRAM)
- **Feedback en catalán**: Evaluación y comentarios adaptados al contexto educativo catalán

## Tecnologías

- **FastAPI**: Framework web asíncrono
- **vLLM**: Motor de inferencia optimizado para LLMs
- **Docker**: Containerización con soporte GPU
- **Pydantic**: Validación de datos

## Inicio Rápido

### Requisitos
- Python 3.11+
- CUDA-compatible GPU (recomendado: ≥16GB VRAM)
- Docker con soporte NVIDIA

### Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/carlosvillu/api-texts-evaluation.git
cd api-texts-evaluation

# Configurar entorno
cp .env.example .env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Ejecutar servicio
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build y ejecutar con GPU
docker build -t text-eval-service .
docker run --gpus all -p 8000:8000 --env-file .env text-eval-service
```

## Uso de la API

### Evaluar textos

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "id_alumno": "A001",
        "curso": "3r ESO",
        "consigna": "Explica la fotosíntesi",
        "respuesta": "La fotosíntesi és quan les plantes fan menjar amb llum..."
      }
    ]
  }'
```

### Streaming de resultados

```javascript
const eventSource = new EventSource(`http://localhost:8000/stream/${job_id}`);
eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Progreso:', data.progress.percentage + '%');
  console.log('Resultados:', data.results);
};
```

## Estructura del Proyecto

```
├── app/
│   ├── main.py              # FastAPI app principal
│   ├── config.py            # Configuración
│   ├── models.py            # Modelos Pydantic
│   ├── routes/              # Endpoints de la API
│   ├── services/            # Lógica de negocio
│   └── utils/               # Utilidades
├── scripts/
│   ├── start.sh            # Script de inicio para RunPod.io
│   └── test_client.py      # Cliente de prueba
├── vllm_inference_model.ipynb  # Notebook de referencia
├── project-definition.md   # Documentación técnica detallada
└── README.md
```

## Configuración

Variables principales en `.env`:

```env
MODEL_NAME=carlosvillu/gemma2-9b-teacher-eval-nota-feedback
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=1024
BATCH_SIZE=10
PORT=8000
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Documentación

- **PRD completo**: Ver `project-definition.md`
- **API docs**: http://localhost:8000/docs (cuando el servicio esté ejecutándose)
- **Notebook original**: `vllm_inference_model.ipynb`

## Performance

- **Throughput objetivo**: ≥5 inferencias/segundo
- **Latencia P95**: <3 segundos por inferencia
- **Memoria GPU**: ~17GB para modelo Gemma2-9B
- **Batch size recomendado**: 10 items

## Limitaciones (v1.0)

- Sin autenticación/autorización
- Sin persistencia permanente (solo memoria)
- Un único modelo de evaluación
- Servicio interno, sin rate limiting

## Contribuir

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## Licencia

MIT