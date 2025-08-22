# Product Requirements Document (PRD)

## Servicio de Inferencia para Evaluación de Textos Educativos - Implementación

### Información del Proyecto

**Proyecto**: Transformación de Notebook a Servicio API  
**Versión**: 1.0 - Proof of Concept  
**Fecha**: Agosto 2025  
**Objetivo**: Convertir `vllm_inference_model.ipynb` en un servicio FastAPI productivo

---

## Fase 1: Preparación del Entorno

**Duración estimada**: 2-3 horas  
**Objetivo**: Crear la estructura base del proyecto y configuración inicial

### 1.1 Estructura del Proyecto

**Tarea**: Crear estructura de directorios

- [x] **1.1.1** Crear subdirectorios:
  - `app/` (código principal)
  - `app/routes/` (endpoints)
  - `app/services/` (lógica de negocio)
  - `app/utils/` (utilidades)
  - `scripts/` (scripts auxiliares)
- [x] **1.1.2** Crear archivos `__init__.py` necesarios

**Criterios de aceptación**:

- ✅ Estructura de directorios creada según especificación
- ✅ Todos los `__init__.py` creados
- ✅ Estructura lista para desarrollo modular

### 1.2 Configuración Base

**Tarea**: Crear archivos de configuración esenciales

- [x] **1.2.1** Crear `requirements.txt` con dependencias:

  ```txt
  fastapi>=0.104.0
  uvicorn[standard]>=0.24.0
  vllm>=0.2.6
  torch>=2.1.0
  pydantic>=2.5.0
  python-multipart>=0.0.6
  datasets>=2.14.0
  ```

- [x] **1.2.2** Crear `.env.example` con variables de entorno:

  ```env
  # Model Configuration
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

  # Performance
  ENABLE_PREFIX_CACHING=true
  BLOCK_SIZE=16
  MAX_NUM_SEQS=128
  COMPILATION_LEVEL=2
  ```

- [x] **1.2.3** Crear `Dockerfile` optimizado para GPU L40:

  ```dockerfile
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

  ENV HF_HOME=/workspace
  ENV TRANSFORMERS_CACHE=/workspace/transformers
  ENV HF_HUB_CACHE=/workspace/hub
  ENV VLLM_USE_PRECOMPILED=1

  EXPOSE 8000

  HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

  CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

- [x] **1.2.4** Crear `docker-compose.yml` para desarrollo:

  ```yaml
  version: "3.8"
  services:
    text-eval-api:
      build: .
      ports:
        - "8000:8000"
      environment:
        - MODEL_NAME=${MODEL_NAME}
        - GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
      volumes:
        - ./app:/app/app
        - huggingface_cache:/workspace
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]

  volumes:
    huggingface_cache:
  ```

**Criterios de aceptación**:

- ✅ Archivos de configuración creados y validados
- ✅ Docker build exitoso (sin ejecutar)
- ✅ Variables de entorno documentadas

**Estado**: ✅ **COMPLETADA** - Fase 1 implementada exitosamente
- Rama `feature/api-implementation` creada
- PR #2: https://github.com/carlosvillu/api-texts-evaluation/pull/2
- Commit: 60d534f - feat: implement Phase 1 - Environment Preparation

---

## Fase 2: Componentes Core

**Duración estimada**: 4-5 horas  
**Objetivo**: Migrar la lógica del notebook a componentes reutilizables

### 2.1 Modelos Pydantic

**Tarea**: Definir esquemas de datos en `app/models.py`

- [x] **2.1.1** Crear modelos de entrada:

  ```python
  from pydantic import BaseModel
  from typing import List, Optional
  from datetime import datetime

  class EvaluationItem(BaseModel):
      id_alumno: str
      curso: str
      consigna: str
      respuesta: str

  class EvaluationRequest(BaseModel):
      items: List[EvaluationItem]
  ```

- [x] **2.1.2** Crear modelos de respuesta:

  ```python
  class EvaluationResult(BaseModel):
      id_alumno: str
      nota: int
      feedback: str

  class JobResponse(BaseModel):
      job_id: str
      total_items: int
      estimated_time_seconds: int
      status: str
      stream_url: str

  class ProgressInfo(BaseModel):
      completed: int
      total: int
      percentage: float
      elapsed_seconds: int
      estimated_remaining_seconds: int
      avg_time_per_item: float
  ```

- [x] **2.1.3** Crear modelos para eventos SSE:

  ```python
  class BatchCompleteEvent(BaseModel):
      event: str = "batch_complete"
      data: dict

  class JobCompleteEvent(BaseModel):
      event: str = "complete"
      data: dict
  ```

**Criterios de aceptación**:

- ✅ Modelos Pydantic definidos y funcionales
- ✅ Validación de datos implementada
- ✅ Esquemas listos para API

### 2.2 Motor de Inferencia

**Tarea**: Migrar código del notebook a `app/services/inference.py`

- [x] **2.2.1** Implementar clase `InferenceEngine`:

  ```python
  import os
  import torch
  import time
  import statistics
  from vllm import LLM, SamplingParams
  from typing import List, AsyncIterator, Dict, Any

  class InferenceEngine:
      def __init__(self, config: dict):
          self.config = config
          self.llm = None
          self.sampling_params = None
          self._initialize_model()

      def _initialize_model(self):
          # Configuración del entorno (del notebook)
          os.environ["VLLM_USE_PRECOMPILED"] = "1"
          torch.cuda.empty_cache()

          # Inicialización del modelo (migrado del notebook)
          self.llm = LLM(
              model=self.config["model_name"],
              gpu_memory_utilization=self.config["gpu_memory_utilization"],
              max_model_len=self.config["max_model_len"],
              tensor_parallel_size=self.config["tensor_parallel_size"],
              dtype=self.config["dtype"],
              enable_prefix_caching=self.config["enable_prefix_caching"],
              disable_log_stats=True,
              block_size=self.config["block_size"],
              max_num_seqs=self.config["max_num_seqs"],
              compilation_config={
                  "level": self.config["compilation_level"],
                  "use_inductor": True,
                  "use_cudagraph": True,
                  "cache_dir": "/tmp/vllm_cache"
              }
          )

          self.sampling_params = SamplingParams(
              temperature=0.1,
              top_p=0.95,
              max_tokens=200,
              skip_special_tokens=True
          )
  ```

- [x] **2.2.2** Migrar función de formateo de prompts:

  ```python
  def _format_prompt(self, item: dict) -> str:
      """Migrado de format_prompts_correctly del notebook"""
      instruction = """Ets una professora experimentada avaluant textos d'estudiants catalans.
  Has d'avaluar amb una nota de 0 a 5 i proporcionar feedback constructiu.

  Escala d'avaluació:
  0 = Molt per sota del nivell
  1 = Per sota del nivell
  2 = Just acceptable
  3 = Nivell esperat
  4 = Per sobre del nivell
  5 = Excel·lent

  Respon NOMÉS amb JSON: {"nota": X, "feedback": "..."}"""

      user_content = f"""{instruction}

  Alumne de {item['curso']} respon a "{item['consigna']}":
  {item['respuesta']}"""

      return f"""<start_of_turn>user
  {user_content}<end_of_turn>
  <start_of_turn>model"""
  ```

- [x] **2.2.3** Implementar procesamiento por batches:

  ```python
  async def process_batch(self, items: List[dict], batch_size: int = 10) -> AsyncIterator[Dict[str, Any]]:
      """Procesar items en batches con yield de resultados"""
      prompts = [self._format_prompt(item) for item in items]

      for i in range(0, len(prompts), batch_size):
          batch_prompts = prompts[i:i+batch_size]
          batch_items = items[i:i+batch_size]
          start_time = time.time()

          outputs = self.llm.generate(batch_prompts, self.sampling_params)

          results = self._parse_outputs(outputs, batch_items)
          elapsed = time.time() - start_time

          yield {
              "batch_results": results,
              "timing": {
                  "batch_time": elapsed,
                  "avg_per_item": elapsed / len(batch_prompts)
              }
          }
  ```

- [x] **2.2.4** Implementar parser de salidas:
  ```python
  def _parse_outputs(self, outputs, items: List[dict]) -> List[dict]:
      """Parsear salidas del modelo a formato estructurado"""
      results = []
      for output, item in zip(outputs, items):
          try:
              generated_text = output.outputs[0].text.strip()
              # Intentar parsear JSON de la respuesta
              import json
              parsed = json.loads(generated_text)
              results.append({
                  "id_alumno": item["id_alumno"],
                  "nota": parsed.get("nota", 0),
                  "feedback": parsed.get("feedback", "Error en evaluación")
              })
          except:
              # Fallback en caso de error
              results.append({
                  "id_alumno": item["id_alumno"],
                  "nota": 0,
                  "feedback": "Error al procesar respuesta del modelo"
              })
      return results
  ```

**Criterios de aceptación**:

- ✅ Código del notebook migrado exitosamente
- ✅ Funcionalidad de inferencia operativa
- ✅ Manejo de errores implementado

### 2.3 Gestor de Estado

**Tarea**: Implementar `StateManager` en `app/services/state.py`

- [x] **2.3.1** Implementar clase base:

  ```python
  import asyncio
  import time
  from uuid import uuid4
  from typing import Dict, Any, Optional

  class StateManager:
      def __init__(self, ttl_seconds: int = 3600):
          self.jobs: Dict[str, Dict[str, Any]] = {}
          self.locks: Dict[str, asyncio.Lock] = {}
          self.ttl = ttl_seconds

      async def create_job(self, job_data: dict) -> str:
          job_id = str(uuid4())
          self.jobs[job_id] = {
              "id": job_id,
              "status": "queued",
              "data": job_data,
              "results": [],
              "created_at": time.time(),
              "processed": 0,
              "total": len(job_data.get("items", [])),
              "start_time": None,
              "end_time": None
          }
          self.locks[job_id] = asyncio.Lock()
          return job_id
  ```

- [x] **2.3.2** Métodos de gestión de trabajos:

  ```python
  async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
      return self.jobs.get(job_id)

  async def update_job_status(self, job_id: str, status: str):
      if job_id in self.jobs:
          async with self.locks[job_id]:
              self.jobs[job_id]["status"] = status
              if status == "processing" and not self.jobs[job_id]["start_time"]:
                  self.jobs[job_id]["start_time"] = time.time()
              elif status == "completed":
                  self.jobs[job_id]["end_time"] = time.time()

  async def add_batch_results(self, job_id: str, results: List[dict]):
      if job_id in self.jobs:
          async with self.locks[job_id]:
              self.jobs[job_id]["results"].extend(results)
              self.jobs[job_id]["processed"] = len(self.jobs[job_id]["results"])
  ```

- [x] **2.3.3** Limpieza automática con TTL:

  ```python
  async def cleanup_expired_jobs(self):
      """Ejecutar periódicamente para limpiar jobs expirados"""
      current_time = time.time()
      expired_jobs = [
          job_id for job_id, job in self.jobs.items()
          if current_time - job["created_at"] > self.ttl
      ]

      for job_id in expired_jobs:
          del self.jobs[job_id]
          if job_id in self.locks:
              del self.locks[job_id]
  ```

**Criterios de aceptación**:

- ✅ StateManager funcional y thread-safe
- ✅ Gestión de trabajos implementada
- ✅ Limpieza automática operativa

### 2.4 Configuración

**Tarea**: Crear `app/config.py` para centralizar configuración

- [x] **2.4.1** Implementar clase de configuración:

  ```python
  import os
  from pydantic_settings import BaseSettings

  class Settings(BaseSettings):
      # Model Configuration
      model_name: str = "carlosvillu/gemma2-9b-teacher-eval-nota-feedback"
      gpu_memory_utilization: float = 0.90
      max_model_len: int = 1024
      tensor_parallel_size: int = 1
      dtype: str = "bfloat16"

      # API Configuration
      host: str = "0.0.0.0"
      port: int = 8000
      batch_size: int = 10
      job_ttl_seconds: int = 3600
      max_concurrent_jobs: int = 5

      # Performance
      enable_prefix_caching: bool = True
      block_size: int = 16
      max_num_seqs: int = 128
      compilation_level: int = 2

      class Config:
          env_file = ".env"

  settings = Settings()
  ```

**Criterios de aceptación**:

- ✅ Configuración centralizada y tipada
- ✅ Variables de entorno integradas
- ✅ Settings accesibles globalmente

### 2.5 Utilidades

**Tarea**: Crear utilidades en `app/utils/timing.py`

- [x] **2.5.1** Implementar calculadora de progreso:

  ```python
  import time
  from typing import Dict, Any

  class ProgressCalculator:
      def __init__(self, total_items: int):
          self.total_items = total_items
          self.start_time = time.time()
          self.processed_items = 0

      def update(self, processed: int) -> Dict[str, Any]:
          self.processed_items = processed
          elapsed = time.time() - self.start_time

          if processed > 0:
              avg_time_per_item = elapsed / processed
              remaining_items = self.total_items - processed
              estimated_remaining = remaining_items * avg_time_per_item
          else:
              avg_time_per_item = 0
              estimated_remaining = 0

          return {
              "completed": processed,
              "total": self.total_items,
              "percentage": round((processed / self.total_items) * 100, 2),
              "elapsed_seconds": round(elapsed, 1),
              "estimated_remaining_seconds": round(estimated_remaining, 1),
              "avg_time_per_item": round(avg_time_per_item, 2)
          }
  ```

**Criterios de aceptación**:

- ✅ Cálculo de progreso preciso
- ✅ Estimaciones de tiempo implementadas
- ✅ Métricas de rendimiento disponibles

**Estado**: ✅ **COMPLETADA** - Fase 2 implementada exitosamente
- Rama `feature/phase-2-core-components` creada
- PR #3: https://github.com/carlosvillu/api-texts-evaluation/pull/3
- Commit: 81cf8c0 - feat: implement Phase 2 - Core Components

---

## Fase 3: API FastAPI

**Duración estimada**: 3-4 horas  
**Objetivo**: Implementar endpoints y funcionalidad web

### 3.1 Aplicación Principal

**Tarea**: Crear `app/main.py` con FastAPI

- [ ] **3.1.1** Configurar aplicación base:

  ```python
  from fastapi import FastAPI, BackgroundTasks
  from contextlib import asynccontextmanager
  import asyncio

  from .services.inference import InferenceEngine
  from .services.state import StateManager
  from .config import settings

  # Variables globales
  inference_engine = None
  state_manager = None

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      # Startup
      global inference_engine, state_manager

      print("🚀 Inicializando servicio de evaluación...")

      # Inicializar componentes
      state_manager = StateManager(ttl_seconds=settings.job_ttl_seconds)
      inference_engine = InferenceEngine({
          "model_name": settings.model_name,
          "gpu_memory_utilization": settings.gpu_memory_utilization,
          "max_model_len": settings.max_model_len,
          "tensor_parallel_size": settings.tensor_parallel_size,
          "dtype": settings.dtype,
          "enable_prefix_caching": settings.enable_prefix_caching,
          "block_size": settings.block_size,
          "max_num_seqs": settings.max_num_seqs,
          "compilation_level": settings.compilation_level
      })

      # Tarea de limpieza periódica
      cleanup_task = asyncio.create_task(cleanup_periodic())

      print("✅ Servicio inicializado correctamente")

      yield

      # Shutdown
      cleanup_task.cancel()
      print("🛑 Servicio detenido")

  app = FastAPI(
      title="Text Evaluation Service",
      description="API para evaluación automatizada de textos educativos",
      version="1.0.0",
      lifespan=lifespan
  )
  ```

- [ ] **3.1.2** Incluir rutas:

  ```python
  from .routes import evaluation, health

  app.include_router(evaluation.router, prefix="", tags=["evaluation"])
  app.include_router(health.router, prefix="", tags=["health"])

  async def cleanup_periodic():
      """Tarea de limpieza periódica"""
      while True:
          await asyncio.sleep(300)  # Cada 5 minutos
          if state_manager:
              await state_manager.cleanup_expired_jobs()
  ```

**Criterios de aceptación**:

- FastAPI configurada y funcional
- Inicialización de componentes exitosa
- Limpieza automática implementada

### 3.2 Endpoints de Evaluación

**Tarea**: Implementar `app/routes/evaluation.py`

- [ ] **3.2.1** Endpoint POST /evaluate:

  ```python
  from fastapi import APIRouter, HTTPException, BackgroundTasks
  from fastapi.responses import StreamingResponse
  import asyncio
  import json

  from ..models import EvaluationRequest, JobResponse
  from ..utils.timing import ProgressCalculator

  router = APIRouter()

  @router.post("/evaluate", response_model=JobResponse)
  async def evaluate_texts(
      request: EvaluationRequest,
      background_tasks: BackgroundTasks
  ):
      """Enviar batch de textos para evaluación"""
      from ..main import state_manager, inference_engine

      if not inference_engine:
          raise HTTPException(status_code=503, detail="Inference engine not ready")

      # Crear job
      job_data = {"items": [item.dict() for item in request.items]}
      job_id = await state_manager.create_job(job_data)

      # Estimar tiempo (aproximación)
      estimated_time = len(request.items) * 2  # 2 segundos por item aprox.

      # Iniciar procesamiento en background
      background_tasks.add_task(process_evaluation_job, job_id)

      return JobResponse(
          job_id=job_id,
          total_items=len(request.items),
          estimated_time_seconds=estimated_time,
          status="queued",
          stream_url=f"/stream/{job_id}"
      )
  ```

- [ ] **3.2.2** Función de procesamiento en background:

  ```python
  async def process_evaluation_job(job_id: str):
      """Procesar job de evaluación en background"""
      from ..main import state_manager, inference_engine

      try:
          job = await state_manager.get_job(job_id)
          if not job:
              return

          await state_manager.update_job_status(job_id, "processing")

          items = job["data"]["items"]
          async for batch_result in inference_engine.process_batch(
              items,
              batch_size=10
          ):
              # Agregar resultados del batch
              await state_manager.add_batch_results(
                  job_id,
                  batch_result["batch_results"]
              )

          await state_manager.update_job_status(job_id, "completed")

      except Exception as e:
          await state_manager.update_job_status(job_id, "error")
          print(f"Error processing job {job_id}: {e}")
  ```

- [ ] **3.2.3** Endpoint GET /stream/{job_id} con SSE:

  ```python
  @router.get("/stream/{job_id}")
  async def stream_results(job_id: str):
      """Stream de resultados via Server-Sent Events"""
      from ..main import state_manager

      async def generate():
          progress_calc = None
          last_processed = 0

          while True:
              job = await state_manager.get_job(job_id)

              if not job:
                  yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                  break

              if not progress_calc:
                  progress_calc = ProgressCalculator(job["total"])

              current_processed = job["processed"]

              # Enviar actualización si hay nuevos resultados
              if current_processed > last_processed:
                  # Obtener nuevos resultados
                  new_results = job["results"][last_processed:current_processed]

                  progress_info = progress_calc.update(current_processed)

                  event_data = {
                      "event": "batch_complete",
                      "data": {
                          "batch_number": (current_processed // 10) + 1,
                          "results": new_results,
                          "progress": progress_info
                      }
                  }

                  yield f"data: {json.dumps(event_data)}\n\n"
                  last_processed = current_processed

              # Verificar si está completo
              if job["status"] == "completed":
                  final_event = {
                      "event": "complete",
                      "data": {
                          "total_processed": current_processed,
                          "total_time_seconds": job.get("end_time", 0) - job.get("start_time", 0),
                          "status": "completed"
                      }
                  }
                  yield f"data: {json.dumps(final_event)}\n\n"
                  break

              elif job["status"] == "error":
                  yield f"data: {json.dumps({'error': 'Processing failed'})}\n\n"
                  break

              # Esperar antes de la siguiente verificación
              await asyncio.sleep(1)

      return StreamingResponse(
          generate(),
          media_type="text/event-stream",
          headers={
              "Cache-Control": "no-cache",
              "Connection": "keep-alive",
          }
      )
  ```

**Criterios de aceptación**:

- Endpoint /evaluate funcional
- Procesamiento en background operativo
- SSE streaming implementado correctamente

### 3.3 Endpoints de Salud

**Tarea**: Implementar `app/routes/health.py`

- [ ] **3.3.1** Health check básico:

  ```python
  from fastapi import APIRouter
  import torch

  router = APIRouter()

  @router.get("/health")
  async def health_check():
      """Health check del servicio"""
      from ..main import inference_engine, state_manager

      gpu_available = torch.cuda.is_available()
      model_loaded = inference_engine is not None
      active_jobs = len(state_manager.jobs) if state_manager else 0

      status = "healthy" if (gpu_available and model_loaded) else "unhealthy"

      return {
          "status": status,
          "model_loaded": model_loaded,
          "gpu_available": gpu_available,
          "active_jobs": active_jobs,
          "gpu_memory": torch.cuda.get_device_properties(0).total_memory if gpu_available else None
      }
  ```

**Criterios de aceptación**:

- Health check operativo
- Información del sistema disponible
- Estado del modelo reportado correctamente

---

## Fase 4: Deployment PoC

**Duración estimada**: 2-3 horas  
**Objetivo**: Preparar el servicio para deployment en RunPod.io

### 4.1 Scripts de Utilidad

**Tarea**: Crear scripts auxiliares

- [ ] **4.1.1** Script de inicio `scripts/start.sh`:

  ```bash
  #!/bin/bash

  echo "🚀 Iniciando Text Evaluation Service..."

  # Verificar GPU
  if ! command -v nvidia-smi &> /dev/null; then
      echo "❌ NVIDIA GPU no detectada"
      exit 1
  fi

  echo "📊 Estado GPU:"
  nvidia-smi

  # Crear directorios necesarios
  mkdir -p /tmp/vllm_cache
  mkdir -p /workspace/transformers
  mkdir -p /workspace/hub

  # Variables de entorno
  export HF_HOME=/workspace
  export TRANSFORMERS_CACHE=/workspace/transformers
  export HF_HUB_CACHE=/workspace/hub
  export VLLM_USE_PRECOMPILED=1

  # Iniciar servicio
  echo "🔥 Iniciando FastAPI..."
  python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
  ```

- [ ] **4.1.2** Cliente de prueba `scripts/test_client.py`:

  ```python
  #!/usr/bin/env python3

  import requests
  import json
  import time
  import sseclient  # pip install sseclient-py

  def test_evaluation_service():
      base_url = "http://localhost:8000"

      # Test data
      test_data = {
          "items": [
              {
                  "id_alumno": "test_001",
                  "curso": "3r ESO",
                  "consigna": "Explica qué és la fotosíntesi",
                  "respuesta": "La fotosíntesi és el procés pel qual les plantes fan menjar amb la llum del sol."
              },
              {
                  "id_alumno": "test_002",
                  "curso": "3r ESO",
                  "consigna": "Descriu el cicle de l'aigua",
                  "respuesta": "L'aigua s'evapora, forma núvols i després plou."
              }
          ]
      }

      print("🧪 Testing Text Evaluation Service")

      # 1. Health check
      print("\n1. Checking health...")
      response = requests.get(f"{base_url}/health")
      print(f"Health: {response.json()}")

      # 2. Submit evaluation
      print("\n2. Submitting evaluation...")
      response = requests.post(f"{base_url}/evaluate", json=test_data)
      job_data = response.json()
      print(f"Job created: {job_data}")

      job_id = job_data["job_id"]

      # 3. Stream results
      print(f"\n3. Streaming results for job {job_id}...")

      stream_url = f"{base_url}/stream/{job_id}"
      response = requests.get(stream_url, stream=True)
      client = sseclient.SSEClient(response)

      for event in client.events():
          if event.data:
              try:
                  data = json.loads(event.data)
                  print(f"Event: {json.dumps(data, indent=2)}")

                  if data.get("event") == "complete":
                      print("✅ Processing completed!")
                      break

              except json.JSONDecodeError:
                  print(f"Raw data: {event.data}")

  if __name__ == "__main__":
      test_evaluation_service()
  ```

**Criterios de aceptación**:

- Scripts ejecutables y funcionales
- Cliente de prueba operativo
- Verificaciones de entorno implementadas

### 4.2 Documentación

**Tarea**: Crear README.md básico

- [ ] **4.2.1** Documentación de deployment:

  ````markdown
  # Text Evaluation Service

  Servicio de evaluación automatizada de textos educativos usando vLLM y FastAPI.

  ## Requisitos

  - NVIDIA GPU con CUDA 12.1+
  - Docker con soporte GPU
  - 24GB VRAM mínimo (optimizado para L40)

  ## Instalación

  1. Clonar repositorio
  2. Copiar `.env.example` a `.env` y configurar variables
  3. Build del contenedor:
     ```bash
     docker build -t text-eval-service .
     ```
  ````

  4. Ejecutar:
     ```bash
     docker run --gpus all -p 8000:8000 text-eval-service
     ```

  ## Uso

  ### Evaluar textos

  ```bash
  curl -X POST "http://localhost:8000/evaluate" \
       -H "Content-Type: application/json" \
       -d @example_request.json
  ```

  ### Stream de resultados

  ```bash
  curl "http://localhost:8000/stream/{job_id}"
  ```

  ## Cliente de prueba

  ```bash
  python scripts/test_client.py
  ```

  ```

  ```

**Criterios de aceptación**:

- Documentación clara y completa
- Instrucciones de deployment válidas
- Ejemplos de uso incluidos

### 4.3 Optimización Final

**Tarea**: Ajustes finales para RunPod.io

- [ ] **4.3.1** Verificar configuración GPU L40:
  - GPU memory utilization: 0.90
  - Compilation level: 2 (balance velocidad/rendimiento)
  - Prefix caching habilitado
  - Batch size: 10 (óptimo para memoria disponible)

- [ ] **4.3.2** Variables de entorno para RunPod:

  ```env
  RUNPOD_GPU_TYPE=L40
  RUNPOD_MEMORY_LIMIT=24GB
  PORT=8000
  ```

- [ ] **4.3.3** Healthcheck robustos:
  - Verificación de carga del modelo
  - Monitoreo de memoria GPU
  - Estado de trabajos activos

**Criterios de aceptación**:

- Configuración optimizada para L40
- Healthchecks robustos implementados
- Servicio listo para production

---

## Criterios de Aceptación General del Proyecto

### Funcionalidad Core

- [ ] ✅ Servicio recibe batches de textos y devuelve evaluaciones
- [ ] ✅ Streaming de resultados via SSE funcional
- [ ] ✅ Gestión de trabajos concurrentes implementada
- [ ] ✅ Health checks operativos

### Performance

- [ ] ✅ Procesamiento ≥5 textos/segundo en GPU L40
- [ ] ✅ Memoria GPU utilizada <22GB para batches de 2000 textos
- [ ] ✅ SSE latency <100ms por evento

### Deployment

- [ ] ✅ Docker container funcional
- [ ] ✅ Variables de entorno configurables
- [ ] ✅ Scripts de inicio y prueba operativos
- [ ] ✅ Documentación completa

### Mantenibilidad

- [ ] ✅ Código modular y bien estructurado
- [ ] ✅ Configuración centralizada
- [ ] ✅ Manejo de errores robusto
- [ ] ✅ Logs informativos

---

## Entregables

1. **Código fuente completo** en estructura definida
2. **Dockerfile** optimizado para GPU L40
3. **Docker-compose.yml** para desarrollo
4. **Scripts de utilidad** (start.sh, test_client.py)
5. **Documentación README.md**
6. **Variables de entorno** configuradas (.env.example)
7. **Cliente de prueba** funcional

---

## Estimación Total: 12-15 horas

Esta es una estimación para un PoC funcional. El enfoque está en migrar exitosamente la funcionalidad del notebook a un servicio web robusto y desplegable.

