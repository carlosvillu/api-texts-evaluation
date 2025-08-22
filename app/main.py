from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

from .services.inference import InferenceEngine
from .services.state import StateManager
from .config import settings

# Global instances
inference_engine = None
state_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global inference_engine, state_manager

    print("🚀 Inicializando servicio de evaluación...")

    # Initialize StateManager
    state_manager = StateManager(ttl_seconds=settings.job_ttl_seconds)
    
    # Initialize InferenceEngine
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

    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(cleanup_periodic())

    print("✅ Servicio inicializado correctamente")

    yield

    # Shutdown
    cleanup_task.cancel()
    print("🛑 Servicio detenido")


async def cleanup_periodic():
    """Periodic cleanup task for expired jobs"""
    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)
        if state_manager:
            await state_manager.cleanup_expired_jobs()


app = FastAPI(
    title="Text Evaluation Service",
    description="API para evaluación automatizada de textos educativos",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
from .routes import evaluation, health

app.include_router(evaluation.router, prefix="", tags=["evaluation"])
app.include_router(health.router, prefix="", tags=["health"])