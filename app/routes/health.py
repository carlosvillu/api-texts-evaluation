from fastapi import APIRouter
import torch

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check of the service"""
    from ..main import inference_engine, state_manager

    gpu_available = torch.cuda.is_available()
    model_loaded = inference_engine is not None
    active_jobs = len(state_manager.jobs) if state_manager else 0

    status = "healthy" if (gpu_available and model_loaded) else "unhealthy"

    health_data = {
        "status": status,
        "model_loaded": model_loaded,
        "gpu_available": gpu_available,
        "active_jobs": active_jobs
    }

    # Add GPU memory info if available
    if gpu_available:
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            health_data["gpu_memory_total"] = gpu_props.total_memory
            health_data["gpu_name"] = gpu_props.name
        except Exception as e:
            health_data["gpu_memory_total"] = None
            health_data["gpu_name"] = None
            health_data["gpu_error"] = str(e)

    return health_data