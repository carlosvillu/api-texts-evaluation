from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio
import json

from ..models import EvaluationRequest, JobResponse
from ..utils.timing import ProgressCalculator
from ..config import settings

router = APIRouter()


@router.post("/evaluate", response_model=JobResponse)
async def evaluate_texts(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """Submit batch of texts for evaluation"""
    from ..main import state_manager, inference_engine

    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not ready")

    # Create job
    job_data = {"items": [item.dict() for item in request.items]}
    job_id = await state_manager.create_job(job_data)

    # Estimate time using configurable estimation per item
    estimated_time = len(request.items) * settings.time_estimation_per_item

    # Start background processing
    background_tasks.add_task(process_evaluation_job, job_id)

    return JobResponse(
        job_id=job_id,
        total_items=len(request.items),
        estimated_time_seconds=estimated_time,
        status="queued",
        stream_url=f"/stream/{job_id}"
    )


async def process_evaluation_job(job_id: str):
    """Process evaluation job in background"""
    from ..main import state_manager, inference_engine

    try:
        job = await state_manager.get_job(job_id)
        if not job:
            return

        await state_manager.update_job_status(job_id, "processing")

        items = job["data"]["items"]
        async for batch_result in inference_engine.process_batch(
            items,
            batch_size=settings.batch_size
        ):
            # Add batch results to job
            await state_manager.add_batch_results(
                job_id,
                batch_result["batch_results"]
            )

        await state_manager.update_job_status(job_id, "completed")

    except Exception as e:
        await state_manager.update_job_status(job_id, "error")
        print(f"Error processing job {job_id}: {e}")


@router.get("/stream/{job_id}")
async def stream_results(job_id: str):
    """Stream results via Server-Sent Events"""
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

            # Send update if there are new results
            if current_processed > last_processed:
                # Get new results
                new_results = job["results"][last_processed:current_processed]

                progress_info = progress_calc.update(current_processed)

                event_data = {
                    "event": "batch_complete",
                    "data": {
                        "batch_number": (current_processed // settings.batch_size) + 1,
                        "results": new_results,
                        "progress": progress_info
                    }
                }

                yield f"data: {json.dumps(event_data)}\n\n"
                last_processed = current_processed

            # Check if completed
            if job["status"] == "completed":
                start_time = job.get("start_time")
                end_time = job.get("end_time")
                total_time = (end_time - start_time) if (start_time and end_time) else 0
                final_event = {
                    "event": "complete",
                    "data": {
                        "total_processed": current_processed,
                        "total_time_seconds": total_time,
                        "status": "completed"
                    }
                }
                yield f"data: {json.dumps(final_event)}\n\n"
                break

            elif job["status"] == "error":
                yield f"data: {json.dumps({'error': 'Processing failed'})}\n\n"
                break

            # Wait before next check
            await asyncio.sleep(settings.sse_polling_interval)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )