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


class BatchCompleteEvent(BaseModel):
    event: str = "batch_complete"
    data: dict


class JobCompleteEvent(BaseModel):
    event: str = "complete"
    data: dict