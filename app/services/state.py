import asyncio
import time
from uuid import uuid4
from typing import Dict, Any, Optional, List


class StateManager:
    def __init__(self, ttl_seconds: int = 3600):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.ttl = ttl_seconds

    async def create_job(self, job_data: dict) -> str:
        """Create a new job and return job_id"""
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

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by job_id"""
        return self.jobs.get(job_id)

    async def update_job_status(self, job_id: str, status: str):
        """Update job status with timing information"""
        if job_id in self.jobs:
            async with self.locks[job_id]:
                self.jobs[job_id]["status"] = status
                if status == "processing" and not self.jobs[job_id]["start_time"]:
                    self.jobs[job_id]["start_time"] = time.time()
                elif status == "completed":
                    self.jobs[job_id]["end_time"] = time.time()

    async def add_batch_results(self, job_id: str, results: List[dict]):
        """Add batch results to job"""
        if job_id in self.jobs:
            async with self.locks[job_id]:
                self.jobs[job_id]["results"].extend(results)
                self.jobs[job_id]["processed"] = len(self.jobs[job_id]["results"])

    async def cleanup_expired_jobs(self):
        """Remove expired jobs based on TTL"""
        current_time = time.time()
        expired_jobs = [
            job_id for job_id, job in self.jobs.items()
            if current_time - job["created_at"] > self.ttl
        ]

        for job_id in expired_jobs:
            del self.jobs[job_id]
            if job_id in self.locks:
                del self.locks[job_id]