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