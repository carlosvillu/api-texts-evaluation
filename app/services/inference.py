import os
import torch
import time
import json
from vllm import LLM, SamplingParams
from typing import List, AsyncIterator, Dict, Any


class InferenceEngine:
    def __init__(self, config: dict):
        self.config = config
        self.llm = None
        self.sampling_params = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize vLLM model with optimized configuration for L40 GPU"""
        # Environment optimizations (migrated from notebook)
        os.environ["VLLM_USE_PRECOMPILED"] = "1"
        torch.cuda.empty_cache()

        # Model initialization (migrated from notebook)
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

    def _format_prompt(self, item: dict) -> str:
        """Format prompt correctly for Gemma model (migrated from format_prompts_correctly)"""
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

    async def process_batch(self, items: List[dict], batch_size: int = 10) -> AsyncIterator[Dict[str, Any]]:
        """Process items in batches with yield of results"""
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

    def _parse_outputs(self, outputs, items: List[dict]) -> List[dict]:
        """Parse model outputs to structured format"""
        results = []
        for output, item in zip(outputs, items):
            try:
                generated_text = output.outputs[0].text.strip()
                # Try to parse JSON response
                parsed = json.loads(generated_text)
                results.append({
                    "id_alumno": item["id_alumno"],
                    "nota": parsed.get("nota", 0),
                    "feedback": parsed.get("feedback", "Error en evaluación")
                })
            except:
                # Fallback in case of error
                results.append({
                    "id_alumno": item["id_alumno"],
                    "nota": 0,
                    "feedback": "Error al procesar respuesta del modelo"
                })
        return results