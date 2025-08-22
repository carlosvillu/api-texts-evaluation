import time
from typing import Dict, Any


class ProgressCalculator:
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.start_time = time.time()
        self.processed_items = 0

    def update(self, processed: int) -> Dict[str, Any]:
        """Update progress and calculate metrics"""
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