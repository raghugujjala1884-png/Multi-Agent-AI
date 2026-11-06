import time
import json


class ToolCache:

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _create_key(self, tool_name: str, arguments: dict) -> str:
        normalized_arguments = json.dumps(
            arguments,
            sort_keys=True
        )

        return f"{tool_name}:{normalized_arguments}"

    def get(self, tool_name: str, arguments: dict):
        key = self._create_key(
            tool_name,
            arguments
        )

        if key not in self.cache:
            self.misses += 1
            return None

        cached_item = self.cache[key]

        age = time.time() - cached_item["timestamp"]

        if age > self.ttl_seconds:
            del self.cache[key]
            self.misses += 1
            return None

        self.hits += 1

        return cached_item["result"]

    def set(
        self,
        tool_name: str,
        arguments: dict,
        result
    ):
        key = self._create_key(
            tool_name,
            arguments
        )

        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }

    def clear(self):
        self.cache.clear()

    def stats(self):
        total_requests = self.hits + self.misses

        hit_rate = (
            self.hits / total_requests * 100
            if total_requests > 0
            else 0
        )

        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.cache),
            "hit_rate": round(hit_rate, 2)
        }