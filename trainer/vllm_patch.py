# ===== Patch LRUCache to fix KeyError in vLLM =====
# Ref: https://github.com/Avabowler/vllm/blob/0fd4bcbd7a01a8c0c947e78865e46007f98a7b5f/vllm/utils.py#L272
from vllm.utils import LRUCache


def hooked_touch(self, key):
    try:
        self._LRUCache__order.move_to_end(key)
    except KeyError:
        self._LRUCache__order[key] = None


LRUCache.touch = hooked_touch
