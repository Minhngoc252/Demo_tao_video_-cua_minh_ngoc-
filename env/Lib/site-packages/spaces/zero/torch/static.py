"""
"""

from types import SimpleNamespace as _SimpleNamespace

import torch as _torch

from ...config import Config


def compute_base_free_memory(total_memory: int):
    pytorch_base_memory = 309002240 # TODO: fine-grain per: torch-version x GPU(-MIG) model
    return total_memory - pytorch_base_memory - Config.zerogpu_cuda_reserved_memory

CUDA_DEVICE_NAME = Config.zerogpu_cuda_device_name
CUDA_TOTAL_MEMORY = Config.zerogpu_cuda_total_memory
CUDA_MEM_GET_INFO = (compute_base_free_memory(CUDA_TOTAL_MEMORY), CUDA_TOTAL_MEMORY)
CUDA_DEVICE_CAPABILITY = (Config.zerogpu_cuda_capability_major, Config.zerogpu_cuda_capability_minor)
CUDA_DEVICE_PROPERTIES = _SimpleNamespace(
    name=CUDA_DEVICE_NAME,
    major=CUDA_DEVICE_CAPABILITY[0],
    minor=CUDA_DEVICE_CAPABILITY[1],
    total_memory=CUDA_TOTAL_MEMORY,
    multi_processor_count=Config.zerogpu_cuda_multi_processor_count,
    # TODO: L2_cache_size
)

if _torch.version.cuda.startswith("12."): # pyright: ignore [reportAttributeAccessIssue, reportOptionalMemberAccess]
    CUDA_MEMORY_STATS_AS_NESTED_DICT = {
        "num_alloc_retries": 0,
        "num_ooms": 0,
        "max_split_size": -1,
        "num_sync_all_streams": 0,
        "num_device_alloc": 0,
        "num_device_free": 0,
        "allocation": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "segment": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "active": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "inactive_split": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "allocated_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "reserved_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "active_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "inactive_split_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "requested_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "oversize_allocations": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        "oversize_segments": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
    }
else: # pragma: no cover (CUDA 11)
    CUDA_MEMORY_STATS_AS_NESTED_DICT = {
        "num_alloc_retries": 0,
        "num_ooms": 0,
        "max_split_size": -1,
        "allocation": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "segment": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "active": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "inactive_split": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "allocated_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "reserved_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "active_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "inactive_split_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "requested_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "oversize_allocations": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        "oversize_segments": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
    }
