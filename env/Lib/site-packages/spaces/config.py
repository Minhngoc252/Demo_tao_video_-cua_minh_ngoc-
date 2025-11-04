"""
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast
from typing import get_args

from typing_extensions import TypeAlias
from typing_extensions import assert_type

from .zero.api import GPUSize


GPUSizeConfig = Literal['auto', 'medium', 'large']


ZEROGPU_HOME = Path.home() / '.zerogpu'


def boolean(value: str | None) -> bool:
    return value is not None and value.lower() in ("1", "t", "true")


class Settings:
    def __init__(self):
        self.zero_gpu = boolean(
            os.getenv('SPACES_ZERO_GPU'))
        self.zero_device_api_url = (
            os.getenv('SPACES_ZERO_DEVICE_API_URL'))
        self.gradio_auto_wrap = boolean(
            os.getenv('SPACES_GRADIO_AUTO_WRAP'))
        self.zero_patch_torch_device = boolean(
            os.getenv('ZERO_GPU_PATCH_TORCH_DEVICE'))
        self.zero_gpu_v2 = boolean(
            os.getenv('ZEROGPU_V2', "true"))
        self.zerogpu_size: GPUSize | Literal['auto'] = cast(GPUSizeConfig,
            os.getenv('ZEROGPU_SIZE', 'large'))
        self.zerogpu_medium_size_threshold = int(
            os.getenv('ZEROGPU_MEDIUM_SIZE_THRESHOLD', 30 * 2**30))
        self.zerogpu_offload_dir = (
            os.getenv('ZEROGPU_OFFLOAD_DIR', str(ZEROGPU_HOME / 'tensors')))
        self.zerogpu_proc_self_cgroup_path = (
            os.getenv('ZEROGPU_PROC_SELF_CGROUP_PATH', '/proc/self/cgroup'))
        self.zerogpu_cuda_device_name = (
            os.getenv('ZEROGPU_CUDA_DEVICE_NAME', "NVIDIA H200 MIG 3g.71gb"))
        self.zerogpu_cuda_total_memory = int(
            os.getenv('ZEROGPU_CUDA_TOTAL_MEMORY', 74625056768))
        self.zerogpu_cuda_reserved_memory = int(
            os.getenv('ZEROGPU_CUDA_RESERVED_MEMORY', 0))
        self.zerogpu_cuda_capability_major = int(
            os.getenv('ZEROGPU_CUDA_CAPABILITY_MAJOR', 9))
        self.zerogpu_cuda_capability_minor = int(
            os.getenv('ZEROGPU_CUDA_CAPABILITY_MINOR', 0))
        self.zerogpu_cuda_multi_processor_count = int(
            os.getenv('ZEROGPU_CUDA_MULTI_PROCESSOR_COUNT', 60))


Config = Settings()


if TYPE_CHECKING:
    assert_type(Config.zerogpu_size, GPUSizeConfig)


if Config.zero_gpu:
    assert Config.zero_device_api_url is not None, (
        'SPACES_ZERO_DEVICE_API_URL env must be set '
        'on ZeroGPU Spaces (identified by SPACES_ZERO_GPU=true)'
    )
    assert Config.zerogpu_size in (values := get_args(GPUSizeConfig)), (
        f"ZEROGPU_SIZE should be one of {', '.join(values)}"
    )
