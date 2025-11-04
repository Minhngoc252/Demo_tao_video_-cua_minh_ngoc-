"""
"""

import mmap
from functools import wraps

import torch


@wraps(torch.empty_like)
def empty_like_raw_alloc(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    empty = torch.empty_like(tensor, **{**kwargs, 'requires_grad': False})
    if (nbytes := empty.untyped_storage().nbytes()) > 0:
        buffer = mmap.mmap(-1, nbytes, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        buffer = torch.frombuffer(buffer, dtype=torch.uint8)
        empty.set_(buffer.untyped_storage(), 0, empty.shape, empty.stride())
    empty.requires_grad_(kwargs.get('requires_grad', False))
    return empty
