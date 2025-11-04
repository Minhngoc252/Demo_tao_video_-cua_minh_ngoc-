"""
"""
from __future__ import annotations

import sys
from types import FunctionType
from typing import TYPE_CHECKING
from typing import Callable


if sys.version_info.minor < 8: # pragma: no cover
    raise RuntimeError("Importing PySpaces requires Python 3.8+")


# Prevent gradio from importing spaces
if (gr := sys.modules.get('gradio')) is not None: # pragma: no cover
    try:
        gr.Blocks
    except AttributeError:
        raise ImportError


from .zero.decorator import GPU
from .gradio import gradio_auto_wrap
from .gradio import disable_gradio_auto_wrap
from .gradio import enable_gradio_auto_wrap


class LazyImported:
    def __init__(self, import_fn: Callable[[], FunctionType]):
        self.import_fn = import_fn
    def __call__(self, *args, **kwargs):
        return self.import_fn()(*args, **kwargs)
    @property
    def __wrapped__(self):
        return self.import_fn()
    @property
    def __doc__(self): # pyright: ignore[reportIncompatibleVariableOverride]
        return self.import_fn().__doc__
    @property
    def __class__(self): # pyright: ignore[reportIncompatibleMethodOverride]
        return FunctionType


def _aoti_capture():
    from .zero.torch.aoti import aoti_capture
    return aoti_capture

def _aoti_compile():
    from .zero.torch.aoti import aoti_compile
    return aoti_compile

def _aoti_apply():
    from .zero.torch.aoti import aoti_apply
    return aoti_apply

def _aoti_blocks_load():
    from .zero.torch.aoti import aoti_blocks_load
    return aoti_blocks_load


aoti_capture = LazyImported(_aoti_capture)
aoti_compile = LazyImported(_aoti_compile)
aoti_apply = LazyImported(_aoti_apply)
aoti_blocks_load = LazyImported(_aoti_blocks_load)


if TYPE_CHECKING:
    from .zero.torch.aoti import aoti_capture
    from .zero.torch.aoti import aoti_compile
    from .zero.torch.aoti import aoti_apply
    from .zero.torch.aoti import aoti_blocks_load


__all__ = [
    'GPU',
    'gradio_auto_wrap',
    'disable_gradio_auto_wrap',
    'enable_gradio_auto_wrap',
    'aoti_capture',
    'aoti_compile',
    'aoti_apply',
    'aoti_blocks_load',
]
