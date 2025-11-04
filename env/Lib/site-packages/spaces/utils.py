"""
"""
from __future__ import annotations

import sys
from functools import partial

import multiprocessing
from multiprocessing.queues import SimpleQueue as _SimpleQueue
from pickle import PicklingError
from typing import Callable
from typing import TypeVar


GRADIO_VERSION_ERROR_MESSAGE = "Make sure Gradio version is at least 3.46"


T = TypeVar('T')


if sys.version_info.minor < 9: # pragma: no cover
    _SimpleQueue.__class_getitem__ = classmethod(lambda cls, _: cls) # type: ignore

class SimpleQueue(_SimpleQueue[T]):
    def __init__(self, *args):
        super().__init__(*args, ctx=multiprocessing.get_context('fork'))
    def put(self, obj: T):
        try:
            super().put(obj)
        except PicklingError:
            raise # pragma: no cover
        # https://bugs.python.org/issue29187
        except Exception as e:
            message = str(e)
            if not "pickle" in message:
                raise # pragma: no cover
            raise PicklingError(message)
    def close(self): # Python 3.8 static typing trick
        super().close() # type: ignore
    def wlock_release(self):
        if (lock := getattr(self, '_wlock', None)) is None:
            return # pragma: no cover
        try:
            lock.release()
        except ValueError:
            pass


def drop_params(fn: Callable[[], T]) -> Callable[..., T]:
    def drop(*args):
        return fn()
    return drop


def gradio_request_var():
    try:
        from gradio.context import LocalContext
    except ImportError: # pragma: no cover
        raise RuntimeError(GRADIO_VERSION_ERROR_MESSAGE)
    return LocalContext.request


debug = partial(print, 'SPACES_ZERO_GPU_DEBUG')
