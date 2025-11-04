"""
"""
# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from importlib import metadata

from packaging import version


MULTI_BACKEND_VERSION = version.parse('0.46.0')


def is_old_bnb() -> bool:
    try:
        version_str = metadata.version('bitsandbytes')
    except ImportError: # pragma: no cover
        return False
    if (bnb_version := version.parse(version_str)) < version.parse('0.46.0'): # pragma: no cover
        message = f"ZeroGPU recommends bitsandbytes >= `{MULTI_BACKEND_VERSION}` "
        message += f"(`{bnb_version}` installed). Falling back to legacy support"
        print(message)
        return True
    return False


if is_old_bnb(): # pragma: no cover

    from . import bitsandbytes_legacy

    _patch = bitsandbytes_legacy.patch
    _unpatch = bitsandbytes_legacy.unpatch
    _move = bitsandbytes_legacy.move

else:

    _patch = lambda: None
    _unpatch = lambda: None
    _move = lambda: None


patch = _patch
unpatch = _unpatch
move = _move
