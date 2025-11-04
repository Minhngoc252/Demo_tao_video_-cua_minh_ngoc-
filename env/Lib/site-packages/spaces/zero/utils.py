"""
"""
from __future__ import annotations

import base64
import ctypes
import json
import shutil
from functools import cache
from pathlib import Path
from uuid import uuid4
from typing import Any

from ..config import ZEROGPU_HOME
from ..config import Config


CLEANUPS_BASE_DIR = ZEROGPU_HOME / 'cleanups'


def register_cleanup(pid: int, target_dir: Path):
    cleanups_dir = CLEANUPS_BASE_DIR / f'{pid}'
    cleanups_dir.mkdir(parents=True, exist_ok=True)
    cleanup = cleanups_dir / f'{uuid4()}'
    cleanup.symlink_to(target_dir, target_is_directory=True)


def apply_cleanups(pid: int):
    cleanups_dir = CLEANUPS_BASE_DIR / f'{pid}'
    try:
        targets = [cleanup.readlink() for cleanup in cleanups_dir.iterdir()]
    except FileNotFoundError:
        return
    for target in targets:
        shutil.rmtree(target, ignore_errors=True)
    shutil.rmtree(cleanups_dir, ignore_errors=True)


@cache
def self_cgroup_device_path() -> str:
    cgroup_content = Path(Config.zerogpu_proc_self_cgroup_path).read_text()
    for line in cgroup_content.strip().split('\n'):
        contents = line.split(':devices:')
        if len(contents) != 2:
            continue # pragma: no cover
        return contents[1]
    raise Exception # pragma: no cover


def malloc_trim():
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def jwt_payload(token: str) -> dict[str, Any]:
    _, payload, _ = token.split('.')
    return json.loads(base64.urlsafe_b64decode(f'{payload}=='))
