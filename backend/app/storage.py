"""Where uploaded files and fitted pipelines are kept.

The app never touches the filesystem directly. Everything goes through
:class:`StorageBackend`, which has exactly four operations, none of which mention
directories or paths-as-strings. The point is not that S3 is coming — it may never — but
that the code which stores a fitted pipeline should not be the code that knows about
file permissions, and separating them costs one small interface.

:class:`LocalStorage` is the only implementation, and it writes under
``config.STORAGE_ROOT``.

Keys, not paths
---------------
Callers name things with a *key* — ``"uploads/3f2a.../data.csv"``. A key is not a path:
it never escapes the storage root, because every segment is validated before it becomes
one. A key assembled from a user-supplied filename is the obvious way this would
otherwise become a directory-traversal bug, so the check lives here rather than at each
call site, where it would eventually be forgotten.
"""

from __future__ import annotations

import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from app.config import STORAGE_ROOT

__all__ = ["StorageBackend", "LocalStorage", "StorageError", "get_storage"]

#: One key segment: letters, digits, dash, underscore, dot. No slashes, no "..".
_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


class StorageError(Exception):
    """A key was malformed, or the object was not there."""


def _validate(key: str) -> tuple[str, ...]:
    """Split a key into segments, rejecting anything that could escape the root."""
    if not key or key.startswith("/"):
        raise StorageError(f"storage key must be relative and non-empty, got {key!r}")
    segments = tuple(key.split("/"))
    for segment in segments:
        if segment in ("", ".", "..") or not _SEGMENT.match(segment):
            raise StorageError(f"illegal segment {segment!r} in storage key {key!r}")
    return segments


class StorageBackend(ABC):
    """Somewhere bytes can be put and got back by key."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> str:
        """Store ``data`` under ``key``. Returns the key."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Return what was stored under ``key``. Raises :class:`StorageError` if absent."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether anything is stored under ``key``."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key``. Removing something absent is not an error."""

    @abstractmethod
    def open(self, key: str, mode: str = "rb") -> BinaryIO:
        """A file object for ``key``.

        Present because :mod:`joblib` wants a handle rather than bytes, and round-tripping
        a fitted pipeline through a ``bytes`` copy to satisfy an interface would be the
        interface serving itself.
        """


class LocalStorage(StorageBackend):
    """Files under one directory on this machine."""

    def __init__(self, root: Path | str = STORAGE_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        """The absolute path a key maps to. Validated, so it cannot leave the root."""
        return self.root.joinpath(*_validate(key))

    def put(self, key: str, data: bytes) -> str:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and move into place, so a crash mid-write cannot
        # leave a half-written pipeline that later loads without complaint.
        scratch = path.with_suffix(path.suffix + ".partial")
        scratch.write_bytes(data)
        scratch.replace(path)
        return key

    def get(self, key: str) -> bytes:
        path = self.path_for(key)
        if not path.is_file():
            raise StorageError(f"nothing stored under {key!r}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def delete(self, key: str) -> None:
        self.path_for(key).unlink(missing_ok=True)

    def open(self, key: str, mode: str = "rb") -> BinaryIO:
        path = self.path_for(key)
        if "r" in mode and not path.is_file():
            raise StorageError(f"nothing stored under {key!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return open(path, mode)  # noqa: SIM115 - the caller owns the handle

    def clear(self) -> None:
        """Delete everything. Used by tests; never called by the app."""
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)


_backend: StorageBackend | None = None


def get_storage() -> StorageBackend:
    """The process-wide storage backend, created on first use."""
    global _backend
    if _backend is None:
        _backend = LocalStorage()
    return _backend


def set_storage(backend: StorageBackend) -> None:
    """Replace the backend. For tests, which point it at a temporary directory."""
    global _backend
    _backend = backend
