"""Keep recently used models in memory between tool calls.

Loading a parameter file (or a PTT archive's ladder) can take longer than the
operation itself, and an agent typically scores, scans and samples the same model
several times in a row. Entries are keyed by the file's path, size and modification
time, so a retrained model is reloaded automatically.

``AGENTDCA_MODEL_CACHE`` sets how many models are kept (default 2; 0 disables).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _capacity() -> int:
    try:
        return max(int(os.environ.get("AGENTDCA_MODEL_CACHE", "2")), 0)
    except ValueError:
        return 2


class ModelCache:
    def __init__(self, capacity: int | None = None) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def capacity(self) -> int:
        return _capacity() if self._capacity is None else self._capacity

    def _get(self, key: tuple[Any, ...], load: Any) -> Any:
        if self.capacity == 0:
            return load()
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
        value = load()
        with self._lock:
            self.misses += 1
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return value

    @staticmethod
    def _file_key(path: str) -> tuple[str, int, int]:
        resolved = Path(path).expanduser().resolve()
        stat = resolved.stat()
        return str(resolved), stat.st_size, stat.st_mtime_ns

    def model(self, path: str, *, alphabet: str | None, device: str, dtype: str) -> Any:
        """A loaded ``DCAModel`` for a parameter file or PTT archive endpoint."""
        from adabmDCA import load_model

        if not Path(path).expanduser().is_file():
            # Let adabmDCA raise its structured ModelLoadError.
            return load_model(path, alphabet=alphabet, device=device, dtype=dtype)
        key = ("model", *self._file_key(path), alphabet, device, dtype)
        return self._get(
            key, lambda: load_model(path, alphabet=alphabet, device=device, dtype=dtype)
        )

    def ptt_sampler(self, path: str, *, device: str) -> Any:
        """A ``PTTSampler`` restored from a training archive, for generation.

        adabmDCA forks the sampler for every sampling call, so the cached one stays intact.
        """
        from adabmDCA import PTTSampler
        from adabmDCA.api.runtime import resolve_runtime

        if not Path(path).expanduser().is_file():
            return path  # adabmDCA reports the missing archive
        resolved = str(resolve_runtime(device, "float32")[0])
        key = ("ptt", *self._file_key(path), resolved)
        return self._get(
            key, lambda: PTTSampler.from_archive(path, device=resolved, mode="generate")
        )

    def info(self) -> dict[str, Any]:
        with self._lock:
            entries = [
                {"kind": key[0], "path": key[1], "options": [str(part) for part in key[4:]]}
                for key in self._entries
            ]
        return {
            "capacity": self.capacity,
            "entries": entries,
            "hits": self.hits,
            "misses": self.misses,
        }

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self._entries.clear()
        return self.info()


MODELS = ModelCache()
