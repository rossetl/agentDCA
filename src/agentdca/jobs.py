"""Foreground runs, and background jobs that outlive the MCP server and the chat.

Long operations (training, reintegration, sampling, entropy) receive two hooks from
adabmDCA: a ``progress`` callback and an ``is_cancelled`` predicate.

- In the foreground they drive MCP progress notifications and client cancellation.
- In the background the tool call is written to a job directory and executed by a
  detached worker process (``python -m agentdca.worker <job dir>``). The worker records
  progress and the final result in ``status.json`` and stops when a ``cancel`` file
  appears, so a job keeps running when the chat or the MCP server ends, and any later
  session can list it, read its result or cancel it.

Jobs live in ``$AGENTDCA_HOME/jobs`` (default ``~/.agentdca/jobs``).
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import anyio.from_thread
import anyio.to_thread

from agentdca._common import json_safe

Progress = Callable[[Any], None]
IsCancelled = Callable[[], bool]
Work = Callable[[Progress, IsCancelled], dict[str, Any]]

PROGRESS_INTERVAL_SECONDS = 1.0
STATUS_INTERVAL_SECONDS = 2.0
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted"}


def progress_snapshot(event: Any) -> dict[str, Any]:
    """Compact, JSON-safe view of an adabmDCA progress event."""
    fields = (
        "stage",
        "epoch",
        "completed",
        "total",
        "gradient_steps",
        "structure_steps",
        "sweeps",
        "pearson",
        "slope",
        "theta",
        "entropy",
        "mean_sequence_identity",
    )
    snapshot = {name: getattr(event, name) for name in fields if hasattr(event, name)}
    metrics = getattr(event, "metrics", None)
    if metrics:
        snapshot["metrics"] = metrics
    return json_safe({key: value for key, value in snapshot.items() if value is not None})


def _progress_numbers(snapshot: dict[str, Any]) -> tuple[float, float | None, str]:
    if "completed" in snapshot:
        progress, total = float(snapshot["completed"]), snapshot.get("total")
    else:
        progress, total = float(snapshot.get("epoch", 0)), None
    parts = [str(snapshot.get("stage", "running"))]
    metrics = snapshot.get("metrics") or {}
    for key in ("Pearson", "pearson", "Density", "LL_val"):
        value = metrics.get(key, snapshot.get(key))
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:.4g}")
    if isinstance(snapshot.get("entropy"), (int, float)):
        parts.append(f"entropy={snapshot['entropy']:.4g}")
    return progress, None if total is None else float(total), " ".join(parts)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    for _ in range(5):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:  # concurrent replace on some filesystems
            time.sleep(0.05)
    return json.loads(path.read_text())


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# Worker-side state


class WorkerContext:
    """Hooks used when a tool runs inside ``agentdca.worker``."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.status_path = directory / "status.json"
        self.cancel_path = directory / "cancel"
        self._last_write = 0.0
        self._events = 0

    def update(self, **fields: Any) -> None:
        status = _read_json(self.status_path) if self.status_path.exists() else {}
        status.update(fields)
        _write_json(self.status_path, status)

    def progress(self, event: Any) -> None:
        self._events += 1
        now = time.monotonic()
        if now - self._last_write < STATUS_INTERVAL_SECONDS:
            return
        self._last_write = now
        self.update(
            latest_progress=progress_snapshot(event),
            progress_events=self._events,
            updated_at=_now(),
        )

    def is_cancelled(self) -> bool:
        return self.cancel_path.exists()


WORKER: WorkerContext | None = None


async def run_foreground(ctx: Any, work: Work) -> dict[str, Any]:
    """Run blocking ``work`` in a thread, forwarding progress and cancellation.

    Inside a background worker the job directory provides the hooks instead.
    """
    if WORKER is not None:
        return work(WORKER.progress, WORKER.is_cancelled)

    cancel = threading.Event()
    last_sent = [0.0]

    def progress(event: Any) -> None:
        if ctx is None:
            return
        now = time.monotonic()
        if now - last_sent[0] < PROGRESS_INTERVAL_SECONDS:
            return
        last_sent[0] = now
        try:
            value, total, message = _progress_numbers(progress_snapshot(event))
            anyio.from_thread.run(ctx.report_progress, value, total, message)
        except Exception:  # noqa: BLE001, S110 - progress must never break a run
            pass

    try:
        return await anyio.to_thread.run_sync(
            lambda: work(progress, cancel.is_set), abandon_on_cancel=True
        )
    except anyio.get_cancelled_exc_class():
        cancel.set()
        raise


# ---------------------------------------------------------------------------
# Job registry


def jobs_home() -> Path:
    root = os.environ.get("AGENTDCA_HOME") or str(Path.home() / ".agentdca")
    path = Path(root).expanduser() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


class JobStore:
    """Background jobs as directories: ``spec.json``, ``status.json``, ``worker.log``."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
            return self._root
        return jobs_home()

    def _directory(self, job_id: str) -> Path:
        if not job_id or Path(job_id).name != job_id:
            raise KeyError(f"Invalid job id '{job_id}'.")
        directory = self.root / job_id
        if not (directory / "spec.json").exists():
            known = ", ".join(job["job_id"] for job in self.list(limit=10)) or "none"
            raise KeyError(f"Unknown job '{job_id}'. Recent jobs: {known}.")
        return directory

    def submit(
        self,
        kind: str,
        tool: str,
        arguments: dict[str, Any],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        job_id = f"{kind}-{stamp}-{secrets.token_hex(3)}"
        directory = self.root / job_id
        directory.mkdir(parents=True)
        _write_json(
            directory / "spec.json",
            {"job_id": job_id, "kind": kind, "tool": tool, "arguments": arguments},
        )
        _write_json(
            directory / "status.json",
            {
                "job_id": job_id,
                "kind": kind,
                "status": "starting",
                "created_at": _now(),
                "parameters": parameters,
            },
        )
        log = (directory / "worker.log").open("ab")
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "cwd": str(directory),
            "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
        }
        if os.name == "nt":  # pragma: no cover - Windows
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008
        else:
            kwargs["start_new_session"] = True
        # The worker records its own pid; only it writes status.json from here on.
        process = subprocess.Popen(  # fixed interpreter and module
            [sys.executable, "-m", "agentdca.worker", str(directory)], **kwargs
        )
        log.close()
        # Reap the worker when it exits so a crashed one does not linger as a zombie
        # that still looks alive to get().
        threading.Thread(target=process.wait, daemon=True).start()
        return self.get(job_id)

    def get(self, job_id: str, *, include_result: bool = True) -> dict[str, Any]:
        directory = self._directory(job_id)
        status = _read_json(directory / "status.json")
        state = status.get("status")
        # A just-spawned worker gets a grace period to record its pid before it counts as lost.
        starting = (
            state == "starting" and time.time() - (directory / "spec.json").stat().st_mtime < 60
        )
        if state not in TERMINAL_STATES and not starting and not _pid_alive(status.get("pid")):
            status["status"] = "interrupted"
            status["error"] = (
                "The worker process ended without recording a result; see worker_log. "
                "Training runs can continue from their checkpoint with "
                "initial_params_path/initial_chains_path or ptt_resume."
            )
            status.setdefault("finished_at", _now())
            _write_json(directory / "status.json", status)
        status["cancel_requested"] = (directory / "cancel").exists()
        status["job_dir"] = str(directory)
        status["worker_log"] = str(directory / "worker.log")
        if not include_result:
            status.pop("result", None)
        return status

    def list(self, *, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
        entries = []
        for directory in sorted(self.root.iterdir(), reverse=True):
            if not (directory / "spec.json").exists():
                continue
            try:
                job = self.get(directory.name, include_result=False)
            except (OSError, KeyError, json.JSONDecodeError):
                continue
            if status is None or job.get("status") == status:
                entries.append(job)
        entries.sort(key=lambda job: job.get("created_at", ""), reverse=True)
        return entries[:limit]

    def cancel(self, job_id: str) -> dict[str, Any]:
        directory = self._directory(job_id)
        job = self.get(job_id, include_result=False)
        if job.get("status") not in TERMINAL_STATES:
            (directory / "cancel").write_text(_now() + "\n")
        return self.get(job_id, include_result=False)

    def kill(self, job_id: str) -> dict[str, Any]:
        """Terminate a worker that does not stop on cancel (last resort)."""
        job = self.get(job_id, include_result=False)
        pid = job.get("pid")
        if job.get("status") not in TERMINAL_STATES and _pid_alive(pid):
            os.kill(pid, signal.SIGTERM)
        return job

    def wait(self, job_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        job = self.get(job_id)
        while job.get("status") not in TERMINAL_STATES and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))
            job = self.get(job_id)
        return job


JOBS = JobStore()
