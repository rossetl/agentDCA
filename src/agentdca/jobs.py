"""Foreground and background execution of long adabmDCA runs.

Long operations (training, reintegration, PTT sampling, entropy) receive two hooks
from adabmDCA: a ``progress`` callback and an ``is_cancelled`` predicate. In the
foreground they drive MCP progress notifications and request cancellation; in the
background they feed an in-memory job record that ``get_dca_job`` reports.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anyio
import anyio.from_thread
import anyio.to_thread

from agentdca._common import describe_error, json_safe

Progress = Callable[[Any], None]
IsCancelled = Callable[[], bool]
Work = Callable[[Progress, IsCancelled], dict[str, Any]]

PROGRESS_INTERVAL_SECONDS = 1.0


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


async def run_foreground(ctx: Any, work: Work) -> dict[str, Any]:
    """Run blocking ``work`` in a worker thread, forwarding progress and cancellation."""
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: str
    parameters: dict[str, Any]
    status: str = "running"
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    progress: dict[str, Any] | None = None
    progress_events: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)

    def to_dict(self, *, include_result: bool = True) -> dict[str, Any]:
        document = {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "parameters": self.parameters,
            "latest_progress": self.progress,
            "progress_events": self.progress_events,
            "cancel_requested": self.cancel_event.is_set(),
        }
        if self.error is not None:
            document["error"] = self.error
        if include_result and self.result is not None:
            document["result"] = self.result
        return document


class JobManager:
    """In-process registry of background runs; jobs live as long as the server process."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def start(self, kind: str, work: Work, parameters: dict[str, Any]) -> Job:
        with self._lock:
            job_id = f"{kind}-{next(self._ids)}-{int(time.time())}"
            job = Job(id=job_id, kind=kind, parameters=json_safe(parameters))
            self._jobs[job_id] = job

        def progress(event: Any) -> None:
            job.progress = progress_snapshot(event)
            job.progress_events += 1

        def target() -> None:
            try:
                job.result = work(progress, job.cancel_event.is_set)
                job.status = "succeeded"
            except Exception as exc:  # noqa: BLE001 - every failure is recorded on the job
                from adabmDCA.exceptions import OperationCancelledError

                if isinstance(exc, OperationCancelledError) or job.cancel_event.is_set():
                    job.status = "cancelled"
                    job.error = describe_error(exc)
                else:
                    job.status = "failed"
                    job.error = describe_error(exc)
                    job.parameters = {
                        **job.parameters,
                        "traceback_tail": traceback.format_exc().splitlines()[-6:],
                    }
            finally:
                job.finished_at = _now()

        job.thread = threading.Thread(target=target, name=f"agentdca-{job_id}", daemon=True)
        job.thread.start()
        return job

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            known = ", ".join(sorted(self._jobs)) or "none"
            raise KeyError(f"Unknown job '{job_id}'. Known jobs: {known}.") from None

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == "running":
            job.cancel_event.set()
        return job

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        job = self.get(job_id)
        if job.thread is not None:
            job.thread.join(timeout)
        return job


JOBS = JobManager()
