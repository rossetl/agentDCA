"""Run one background job: ``python -m agentdca.worker <job directory>``.

The job directory holds ``spec.json`` (tool name and arguments) written by the MCP
server. This process replays that tool call with the job directory's progress and
cancellation hooks, and records the outcome in ``status.json``.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
import traceback
from pathlib import Path

from agentdca import jobs


def run(directory: Path) -> int:
    spec = jobs._read_json(directory / "spec.json")
    context = jobs.WorkerContext(directory)
    jobs.WORKER = context
    context.update(status="running", pid=os.getpid(), started_at=jobs._now())
    try:
        from agentdca import server

        tool = getattr(server, spec["tool"])
        arguments = dict(spec["arguments"])
        if "background" in inspect.signature(tool).parameters:
            arguments["background"] = False
        result = tool(**arguments)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
    except BaseException as exc:  # noqa: BLE001 - every outcome is recorded
        cancelled = context.is_cancelled() or "OperationCancelled" in repr(exc)
        context.update(
            status="cancelled" if cancelled else "failed",
            error=str(exc) or type(exc).__name__,
            traceback_tail=traceback.format_exc().splitlines()[-8:],
            finished_at=jobs._now(),
        )
        return 1
    context.update(status="succeeded", result=result, finished_at=jobs._now())
    return 0


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m agentdca.worker <job directory>")
    raise SystemExit(run(Path(sys.argv[1])))


if __name__ == "__main__":
    main()
