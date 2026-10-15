"""Shared helpers: JSON safety, bounded payloads, output guards, alphabets and errors."""

from __future__ import annotations

import functools
import inspect
import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import numpy as np
from mcp.server.mcpserver.exceptions import ToolError

F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_LIST_PREVIEW = 20


def json_safe(value: Any) -> Any:
    """Convert nested scientific values to strict JSON-compatible objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return json_safe(value.detach().cpu().tolist())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return json_safe(asdict(value))
    return str(value)


def bounded(value: Any, *, max_items: int = DEFAULT_LIST_PREVIEW) -> Any:
    """Replace long lists with a count and a short preview, recursively."""
    if isinstance(value, Mapping):
        return {key: bounded(item, max_items=max_items) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > max_items:
            return {
                "count": len(value),
                "preview": [bounded(item, max_items=max_items) for item in value[:max_items]],
                "truncated": True,
            }
        return [bounded(item, max_items=max_items) for item in value]
    return value


def compact(value: Any, *, max_items: int = DEFAULT_LIST_PREVIEW) -> Any:
    """JSON-safe and bounded."""
    return bounded(json_safe(value), max_items=max_items)


def inline_limit(max_inline: int) -> int:
    if max_inline < 0:
        raise ValueError("max_inline_sequences cannot be negative.")
    return max_inline


# ---------------------------------------------------------------------------
# Output guards


def prepare_file_output(path: str, *, overwrite: bool) -> Path:
    output = Path(path).expanduser()
    if output.exists():
        if output.is_dir():
            raise ValueError(f"Output path '{output}' is a directory.")
        if not overwrite:
            raise FileExistsError(
                f"Output file '{output}' already exists. Set overwrite=true to replace it."
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def validated_label(label: str | None) -> str | None:
    if label is None:
        return None
    if not label.strip():
        raise ValueError("Label cannot be empty.")
    if label in {".", ".."} or Path(label).name != label or "/" in label or "\\" in label:
        raise ValueError("Label must be a single filename-safe component.")
    return label


def prepare_output_dir(
    path: str,
    *,
    managed_names: Iterable[str] = (),
    managed_prefixes: Iterable[str] = (),
    overwrite: bool,
) -> Path:
    """Create ``path`` and refuse to reuse it if it holds files this run would write."""
    output = Path(path).expanduser()
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path '{output}' is not a directory.")
    existing: set[str] = set()
    if output.is_dir():
        existing.update(name for name in managed_names if (output / name).exists())
        prefixes = tuple(prefix for prefix in managed_prefixes if prefix)
        if prefixes:
            existing.update(
                entry.name for entry in output.iterdir() if entry.name.startswith(prefixes)
            )
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory '{output}' contains existing artifacts: "
            f"{', '.join(sorted(existing))}. Set overwrite=true to replace them, "
            "or use a different label or directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def training_artifact_names(label: str | None, *, ptt: bool = False) -> dict[str, str]:
    """File names written by ``adabmDCA.train_model`` plus the agentDCA summary."""
    prefix = f"{label}_" if label else ""
    names = {
        "log": f"{label or 'adabmDCA'}.log",
        "history": f"{prefix}history.csv",
        "events": f"{prefix}events.jsonl",
        "params": f"{prefix}params.dat.gz",
        "chains": f"{prefix}chains.fasta",
        "weights": f"{prefix}weights.dat",
        "summary": f"{prefix}training.json",
    }
    if ptt:
        names["ptt_archive"] = f"{prefix}ptt.h5"
    return names


# ---------------------------------------------------------------------------
# Alphabets


def resolve_alphabet(
    alphabet: str | None,
    *,
    model: str | None = None,
    data: str | None = None,
) -> str | None:
    """Resolve ``"auto"`` like the adabmDCA CLI does, from a model file and/or alignment.

    Returns the alphabet unchanged when it is explicit. PTT archives resolve to their
    stored token string.
    """
    if alphabet not in (None, "auto"):
        return alphabet
    if model is None and data is None:
        return None
    from adabmDCA.scripts._frontend import resolve_alphabet as _cli_resolve

    args = SimpleNamespace(
        alphabet="auto",
        path_params=None if model is None else str(Path(model).expanduser()),
        data=None if data is None else str(Path(data).expanduser()),
    )
    _cli_resolve(args)
    return args.alphabet


# ---------------------------------------------------------------------------
# Errors


def describe_error(exc: BaseException) -> str:
    """Readable, structured message for an exception raised by adabmDCA or agentDCA."""
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    details = getattr(exc, "details", None)
    text = f"{type(exc).__name__}"
    if code:
        text += f" [{code}]"
    text += f": {message}"
    if details:
        text += f"\ndetails: {json.dumps(json_safe(details), sort_keys=True)}"
    return text


_EXPECTED_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    TypeError,
    KeyError,
    OSError,
    ImportError,
    RuntimeError,
)


def _is_expected(exc: BaseException) -> bool:
    from adabmDCA.exceptions import AdabmDCAError

    return isinstance(exc, (AdabmDCAError, *_EXPECTED_ERRORS))


def surface_errors(fn: F) -> F:
    """Re-raise anticipated failures as ``ToolError`` so the agent sees the message.

    The MCP server hides the text of any exception that is not a ``ToolError``.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except ToolError:
                raise
            except Exception as exc:
                if _is_expected(exc):
                    raise ToolError(describe_error(exc)) from exc
                raise

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            if _is_expected(exc):
                raise ToolError(describe_error(exc)) from exc
            raise

    return wrapper  # type: ignore[return-value]
