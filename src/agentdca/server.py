"""MCP tools backed by the high-level adabmDCA Python API."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mcp.server import MCPServer

mcp = MCPServer(
    "agentDCA",
    instructions=(
        "Use these tools to inspect and operate adabmDCA models and alignments. "
        "Inspect inputs before expensive sampling or training operations."
    ),
)


def _metadata_dict(metadata: Any) -> dict[str, Any]:
    return metadata.to_dict()


def _json_safe(value: Any) -> Any:
    """Convert nested scientific values to strict JSON-compatible objects."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _prepare_file_output(path: str, *, overwrite: bool) -> Path:
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


def _prepare_training_output(path: str, *, overwrite: bool) -> Path:
    output = Path(path).expanduser()
    if output.exists() and not output.is_dir():
        raise ValueError(f"Training output path '{output}' is not a directory.")
    managed_names = {
        "adabmDCA.log",
        "params.dat",
        "chains.fasta",
        "weights.dat",
        "training_history.json",
    }
    existing = sorted(name for name in managed_names if (output / name).exists())
    if existing and not overwrite:
        raise FileExistsError(
            f"Training output directory '{output}' contains existing artifacts: "
            f"{', '.join(existing)}. Set overwrite=true to replace them."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _inline_limit(max_inline_sequences: int) -> int:
    if max_inline_sequences < 0:
        raise ValueError("max_inline_sequences cannot be negative.")
    return max_inline_sequences


def _write_energy_csv(path: Path, result: Any) -> None:
    names = result.names or tuple(f"sequence_{index + 1}" for index in range(len(result.sequences)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "sequence", "energy"])
        for name, sequence, energy in zip(names, result.sequences, result.energies):
            writer.writerow([name, sequence, float(energy)])


def _write_sample_fasta(path: Path, result: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, (sequence, energy) in enumerate(zip(result.sequences, result.energies), start=1):
            handle.write(f">sequence_{index} | DCAenergy: {float(energy):.3f}\n{sequence}\n")


def _top_contacts(
    scores: np.ndarray,
    *,
    top_k: int,
    min_sequence_separation: int,
) -> list[dict[str, int | float]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if min_sequence_separation < 1:
        raise ValueError("min_sequence_separation must be at least 1")

    positions_i, positions_j = np.triu_indices(scores.shape[0], k=min_sequence_separation)
    pair_scores = scores[positions_i, positions_j]
    order = np.argsort(pair_scores)[::-1][:top_k]
    return [
        {
            "position_i": int(positions_i[index]),
            "position_j": int(positions_j[index]),
            "position_i_1based": int(positions_i[index] + 1),
            "position_j_1based": int(positions_j[index] + 1),
            "score": float(pair_scores[index]),
        }
        for index in order
    ]


@mcp.tool()
def inspect_dca_model(
    model_path: str,
    alphabet: str = "protein",
    device: str = "auto",
    dtype: str = "float32",
) -> dict[str, Any]:
    """Inspect a saved adabmDCA parameter file without running an analysis."""
    from adabmDCA import inspect_model

    metadata = inspect_model(model_path, alphabet=alphabet, device=device, dtype=dtype)
    return _metadata_dict(metadata)


@mcp.tool()
def score_dca_sequences(
    model_path: str,
    sequences: list[str] | None = None,
    fasta_path: str | None = None,
    alphabet: str = "protein",
    device: str = "auto",
    dtype: str = "float32",
    remove_duplicates: bool = False,
    output_csv: str | None = None,
    overwrite: bool = False,
    max_inline_sequences: int = 100,
) -> dict[str, Any]:
    """Compute energies, optionally write all records, and return a bounded preview."""
    from adabmDCA import score_sequences

    limit = _inline_limit(max_inline_sequences)
    output = None if output_csv is None else _prepare_file_output(output_csv, overwrite=overwrite)
    result = score_sequences(
        sequences=sequences,
        fasta_path=fasta_path,
        model=model_path,
        alphabet=alphabet,
        device=device,
        dtype=dtype,
        remove_duplicates=remove_duplicates,
    )
    if output is not None:
        _write_energy_csv(output, result)
    count = len(result.sequences)
    returned = min(count, limit)
    names = result.names or tuple(f"sequence_{index + 1}" for index in range(count))
    return _json_safe(
        {
            "names": list(names[:returned]),
            "sequences": list(result.sequences[:returned]),
            "energies": [float(value) for value in result.energies[:returned]],
            "num_sequences": count,
            "returned_sequences": returned,
            "truncated": returned < count,
            "output_csv": None if output is None else str(output),
            "model": _metadata_dict(result.model),
            "warnings": list(result.warnings),
        }
    )


@mcp.tool()
def predict_dca_contacts(
    model_path: str | None = None,
    fasta_path: str | None = None,
    alphabet: str = "protein",
    pseudocount: float = 0.5,
    top_k: int = 100,
    min_sequence_separation: int = 5,
    device: str = "auto",
    dtype: str = "float32",
) -> dict[str, Any]:
    """Return the highest-scoring residue pairs from a model or mean-field alignment analysis."""
    from adabmDCA import predict_contacts

    result = predict_contacts(
        model=model_path,
        fasta_path=fasta_path,
        alphabet=alphabet,
        pseudocount=pseudocount,
        device=device,
        dtype=dtype,
    )
    return {
        "method": result.method,
        "alphabet": result.alphabet,
        "tokens": result.tokens,
        "sequence_length": int(result.scores.shape[0]),
        "contacts": _top_contacts(
            result.scores,
            top_k=top_k,
            min_sequence_separation=min_sequence_separation,
        ),
        "model": None if result.model is None else _metadata_dict(result.model),
        "warnings": list(result.warnings),
    }


@mcp.tool()
def scan_dca_mutations(
    model_path: str,
    wild_type: str,
    alphabet: str = "protein",
    include_gap: bool = False,
    top_k: int = 100,
    ranking: Literal["most_favorable", "largest_effect"] = "most_favorable",
    device: str = "auto",
    dtype: str = "float32",
) -> dict[str, Any]:
    """Score single substitutions and return a ranked subset of mutation effects."""
    from adabmDCA import scan_mutations

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    result = scan_mutations(
        wild_type,
        model=model_path,
        alphabet=alphabet,
        include_gap=include_gap,
        device=device,
        dtype=dtype,
    )
    records = list(result.mutations)
    if ranking == "largest_effect":
        records.sort(key=lambda record: abs(record.delta_energy), reverse=True)
    else:
        records.sort(key=lambda record: record.delta_energy)
    return {
        "wild_type": result.wild_type,
        "wild_type_energy": result.wild_type_energy,
        "ranking": ranking,
        "mutations": [
            {
                "mutation": record.label,
                "position": record.position,
                "position_1based": record.position_1based,
                "wild_type": record.wild_type,
                "mutant": record.mutant,
                "delta_energy": record.delta_energy,
                "sequence": record.sequence,
            }
            for record in records[:top_k]
        ],
        "total_mutations": len(result.mutations),
        "model": _metadata_dict(result.model),
    }


@mcp.tool()
def sample_dca_sequences(
    model_path: str,
    n_sequences: int,
    n_sweeps: int = 1000,
    sampler: Literal["gibbs", "metropolis"] = "gibbs",
    beta: float = 1.0,
    seed: int = 0,
    alphabet: str = "protein",
    device: str = "auto",
    dtype: str = "float32",
    output_fasta: str | None = None,
    overwrite: bool = False,
    max_inline_sequences: int = 100,
) -> dict[str, Any]:
    """Generate sequences, optionally write FASTA, and return a bounded preview."""
    from adabmDCA import sample_sequences

    limit = _inline_limit(max_inline_sequences)
    output = (
        None if output_fasta is None else _prepare_file_output(output_fasta, overwrite=overwrite)
    )
    result = sample_sequences(
        model=model_path,
        n_sequences=n_sequences,
        n_sweeps=n_sweeps,
        sampler=sampler,
        beta=beta,
        seed=seed,
        alphabet=alphabet,
        device=device,
        dtype=dtype,
    )
    if output is not None:
        _write_sample_fasta(output, result)
    count = len(result.sequences)
    returned = min(count, limit)
    return _json_safe(
        {
            "sequences": list(result.sequences[:returned]),
            "energies": [float(value) for value in result.energies[:returned]],
            "num_sequences": count,
            "returned_sequences": returned,
            "truncated": returned < count,
            "output_fasta": None if output is None else str(output),
            "num_sweeps": result.num_sweeps,
            "sampler": result.sampler,
            "beta": result.beta,
            "seed": result.seed,
            "model": _metadata_dict(result.model),
            "warnings": list(result.warnings),
        }
    )


@mcp.tool()
def preprocess_dca_alignment(
    input_path: str,
    output_path: str,
    input_format: Literal["auto", "fasta", "stockholm"] = "auto",
    alignment_index: int | None = None,
    remove_insertions: bool = False,
    max_gap_fraction: float | None = None,
    remove_duplicates: bool = False,
    alphabet: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert and clean an alignment, writing canonical aligned FASTA output."""
    from adabmDCA import preprocess_alignment

    output = _prepare_file_output(output_path, overwrite=overwrite)
    result = preprocess_alignment(
        input_path,
        output_path=output,
        input_format=input_format,
        alignment_index=alignment_index,
        remove_insertions=remove_insertions,
        max_gap_fraction=max_gap_fraction,
        remove_duplicates=remove_duplicates,
        alphabet=alphabet,
    )
    return {
        "output_path": str(result.output_path),
        "keep_mask": list(result.keep_mask),
        "report": result.report.to_dict(),
    }


@mcp.tool()
def train_dca_model(
    data_path: str,
    output_dir: str,
    model_type: Literal["bmDCA", "eaDCA", "edDCA", "edgeDCA"] = "bmDCA",
    alphabet: str = "protein",
    learning_rate: float = 0.01,
    n_sweeps: int = 10,
    n_chains: int = 10_000,
    target_pearson: float = 0.95,
    max_epochs: int = 50_000,
    seed: int = 0,
    device: str = "auto",
    dtype: str = "float32",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train a DCA model and persist its artifacts. This may be computationally expensive."""
    from adabmDCA import train_model

    output = _prepare_training_output(output_dir, overwrite=overwrite)
    latest_progress: dict[str, Any] = {}

    def record_progress(event: Any) -> None:
        latest_progress.clear()
        latest_progress.update({"epoch": event.epoch, "metrics": event.metrics})

    result = train_model(
        data_path,
        output_dir=output,
        model_type=model_type,
        alphabet=alphabet,
        learning_rate=learning_rate,
        n_sweeps=n_sweeps,
        n_chains=n_chains,
        target_pearson=target_pearson,
        max_epochs=max_epochs,
        seed=seed,
        device=device,
        dtype=dtype,
        progress=record_progress,
    )
    history_path = output / "training_history.json"
    history_path.write_text(
        json.dumps(_json_safe(result.history), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifacts = {name: str(path) for name, path in result.artifacts.items()}
    artifacts["history"] = str(history_path)
    final_progress = _json_safe(latest_progress)
    epochs_completed = int(latest_progress.get("epoch", 0))
    final_pearson = latest_progress.get("metrics", {}).get("Pearson")
    converged = (
        final_pearson is not None
        and math.isfinite(float(final_pearson))
        and (float(final_pearson) >= target_pearson)
    )
    stop_reason = "target_pearson" if converged else "max_epochs"
    warnings = list(result.warnings)
    if not converged:
        warnings.append(
            f"Training stopped after {epochs_completed} epochs without reaching "
            f"target Pearson {target_pearson}."
        )
    return _json_safe(
        {
            "model": _metadata_dict(result.model.metadata),
            "pseudocount": result.pseudocount,
            "num_sequences": result.num_sequences,
            "effective_sequences": result.effective_sequences,
            "artifacts": artifacts,
            "epochs_completed": epochs_completed,
            "converged": converged,
            "stop_reason": stop_reason,
            "target_pearson": target_pearson,
            "final_progress": final_progress,
            "warnings": warnings,
        }
    )


def main() -> None:
    """Run the local stdio MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
