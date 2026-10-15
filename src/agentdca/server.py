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


def _validated_label(label: str | None) -> str | None:
    if label is None:
        return None
    if not label.strip():
        raise ValueError("Training label cannot be empty.")
    if label in {".", ".."} or Path(label).name != label or "/" in label or "\\" in label:
        raise ValueError("Training label must be a single filename-safe component.")
    return label


def _training_artifact_names(label: str | None) -> dict[str, str]:
    if label is None:
        return {
            "log": "adabmDCA.log",
            "params": "params.dat",
            "chains": "chains.fasta",
            "weights": "weights.dat",
            "history": "training_history.json",
        }
    return {
        "log": f"{label}.log",
        "params": f"{label}_params.dat",
        "chains": f"{label}_chains.fasta",
        "weights": f"{label}_weights.dat",
        "history": f"{label}_training_history.json",
    }


def _prepare_training_output(
    path: str,
    *,
    overwrite: bool,
    label: str | None = None,
) -> tuple[Path, Path]:
    output = Path(path).expanduser()
    if output.exists() and not output.is_dir():
        raise ValueError(f"Training output path '{output}' is not a directory.")
    label = _validated_label(label)
    artifact_names = _training_artifact_names(label)
    managed_names = set(artifact_names.values())
    existing = sorted(name for name in managed_names if (output / name).exists())
    if existing and not overwrite:
        raise FileExistsError(
            f"Training output directory '{output}' contains existing artifacts: "
            f"{', '.join(existing)}. Set overwrite=true to replace them."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output, output / artifact_names["history"]


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


def _normalize_contact_scores(
    scores: Any,
    *,
    symmetrize: bool,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    matrix = np.asarray(scores)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Contact scores must be a square matrix, got shape {matrix.shape}.")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("Contact scores must be numeric.")
    difference = np.abs(matrix - matrix.T)
    diagnostics: dict[str, float | bool] = {
        "symmetrized": symmetrize,
        "max_abs_asymmetry_before": float(np.nanmax(difference)),
        "mean_abs_asymmetry_before": float(np.nanmean(difference)),
    }
    if symmetrize:
        matrix = (matrix + matrix.T) / 2
    return matrix, diagnostics


def _prepare_contact_outputs(
    *,
    output_csv: str | None,
    output_npy: str | None,
    overwrite: bool,
) -> tuple[Path | None, Path | None]:
    requested = [path for path in (output_csv, output_npy) if path is not None]
    normalized = [str(Path(path).expanduser().resolve()) for path in requested]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Contact CSV and NPY outputs must use different paths.")
    csv_output = (
        None if output_csv is None else _prepare_file_output(output_csv, overwrite=overwrite)
    )
    npy_output = (
        None if output_npy is None else _prepare_file_output(output_npy, overwrite=overwrite)
    )
    return csv_output, npy_output


def _write_contact_outputs(
    scores: np.ndarray,
    *,
    output_csv: Path | None,
    output_npy: Path | None,
) -> None:
    if output_csv is not None:
        np.savetxt(output_csv, scores, delimiter=",")
    if output_npy is not None:
        with output_npy.open("wb") as handle:
            np.save(handle, scores)


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
    symmetrize: bool = True,
    output_csv: str | None = None,
    output_npy: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rank contacts and optionally save the complete contact-score matrix."""
    from adabmDCA import predict_contacts

    csv_output, npy_output = _prepare_contact_outputs(
        output_csv=output_csv,
        output_npy=output_npy,
        overwrite=overwrite,
    )
    result = predict_contacts(
        model=model_path,
        fasta_path=fasta_path,
        alphabet=alphabet,
        pseudocount=pseudocount,
        device=device,
        dtype=dtype,
    )
    scores, symmetry = _normalize_contact_scores(result.scores, symmetrize=symmetrize)
    _write_contact_outputs(scores, output_csv=csv_output, output_npy=npy_output)
    return _json_safe(
        {
            "method": result.method,
            "alphabet": result.alphabet,
            "tokens": result.tokens,
            "sequence_length": int(scores.shape[0]),
            "contacts": _top_contacts(
                scores,
                top_k=top_k,
                min_sequence_separation=min_sequence_separation,
            ),
            "matrix": {
                "shape": list(scores.shape),
                "output_csv": None if csv_output is None else str(csv_output),
                "output_npy": None if npy_output is None else str(npy_output),
                **symmetry,
            },
            "model": None if result.model is None else _metadata_dict(result.model),
            "warnings": list(result.warnings),
        }
    )


@mcp.tool()
def compare_dca_contacts(
    model_path: str,
    fasta_path: str,
    output_plot: str,
    alphabet: str = "protein",
    pseudocount: float = 0.5,
    top_k: int = 100,
    min_sequence_separation: int = 5,
    device: str = "auto",
    dtype: str = "float32",
    symmetrize: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Plot mean-field and model contact matrices side by side and return ranked contacts."""
    from adabmDCA import predict_contacts
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    plot_output = _prepare_file_output(output_plot, overwrite=overwrite)
    if plot_output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("Contact comparison plot must use a .png, .pdf, or .svg extension.")

    model_result = predict_contacts(
        model=model_path,
        alphabet=alphabet,
        device=device,
        dtype=dtype,
    )
    mean_field_result = predict_contacts(
        fasta_path=fasta_path,
        alphabet=alphabet,
        pseudocount=pseudocount,
        device=device,
        dtype=dtype,
    )
    model_scores, model_symmetry = _normalize_contact_scores(
        model_result.scores,
        symmetrize=symmetrize,
    )
    mean_field_scores, mean_field_symmetry = _normalize_contact_scores(
        mean_field_result.scores,
        symmetrize=symmetrize,
    )
    if model_scores.shape != mean_field_scores.shape:
        raise ValueError(
            "Model and mean-field contact matrices have incompatible shapes: "
            f"{model_scores.shape} and {mean_field_scores.shape}."
        )

    finite_values = np.concatenate(
        [
            model_scores[np.isfinite(model_scores)],
            mean_field_scores[np.isfinite(mean_field_scores)],
        ]
    )
    if finite_values.size == 0:
        raise ValueError("Contact matrices contain no finite scores to plot.")
    vmin, vmax = (float(value) for value in np.percentile(finite_values, [1, 99]))
    if math.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.01, 1e-6)
        vmin -= padding
        vmax += padding

    figure = Figure(figsize=(12, 5.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    image = None
    for axis, title, scores in zip(
        axes,
        ("Mean-field DCA", "DCA model"),
        (mean_field_scores, model_scores),
    ):
        image = axis.imshow(scores, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("Residue position")
        axis.set_ylabel("Residue position")
        axis.set_aspect("equal")
    figure.colorbar(image, ax=axes, label="APC-corrected contact score", shrink=0.86)
    figure.suptitle("DCA contact-score matrices", fontsize=14)
    figure.savefig(plot_output)

    return _json_safe(
        {
            "output_plot": str(plot_output),
            "sequence_length": int(model_scores.shape[0]),
            "color_scale": {"vmin": vmin, "vmax": vmax, "percentiles": [1, 99]},
            "mean_field": {
                "method": mean_field_result.method,
                "symmetry": mean_field_symmetry,
                "contacts": _top_contacts(
                    mean_field_scores,
                    top_k=top_k,
                    min_sequence_separation=min_sequence_separation,
                ),
                "warnings": list(mean_field_result.warnings),
            },
            "model": {
                "method": model_result.method,
                "metadata": (
                    None if model_result.model is None else _metadata_dict(model_result.model)
                ),
                "symmetry": model_symmetry,
                "contacts": _top_contacts(
                    model_scores,
                    top_k=top_k,
                    min_sequence_separation=min_sequence_separation,
                ),
                "warnings": list(model_result.warnings),
            },
        }
    )


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
    label: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train a DCA model and persist its artifacts. This may be computationally expensive."""
    from adabmDCA import train_model

    label = _validated_label(label)
    output, history_path = _prepare_training_output(
        output_dir,
        overwrite=overwrite,
        label=label,
    )
    latest_progress: dict[str, Any] = {}

    def record_progress(event: Any) -> None:
        latest_progress.clear()
        latest_progress.update({"epoch": event.epoch, "metrics": event.metrics})

    result = train_model(
        data_path,
        output_dir=output,
        label=label,
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
    history_path.write_text(
        json.dumps(_json_safe(result.history), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifacts = {name: str(path) for name, path in result.artifacts.items()}
    artifacts["history"] = str(history_path)
    final_progress = _json_safe(latest_progress)
    epochs_completed = int(latest_progress.get("epoch", 0))
    final_pearson = latest_progress.get("metrics", {}).get("Pearson")
    inferred_converged = (
        final_pearson is not None
        and math.isfinite(float(final_pearson))
        and (float(final_pearson) >= target_pearson)
    )
    native_converged = getattr(result, "converged", None)
    converged = inferred_converged if native_converged is None else bool(native_converged)
    native_stop_reason = getattr(result, "stop_reason", None)
    stop_reason = (
        str(native_stop_reason)
        if native_stop_reason is not None
        else ("target_pearson" if converged else "unknown")
    )
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
            "label": label,
            "epochs_completed": epochs_completed,
            "converged": converged,
            "stop_reason": stop_reason,
            "gradient_steps": getattr(result, "gradient_steps", None),
            "structure_steps": getattr(result, "structure_steps", None),
            "sweeps": getattr(result, "sweeps", None),
            "target_pearson": target_pearson,
            "final_progress": final_progress,
            "input_report": getattr(result, "input_report", None),
            "warnings": warnings,
        }
    )


def main() -> None:
    """Run the local stdio MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
