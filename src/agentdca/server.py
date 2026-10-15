"""MCP tools backed by the high-level adabmDCA 2.x Python API."""

from __future__ import annotations

import importlib.util
import io
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.utilities.types import Image

from agentdca._common import (
    compact,
    inline_limit,
    json_safe,
    prepare_file_output,
    prepare_output_dir,
    resolve_alphabet,
    surface_errors,
    training_artifact_names,
    validated_label,
)
from agentdca.jobs import JOBS, IsCancelled, Progress, run_foreground
from agentdca.models import MODELS

mcp = MCPServer(
    "agentDCA",
    instructions=(
        "Use these tools to inspect and operate adabmDCA models and alignments. "
        "Inspect inputs before expensive sampling or training operations. "
        "Long runs (training, reintegration, PTT sampling, entropy) accept background=true "
        "and are then followed with get_dca_job / cancel_dca_job."
    ),
)

Device = Literal["auto", "cpu", "cuda", "mps"]
Sampler = Literal["metropolized_gibbs", "gibbs", "metropolis"]
ModelType = Literal["bmDCA", "eaDCA", "edDCA", "edgeDCA"]

# Backwards-compatible aliases used by the test-suite and external scripts.
_json_safe = json_safe
_prepare_file_output = prepare_file_output


def _metadata_dict(metadata: Any) -> dict[str, Any] | None:
    return None if metadata is None else json_safe(metadata.to_dict())


def _path(value: str | None) -> str | None:
    return None if value is None else str(Path(value).expanduser())


def _energy_stats(energies: Any) -> dict[str, float | None]:
    values = np.asarray(energies, dtype=float)
    if values.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return json_safe(
        {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    )


def _preview(values: Any, limit: int) -> list[Any]:
    if values is None:
        return []
    return [json_safe(value) for value in list(values)[:limit]]


def _call_arguments(values: dict[str, Any]) -> dict[str, Any]:
    """The JSON arguments of a tool call, replayed by a background worker."""
    return {key: value for key, value in values.items() if key != "ctx"}


async def _dispatch(
    ctx: Context | None,
    *,
    kind: str,
    tool: str,
    arguments: dict[str, Any],
    background: bool,
    parameters: dict[str, Any],
    work: Any,
) -> dict[str, Any]:
    if background:
        job = JOBS.submit(kind, tool, arguments, parameters)
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "job_dir": job["job_dir"],
            "message": (
                f"{kind} is running in a detached worker that survives this chat. Follow it "
                f"with get_dca_job('{job['job_id']}') (also from later sessions) and stop it "
                "with cancel_dca_job."
            ),
        }
    return await run_foreground(ctx, work)


MAX_IMAGE_PIXELS = 1600


def _image(path: str | Path) -> Image:
    """A PNG of the plot, downscaled so its longest side is at most MAX_IMAGE_PIXELS."""
    from PIL import Image as PILImage

    with PILImage.open(path) as picture:
        picture.thumbnail((MAX_IMAGE_PIXELS, MAX_IMAGE_PIXELS))
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG", optimize=True)
    return Image(data=buffer.getvalue(), format="png")


def _with_images(
    summary: dict[str, Any],
    paths: list[str | Path],
    *,
    return_images: bool,
    max_images: int,
) -> dict[str, Any] | list[Any]:
    """Attach plots as MCP image content, so the agent can look at them directly."""
    from agentdca import jobs

    selected = [Path(path) for path in paths if str(path).lower().endswith(".png")]
    selected = [path for path in selected if path.is_file()][: max(max_images, 0)]
    if not return_images or not selected or jobs.WORKER is not None:
        return summary
    summary = {**summary, "images_returned": [str(path) for path in selected]}
    return [summary, *(_image(path) for path in selected)]


# ---------------------------------------------------------------------------
# Contacts


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
        None if output_csv is None else prepare_file_output(output_csv, overwrite=overwrite)
    )
    npy_output = (
        None if output_npy is None else prepare_file_output(output_npy, overwrite=overwrite)
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


def _color_limits(*matrices: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    if finite.size == 0:
        raise ValueError("Contact matrices contain no finite scores to plot.")
    vmin, vmax = (float(value) for value in np.percentile(finite, [1, 99]))
    if math.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.01, 1e-6)
        vmin -= padding
        vmax += padding
    return vmin, vmax


# ---------------------------------------------------------------------------
# Model inspection, scoring, contacts, mutations


@mcp.tool()
@surface_errors
def inspect_dca_model(
    model_path: str,
    alphabet: str = "auto",
    device: Device = "auto",
    dtype: str = "float32",
) -> dict[str, Any]:
    """Inspect a saved adabmDCA parameter file (.dat, .dat.gz) or PTT archive (.h5).

    alphabet="auto" detects protein/RNA/DNA from the file, like the adabmDCA CLI.
    """
    from adabmDCA import inspect_model

    model_path = _path(model_path)
    resolved = resolve_alphabet(alphabet, model=model_path)
    model = MODELS.model(model_path, alphabet=resolved, device=device, dtype=dtype)
    metadata = inspect_model(model)
    return _metadata_dict(metadata)


@mcp.tool()
@surface_errors
def score_dca_sequences(
    model_path: str,
    sequences: list[str] | None = None,
    fasta_path: str | None = None,
    alphabet: str = "auto",
    device: Device = "auto",
    dtype: str = "float32",
    remove_duplicates: bool = False,
    local_lambda: float | None = 1.0,
    output_csv: str | None = None,
    overwrite: bool = False,
    max_inline_sequences: int = 100,
) -> dict[str, Any]:
    """Compute DCA energies (lower = more probable under the model) for aligned sequences.

    Also returns the summed context-dependent entropy (cde_sum) and the local free energy
    energy - local_lambda * cde_sum; set local_lambda=null to skip them. Writes every record
    to output_csv when given and returns a bounded preview.
    """
    from adabmDCA import score_sequences

    limit = inline_limit(max_inline_sequences)
    output = None if output_csv is None else prepare_file_output(output_csv, overwrite=overwrite)
    model_path, fasta_path = _path(model_path), _path(fasta_path)
    resolved = resolve_alphabet(alphabet, model=model_path, data=fasta_path)
    result = score_sequences(
        sequences=sequences,
        fasta_path=fasta_path,
        model=MODELS.model(model_path, alphabet=resolved, device=device, dtype=dtype),
        alphabet=resolved,
        device=device,
        dtype=dtype,
        remove_duplicates=remove_duplicates,
        local_lambda=local_lambda,
    )
    if output is not None:
        result.to_csv(output)
    count = len(result.sequences)
    returned = min(count, limit)
    names = result.names or tuple(f"sequence_{index + 1}" for index in range(count))
    return json_safe(
        {
            "names": list(names[:returned]),
            "sequences": list(result.sequences[:returned]),
            "energies": _preview(result.energies, returned),
            "cde_sum": _preview(result.cde_sum, returned),
            "local_free_energies": _preview(result.local_free_energies, returned),
            "local_lambda": result.local_lambda,
            "energy_stats": _energy_stats(result.energies),
            "num_sequences": count,
            "returned_sequences": returned,
            "truncated": returned < count,
            "output_csv": None if output is None else str(output),
            "model": _metadata_dict(result.model),
            "warnings": list(result.warnings),
        }
    )


@mcp.tool()
@surface_errors
def predict_dca_contacts(
    model_path: str | None = None,
    fasta_path: str | None = None,
    alphabet: str = "auto",
    pseudocount: float = 0.5,
    top_k: int = 100,
    min_sequence_separation: int = 5,
    device: Device = "auto",
    dtype: str = "float32",
    symmetrize: bool = True,
    output_csv: str | None = None,
    output_npy: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rank APC-corrected contact scores from a model, or mean-field DCA from an alignment.

    Give exactly one of model_path or fasta_path. Optionally saves the full matrix.
    """
    from adabmDCA import predict_contacts

    csv_output, npy_output = _prepare_contact_outputs(
        output_csv=output_csv,
        output_npy=output_npy,
        overwrite=overwrite,
    )
    model_path, fasta_path = _path(model_path), _path(fasta_path)
    resolved = resolve_alphabet(alphabet, model=model_path, data=fasta_path)
    result = predict_contacts(
        model=None
        if model_path is None
        else MODELS.model(model_path, alphabet=resolved, device=device, dtype=dtype),
        fasta_path=fasta_path,
        alphabet=resolved,
        pseudocount=pseudocount,
        device=device,
        dtype=dtype,
    )
    scores, symmetry = _normalize_contact_scores(result.scores, symmetrize=symmetrize)
    _write_contact_outputs(scores, output_csv=csv_output, output_npy=npy_output)
    return json_safe(
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
            "model": _metadata_dict(result.model),
            "warnings": list(result.warnings),
        }
    )


@mcp.tool(structured_output=False)  # may return images
@surface_errors
def compare_dca_contacts(
    model_path: str,
    fasta_path: str,
    output_plot: str,
    alphabet: str = "auto",
    pseudocount: float = 0.5,
    top_k: int = 100,
    min_sequence_separation: int = 5,
    color_scale: Literal["shared", "per_panel"] = "shared",
    device: Device = "auto",
    dtype: str = "float32",
    symmetrize: bool = True,
    overwrite: bool = False,
    return_image: bool = True,
) -> dict[str, Any] | list[Any]:
    """Plot mean-field and model contact matrices side by side and return ranked contacts.

    color_scale="shared" (default) makes the panels directly comparable; "per_panel" gives
    each panel its own 1st-99th percentile range, which helps when one model's scores are
    much smaller. A .png plot is also returned as an image (return_image=false to skip).
    """
    from adabmDCA import predict_contacts
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    plot_output = prepare_file_output(output_plot, overwrite=overwrite)
    if plot_output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("Contact comparison plot must use a .png, .pdf, or .svg extension.")

    model_path, fasta_path = _path(model_path), _path(fasta_path)
    resolved = resolve_alphabet(alphabet, model=model_path, data=fasta_path)
    model_result = predict_contacts(
        model=MODELS.model(model_path, alphabet=resolved, device=device, dtype=dtype)
    )
    mean_field_result = predict_contacts(
        fasta_path=fasta_path,
        alphabet=resolved,
        pseudocount=pseudocount,
        device=device,
        dtype=dtype,
    )
    model_scores, model_symmetry = _normalize_contact_scores(
        model_result.scores, symmetrize=symmetrize
    )
    mean_field_scores, mean_field_symmetry = _normalize_contact_scores(
        mean_field_result.scores, symmetrize=symmetrize
    )
    if model_scores.shape != mean_field_scores.shape:
        raise ValueError(
            "Model and mean-field contact matrices have incompatible shapes: "
            f"{model_scores.shape} and {mean_field_scores.shape}."
        )

    panels = (("Mean-field DCA", mean_field_scores), ("DCA model", model_scores))
    if color_scale == "shared":
        shared = _color_limits(mean_field_scores, model_scores)
        limits = [shared, shared]
    else:
        limits = [_color_limits(scores) for _, scores in panels]

    figure = Figure(figsize=(12, 5.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    image = None
    for axis, (title, scores), (vmin, vmax) in zip(axes, panels, limits):
        image = axis.imshow(scores, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("Residue position")
        axis.set_ylabel("Residue position")
        axis.set_aspect("equal")
        if color_scale == "per_panel":
            figure.colorbar(image, ax=axis, label="APC-corrected contact score", shrink=0.86)
    if color_scale == "shared":
        figure.colorbar(image, ax=axes, label="APC-corrected contact score", shrink=0.86)
    figure.suptitle("DCA contact-score matrices", fontsize=14)
    figure.savefig(plot_output)

    def scale(index: int) -> dict[str, Any]:
        return {"vmin": limits[index][0], "vmax": limits[index][1], "percentiles": [1, 99]}

    summary = json_safe(
        {
            "output_plot": str(plot_output),
            "sequence_length": int(model_scores.shape[0]),
            "color_scale": {
                "mode": color_scale,
                "mean_field": scale(0),
                "model": scale(1),
            },
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
                "metadata": _metadata_dict(model_result.model),
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
    return _with_images(summary, [plot_output], return_images=return_image, max_images=1)


@mcp.tool()
@surface_errors
def scan_dca_mutations(
    model_path: str,
    wild_type: str,
    alphabet: str = "auto",
    include_gap: bool = False,
    top_k: int = 100,
    ranking: Literal["most_favorable", "largest_effect"] = "most_favorable",
    device: Device = "auto",
    dtype: str = "float32",
    output_csv: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Score every single substitution of a wild type (deep mutational scan).

    Returns a ranked subset; output_csv saves the complete scan.
    """
    from adabmDCA import scan_mutations

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    output = None if output_csv is None else prepare_file_output(output_csv, overwrite=overwrite)
    model_path = _path(model_path)
    resolved = resolve_alphabet(alphabet, model=model_path)
    result = scan_mutations(
        wild_type,
        model=MODELS.model(model_path, alphabet=resolved, device=device, dtype=dtype),
        alphabet=resolved,
        include_gap=include_gap,
        device=device,
        dtype=dtype,
    )
    if output is not None:
        result.to_csv(output)
    records = list(result.mutations)
    if ranking == "largest_effect":
        records.sort(key=lambda record: abs(record.delta_energy), reverse=True)
    else:
        records.sort(key=lambda record: record.delta_energy)
    return json_safe(
        {
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
            "delta_energy_stats": _energy_stats(result.delta_energies),
            "output_csv": None if output is None else str(output),
            "model": _metadata_dict(result.model),
            "warnings": list(result.warnings),
        }
    )


# ---------------------------------------------------------------------------
# Sampling


def _hamming_potential(target: str):
    target_array = np.frombuffer(target.encode("ascii"), dtype=np.uint8)

    def potential(sequences: Any, strength: float) -> np.ndarray:
        distances = np.array(
            [
                np.count_nonzero(
                    np.frombuffer(sequence.encode("ascii"), dtype=np.uint8) != target_array
                )
                for sequence in sequences
            ],
            dtype=float,
        )
        return strength * distances

    return potential


def _load_steering_script(spec: str):
    path_text, _, function_name = spec.rpartition(":")
    if not path_text or not function_name:
        raise ValueError("steering_script must look like '/path/to/potential.py:function_name'.")
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Steering script '{path}' was not found.")
    module_spec = importlib.util.spec_from_file_location(f"agentdca_steering_{path.stem}", path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot import steering script '{path}'.")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    try:
        return getattr(module, function_name)
    except AttributeError:
        raise AttributeError(f"Steering script '{path}' defines no '{function_name}'.") from None


SAMPLING_PLOT_ORDER = (
    "cij_scatter_plot",
    "pca_1_2_plot",
    "distances_plot",
    "privet_plot",
    "ptt_mixing_plot",
    "data_vs_samples_plot",
    "energy_cde_plot",
    "pearson_plot",
    "ptt_ladder_health_plot",
    "ptt_renewal_plot",
    "autocorrelation_plot",
    "pca_3_4_plot",
)


def _sampling_summary(
    result: Any,
    *,
    limit: int,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    count = len(result.sequences)
    returned = min(count, limit)
    distance = result.distance_comparison or {}
    privet = distance.get("privet") or {}
    privet_summary = {
        key: privet[key]
        for key in ("n_generated", "n_flagged", "n_memorized", "n_pleaks", "excess_train")
        if key in privet
    }
    ptt = result.ptt_diagnostics or {}
    pearson_history = (result.sampling_history or {}).get("pearson") or [None]
    summary = {
        "sequences": list(result.sequences[:returned]),
        "energies": _preview(result.energies, returned),
        "energy_stats": _energy_stats(result.energies),
        "num_sequences": count,
        "returned_sequences": returned,
        "truncated": returned < count,
        "num_sweeps": result.num_sweeps,
        "sampler": result.sampler,
        "beta": result.beta,
        "seed": result.seed,
        "sampling_dtype": result.sampling_dtype,
        "local_lambda_fit": result.local_lambda_fit,
        "final_cij_pearson": list(pearson_history)[-1],
        "ptt_mixing": ptt.get("mixing"),
        "distances": distance.get("summary"),
        "privet": privet_summary or None,
        "steering": result.steering or None,
        "artifacts": artifacts,
        "model": _metadata_dict(result.model),
        "warnings": list(result.warnings),
    }
    return compact(summary, max_items=max(returned, 20))


@mcp.tool(structured_output=False)  # may return images
@surface_errors
async def sample_dca_sequences(
    model_path: str,
    n_sequences: int,
    n_sweeps: int = 1000,
    sampler: Sampler = "metropolized_gibbs",
    beta: float = 1.0,
    seed: int = 0,
    alphabet: str = "auto",
    device: Device = "auto",
    dtype: str | None = None,
    reference_fasta: str | None = None,
    test_fasta: str | None = None,
    weights_path: str | None = None,
    collect_diagnostics: bool = False,
    mixing_multiplier: int = 2,
    ptt: bool = False,
    ptt_local_sweeps: int = 10,
    ptt_max_rounds: int = 20_000,
    ptt_mixing_method: Literal["renewal", "autocorrelation"] = "renewal",
    ptt_renewal_tolerance: float = 0.01,
    ptt_stationary: bool = False,
    ptt_local_kernel: Literal["archived", "metropolized_gibbs", "gibbs", "metropolis"] = "archived",
    ptt_return_full_ladder: bool = False,
    steering_target: str | None = None,
    steering_script: str | None = None,
    steering_strength: float = 1.0,
    output_dir: str | None = None,
    label: str | None = None,
    output_fasta: str | None = None,
    overwrite: bool = False,
    max_inline_sequences: int = 100,
    return_images: bool = True,
    max_images: int = 4,
    background: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any] | list[Any]:
    """Generate sequences from a DCA model, optionally with diagnostics, PTT or steering.

    - Ordinary MCMC: runs n_sweeps sweeps; with reference_fasta it first measures the mixing
      time and runs mixing_multiplier mixing times instead.
    - collect_diagnostics=true (needs reference_fasta) compares samples with the data: Cij
      Pearson, PCA, cluster occupancy, Hamming distances and, with test_fasta, the PRIVET
      overfitting check. With output_dir, diagnostic PNGs are also written.
    - ptt=true samples a PTT archive (model_path = the training ptt.h5) and needs
      reference_fasta; it ignores n_sweeps and beta and runs until its mixing check passes.
    - Steering biases sampling by V(x, s): steering_target pulls samples toward an aligned
      target sequence (V = s * Hamming distance); steering_script='file.py:function' loads a
      user-written V(sequences, s). The result carries importance weights back to the
      unsteered model. steering_script executes that Python file in this server.
    - output_dir writes the package bundle (samples FASTA/CSV, JSON summary, logs);
      output_fasta writes only a FASTA.
    - With diagnostics and output_dir, up to max_images plots (Cij scatter, PCA, distances,
      PRIVET, PTT mixing) are returned as images for direct inspection.
    """
    call_arguments = _call_arguments(locals())
    from adabmDCA import sample_sequences

    limit = inline_limit(max_inline_sequences)
    label = validated_label(label)
    if steering_target is not None and steering_script is not None:
        raise ValueError("Give at most one of steering_target and steering_script.")
    if collect_diagnostics and reference_fasta is None:
        raise ValueError("collect_diagnostics=true needs reference_fasta.")
    if ptt and reference_fasta is None:
        raise ValueError("PTT sampling needs reference_fasta (the training alignment).")
    model_path = _path(model_path)
    reference_fasta, test_fasta, weights_path = (
        _path(reference_fasta),
        _path(test_fasta),
        _path(weights_path),
    )
    fasta_output = (
        None if output_fasta is None else prepare_file_output(output_fasta, overwrite=overwrite)
    )
    folder = None
    if output_dir is not None:
        prefix = f"{label}_" if label else ""
        folder = prepare_output_dir(
            output_dir,
            managed_names=[
                f"{prefix}samples.fasta",
                f"{prefix}samples.csv",
                f"{prefix}sampling.json",
            ],
            overwrite=overwrite,
        )
    resolved = resolve_alphabet(alphabet, model=model_path, data=reference_fasta)
    potential = None
    if steering_target is not None:
        potential = _hamming_potential(steering_target)
    elif steering_script is not None:
        potential = _load_steering_script(steering_script)

    parameters = {
        "model_path": model_path,
        "n_sequences": n_sequences,
        "ptt": ptt,
        "sampler": sampler,
        "seed": seed,
        "output_dir": None if folder is None else str(folder),
    }

    def work(progress: Progress, is_cancelled: IsCancelled) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if potential is not None:
            kwargs.update(steering_potential=potential, steering_strength=steering_strength)
        if ptt:
            source = MODELS.ptt_sampler(model_path, device=device)
        else:
            # adabmDCA keeps bfloat16 models in float32 and casts only the sampler.
            model_dtype = "float32" if dtype in (None, "bfloat16") else dtype
            source = MODELS.model(model_path, alphabet=resolved, device=device, dtype=model_dtype)
        result = sample_sequences(
            model=source,
            n_sequences=n_sequences,
            ptt=ptt,
            ptt_local_sweeps=ptt_local_sweeps,
            ptt_max_rounds=ptt_max_rounds,
            ptt_mixing_method=ptt_mixing_method,
            ptt_renewal_tolerance=ptt_renewal_tolerance,
            ptt_stationary=ptt_stationary,
            ptt_local_kernel=None if ptt_local_kernel == "archived" else ptt_local_kernel,
            ptt_return_full_ladder=ptt_return_full_ladder,
            n_sweeps=n_sweeps,
            # PTT archives keep their training sampler unless a local kernel is chosen.
            sampler=None if ptt else sampler,
            beta=beta,
            seed=seed,
            reference_fasta=reference_fasta,
            test_fasta=test_fasta,
            weights_path=weights_path,
            mixing_multiplier=mixing_multiplier,
            collect_diagnostics=collect_diagnostics,
            alphabet=resolved,
            device=device,
            dtype=dtype,
            progress=progress,
            is_cancelled=is_cancelled,
            **kwargs,
        )
        artifacts: dict[str, Any] = {}
        if folder is not None:
            artifacts.update(result.save_bundle(folder, label=label))
            if collect_diagnostics:
                artifacts.update(result.save_diagnostic_plots(folder, label=label))
        if fasta_output is not None:
            artifacts["output_fasta"] = result.to_fasta(fasta_output)
        return _sampling_summary(result, limit=limit, artifacts=json_safe(artifacts))

    summary = await _dispatch(
        ctx,
        kind="sampling",
        tool="sample_dca_sequences",
        arguments=call_arguments,
        background=background,
        parameters=parameters,
        work=work,
    )
    if background:
        return summary
    artifacts = summary.get("artifacts") or {}
    plots = [artifacts[key] for key in SAMPLING_PLOT_ORDER if key in artifacts]
    return _with_images(summary, plots, return_images=return_images, max_images=max_images)


# ---------------------------------------------------------------------------
# Alignments


@mcp.tool()
@surface_errors
def preprocess_dca_alignment(
    input_path: str,
    output_path: str,
    input_format: Literal["auto", "fasta", "stockholm"] = "auto",
    alignment_index: int | None = None,
    remove_insertions: bool = False,
    max_gap_fraction: float | None = None,
    remove_duplicates: bool = False,
    alphabet: str | None = None,
    unknown_tokens: Literal["gap", "remove", "error"] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert (Stockholm/FASTA) and clean an alignment, writing aligned FASTA.

    Every filter is off unless requested. alphabet="auto" detects protein/RNA/DNA and checks
    tokens; unknown_tokens decides what happens to symbols outside the alphabet.
    """
    from adabmDCA import preprocess_alignment

    output = prepare_file_output(output_path, overwrite=overwrite)
    kwargs: dict[str, Any] = {}
    if unknown_tokens is not None:
        kwargs["unknown_tokens"] = unknown_tokens
    result = preprocess_alignment(
        _path(input_path),
        output_path=output,
        input_format=input_format,
        alignment_index=alignment_index,
        remove_insertions=remove_insertions,
        max_gap_fraction=max_gap_fraction,
        remove_duplicates=remove_duplicates,
        alphabet=alphabet,
        **kwargs,
    )
    keep = list(result.keep_mask)
    removed = [index for index, kept in enumerate(keep) if not kept]
    return compact(
        {
            "output_path": str(result.output_path),
            "input_sequences": len(keep),
            "kept_sequences": len(keep) - len(removed),
            "removed_indices": removed,
            "report": result.report.to_dict(),
        }
    )


@mcp.tool()
@surface_errors
def split_dca_alignment(
    alignment_path: str,
    output_prefix: str,
    method: Literal["clustering", "cobalt"] = "clustering",
    identity: float = 0.8,
    train_fraction: float = 0.8,
    t1: float = 0.5,
    t2: float = 0.5,
    t3: float = 1.0,
    max_train: int | None = None,
    max_test: int | None = None,
    attempts: int = 1,
    alphabet: str = "auto",
    seed: int = 0,
    device: Device = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Split an alignment into training and test sets with limited homology between them.

    Writes <output_prefix>.train.fasta, .test.fasta and .split.json. "clustering" assigns whole
    identity clusters to one side; "cobalt" (Petti & Eddy 2022) bounds identities with t1-t3.
    Use the test set as validation_path for training or test_fasta for sampling diagnostics.
    """
    from adabmDCA import split_alignment

    alignment_path = _path(alignment_path)
    prefix = Path(output_prefix).expanduser()
    for suffix in (".train.fasta", ".test.fasta", ".split.json"):
        prepare_file_output(str(prefix.with_suffix(prefix.suffix + suffix)), overwrite=overwrite)
    resolved = resolve_alphabet(alphabet, data=alignment_path)
    result = split_alignment(
        alignment_path,
        method=method,
        identity=identity,
        train_fraction=train_fraction,
        t1=t1,
        t2=t2,
        t3=t3,
        max_train=max_train,
        max_test=max_test,
        attempts=attempts,
        alphabet=resolved,
        seed=seed,
        device=device,
    )
    artifacts = result.save_bundle(prefix)
    return json_safe(
        {
            "method": result.method,
            "training_sequences": len(result.training.sequences),
            "test_sequences": len(result.test.sequences),
            "score": result.score,
            "attempts": result.attempts,
            "identity": result.identity,
            "train_fraction": result.train_fraction,
            "seed": result.seed,
            "alphabet": resolved,
            "artifacts": artifacts,
        }
    )


# ---------------------------------------------------------------------------
# Training


def _training_strategy(
    ptt: bool | None, *, model_type: str, resume: bool, warm_start: bool
) -> tuple[bool, str]:
    """PTT or PCD, with the reason, following the adabmDCA CLI's default of PTT."""
    if resume:
        return True, "ptt_resume continues a PTT archive"
    if ptt is not None:
        return ptt, "requested"
    if model_type not in ("bmDCA", "eaDCA"):
        return False, f"PTT supports bmDCA and eaDCA only, so {model_type} trains with PCD"
    if warm_start:
        return False, "PTT cannot start from initial parameters or chains, so the run uses PCD"
    return True, "default"


def _ptt_config(ptt: bool, ptt_options: dict[str, Any] | None):
    if not ptt:
        if ptt_options:
            raise ValueError("ptt_options requires ptt=true.")
        return None
    from adabmDCA import PTTConfig

    return PTTConfig(**(ptt_options or {}))


def _history_tail(history: dict[str, Any], rows: int) -> dict[str, Any]:
    return {key: list(values)[-rows:] for key, values in (history or {}).items()}


def _training_summary(
    result: Any, *, label: str | None, artifacts: dict[str, Any]
) -> dict[str, Any]:
    initialization = result.initialization
    warnings = list(result.warnings)
    if not result.converged:
        warnings.append(f"Training stopped without converging (stop_reason={result.stop_reason}).")
    return compact(
        {
            "model": _metadata_dict(result.model.metadata),
            "label": label,
            "converged": result.converged,
            "stop_reason": result.stop_reason,
            "gradient_steps": result.gradient_steps,
            "structure_steps": result.structure_steps,
            "sweeps": result.sweeps,
            "final_metrics": result.final_metrics,
            "history_tail": _history_tail(result.history, 5),
            "pseudocount": result.pseudocount,
            "num_sequences": result.num_sequences,
            "effective_sequences": result.effective_sequences,
            "training_data": None if initialization is None else initialization.training,
            "validation_data": None if initialization is None else initialization.validation,
            "runtime": None
            if initialization is None
            else {
                "device": initialization.device,
                "dtype": initialization.dtype,
                "n_chains": initialization.n_chains,
            },
            "config": result.config,
            "partition_estimate": result.partition_estimate,
            "input_report": result.input_report,
            "artifacts": artifacts,
            "warnings": warnings,
        }
    )


@mcp.tool()
@surface_errors
async def train_dca_model(
    data_path: str,
    output_dir: str,
    model_type: ModelType = "bmDCA",
    alphabet: str = "auto",
    label: str | None = None,
    validation_path: str | None = None,
    weights_path: str | None = None,
    learning_rate: float | None = None,
    n_sweeps: int | None = None,
    sampler: Sampler | None = None,
    n_chains: int | None = None,
    target_pearson: float | None = None,
    max_epochs: int | None = None,
    max_gradient_steps: int | None = None,
    max_structure_steps: int | None = None,
    pseudocount: float | None = None,
    l2_regularization: float | None = None,
    clustering_seqid: float | None = None,
    no_reweighting: bool = False,
    target_density: float | None = None,
    decimation_rate: float | None = None,
    activation_steps: int | None = None,
    activation_fraction: float | None = None,
    checkpoint_interval: int | None = None,
    initial_params_path: str | None = None,
    initial_chains_path: str | None = None,
    ptt: bool | None = None,
    ptt_options: dict[str, Any] | None = None,
    ptt_resume: str | None = None,
    seed: int = 0,
    device: Device = "auto",
    dtype: Literal["float32", "float64", "bfloat16"] = "float32",
    overwrite: bool = False,
    background: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Train a DCA model with adabmDCA and write its artifacts. Computationally expensive.

    Options left null use adabmDCA's defaults (which differ for PTT). Model types: bmDCA
    (dense), eaDCA (edge activation; target_density/activation_*), edDCA (decimation;
    target_density/decimation_rate), edgeDCA (sparse, PCD only).
    - Strategy, as in the adabmDCA CLI: Parallel Trajectory Tempering (PTT) by default.
      ptt=null picks PTT for bmDCA/eaDCA and PCD for edDCA/edgeDCA or when restarting from
      initial_params_path/initial_chains_path (PTT supports neither). ptt=false forces
      PCD: much faster, but nothing checks that its chains stay at equilibrium.
    - validation_path tracks held-out log-likelihood (from split_dca_alignment); with PTT
      the run then stops at the validation plateau unless ptt_options sets
      {"validation_stop": false}.
    - ptt_options are PTTConfig fields. ptt_resume continues from a ptt.h5 archive (writes
      into the same files).
    - initial_params_path / initial_chains_path restart or fine-tune from a PCD run.
    - background=true returns a job id immediately; follow it with get_dca_job.
    Writes <label>_params.dat.gz, history.csv, events.jsonl, chains, weights, a JSON summary
    (training.json) and, with PTT, ptt.h5.
    """
    call_arguments = _call_arguments(locals())
    from adabmDCA import train_model

    label = validated_label(label)
    data_path = _path(data_path)
    use_ptt, strategy_reason = _training_strategy(
        ptt,
        model_type=model_type,
        resume=ptt_resume is not None,
        warm_start=initial_params_path is not None or initial_chains_path is not None,
    )
    if use_ptt and validation_path is not None:
        ptt_options = {"validation_stop": True, **(ptt_options or {})}
    names = training_artifact_names(label, ptt=use_ptt)
    output = prepare_output_dir(
        output_dir,
        managed_names=names.values(),
        overwrite=overwrite or ptt_resume is not None,
    )
    resolved = resolve_alphabet(
        alphabet, model=_path(ptt_resume) or _path(initial_params_path), data=data_path
    )
    ptt_config = _ptt_config(use_ptt, ptt_options)

    optional = {
        "learning_rate": learning_rate,
        "n_sweeps": n_sweeps,
        "sampler": sampler,
        "n_chains": n_chains,
        "target_pearson": target_pearson,
        "max_epochs": max_epochs,
        "l2_regularization": l2_regularization,
        "clustering_seqid": clustering_seqid,
        "target_density": target_density,
        "decimation_rate": decimation_rate,
        "activation_steps": activation_steps,
        "activation_fraction": activation_fraction,
        "checkpoint_interval": checkpoint_interval,
    }
    kwargs = {key: value for key, value in optional.items() if value is not None}
    parameters = {
        "data_path": data_path,
        "output_dir": str(output),
        "model_type": model_type,
        "label": label,
        "strategy": "ptt" if use_ptt else "pcd",
        **kwargs,
    }

    def work(progress: Progress, is_cancelled: IsCancelled) -> dict[str, Any]:
        result = train_model(
            data_path,
            model_type=model_type,
            ptt=ptt_config,
            ptt_resume=_path(ptt_resume),
            validation_path=_path(validation_path),
            weights_path=_path(weights_path),
            output_dir=output,
            label=label,
            initial_params_path=_path(initial_params_path),
            initial_chains_path=_path(initial_chains_path),
            alphabet=resolved,
            max_gradient_steps=max_gradient_steps,
            max_structure_steps=max_structure_steps,
            pseudocount=pseudocount,
            seed=seed,
            no_reweighting=no_reweighting,
            device=device,
            dtype=dtype,
            progress=progress,
            stage_progress=progress,
            is_cancelled=is_cancelled,
            **kwargs,
        )
        artifacts = {name: str(path) for name, path in result.artifacts.items()}
        artifacts.update(
            {name: str(path) for name, path in result.save_bundle(output, label=label).items()}
        )
        summary = _training_summary(result, label=label, artifacts=artifacts)
        strategy = {"strategy": "ptt" if use_ptt else "pcd", "strategy_reason": strategy_reason}
        return {**strategy, **summary}

    return await _dispatch(
        ctx,
        kind="training",
        tool="train_dca_model",
        arguments=call_arguments,
        background=background,
        parameters=parameters,
        work=work,
    )


@mcp.tool()
@surface_errors
async def reintegrate_dca_model(
    natural_alignment: str,
    experimental_alignment: str,
    adjustments: str | list[float],
    output_dir: str,
    label: str = "reintegrated",
    lambda_value: float | None = None,
    natural_weights: str | None = None,
    model_type: ModelType = "bmDCA",
    alphabet: str = "auto",
    target_pearson: float | None = None,
    max_epochs: int | None = None,
    n_chains: int | None = None,
    learning_rate: float | None = None,
    pseudocount: float | None = None,
    initial_params_path: str | None = None,
    initial_chains_path: str | None = None,
    ptt: bool = False,
    ptt_options: dict[str, Any] | None = None,
    ptt_resume: str | None = None,
    seed: int = 0,
    device: Device = "auto",
    dtype: Literal["float32", "float64", "bfloat16"] = "float32",
    overwrite: bool = False,
    background: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Retrain a DCA model with experimental feedback on tested sequences.

    adjustments gives one signed score per experimental sequence (a file path or a list),
    e.g. +1 functional and -1 non-functional. lambda_value weighs the experimental data
    (default 1/max|adjustment|). Files are labelled '<label>-lambda_<value>'.
    """
    call_arguments = _call_arguments(locals())
    from adabmDCA import TrainingConfig, reintegrate_model

    label = validated_label(label) or "reintegrated"
    natural_alignment = _path(natural_alignment)
    experimental_alignment = _path(experimental_alignment)
    output = prepare_output_dir(
        output_dir,
        managed_prefixes=[f"{label}-lambda_"],
        overwrite=overwrite or ptt_resume is not None,
    )
    resolved = resolve_alphabet(alphabet, data=natural_alignment)
    optional = {
        "target_pearson": target_pearson,
        "max_epochs": max_epochs,
        "n_chains": n_chains,
        "learning_rate": learning_rate,
        "pseudocount": pseudocount,
    }
    config = TrainingConfig(
        model_type=model_type,
        alphabet=resolved or "protein",
        seed=seed,
        device=device,
        dtype=dtype,
        ptt=_ptt_config(ptt or ptt_resume is not None, ptt_options),
        **{key: value for key, value in optional.items() if value is not None},
    )
    adjustment_input = (
        _path(adjustments) if isinstance(adjustments, str) else np.asarray(adjustments, dtype=float)
    )
    parameters = {
        "natural_alignment": natural_alignment,
        "experimental_alignment": experimental_alignment,
        "output_dir": str(output),
        "label": label,
        "lambda_value": lambda_value,
        "model_type": model_type,
    }

    def work(progress: Progress, is_cancelled: IsCancelled) -> dict[str, Any]:
        result = reintegrate_model(
            natural_alignment,
            experimental_alignment,
            adjustment_input,
            config=config,
            natural_weights=_path(natural_weights),
            lambda_value=lambda_value,
            output_dir=output,
            label=label,
            initial_params_path=_path(initial_params_path),
            initial_chains_path=_path(initial_chains_path),
            ptt_resume=_path(ptt_resume),
            progress=progress,
            stage_progress=progress,
            is_cancelled=is_cancelled,
        )
        artifacts = {name: str(path) for name, path in result.artifacts.items()}
        artifacts.update({name: str(path) for name, path in result.save_bundle(output).items()})
        summary = _training_summary(result.training, label=result.label, artifacts=artifacts)
        summary["reintegration"] = json_safe(
            {
                "lambda_value": result.lambda_value,
                "scaling_factor": result.scaling_factor,
                "combined_sequences": len(result.alignment.sequences),
                "label": result.label,
            }
        )
        return summary

    return await _dispatch(
        ctx,
        kind="reintegration",
        tool="reintegrate_dca_model",
        arguments=call_arguments,
        background=background,
        parameters=parameters,
        work=work,
    )


@mcp.tool(structured_output=False)  # may return images
@surface_errors
def plot_dca_training_history(
    history_path: str,
    output_dir: str | None = None,
    return_image: bool = True,
) -> dict[str, Any] | list[Any]:
    """Plot every recorded training quantity (Pearson, slope, log-likelihood, density, PTT ...).

    history_path is the run's history.csv (or <label>_history.csv). Writes one PNG per panel
    plus <label>_overview.png, by default in <label>_plots next to the history file.
    Also returns the last recorded row and, unless return_image=false, the overview plot as
    an image. Works on a run that is still training (e.g. a background job).
    """
    from adabmDCA.plot_training_log import plot_training_history, read_training_history

    history_path = _path(history_path)
    written = plot_training_history(history_path, output_dir=_path(output_dir))
    history, context = read_training_history(history_path)
    last = history.iloc[-1].to_dict() if not history.empty else {}
    summary = compact(
        {
            "plots": written,
            "updates": len(history),
            "columns": list(history.columns),
            "last_row": last,
            "model": context.get("model"),
            "label": context.get("label"),
        },
        max_items=40,
    )
    overview = [written["overview"]] if "overview" in written else []
    return _with_images(summary, overview, return_images=return_image, max_images=1)


@mcp.tool()
@surface_errors
def diagnose_dca_run(
    run_dir: str,
    label: str | None = None,
    sampling_json: str | None = None,
) -> dict[str, Any]:
    """Diagnose a training run from its files: verdict, findings and the indicators behind them.

    Reads <label_>history.csv, <label_>events.jsonl, <label_>training.json and the log in
    run_dir, and optionally the sampling.json written by sample_dca_sequences (with
    reference_fasta, test_fasta and collect_diagnostics) to check the finished model.
    verdict is healthy, needs_attention, failed, running or incomplete (stopped without a
    summary, e.g. cancelled). Each finding has a severity, the failure it points to (F1-F8
    of adabmDCA's diagnostics guide) and the recommended change, written as adabmDCA CLI
    options (the dca-analysis skill maps them to this server's parameters). Works on a run
    that is still going. Same report as `adabmDCA diagnose`.
    """
    from adabmDCA.api.diagnostics import diagnose_run

    if label is not None:
        label = validated_label(label)
    return json_safe(diagnose_run(_path(run_dir), label=label, sampling_json=_path(sampling_json)))


# ---------------------------------------------------------------------------
# Entropy


@mcp.tool()
@surface_errors
async def estimate_dca_entropy(
    model_path: str,
    method: Literal["thermodynamic_integration", "ptt"] = "thermodynamic_integration",
    natural_alignment: str | None = None,
    target_alignment: str | None = None,
    output_dir: str | None = None,
    label: str = "entropy",
    n_chains: int = 8_192,
    n_sweeps: int = 100,
    n_steps: int = 100,
    theta_max: float = 5.0,
    target_fraction: float = 0.1,
    sampler: Sampler = "metropolized_gibbs",
    alphabet: str = "auto",
    seed: int = 0,
    device: Device = "auto",
    dtype: str = "float32",
    overwrite: bool = False,
    background: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Estimate the entropy (nats) of a DCA model.

    - "thermodynamic_integration": any parameter file; needs natural_alignment and a
      target_alignment whose first sequence is the integration target.
    - "ptt": a PTT archive (ptt.h5); cheap, uses the stored PTT bridges (n_sweeps sweeps).
    """
    call_arguments = _call_arguments(locals())
    label = validated_label(label) or "entropy"
    model_path = _path(model_path)
    folder = None
    if output_dir is not None:
        folder = prepare_output_dir(
            output_dir,
            managed_names=[f"{label}.json", f"{label}.csv", f"{label}.log"],
            overwrite=overwrite,
        )
    parameters = {"model_path": model_path, "method": method, "output_dir": _path(output_dir)}

    if method == "ptt":
        from adabmDCA import estimate_ptt_entropy

        def work(progress: Progress, is_cancelled: IsCancelled) -> dict[str, Any]:
            result = estimate_ptt_entropy(
                model=model_path,
                n_sweeps=n_sweeps,
                seed=seed,
                device="cpu" if device == "auto" else device,
                alphabet=None if alphabet == "auto" else alphabet,
                output_dir=folder,
                label=label,
                is_cancelled=is_cancelled,
            )
            return json_safe(result.to_dict()["data"])

    else:
        if natural_alignment is None or target_alignment is None:
            raise ValueError(
                "Thermodynamic integration needs natural_alignment and target_alignment."
            )
        from adabmDCA import estimate_entropy

        natural_alignment, target_alignment = _path(natural_alignment), _path(target_alignment)
        resolved = resolve_alphabet(alphabet, model=model_path, data=natural_alignment)

        def work(progress: Progress, is_cancelled: IsCancelled) -> dict[str, Any]:
            result = estimate_entropy(
                model=model_path,
                natural_alignment=natural_alignment,
                target_alignment=target_alignment,
                n_chains=n_chains,
                n_sweeps=n_sweeps,
                n_steps=n_steps,
                theta_max=theta_max,
                target_fraction=target_fraction,
                sampler=sampler,
                alphabet=resolved,
                seed=seed,
                device=device,
                dtype=dtype,
                output_dir=folder,
                label=label,
                progress=progress,
                is_cancelled=is_cancelled,
            )
            return compact(
                {
                    "entropy": result.entropy,
                    "free_energy": result.free_energy,
                    "theta_max": result.theta_max,
                    "target_fraction": result.target_fraction,
                    "integration_steps": len(next(iter(result.history.values()), [])),
                    "model": _metadata_dict(result.model),
                    "artifacts": result.artifacts,
                }
            )

    return await _dispatch(
        ctx,
        kind="entropy",
        tool="estimate_dca_entropy",
        arguments=call_arguments,
        background=background,
        parameters=parameters,
        work=work,
    )


# ---------------------------------------------------------------------------
# Background jobs


@mcp.tool()
@surface_errors
def get_dca_job(job_id: str, wait_seconds: float = 0.0) -> dict[str, Any]:
    """Report a background job: status, latest progress and, when finished, its result.

    Jobs persist on disk, so this works for jobs started in earlier chats. wait_seconds
    (max 300) blocks until the job finishes or the time elapses. Status is one of starting,
    running, succeeded, failed, cancelled or interrupted (worker died; see worker_log).
    """
    if wait_seconds < 0:
        raise ValueError("wait_seconds cannot be negative.")
    if wait_seconds:
        return JOBS.wait(job_id, timeout=min(wait_seconds, 300.0))
    return JOBS.get(job_id)


@mcp.tool()
@surface_errors
def list_dca_jobs(
    limit: int = 20,
    status: Literal["starting", "running", "succeeded", "failed", "cancelled", "interrupted"]
    | None = None,
) -> dict[str, Any]:
    """List background jobs, newest first, including those started in earlier chats."""
    from agentdca.jobs import jobs_home

    return {"jobs_home": str(jobs_home()), "jobs": JOBS.list(limit=limit, status=status)}


@mcp.tool()
@surface_errors
def cancel_dca_job(job_id: str, force: bool = False) -> dict[str, Any]:
    """Ask a background job to stop; adabmDCA stops at its next safe point.

    force=true terminates the worker process immediately (use only if cancel is ignored).
    """
    if force:
        return JOBS.kill(job_id)
    return JOBS.cancel(job_id)


@mcp.tool(structured_output=False)  # may return images
@surface_errors
def show_dca_plots(paths: list[str], max_images: int = 4) -> list[Any]:
    """Return existing PNG plots as images so they can be inspected directly.

    Use it for plots written by background jobs or earlier runs (e.g. a job result's
    artifacts, or plot_dca_training_history output).
    """
    resolved = [Path(path).expanduser() for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Plot files not found: {', '.join(missing)}")
    not_png = [str(path) for path in resolved if path.suffix.lower() != ".png"]
    if not_png:
        raise ValueError(f"Only PNG plots can be shown: {', '.join(not_png)}")
    selected = resolved[: max(max_images, 0)]
    return [
        {"shown": [str(path) for path in selected], "skipped": len(resolved) - len(selected)},
        *(_image(path) for path in selected),
    ]


def main() -> None:
    """Run the local stdio MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
