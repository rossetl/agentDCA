import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import adabmDCA.exceptions  # noqa: F401  (real exceptions stay importable under fakes)
import numpy as np
import pytest
import tomllib
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image

from agentdca._common import (
    bounded,
    json_safe,
    prepare_file_output,
    prepare_output_dir,
    resolve_alphabet,
    surface_errors,
    training_artifact_names,
    validated_label,
)
from agentdca.diagnose import diagnose_run
from agentdca.jobs import JobStore, WorkerContext, progress_snapshot
from agentdca.models import ModelCache
from agentdca.server import (
    _hamming_potential,
    _top_contacts,
    compare_dca_contacts,
    mcp,
    predict_dca_contacts,
    sample_dca_sequences,
    score_dca_sequences,
    train_dca_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _PassthroughModels:
    """Stand-in for the model cache: hands the path to the (faked) adabmDCA functions."""

    def model(self, path, **_):
        return path

    def ptt_sampler(self, path, **_):
        return path


@pytest.fixture(autouse=True)
def _no_model_cache(monkeypatch):
    monkeypatch.setattr("agentdca.server.MODELS", _PassthroughModels())


def test_plugin_launch_uses_locked_adabmdca_develop_dependency():
    mcp_config = json.loads((PROJECT_ROOT / ".mcp.json").read_text())
    server_config = mcp_config["mcpServers"]["agentdca"]
    args = server_config["args"]
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())

    assert (
        "adabmDCA[cpu] @ git+https://github.com/spqb/adabmDCApy.git@develop"
        in project["project"]["dependencies"]
    )
    assert args[:2] == ["run", "--locked"]
    assert "--upgrade-package" not in args
    adabmdca = next(package for package in lock["package"] if package["name"] == "adabmdca")
    assert adabmdca["source"]["git"].startswith(
        "https://github.com/spqb/adabmDCApy.git?rev=develop#"
    )
    assert int(adabmdca["version"].split(".")[0]) >= 2
    assert server_config["env"]["MPLCONFIGDIR"] == "/tmp/agentdca-matplotlib"


def test_plugin_manifests_share_version():
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    version = project["project"]["version"]
    codex = json.loads((PROJECT_ROOT / ".codex-plugin" / "plugin.json").read_text())
    claude = json.loads((PROJECT_ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert codex["version"].split("+")[0] == version
    assert claude["version"] == version
    # Claude Code also loads the root .mcp.json (written for Codex's ${PLUGIN_ROOT});
    # the same-named inline server replaces it with Claude Code's root variable.
    codex_server = json.loads((PROJECT_ROOT / ".mcp.json").read_text())["mcpServers"]
    claude_server = claude["mcpServers"]["agentdca"]
    assert set(codex_server) == {"agentdca"}
    assert "${CLAUDE_PLUGIN_ROOT}/pyproject.toml" in claude_server["args"]
    assert claude_server["args"][:2] == codex_server["agentdca"]["args"][:2]
    marketplace = json.loads((PROJECT_ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert marketplace["plugins"][0]["name"] == claude["name"]


def _metadata():
    return SimpleNamespace(
        to_dict=lambda: {
            "length": 2,
            "alphabet": "protein",
            "tokens": "-A",
            "device": "cpu",
            "dtype": "float32",
        }
    )


def _fake_adabmdca(monkeypatch, **functions):
    monkeypatch.setitem(sys.modules, "adabmDCA", SimpleNamespace(**functions))


# ---------------------------------------------------------------------------
# Helpers


def test_top_contacts_returns_ranked_upper_triangle():
    scores = np.array([[0.0, 0.2, 0.9], [0.2, 0.0, 0.5], [0.9, 0.5, 0.0]])

    contacts = _top_contacts(scores, top_k=2, min_sequence_separation=1)

    assert [(item["position_i"], item["position_j"]) for item in contacts] == [(0, 2), (1, 2)]
    assert contacts[0]["position_i_1based"] == 1
    assert contacts[0]["position_j_1based"] == 3


@pytest.mark.parametrize("top_k,separation", [(0, 1), (1, 0)])
def test_top_contacts_validates_bounds(top_k, separation):
    with pytest.raises(ValueError):
        _top_contacts(np.zeros((2, 2)), top_k=top_k, min_sequence_separation=separation)


def test_json_safe_replaces_non_finite_scientific_values():
    result = json_safe({"array": np.array([1.0, np.nan, np.inf]), "scalar": np.float32(-np.inf)})

    assert result == {"array": [1.0, None, None], "scalar": None}
    json.dumps(result, allow_nan=False)


def test_bounded_summarizes_long_lists():
    value = bounded({"short": [1, 2], "long": list(range(50))}, max_items=5)

    assert value["short"] == [1, 2]
    assert value["long"] == {"count": 50, "preview": [0, 1, 2, 3, 4], "truncated": True}


def test_output_guards_refuse_overwrite(tmp_path):
    output_file = tmp_path / "scores.csv"
    output_file.write_text("existing")
    with pytest.raises(FileExistsError, match="overwrite=true"):
        prepare_file_output(str(output_file), overwrite=False)

    output_dir = tmp_path / "model"
    output_dir.mkdir()
    (output_dir / "params.dat.gz").write_text("existing")
    names = training_artifact_names(None).values()
    with pytest.raises(FileExistsError, match="params.dat.gz"):
        prepare_output_dir(str(output_dir), managed_names=names, overwrite=False)
    assert prepare_output_dir(str(output_dir), managed_names=names, overwrite=True) == output_dir


def test_training_artifact_names_follow_adabmdca_2():
    assert training_artifact_names(None)["params"] == "params.dat.gz"
    assert training_artifact_names(None)["log"] == "adabmDCA.log"
    labeled = training_artifact_names("edge", ptt=True)
    assert labeled["params"] == "edge_params.dat.gz"
    assert labeled["history"] == "edge_history.csv"
    assert labeled["log"] == "edge.log"
    assert labeled["ptt_archive"] == "edge_ptt.h5"


def test_labeled_runs_can_share_a_directory(tmp_path):
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    (output_dir / "params.dat.gz").write_text("existing default model")

    names = training_artifact_names("edgeDCA").values()
    assert prepare_output_dir(str(output_dir), managed_names=names, overwrite=False) == output_dir


def test_prefix_guard_detects_reintegration_outputs(tmp_path):
    (tmp_path / "reintegrated-lambda_0.50_params.dat.gz").write_text("x")
    with pytest.raises(FileExistsError, match="reintegrated-lambda_0.50"):
        prepare_output_dir(
            str(tmp_path), managed_prefixes=["reintegrated-lambda_"], overwrite=False
        )


@pytest.mark.parametrize("label", ["", "   ", "../model", "nested/model", "nested\\model"])
def test_label_must_be_filename_safe(label):
    with pytest.raises(ValueError, match="Label"):
        validated_label(label)


def test_explicit_alphabet_is_kept():
    assert resolve_alphabet("rna", model="anything") == "rna"
    assert resolve_alphabet("auto") is None


def test_surface_errors_exposes_adabmdca_messages():
    from adabmDCA.exceptions import ModelLoadError

    @surface_errors
    def broken():
        raise ModelLoadError("missing file", details={"path": "/x"})

    with pytest.raises(ToolError, match=r"ModelLoadError \[model_load_error\]: missing file"):
        broken()


def test_hamming_potential_counts_mismatches():
    potential = _hamming_potential("AC-G")
    assert list(potential(["AC-G", "AA-G", "----"], 2.0)) == [0.0, 2.0, 6.0]


def test_progress_snapshot_keeps_metrics_and_drops_none():
    event = SimpleNamespace(epoch=4, metrics={"Pearson": np.float32(0.5)}, stage="optimization")
    assert progress_snapshot(event) == {
        "stage": "optimization",
        "epoch": 4,
        "metrics": {"Pearson": 0.5},
    }


def test_job_store_runs_a_detached_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDCA_HOME", str(tmp_path))
    store = JobStore()

    job = store.submit("test", "list_dca_jobs", {"limit": 1}, {"purpose": "test"})
    assert job["status"] in {"starting", "running", "succeeded"}
    finished = store.wait(job["job_id"], timeout=120)

    assert finished["status"] == "succeeded", Path(finished["worker_log"]).read_text()
    assert "jobs" in finished["result"]
    assert store.list(limit=5)[0]["job_id"] == job["job_id"]
    assert "result" not in store.list(limit=5)[0]
    with pytest.raises(KeyError, match="Unknown job"):
        store.get("missing")


def test_job_store_marks_dead_workers_interrupted(tmp_path):
    store = JobStore(tmp_path)
    directory = tmp_path / "training-x"
    directory.mkdir()
    (directory / "spec.json").write_text(json.dumps({"tool": "train_dca_model"}))
    (directory / "status.json").write_text(json.dumps({"status": "running", "pid": 2**22 + 7}))

    job = store.get("training-x")

    assert job["status"] == "interrupted"
    assert "checkpoint" in job["error"]
    assert store.cancel("training-x")["cancel_requested"] is False


def test_worker_context_records_progress_and_cancellation(tmp_path):
    context = WorkerContext(tmp_path)
    context.update(status="running")
    context.progress(SimpleNamespace(epoch=3, metrics={"Pearson": 0.4}))
    status = json.loads((tmp_path / "status.json").read_text())

    assert status["latest_progress"]["metrics"]["Pearson"] == 0.4
    assert context.is_cancelled() is False
    (tmp_path / "cancel").write_text("now")
    assert context.is_cancelled() is True


# ---------------------------------------------------------------------------
# Tools with a fake adabmDCA


def test_score_writes_all_records_and_bounds_inline_payload(tmp_path, monkeypatch):
    written = {}

    def to_csv(path):
        written["path"] = path

    result = SimpleNamespace(
        names=("first", "second", "third"),
        sequences=("AA", "A-", "--"),
        energies=np.array([1.0, 2.0, 3.0]),
        cde_sum=np.array([0.1, 0.2, 0.3]),
        local_free_energies=np.array([0.9, 1.8, 2.7]),
        local_lambda=1.0,
        model=_metadata(),
        warnings=(),
        to_csv=to_csv,
    )
    _fake_adabmdca(monkeypatch, score_sequences=lambda **kwargs: result)
    output = tmp_path / "scores.csv"

    response = score_dca_sequences(
        "params.dat",
        sequences=list(result.sequences),
        alphabet="protein",
        output_csv=str(output),
        max_inline_sequences=2,
    )

    assert response["num_sequences"] == 3
    assert response["returned_sequences"] == 2
    assert response["truncated"] is True
    assert response["sequences"] == ["AA", "A-"]
    assert response["cde_sum"] == pytest.approx([0.1, 0.2])
    assert response["energy_stats"]["max"] == 3.0
    assert written["path"] == output


def _sampling_result():
    result = SimpleNamespace(
        sequences=("AA", "A-", "--"),
        energies=np.array([1.0, 2.0, 3.0]),
        num_sweeps=10,
        sampler="metropolized_gibbs",
        beta=1.0,
        seed=7,
        sampling_dtype="float32",
        local_lambda_fit=None,
        sampling_history={},
        ptt_diagnostics={},
        distance_comparison={},
        steering={},
        model=_metadata(),
        warnings=(),
    )

    def to_fasta(path):
        content = "".join(f">sequence_{i}\n{s}\n" for i, s in enumerate(result.sequences, 1))
        Path(path).write_text(content)
        return Path(path)

    result.to_fasta = to_fasta
    return result


def test_sample_writes_all_sequences_and_bounds_inline_payload(tmp_path, monkeypatch):
    received = {}

    def fake_sample(**kwargs):
        received.update(kwargs)
        return _sampling_result()

    _fake_adabmdca(monkeypatch, sample_sequences=fake_sample)
    output = tmp_path / "samples.fasta"

    response = asyncio.run(
        sample_dca_sequences(
            "params.dat",
            n_sequences=3,
            alphabet="protein",
            output_fasta=str(output),
            max_inline_sequences=1,
        )
    )

    assert response["num_sequences"] == 3
    assert response["returned_sequences"] == 1
    assert response["sequences"] == ["AA"]
    assert output.read_text().count(">sequence_") == 3
    assert received["sampler"] == "metropolized_gibbs"
    assert received["ptt"] is False


def test_sample_validates_ptt_and_steering_inputs(monkeypatch):
    _fake_adabmdca(monkeypatch, sample_sequences=lambda **kwargs: _sampling_result())
    with pytest.raises(ToolError, match="reference_fasta"):
        asyncio.run(sample_dca_sequences("ptt.h5", n_sequences=2, ptt=True, alphabet="protein"))
    with pytest.raises(ToolError, match="at most one"):
        asyncio.run(
            sample_dca_sequences(
                "params.dat",
                n_sequences=2,
                alphabet="protein",
                steering_target="AA",
                steering_script="x.py:f",
            )
        )


def test_contacts_write_complete_symmetrized_matrix(tmp_path, monkeypatch):
    scores = np.array([[0.0, 1.0, 4.0], [1.2, 0.0, 3.0], [3.6, 3.4, 0.0]])
    result = SimpleNamespace(
        scores=scores,
        method="mean_field",
        alphabet="protein",
        tokens="-A",
        model=None,
        warnings=(),
    )
    _fake_adabmdca(monkeypatch, predict_contacts=lambda **kwargs: result)
    output_csv = tmp_path / "contacts.csv"
    output_npy = tmp_path / "contacts.npy"

    response = predict_dca_contacts(
        fasta_path="alignment.fasta",
        alphabet="protein",
        top_k=1,
        min_sequence_separation=1,
        output_csv=str(output_csv),
        output_npy=str(output_npy),
    )

    saved_csv = np.loadtxt(output_csv, delimiter=",")
    saved_npy = np.load(output_npy)
    assert np.allclose(saved_csv, saved_csv.T)
    assert np.allclose(saved_npy, saved_csv)
    assert saved_npy[0, 2] == pytest.approx(3.8)
    assert response["contacts"][0]["score"] == pytest.approx(3.8)
    assert response["matrix"]["symmetrized"] is True
    assert response["matrix"]["max_abs_asymmetry_before"] == pytest.approx(0.4)


@pytest.mark.parametrize("color_scale", ["shared", "per_panel"])
def test_compare_contacts_writes_side_by_side_plot(tmp_path, monkeypatch, color_scale):
    model_result = SimpleNamespace(
        scores=np.array([[0.0, 20.0], [20.0, 0.0]]),
        method="model",
        alphabet="protein",
        tokens="-A",
        model=_metadata(),
        warnings=(),
    )
    mean_field_result = SimpleNamespace(
        scores=np.array([[0.0, 1.0], [1.00002, 0.0]]),
        method="mean_field",
        alphabet="protein",
        tokens="-A",
        model=None,
        warnings=(),
    )

    def fake_predict_contacts(**kwargs):
        return model_result if kwargs.get("model") else mean_field_result

    _fake_adabmdca(monkeypatch, predict_contacts=fake_predict_contacts)
    output = tmp_path / "comparison.png"

    response, image = compare_dca_contacts(
        "params.dat",
        "alignment.fasta",
        str(output),
        alphabet="protein",
        top_k=1,
        min_sequence_separation=1,
        color_scale=color_scale,
    )

    assert output.read_bytes().startswith(b"\x89PNG")
    assert isinstance(image, Image)
    assert image.to_image_content().mime_type == "image/png"
    assert response["images_returned"] == [str(output)]
    assert response["color_scale"]["mode"] == color_scale
    same = response["color_scale"]["mean_field"] == response["color_scale"]["model"]
    assert same is (color_scale == "shared")
    assert response["model"]["metadata"]["device"] == "cpu"


def _training_result(tmp_path, *, converged):
    def save_bundle(directory, label=None):
        return {"summary": Path(directory) / f"{label + '_' if label else ''}training.json"}

    return SimpleNamespace(
        model=SimpleNamespace(metadata=_metadata()),
        history={"Pearson": [0.1, 0.2], "LL_val": [np.nan, np.nan]},
        pseudocount=0.01,
        num_sequences=10,
        effective_sequences=8.5,
        artifacts={"params": tmp_path / "model" / "params.dat.gz"},
        converged=converged,
        stop_reason="target_pearson" if converged else "max_gradient_steps",
        gradient_steps=5,
        structure_steps=2,
        sweeps=50,
        final_metrics={"Pearson": 0.2, "LL_val": np.nan},
        config=None,
        initialization=None,
        partition_estimate=None,
        input_report={"retained_indices": list(range(420))},
        warnings=(),
        save_bundle=save_bundle,
    )


def test_training_reports_native_convergence_and_bounds_input_report(tmp_path, monkeypatch):
    received = {}

    def fake_train_model(data_path, **kwargs):
        received.update(kwargs)
        kwargs["progress"](SimpleNamespace(epoch=5, metrics={"Pearson": 0.2}))
        return _training_result(tmp_path, converged=False)

    _fake_adabmdca(monkeypatch, train_model=fake_train_model)

    response = asyncio.run(
        train_dca_model(
            "alignment.fasta",
            str(tmp_path / "model"),
            alphabet="protein",
            label="edgeDCA",
            max_epochs=5,
        )
    )

    assert received["label"] == "edgeDCA"
    assert received["max_epochs"] == 5
    assert "learning_rate" not in received  # unset options keep adabmDCA defaults
    assert received["ptt"] is None
    assert response["converged"] is False
    assert response["stop_reason"] == "max_gradient_steps"
    assert response["final_metrics"]["LL_val"] is None
    assert response["input_report"]["retained_indices"]["count"] == 420
    assert response["artifacts"]["summary"].endswith("edgeDCA_training.json")
    assert "without converging" in response["warnings"][-1]
    json.dumps(response, allow_nan=False)


def test_training_refuses_existing_artifacts(tmp_path, monkeypatch):
    _fake_adabmdca(
        monkeypatch, train_model=lambda *a, **k: _training_result(tmp_path, converged=True)
    )
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "history.csv").write_text("x")

    with pytest.raises(ToolError, match="history.csv"):
        asyncio.run(train_dca_model("a.fasta", str(tmp_path / "model"), alphabet="protein"))


def test_training_in_background_submits_replayable_call(tmp_path, monkeypatch):
    from agentdca import server

    submitted = {}

    def fake_submit(kind, tool, arguments, parameters):
        submitted.update(kind=kind, tool=tool, arguments=arguments)
        return {"job_id": "training-1", "status": "starting", "job_dir": str(tmp_path)}

    _fake_adabmdca(monkeypatch, train_model=lambda *a, **k: pytest.fail("ran in foreground"))
    monkeypatch.setattr(server.JOBS, "submit", fake_submit)

    started = asyncio.run(
        train_dca_model("a.fasta", str(tmp_path / "model"), alphabet="protein", background=True)
    )

    assert started["job_id"] == "training-1"
    assert submitted["tool"] == "train_dca_model"
    assert submitted["arguments"]["data_path"] == "a.fasta"
    assert "ctx" not in submitted["arguments"]
    json.dumps(submitted["arguments"])


def test_mcp_registers_all_public_tools():
    tools = asyncio.run(mcp.list_tools())

    assert {tool.name for tool in tools} == {
        "inspect_dca_model",
        "score_dca_sequences",
        "predict_dca_contacts",
        "compare_dca_contacts",
        "scan_dca_mutations",
        "sample_dca_sequences",
        "preprocess_dca_alignment",
        "split_dca_alignment",
        "train_dca_model",
        "reintegrate_dca_model",
        "plot_dca_training_history",
        "estimate_dca_entropy",
        "get_dca_job",
        "list_dca_jobs",
        "cancel_dca_job",
        "show_dca_plots",
        "diagnose_dca_run",
    }
    for tool in tools:
        assert "ctx" not in tool.input_schema.get("properties", {})
    image_tools = {
        "compare_dca_contacts",
        "sample_dca_sequences",
        "plot_dca_training_history",
        "show_dca_plots",
    }
    # Image content cannot go through structured output.
    assert {tool.name for tool in tools if tool.output_schema is None} >= image_tools


def test_model_cache_reuses_until_the_file_changes(tmp_path, monkeypatch):
    loads = []

    def fake_load(path, **options):
        loads.append(path)
        return object()

    monkeypatch.setitem(sys.modules, "adabmDCA", SimpleNamespace(load_model=fake_load))
    first, second = tmp_path / "a.dat", tmp_path / "b.dat"
    first.write_text("x")
    second.write_text("y")
    cache = ModelCache(capacity=1)
    options = {"alphabet": "protein", "device": "cpu", "dtype": "float32"}

    model = cache.model(str(first), **options)
    assert cache.model(str(first), **options) is model
    assert cache.model(str(first), **{**options, "dtype": "float64"}) is not model
    first.write_text("retrained")
    assert cache.model(str(first), **options) is not model
    cache.model(str(second), **options)
    assert len(cache.info()["entries"]) == 1
    assert len(loads) == 4

    disabled = ModelCache(capacity=0)
    assert disabled.model(str(second), **options) is not disabled.model(str(second), **options)


def _write_ptt_run(directory, *, updates=60, lr_bias=0.01, kl=0.001, events=(), summary=None):
    directory.mkdir()
    header = "step,sweeps,time_s,pearson,slope,ll_train,acceptance_min,lr_bias,lr_coupling,kl,"
    header += "lag_drift,lag_drift_tail,pearson_val,ll_val"
    rows = [header]
    for step in range(updates):
        rows.append(
            f"{step},{10 * step},{step},{0.5 + step / 200},1,-1.5,0.4,{lr_bias},0.01,{kl},"
            f"0.01,0.02,{0.5 + step / 250},{-1.6 + step / 1000}"
        )
    (directory / "history.csv").write_text("\n".join(rows) + "\n")
    lines = [{"event": "ptt_optimization", "step": 0}, *events]
    (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    if summary is not None:
        (directory / "training.json").write_text(json.dumps({"data": summary}))


def test_diagnose_healthy_ptt_run(tmp_path):
    run = tmp_path / "run"
    summary = {"stop_reason": "validation_plateau", "converged": True, "warnings": []}
    _write_ptt_run(run, summary=summary)

    report = diagnose_run(run)

    assert report["verdict"] == "healthy"
    assert report["training_kind"] == "ptt"
    assert report["training"]["updates"] == 59
    assert [finding["code"] for finding in report["findings"]] == ["unchecked_model"]


def test_diagnose_flags_trapped_configurations_and_large_learning_rate(tmp_path):
    run = tmp_path / "run"
    failed_check = {
        "event": "ptt_mixing",
        "step": 40,
        "status": "budget_exceeded",
        "immobile": [{"replica": 2, "immobile": 61, "blocking": True}],
    }
    recoveries = [{"event": "ptt_recovery", "step": 40}] * 3
    _write_ptt_run(
        run,
        lr_bias=1e-14,
        kl=0.01,
        events=[failed_check, *recoveries],
        summary={"stop_reason": None, "converged": False},
    )

    report = diagnose_run(run)

    assert report["verdict"] == "failed"
    failures = {finding["code"]: finding.get("failure") for finding in report["findings"]}
    assert failures["trapped_configurations"] == "F1"
    assert failures["learning_rate_too_large"] == "F5"
    assert report["training"]["blocking_immobile_configurations"] == 61


def test_diagnose_compares_sampling_with_training(tmp_path):
    run = tmp_path / "run"
    _write_ptt_run(run, summary={"stop_reason": "validation_plateau", "converged": True})
    sampling = tmp_path / "sampling.json"
    sampling.write_text(
        json.dumps(
            {
                "data": {
                    "ptt_diagnostics": {
                        "converged": True,
                        "final_pearson": 0.70,
                        "mixing": {"status": "converged", "trapped_fraction": 0.0},
                    },
                    "distance_comparison": {
                        "summary": {"generated_to_natural": {"median": 0.3, "identical": 0.01}},
                        "privet": {"n_flagged": 2, "n_memorized": 1},
                    },
                }
            }
        )
    )

    report = diagnose_run(run, sampling_json=sampling)

    codes = {finding["code"] for finding in report["findings"]}
    assert {"sampling_pearson_drop", "memorization"} <= codes
    assert "unchecked_model" not in codes
    assert report["sampling"]["pearson_drop"] == pytest.approx(0.095, abs=1e-3)
    assert report["verdict"] == "needs_attention"


def test_diagnose_reports_a_running_job_and_a_missing_run(tmp_path):
    run = tmp_path / "run"
    _write_ptt_run(run)
    assert diagnose_run(run)["verdict"] == "running"
    (tmp_path / "labeled").mkdir()
    (tmp_path / "labeled" / "x_history.csv").write_text("step\n")
    with pytest.raises(FileNotFoundError, match="pass the matching label"):
        diagnose_run(tmp_path / "labeled")
