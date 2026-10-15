import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import adabmDCA.exceptions  # noqa: F401  (real exceptions stay importable under fakes)
import numpy as np
import pytest
import tomllib
from mcp.server.mcpserver.exceptions import ToolError

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
from agentdca.jobs import JobManager, progress_snapshot
from agentdca.server import (
    _hamming_potential,
    _top_contacts,
    compare_dca_contacts,
    get_dca_job,
    mcp,
    predict_dca_contacts,
    sample_dca_sequences,
    score_dca_sequences,
    train_dca_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_job_manager_runs_and_cancels():
    from adabmDCA.exceptions import OperationCancelledError

    jobs = JobManager()

    def slow(progress, is_cancelled):
        for step in range(500):
            progress(SimpleNamespace(completed=step, total=500))
            if is_cancelled():
                raise OperationCancelledError("stopped")
            time.sleep(0.01)
        return {"done": True}

    job = jobs.start("test", slow, {})
    time.sleep(0.05)
    jobs.cancel(job.id)
    jobs.wait(job.id, timeout=5)
    assert job.status == "cancelled"
    assert job.progress["total"] == 500

    quick = jobs.start("test", lambda progress, is_cancelled: {"value": 1}, {})
    jobs.wait(quick.id, timeout=5)
    assert quick.status == "succeeded"
    assert quick.to_dict()["result"] == {"value": 1}
    with pytest.raises(KeyError, match="Unknown job"):
        jobs.get("missing")


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

    response = compare_dca_contacts(
        "params.dat",
        "alignment.fasta",
        str(output),
        alphabet="protein",
        top_k=1,
        min_sequence_separation=1,
        color_scale=color_scale,
    )

    assert output.read_bytes().startswith(b"\x89PNG")
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


def test_training_in_background_returns_job(tmp_path, monkeypatch):
    _fake_adabmdca(
        monkeypatch, train_model=lambda *a, **k: _training_result(tmp_path, converged=True)
    )

    started = asyncio.run(
        train_dca_model("a.fasta", str(tmp_path / "model"), alphabet="protein", background=True)
    )
    report = get_dca_job(started["job_id"], wait_seconds=5)

    assert report["status"] == "succeeded"
    assert report["result"]["converged"] is True


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
    }
    for tool in tools:
        assert "ctx" not in tool.input_schema.get("properties", {})
