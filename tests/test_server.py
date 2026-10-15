import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tomllib

from agentdca.server import (
    _json_safe,
    _prepare_file_output,
    _prepare_training_output,
    _top_contacts,
    compare_dca_contacts,
    mcp,
    predict_dca_contacts,
    sample_dca_sequences,
    score_dca_sequences,
    train_dca_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_plugin_launch_uses_locked_adabmdca_agent_api_dependency():
    mcp_config = json.loads((PROJECT_ROOT / ".mcp.json").read_text())
    server_config = mcp_config["mcpServers"]["agentdca"]
    args = server_config["args"]
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())

    assert (
        "adabmDCA @ git+https://github.com/spqb/adabmDCApy.git@agent-api"
        in project["project"]["dependencies"]
    )
    assert args[:2] == ["run", "--locked"]
    assert "--upgrade-package" not in args
    adabmdca = next(package for package in lock["package"] if package["name"] == "adabmdca")
    assert adabmdca["source"]["git"].startswith(
        "https://github.com/spqb/adabmDCApy.git?rev=agent-api#"
    )
    assert server_config["env"]["MPLCONFIGDIR"] == "/tmp/agentdca-matplotlib"


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


def test_top_contacts_returns_ranked_upper_triangle():
    scores = np.array(
        [
            [0.0, 0.2, 0.9],
            [0.2, 0.0, 0.5],
            [0.9, 0.5, 0.0],
        ]
    )

    contacts = _top_contacts(scores, top_k=2, min_sequence_separation=1)

    assert [(item["position_i"], item["position_j"]) for item in contacts] == [
        (0, 2),
        (1, 2),
    ]
    assert contacts[0]["position_i_1based"] == 1
    assert contacts[0]["position_j_1based"] == 3


@pytest.mark.parametrize("top_k,separation", [(0, 1), (1, 0)])
def test_top_contacts_validates_bounds(top_k, separation):
    with pytest.raises(ValueError):
        _top_contacts(
            np.zeros((2, 2)),
            top_k=top_k,
            min_sequence_separation=separation,
        )


def test_json_safe_replaces_non_finite_scientific_values():
    result = _json_safe(
        {
            "array": np.array([1.0, np.nan, np.inf]),
            "scalar": np.float32(-np.inf),
        }
    )

    assert result == {"array": [1.0, None, None], "scalar": None}
    json.dumps(result, allow_nan=False)


def test_output_guards_refuse_overwrite(tmp_path):
    output_file = tmp_path / "scores.csv"
    output_file.write_text("existing")
    with pytest.raises(FileExistsError, match="overwrite=true"):
        _prepare_file_output(str(output_file), overwrite=False)

    output_dir = tmp_path / "model"
    output_dir.mkdir()
    (output_dir / "params.dat").write_text("existing")
    with pytest.raises(FileExistsError, match="params.dat"):
        _prepare_training_output(str(output_dir), overwrite=False)


def test_labeled_training_can_share_directory_without_overwriting(tmp_path):
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    (output_dir / "params.dat").write_text("existing default model")

    output, history = _prepare_training_output(
        str(output_dir),
        overwrite=False,
        label="edgeDCA",
    )

    assert output == output_dir
    assert history == output_dir / "edgeDCA_training_history.json"
    (output_dir / "edgeDCA_params.dat").write_text("existing labeled model")
    with pytest.raises(FileExistsError, match="edgeDCA_params.dat"):
        _prepare_training_output(str(output_dir), overwrite=False, label="edgeDCA")


@pytest.mark.parametrize("label", ["", "   ", "../model", "nested/model", "nested\\model"])
def test_training_label_must_be_filename_safe(tmp_path, label):
    with pytest.raises(ValueError, match="label"):
        _prepare_training_output(str(tmp_path / "model"), overwrite=False, label=label)


def test_score_writes_all_records_and_bounds_inline_payload(tmp_path, monkeypatch):
    result = SimpleNamespace(
        names=("first", "second", "third"),
        sequences=("AA", "A-", "--"),
        energies=np.array([1.0, 2.0, 3.0]),
        model=_metadata(),
        warnings=(),
    )
    monkeypatch.setitem(
        sys.modules,
        "adabmDCA",
        SimpleNamespace(score_sequences=lambda **kwargs: result),
    )
    output = tmp_path / "scores.csv"

    response = score_dca_sequences(
        "params.dat",
        sequences=list(result.sequences),
        output_csv=str(output),
        max_inline_sequences=2,
    )

    assert response["num_sequences"] == 3
    assert response["returned_sequences"] == 2
    assert response["truncated"] is True
    assert response["sequences"] == ["AA", "A-"]
    assert output.read_text().splitlines() == [
        "name,sequence,energy",
        "first,AA,1.0",
        "second,A-,2.0",
        "third,--,3.0",
    ]


def test_sample_writes_all_sequences_and_bounds_inline_payload(tmp_path, monkeypatch):
    result = SimpleNamespace(
        sequences=("AA", "A-", "--"),
        energies=np.array([1.0, 2.0, 3.0]),
        num_sweeps=10,
        sampler="gibbs",
        beta=1.0,
        seed=7,
        model=_metadata(),
        warnings=(),
    )
    monkeypatch.setitem(
        sys.modules,
        "adabmDCA",
        SimpleNamespace(sample_sequences=lambda **kwargs: result),
    )
    output = tmp_path / "samples.fasta"

    response = sample_dca_sequences(
        "params.dat",
        n_sequences=3,
        output_fasta=str(output),
        max_inline_sequences=1,
    )

    assert response["num_sequences"] == 3
    assert response["returned_sequences"] == 1
    assert response["truncated"] is True
    assert response["sequences"] == ["AA"]
    assert output.read_text().count(">sequence_") == 3


def test_contacts_write_complete_symmetrized_matrix(tmp_path, monkeypatch):
    scores = np.array(
        [
            [0.0, 1.0, 4.0],
            [1.2, 0.0, 3.0],
            [3.6, 3.4, 0.0],
        ]
    )
    result = SimpleNamespace(
        scores=scores,
        method="mean_field",
        alphabet="protein",
        tokens="-A",
        model=None,
        warnings=(),
    )
    monkeypatch.setitem(
        sys.modules,
        "adabmDCA",
        SimpleNamespace(predict_contacts=lambda **kwargs: result),
    )
    output_csv = tmp_path / "contacts.csv"
    output_npy = tmp_path / "contacts.npy"

    response = predict_dca_contacts(
        fasta_path="alignment.fasta",
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
    assert response["matrix"]["output_csv"] == str(output_csv)
    assert response["matrix"]["output_npy"] == str(output_npy)


def test_compare_contacts_writes_side_by_side_plot(tmp_path, monkeypatch):
    model_result = SimpleNamespace(
        scores=np.array([[0.0, 2.0], [2.0, 0.0]]),
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

    monkeypatch.setitem(
        sys.modules,
        "adabmDCA",
        SimpleNamespace(predict_contacts=fake_predict_contacts),
    )
    output = tmp_path / "comparison.png"

    response = compare_dca_contacts(
        "params.dat",
        "alignment.fasta",
        str(output),
        top_k=1,
        min_sequence_separation=1,
    )

    assert output.read_bytes().startswith(b"\x89PNG")
    assert response["output_plot"] == str(output)
    assert response["sequence_length"] == 2
    assert response["mean_field"]["symmetry"]["max_abs_asymmetry_before"] == pytest.approx(0.00002)
    assert response["model"]["metadata"]["device"] == "cpu"


def test_training_sanitizes_progress_and_reports_non_convergence(tmp_path, monkeypatch):
    def fake_train_model(data_path, *, progress, **kwargs):
        progress(
            SimpleNamespace(
                epoch=5,
                metrics={"Pearson": 0.2, "LL_val": np.nan, "Slope_val": np.inf},
            )
        )
        return SimpleNamespace(
            model=SimpleNamespace(metadata=_metadata()),
            history={"Pearson": [0.1, 0.2], "LL_val": [np.nan, np.nan]},
            pseudocount=0.01,
            num_sequences=10,
            effective_sequences=8.5,
            artifacts={},
            converged=False,
            stop_reason="max_gradient_steps",
            gradient_steps=5,
            structure_steps=2,
            sweeps=50,
            input_report={"kept_sequences": 10},
            warnings=(),
        )

    monkeypatch.setitem(
        sys.modules,
        "adabmDCA",
        SimpleNamespace(train_model=fake_train_model),
    )
    output = tmp_path / "model"

    response = train_dca_model(
        "alignment.fasta",
        str(output),
        max_epochs=5,
        target_pearson=0.95,
    )

    assert response["epochs_completed"] == 5
    assert response["converged"] is False
    assert response["stop_reason"] == "max_gradient_steps"
    assert response["gradient_steps"] == 5
    assert response["structure_steps"] == 2
    assert response["sweeps"] == 50
    assert response["input_report"] == {"kept_sequences": 10}
    assert response["final_progress"]["metrics"]["LL_val"] is None
    assert response["final_progress"]["metrics"]["Slope_val"] is None
    assert "without reaching target Pearson" in response["warnings"][-1]
    history = json.loads((output / "training_history.json").read_text())
    assert history["LL_val"] == [None, None]
    json.dumps(response, allow_nan=False)


def test_training_forwards_label_and_uses_labeled_history(tmp_path, monkeypatch):
    received = {}

    def fake_train_model(data_path, *, progress, **kwargs):
        received.update(kwargs)
        progress(SimpleNamespace(epoch=3, metrics={"Pearson": 0.8}))
        return SimpleNamespace(
            model=SimpleNamespace(metadata=_metadata()),
            history={"Pearson": [0.4, 0.6, 0.8]},
            pseudocount=0.01,
            num_sequences=10,
            effective_sequences=8.5,
            artifacts={"params": tmp_path / "model" / "edgeDCA_params.dat"},
            converged=True,
            stop_reason="target_pearson",
            gradient_steps=3,
            structure_steps=3,
            sweeps=30,
            input_report={},
            warnings=(),
        )

    monkeypatch.setitem(
        sys.modules,
        "adabmDCA",
        SimpleNamespace(train_model=fake_train_model),
    )

    response = train_dca_model(
        "alignment.fasta",
        str(tmp_path / "model"),
        label="edgeDCA",
        target_pearson=0.7,
    )

    history = tmp_path / "model" / "edgeDCA_training_history.json"
    assert received["label"] == "edgeDCA"
    assert history.exists()
    assert response["label"] == "edgeDCA"
    assert response["artifacts"]["history"] == str(history)
    assert response["converged"] is True
    assert response["stop_reason"] == "target_pearson"


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
        "train_dca_model",
    }
