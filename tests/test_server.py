import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from agentdca.server import (
    _json_safe,
    _prepare_file_output,
    _prepare_training_output,
    _top_contacts,
    mcp,
    sample_dca_sequences,
    score_dca_sequences,
    train_dca_model,
)


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
    assert response["stop_reason"] == "max_epochs"
    assert response["final_progress"]["metrics"]["LL_val"] is None
    assert response["final_progress"]["metrics"]["Slope_val"] is None
    assert "without reaching target Pearson" in response["warnings"][-1]
    history = json.loads((output / "training_history.json").read_text())
    assert history["LL_val"] == [None, None]
    json.dumps(response, allow_nan=False)


def test_mcp_registers_all_public_tools():
    tools = asyncio.run(mcp.list_tools())

    assert {tool.name for tool in tools} == {
        "inspect_dca_model",
        "score_dca_sequences",
        "predict_dca_contacts",
        "scan_dca_mutations",
        "sample_dca_sequences",
        "preprocess_dca_alignment",
        "train_dca_model",
    }
