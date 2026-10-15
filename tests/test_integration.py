"""End-to-end checks of every tool family against the real adabmDCA package (CPU)."""

import asyncio
import json

import numpy as np
import pytest
from mcp import Client

from agentdca.server import mcp

pytestmark = pytest.mark.integration

LETTERS = np.array(list("ACDEFGHIKL-"))


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    folder = tmp_path_factory.mktemp("agentdca")
    rng = np.random.default_rng(0)
    base = rng.integers(0, 10, size=12)
    lines = []
    for index in range(120):
        sequence = base.copy()
        mutated = rng.random(12) < 0.25
        sequence[mutated] = rng.integers(0, 11, size=mutated.sum())
        lines.append(f">seq{index}\n{''.join(LETTERS[sequence])}\n")
    alignment = folder / "family.fasta"
    alignment.write_text("".join(lines))
    return folder, alignment, "".join(LETTERS[base])


def _call(name, arguments):
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments)

    result = asyncio.run(run())
    text = result.content[0].text
    assert not result.is_error, text
    return json.loads(text)


def _error(name, arguments):
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments)

    result = asyncio.run(run())
    assert result.is_error
    return result.content[0].text


def test_errors_reach_the_agent():
    message = _error(
        "inspect_dca_model", {"model_path": "/missing/params.dat", "alphabet": "protein"}
    )
    assert "ModelLoadError" in message and "/missing/params.dat" in message


def test_full_workflow(workspace):
    folder, alignment, wild_type = workspace
    split = _call(
        "split_dca_alignment",
        {"alignment_path": str(alignment), "output_prefix": str(folder / "split" / "family")},
    )
    assert split["alphabet"] == "protein"
    assert split["training_sequences"] + split["test_sequences"] > 0

    training = _call(
        "train_dca_model",
        {
            "data_path": str(alignment),
            "output_dir": str(folder / "model"),
            "n_chains": 64,
            "max_epochs": 3,
            "device": "cpu",
        },
    )
    params = folder / "model" / "params.dat.gz"
    assert training["artifacts"]["params"] == str(params)
    assert training["stop_reason"]
    assert (folder / "model" / "training.json").exists()

    metadata = _call("inspect_dca_model", {"model_path": str(params)})
    assert metadata["alphabet"] == "protein" and metadata["length"] == 12

    scores = _call("score_dca_sequences", {"model_path": str(params), "fasta_path": str(alignment)})
    assert scores["num_sequences"] > 0 and scores["cde_sum"]

    contacts = _call("predict_dca_contacts", {"model_path": str(params), "top_k": 3})
    assert len(contacts["contacts"]) == 3

    scan = _call("scan_dca_mutations", {"model_path": str(params), "wild_type": wild_type})
    assert scan["total_mutations"] == 12 * 19  # 19 non-gap substitutions per site

    samples = _call(
        "sample_dca_sequences",
        {
            "model_path": str(params),
            "n_sequences": 32,
            "n_sweeps": 5,
            "reference_fasta": str(alignment),
            "collect_diagnostics": True,
            "output_dir": str(folder / "samples"),
            "device": "cpu",
        },
    )
    assert samples["num_sequences"] == 32
    assert samples["artifacts"]["samples"].endswith("samples.fasta")

    plots = _call(
        "plot_dca_training_history", {"history_path": str(folder / "model" / "history.csv")}
    )
    assert plots["plots"]["overview"].endswith("_overview.png")
