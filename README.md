# agentDCA

Open-source, early-stage agent plugin and MCP server for
[adabmDCA](https://github.com/spqb/adabmDCApy).

agentDCA exposes the high-level adabmDCA Python API as structured tools that an
agent can discover and call. The initial tool set covers:

- model inspection;
- sequence energy scoring;
- contact prediction;
- complete contact-matrix export and side-by-side mean-field/model heatmaps;
- single-mutation scanning;
- model-based sequence sampling;
- Stockholm/FASTA preprocessing;
- DCA model training.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for the bundled plugin launch command
- A CUDA, MPS, or CPU PyTorch runtime; `device="auto"` selects in that order

The project resolves adabmDCA from its `agent-api` branch and records the exact
commit in `uv.lock`. MCP startup uses that locked commit; refresh it explicitly
during development when a newer `agent-api` revision is wanted.

## Install in Codex

Because the repository is public, no GitHub authentication is required. Confirm
that `uv` is available on the machine running Codex:

```bash
uv --version
```

Register the repository as a Codex marketplace and install the plugin:

```bash
codex plugin marketplace add rossetl/agentDCA --ref main
codex plugin add agentdca@agentdca
```

If you previously registered the marketplace with the private SSH URL, replace
that saved source with the public one before reinstalling:

```bash
codex plugin marketplace remove agentdca
codex plugin marketplace add rossetl/agentDCA --ref main
codex plugin add agentdca@agentdca
```

To install through the desktop interface instead, restart the Codex/ChatGPT
desktop app after adding the marketplace, open the Plugins Directory, select
the `agentDCA` source, and install `agentDCA`.

Start a new task after installation so Codex loads the bundled skill and MCP
tools. For example:

```text
Use agentDCA to inspect this DCA model and summarize its metadata.
```

The first MCP server launch uses `uv` to resolve the locked Python environment,
so it may take longer while it downloads its dependencies. The bundled MCP
configuration allows up to 120 seconds for this initial startup. Refresh the
locked adabmDCA revision explicitly with:

```bash
uv lock --upgrade-package adabmDCA
```

Verify the configured marketplace or refresh it later with:

```bash
codex plugin marketplace list
codex plugin marketplace upgrade agentdca
```

Plugin releases should increment the version in
`.codex-plugin/plugin.json` so Codex installs the updated package cleanly.

## Development setup

```bash
git clone https://github.com/rossetl/agentDCA.git
cd agentDCA
uv sync --extra dev
uv run pytest
uv run ruff check .
```

For local plugin development, register this checkout as the marketplace once:

```bash
codex plugin marketplace add /absolute/path/to/agentdca
codex plugin add agentdca@agentdca
```

The marketplace entry points back to this repository. After editing the plugin,
refresh its manifest cache-buster and reinstall it, then start a new Codex
thread:

```bash
python3 /path/to/plugin-creator/scripts/update_plugin_cachebuster.py .
codex plugin add agentdca@agentdca
```

The reinstall step is required for Codex to reload changed skills and MCP
configuration; Python source changes are read from this checkout once the local
plugin is installed.

Run the MCP server over stdio:

```bash
uv run agentdca-mcp
```

The server writes protocol messages to standard output. Use standard error for
development logging so the MCP transport is not corrupted.

Verify that every public tool is registered before packaging a release:

```bash
uv run --extra dev pytest
```

## Plugin structure

```text
.agents/plugins/marketplace.json  Codex marketplace catalog
.codex-plugin/plugin.json  Plugin identity and presentation metadata
.mcp.json                  Bundled MCP server launch configuration
skills/dca-analysis/       Agent workflow and interpretation guidance
src/agentdca/server.py     MCP tool implementation
```

The bundled configuration launches the server with:

```text
uv run --locked --project ${PLUGIN_ROOT}/pyproject.toml agentdca-mcp
```

`--locked` ensures the server uses the exact adabmDCA commit recorded in
`uv.lock` and performs no dependency update at startup. The MCP startup timeout
is 120 seconds to accommodate the first dependency installation; individual
tools may run for up to one hour.

## Output safety

- Writing tools refuse to replace existing managed artifacts unless
  `overwrite=true` is explicit.
- Sequence scoring can persist every record with `output_csv`.
- Sequence sampling can persist every generated record with `output_fasta`.
- Scoring and sampling return at most `max_inline_sequences=100` records by
  default while reporting the total count and whether the payload was truncated.
- Training writes `training_history.json` and reports `converged`,
  `stop_reason`, native training counters, and `epochs_completed`.
- Labeled training writes collision-checked artifacts such as
  `edgeDCA_params.dat` and `edgeDCA_training_history.json`, allowing multiple
  models to share one output directory.
- Contact prediction can write the complete score matrix as CSV and/or NPY;
  the default response remains a bounded ranked list. Small device-level
  numerical asymmetries are reported and symmetrized by default.
- `compare_dca_contacts` writes a shared-scale PNG, PDF, or SVG heatmap with
  mean-field and trained-model contact matrices side by side.
- Non-finite scientific metrics are returned as JSON `null`, never non-standard
  `NaN` or infinity values.

## Design principles

- The plugin is a thin adapter; scientific logic remains in adabmDCA.
- Tool outputs are JSON-serializable and contain explicit metadata.
- Expensive or writing operations remain explicit.
- Large matrices and mutation libraries are returned as bounded ranked lists.
- CLI, notebook, and MCP behavior share the same package APIs.

## Status

This repository is public and under active development. The tool schemas and
installation workflow may change before the first stable release.
