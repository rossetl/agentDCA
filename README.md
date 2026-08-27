# agentDCA

Open-source, early-stage agent plugin and MCP server for
[adabmDCA](https://github.com/spqb/adabmDCApy).

agentDCA exposes the high-level adabmDCA Python API as structured tools that an
agent can discover and call. The initial tool set covers:

- model inspection;
- sequence energy scoring;
- contact prediction;
- single-mutation scanning;
- model-based sequence sampling;
- Stockholm/FASTA preprocessing;
- DCA model training.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for the bundled plugin launch command
- A CUDA, MPS, or CPU PyTorch runtime; `device="auto"` selects in that order

The project currently installs adabmDCA from its `agent-api` branch. This can
move to a released package constraint after those APIs are published.

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

The first MCP server launch uses `uv` to resolve the Python environment and may
take longer while it downloads adabmDCA and its dependencies. Verify the
configured marketplace or refresh it later with:

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

## Output safety

- Writing tools refuse to replace existing managed artifacts unless
  `overwrite=true` is explicit.
- Sequence scoring can persist every record with `output_csv`.
- Sequence sampling can persist every generated record with `output_fasta`.
- Scoring and sampling return at most `max_inline_sequences=100` records by
  default while reporting the total count and whether the payload was truncated.
- Training writes `training_history.json` and reports `converged`,
  `stop_reason`, and `epochs_completed`.
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
