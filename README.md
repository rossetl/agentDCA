# agentDCA

Open-source agent plugin and MCP server for
[adabmDCA](https://github.com/spqb/adabmDCApy) 2.x, packaged for both Claude Code and
Codex.

agentDCA exposes the high-level adabmDCA Python API as structured tools that an agent can
discover and call, plus a `dca-analysis` skill that teaches the agent how to use them.

adabmDCA ships its own `adabmdca` agent skill for agents with a shell, which covers the
workflow, settings and interpretation through the CLI. agentDCA focuses on what a skill
alone cannot give:

- **Jobs that outlive the chat.** Long runs execute in their own process, keep going when
  the chat or the MCP server ends, and can be listed, followed and cancelled from any
  later session.
- **A verdict on a run.** `diagnose_dca_run` turns a run's history, events, summary and
  sampling check into findings that name the failure and the fix, following adabmDCA's
  diagnostics guide.
- **Plots the agent can see.** Training curves, contact maps and sampling diagnostics come
  back as images.
- **Structured results and warm models.** Bounded, JSON-safe results with explicit
  metadata, and recently used models kept in memory between calls.
- **Clients without a shell**, such as Claude Desktop.

| Tool | What it does |
|---|---|
| `preprocess_dca_alignment` | Convert Stockholm/FASTA and optionally filter insertions, gaps, duplicates and unknown tokens |
| `split_dca_alignment` | Train/test split with limited homology (clustering or Cobalt) |
| `train_dca_model` | Train bmDCA, eaDCA, edDCA or edgeDCA, with Parallel Trajectory Tempering (PTT, the default as in the CLI) or PCD; validation set, weights, restart and resume |
| `reintegrate_dca_model` | Retrain with signed experimental feedback on tested sequences |
| `plot_dca_training_history` | Plot every recorded training quantity from `history.csv`; returns the overview as an image |
| `diagnose_dca_run` | Verdict and findings (F1-F8) for a training run, optionally with its sampling check |
| `show_dca_plots` | Return saved PNG figures as images |
| `inspect_dca_model` | Model metadata for a parameter file or PTT archive |
| `score_dca_sequences` | DCA energies, context-dependent entropy and local free energies |
| `predict_dca_contacts` | Ranked APC-corrected contacts from a model or mean-field DCA; full matrix export |
| `compare_dca_contacts` | Side-by-side mean-field/model contact heatmaps (shared or per-panel scale), returned as an image |
| `scan_dca_mutations` | All single substitutions of a wild type, ranked; full CSV export |
| `sample_dca_sequences` | MCMC or PTT sampling, diagnostics against data (Cij, PCA, distances, PRIVET) with plots returned as images, steered sampling |
| `estimate_dca_entropy` | Entropy by thermodynamic integration or from a PTT archive |
| `get_dca_job`, `list_dca_jobs`, `cancel_dca_job` | Follow, find and stop runs started with `background=true`, from any session |

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for the bundled plugin launch command
- A CUDA, MPS, or CPU PyTorch runtime; `device="auto"` selects in that order

agentDCA depends on `adabmDCA[cpu]` from the adabmDCA `develop` branch (2.x). The `cpu`
extra installs numba, which adabmDCA 2.x needs for its CPU samplers. The exact commit is
recorded in `uv.lock`, and MCP startup uses that locked commit. Refresh it explicitly when a
newer adabmDCA revision is wanted:

```bash
uv lock --upgrade-package adabmDCA
```

The first MCP server launch uses `uv` to resolve the locked environment, so it may take
longer while it downloads PyTorch and the other dependencies.

## Install in Claude Code

```bash
claude plugin marketplace add rossetl/agentDCA
claude plugin install agentdca@agentdca
```

Or, inside a session (Claude Code 2.1.275 or later):

```text
/plugin install agentdca --marketplace rossetl/agentDCA
```

Start a new session so Claude Code loads the skill and the MCP tools. If the first launch
times out while dependencies download, start Claude Code with a longer MCP startup timeout,
for example `MCP_TIMEOUT=300000 claude`.

## Install in Codex

Confirm that `uv` is available on the machine running Codex, then register the repository
as a Codex marketplace and install the plugin:

```bash
codex plugin marketplace add rossetl/agentDCA --ref main
codex plugin add agentdca@agentdca
```

If you previously registered the marketplace with the private SSH URL, replace that saved
source with the public one before reinstalling:

```bash
codex plugin marketplace remove agentdca
codex plugin marketplace add rossetl/agentDCA --ref main
codex plugin add agentdca@agentdca
```

To install through the desktop interface instead, restart the Codex/ChatGPT desktop app
after adding the marketplace, open the Plugins Directory, select the `agentDCA` source, and
install `agentDCA`. Start a new task after installation so Codex loads the skill and tools.

The bundled Codex MCP configuration allows 120 seconds for the first startup and one hour
per tool call. Verify or refresh the marketplace with:

```bash
codex plugin marketplace list
codex plugin marketplace upgrade agentdca
```

## Example prompts

```text
Split family.fasta into training and test sets, train an eaDCA model with PTT using the
test set for validation, and tell me when it converges.

Generate 2000 sequences from model/params.dat.gz, compare them with the training and test
alignments, and tell me whether the model memorizes training sequences.

Score every single mutant of this wild type and list the 20 most favorable substitutions.

Is the run in model/ healthy? Sample it against the test set and tell me whether I can
rely on it.
```

## Long runs

Training, reintegration, sampling and entropy estimation accept `background=true`. The
tool then returns a `job_id` immediately and runs the call in a detached worker process
(`python -m agentdca.worker`), so the run continues when the chat, the client or the MCP
server stops.

- `get_dca_job(job_id, wait_seconds=...)` reports the latest progress (stage, Pearson,
  density) and, once finished, the full result.
- `list_dca_jobs` finds jobs started in earlier sessions.
- `cancel_dca_job` asks adabmDCA to stop at its next safe point and keep its checkpoint;
  `force=true` terminates a worker that does not respond.
- A worker that dies without recording a result (machine restart, out of memory) is
  reported as `interrupted`, with its log.

Each job is a directory with `spec.json`, `status.json` and `worker.log` under
`$AGENTDCA_HOME/jobs` (default `~/.agentdca/jobs`). In the foreground, the same runs send
MCP progress notifications and stop when the client cancels the call.

## Model cache

The server keeps recently loaded models (parameter files and PTT samplers) in memory, so
repeated scoring, scans, contact maps and sampling on one model skip the reload. Entries
are keyed by path, size and modification time, so a retrained model is picked up
automatically. `AGENTDCA_MODEL_CACHE` sets the number of models kept (default 2; 0
disables the cache).

## Output safety

- Writing tools refuse to replace existing outputs unless `overwrite=true` is explicit.
  Training checks every file adabmDCA 2.x writes (`params.dat.gz`, `history.csv`,
  `events.jsonl`, chains, weights, log, `ptt.h5`) plus its `training.json` summary.
- A `label` prefixes every output, so several runs can share one directory.
- Scoring, mutation scans and sampling return at most `max_inline_sequences=100` records (or
  the top `top_k`), and write complete results to files on request.
- Long lists inside reports are returned as `{count, preview}`.
- Non-finite scientific metrics are returned as JSON `null`.
- Errors carry adabmDCA's error class, code, message and details instead of a generic
  failure.
- `steering_script` executes a user-provided Python file inside the server; only point it at
  code you trust.

## Development setup

```bash
git clone https://github.com/rossetl/agentDCA.git
cd agentDCA
uv sync --extra dev
uv run pytest            # unit tests plus end-to-end runs on a small synthetic alignment
uv run pytest -m "not integration"   # fast unit tests only
uv run ruff check .
```

Run the MCP server over stdio:

```bash
uv run agentdca-mcp
```

The server writes protocol messages to standard output; use standard error for logging.

For local Codex plugin development, register this checkout as the marketplace, then
reinstall after editing skills or MCP configuration and start a new thread:

```bash
codex plugin marketplace add /absolute/path/to/agentdca
codex plugin add agentdca@agentdca
```

For Claude Code, validate the manifests with `claude plugin validate .` and load the
checkout with `claude --plugin-dir /absolute/path/to/agentdca`.

Plugin releases should increment the version in `pyproject.toml`,
`.claude-plugin/plugin.json` and `.codex-plugin/plugin.json` together; a test checks they
match.

## Plugin structure

```text
.claude-plugin/plugin.json        Claude Code plugin manifest (MCP server with ${CLAUDE_PLUGIN_ROOT})
.claude-plugin/marketplace.json   Claude Code marketplace catalog
.codex-plugin/plugin.json         Codex plugin manifest
.agents/plugins/marketplace.json  Codex marketplace catalog
.mcp.json                         MCP launch configuration (Codex, ${PLUGIN_ROOT})
skills/dca-analysis/              Agent workflow and interpretation guidance
src/agentdca/server.py            MCP tools
src/agentdca/jobs.py              Foreground progress/cancellation, on-disk job registry
src/agentdca/worker.py            Detached worker that runs a background job
src/agentdca/diagnose.py          Run diagnosis (diagnose_dca_run)
src/agentdca/models.py            In-memory model cache
src/agentdca/_common.py           Output guards, alphabet detection, JSON safety, errors
```

Claude Code also reads the root `.mcp.json`; the same-named server in
`.claude-plugin/plugin.json` replaces it with Claude Code's plugin-root variable.

## Design principles

- The plugin is a thin adapter; scientific logic and defaults remain in adabmDCA.
- Tool outputs are JSON-serializable and contain explicit metadata.
- Expensive or writing operations remain explicit.
- Large matrices and mutation libraries are returned as bounded ranked lists.
- CLI, notebook, and MCP behavior share the same package APIs.

## Status

This repository is public and under active development. The tool schemas and installation
workflow may change before the first stable release.
