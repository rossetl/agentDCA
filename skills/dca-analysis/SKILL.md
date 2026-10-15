---
name: dca-analysis
description: Drive adabmDCA through the agentDCA MCP tools - train Potts/Boltzmann machine DCA models on protein, RNA or DNA alignments as background jobs that survive the chat, follow and diagnose runs, look at plots, score, scan mutations, predict contacts, sample and estimate entropy. Use when the agentDCA tools are available, especially without a shell or for long runs.
---

# DCA analysis with agentDCA

agentDCA exposes adabmDCA 2.x as MCP tools. The science (what to train, which settings,
how to read the outputs) is the same as with the adabmDCA CLI and is covered by the
**`adabmdca` skill**, which this plugin ships alongside this one
([../adabmdca/SKILL.md](../adabmdca/SKILL.md), maintained in adabmDCApy). Read it for the
workflow, settings, interpretation and diagnostics; it speaks in CLI options, so
translate them with the table below. This skill covers what the tools add and how to use
them. Where the two differ on how to run something, follow this skill: use the tools,
not a shell.

## Tools

| Goal | Tool |
|---|---|
| Clean an alignment, split train/test | `preprocess_dca_alignment`, `split_dca_alignment` |
| Train, retrain with experimental labels | `train_dca_model`, `reintegrate_dca_model` |
| Follow, stop and list runs | `get_dca_job`, `list_dca_jobs`, `cancel_dca_job` |
| Is the run healthy? | `diagnose_dca_run` |
| Training curves, any saved figure | `plot_dca_training_history`, `show_dca_plots` |
| Model metadata | `inspect_dca_model` |
| Energies, mutation scan | `score_dca_sequences`, `scan_dca_mutations` |
| Contacts | `predict_dca_contacts`, `compare_dca_contacts` |
| Generate sequences (MCMC, PTT, steered) | `sample_dca_sequences` |
| Entropy | `estimate_dca_entropy` |

## CLI options as tool parameters

The `adabmdca` skill, the adabmDCA docs and `diagnose_dca_run` findings give advice as
CLI options. Their tool equivalents:

| CLI | Tool parameter |
|---|---|
| `train -d` / `-v` / `-o` / `-l` / `-w` | `train_dca_model(data_path, validation_path, output_dir, label, weights_path)` |
| `-m` | `model_type` |
| `--strategy ptt` / `pcd` | `ptt=true` / `false` (default: PTT, as in the CLI) |
| `--lr`, `--nsweeps`, `--nchains`, `--sampler` | `learning_rate`, `n_sweeps`, `n_chains`, `sampler` |
| `--target`, `--nepochs` | `target_pearson`, `max_epochs` |
| `--max-gradient-steps`, `--max-structure-steps` | same names with underscores |
| `--pseudocount`, `--l2_reg`, `--clustering_seqid`, `--no_reweighting` | `pseudocount`, `l2_regularization`, `clustering_seqid`, `no_reweighting` |
| `--density`, `--drate`, `--gsteps`, `--factivate` | `target_density`, `decimation_rate`, `activation_steps`, `activation_fraction` |
| `-p params.dat.gz -c chains.fasta` (restart PCD) | `initial_params_path`, `initial_chains_path` |
| `--ptt-resume ptt.h5` | `ptt_resume` |
| `--ptt-<name>` (e.g. `--ptt-target-acceptance 0.35`) | `ptt_options={"<name>": ...}` with underscores (`{"target_acceptance": 0.35}`) |
| `--no-ptt-validation-stop` | `ptt_options={"validation_stop": false}` |
| `sample -p` / `-d` / `-v` / `-o` / `--ngen` | `sample_dca_sequences(model_path, reference_fasta, test_fasta, output_dir, n_sequences)` |
| `sample --plot` | `collect_diagnostics=true` (with `output_dir`, the plots are written and returned as images) |
| `sample --strategy ptt` / `pcd` | `ptt=true` / `false` |
| `--ptt-local-sweeps`, `--ptt-max-rounds`, `--ptt-stationary`, `--ptt-local-kernel`, `--ptt-return-full-ladder` | same names with underscores |
| `--max_nsweeps`, `--nmix` (plain MCMC) | `n_sweeps`, `mixing_multiplier` |
| `energies`, `dms`, `contacts` | `score_dca_sequences`, `scan_dca_mutations`, `predict_dca_contacts` |
| `entropy` (`--strategy ptt` / `pcd`) | `estimate_dca_entropy(method="ptt" / "thermodynamic_integration")` |
| `preprocess`, `split-data`, `reintegrate` | `preprocess_dca_alignment`, `split_dca_alignment`, `reintegrate_dca_model` |
| `plot-training-log`, `diagnose` | `plot_dca_training_history`, `diagnose_dca_run` |
| `--alphabet`, `--seed`, `--device`, `--dtype` | `alphabet` (default `auto`), `seed`, `device`, `dtype` |

Running a CLI command in a shell instead would bypass the job registry, the output
guards and the images, so use the tools when they are available.

Defaults:

- `train_dca_model` trains with PTT, like the CLI, and stops at the validation plateau
  when a `validation_path` is given. It falls back to PCD for edDCA/edgeDCA and when
  restarting from `initial_params_path`/`initial_chains_path`; the result's `strategy`
  and `strategy_reason` say which ran. Pass `ptt=false` only for a quick first look.
- Sample with PTT whenever the model has a PTT archive: if the run directory holds a
  `ptt.h5` (or `<label>_ptt.h5`), call `sample_dca_sequences` with `ptt=true`,
  `model_path` set to that archive and a `reference_fasta` (the training alignment).
  PTT sampling mixes through the ladder built in training. Use plain MCMC (the tool's
  default) only for a model without an archive, such as a PCD run or a bare
  `params.dat.gz`, or when the user asks for it.

`alphabet="auto"` (the default) detects the alphabet like the CLI.

## Long runs are jobs that outlive the chat

1. Pass `background=true` to training, reintegration, sampling and entropy for anything
   longer than a minute or two. The tool returns a `job_id` at once; the run executes in
   its own process and keeps going if the chat, the client or the MCP server stops.
2. Follow it with `get_dca_job(job_id, wait_seconds=60)`: status, the latest progress
   (stage, Pearson, density, entropy) and, when finished, the full result. A later chat
   finds it with `list_dca_jobs`.
3. While it runs, `plot_dca_training_history` on its `history.csv` and `diagnose_dca_run`
   on its output directory show how it is going.
4. `cancel_dca_job` asks adabmDCA to stop at its next safe point and keep its checkpoint.
   Resume with `initial_params_path`/`initial_chains_path`, or `ptt_resume` for PTT.
5. Status `interrupted` means the worker died without a result (machine restart, out of
   memory): read `worker_log` and resume from the checkpoint.

Foreground calls also report progress and stop when the client cancels them.

## Check every model before relying on it

1. When training ends, call `diagnose_dca_run(run_dir, label=...)`. It reads the history,
   events, summary and log, and returns a `verdict` (`healthy`, `needs_attention`,
   `failed`, `running`, `incomplete`) with findings. Each finding names the failure it
   points to (F1-F8 of adabmDCA's diagnostics guide) and the change to make. Report the
   verdict and the actions; do not restate every indicator.
2. Sample the model (with PTT when it has a `ptt.h5`, see above) with `reference_fasta`
   (training data), `test_fasta` (held-out),
   `collect_diagnostics=true` and an `output_dir`, then call `diagnose_dca_run` again with
   `sampling_json` set to the written `sampling.json`. This compares the sampled Pearson
   with training, the ladder health, nearest-neighbour distances and PRIVET memorization.
3. Act on `error` and `warning` findings before using the model. A `failed` verdict with
   F1 means restarting from scratch with more sweeps, not resuming.

## Looking at results

- `plot_dca_training_history`, `compare_dca_contacts` and `sample_dca_sequences` (with
  diagnostics and `output_dir`) return their main figures as images, so you can read the
  plots directly. Pass `return_image(s)=false` to save context.
- `show_dca_plots(paths)` returns any saved PNG (for example the other diagnostic plots
  listed in a sampling result's `artifacts`).
- Describe what a plot shows in words for the user; they may not see the image.

## Repeated analyses are cheap

The server keeps the last loaded models in memory (`AGENTDCA_MODEL_CACHE`, default 2),
keyed by file, size and modification time. Scoring, scanning, contacts and sampling on
the same model reuse it; a retrained file is reloaded automatically.

## Safety

- Outputs are never replaced unless `overwrite=true`; use a new `output_dir` or `label`
  per run, and only overwrite when the user asks.
- Do not start training, reintegration or large sampling without a request and an output
  directory.
- Responses carry at most `max_inline_sequences` records; write complete results with
  `output_csv`, `output_fasta` or `output_dir` and report the paths.
- `steering_script` runs a Python file inside the server: only use one the user wrote or
  approved.
- Errors carry adabmDCA's code, message and details: report the message and the setting
  it names; do not retry blindly.
