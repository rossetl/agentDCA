---
name: dca-analysis
description: Analyze biological sequence alignments and Direct Coupling Analysis (Potts / Boltzmann machine) models with agentDCA and adabmDCA 2.x. Use for alignment cleanup and train/test splits, DCA training (bmDCA, eaDCA, edDCA, edgeDCA, PTT), sequence scoring, contact prediction, mutation scans, sequence generation with diagnostics or steering, entropy estimation, and retraining with experimental feedback.
---

# DCA analysis

Use the agentDCA MCP tools to run reproducible adabmDCA workflows. Every tool is a thin
adapter: scientific behaviour and defaults come from adabmDCA.

## Tools at a glance

| Goal | Tool |
|---|---|
| Convert Stockholm/FASTA, filter gaps/duplicates/insertions | `preprocess_dca_alignment` |
| Train/test split with limited homology | `split_dca_alignment` |
| Train a model | `train_dca_model` |
| Retrain with experimental labels | `reintegrate_dca_model` |
| Plot training curves | `plot_dca_training_history` |
| Model metadata | `inspect_dca_model` |
| Energies of sequences | `score_dca_sequences` |
| Contacts (model or mean-field) | `predict_dca_contacts`, `compare_dca_contacts` |
| Single-mutant scan | `scan_dca_mutations` |
| Generate sequences (MCMC, PTT, steered) | `sample_dca_sequences` |
| Model entropy | `estimate_dca_entropy` |
| Follow long runs | `get_dca_job`, `list_dca_jobs`, `cancel_dca_job` |

## Standard workflow

1. **Inspect inputs.** Check the alignment format, length and alphabet. `alphabet="auto"`
   (the default) detects protein, RNA or DNA from the data or model file; pass an explicit
   alphabet only when detection is ambiguous or a custom token string is needed.
2. **Preprocess only on request.** Do not remove insertions, duplicates or gap-rich sequences
   unless the user asks; every filter is off by default.
3. **Split before training when generalisation matters.** `split_dca_alignment` writes
   `<prefix>.train.fasta` and `<prefix>.test.fasta`. Pass the test set as `validation_path`
   when training and as `test_fasta` when assessing samples.
4. **Train.** Leave options unset to use adabmDCA defaults. Choose the model:
   - `bmDCA`: dense couplings, the default.
   - `eaDCA`: sparse, couplings activated progressively (`target_density`, `activation_*`).
   - `edDCA`: sparse by decimating a dense model (`target_density`, `decimation_rate`).
   - `edgeDCA`: sparse, PCD only (not with PTT).
   - `ptt=true` (bmDCA or eaDCA) trains with Parallel Trajectory Tempering: slower per step,
     but keeps chains equilibrated and yields a `ptt.h5` archive for exact-mixing sampling and
     cheap entropy. `ptt_options` takes `PTTConfig` fields, e.g. `{"validation_stop": true}`
     (needs `validation_path`) to stop at the validation log-likelihood plateau.
   - Restart or fine-tune with `initial_params_path` / `initial_chains_path`; continue an
     interrupted PTT run with `ptt_resume`.
5. **Run long jobs in the background.** For anything beyond a few minutes pass
   `background=true`, then poll `get_dca_job(job_id, wait_seconds=...)`. Report progress
   (Pearson, density, stage) to the user and use `cancel_dca_job` if asked to stop; adabmDCA
   stops at a safe point and keeps its checkpoint.
6. **Check the fit.** Report `converged` and `stop_reason`; `converged=false` is an incomplete
   fit even when the run finished. Use `plot_dca_training_history` on `history.csv`.
7. **Analyse.** Score, predict contacts, scan mutations or sample from `params.dat.gz` (or
   `<label>_params.dat.gz`).
8. **Validate generated sequences.** Sample with `reference_fasta` (training data),
   `test_fasta` (held-out data), `collect_diagnostics=true` and an `output_dir`. Report the
   final Cij Pearson, nearest-neighbour distances (generated vs. held-out to training) and the
   PRIVET overfitting check (`n_flagged` / `n_memorized`).

## Sampling modes

- Ordinary MCMC (default sampler `metropolized_gibbs`). Without a reference it runs exactly
  `n_sweeps`; with `reference_fasta` it measures the mixing time and runs
  `mixing_multiplier` mixing times.
- PTT: `ptt=true`, `model_path` = the training `ptt.h5`, `reference_fasta` required. It ignores
  `n_sweeps` and `beta`; report `ptt_mixing.status`.
- Steered: `steering_target` (aligned sequence) pulls samples towards it with
  `V = steering_strength * Hamming distance`; `steering_script="file.py:function"` loads a
  user-written potential `V(sequences, s)`. Only use a script the user wrote or approved,
  since it executes in the server. Report the effective sample size in `steering`.

## Entropy

- `method="ptt"` on a `ptt.h5` archive: cheap, uses the PTT bridges.
- `method="thermodynamic_integration"` on any parameter file: needs `natural_alignment` and a
  `target_alignment` whose first sequence is the integration target. Expensive; prefer
  `background=true`.

## Interpretation

- Lower DCA energy means higher probability under the fitted model, not universal fitness.
- `local_free_energy = energy - local_lambda * cde_sum` corrects energies for local
  sequence entropy; mention which quantity a ranking uses.
- Negative mutation `delta_energy` means the mutant is more probable than the wild type.
- Contact scores are APC-corrected Frobenius norms that rank residue pairs; they are not
  calibrated probabilities. Positions are given zero- and one-based.
- In `compare_dca_contacts`, keep `color_scale="shared"` for comparison; use
  `"per_panel"` when one matrix has much smaller scores and looks flat.

## Safety and reproducibility

- Never overwrite: keep `overwrite=false` unless the user explicitly authorises replacement.
  Use distinct `label`s so several runs can share an output directory.
- Do not start training, reintegration or large sampling without an explicit request and an
  output directory.
- Preserve user-specified seeds and runtime settings; prefer `device="auto"`.
- Report every written path, the device used, and the returned `config` for training runs.
- For more than 100 sequences, write them to a file (`output_csv`, `output_fasta`,
  `output_dir`); responses carry only a bounded preview and long lists come back as
  `{count, preview}`.
- Tool errors carry adabmDCA's error code, message and details; surface them verbatim with
  the failing path or parameter.
