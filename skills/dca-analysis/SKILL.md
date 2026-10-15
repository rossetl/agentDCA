---
name: dca-analysis
description: Analyze biological sequence alignments and Direct Coupling Analysis models with agentDCA. Use for alignment cleanup, DCA training, sequence scoring, contact prediction, mutation scans, and model-based sequence generation.
---

# DCA analysis

Use the agentDCA MCP tools to perform reproducible DCA workflows.

## Workflow

1. Inspect input formats, model metadata, and sequence lengths before expensive operations.
2. Preprocess Stockholm or aligned FASTA inputs when requested. Do not remove insertions, duplicates, or gap-rich sequences unless the user requests those transformations.
3. Use `inspect_dca_model` before scoring, mutation scanning, contact prediction, or sampling when the model configuration is unknown.
4. Use `score_dca_sequences` for statistical energies and `scan_dca_mutations` for single-mutant effects.
5. Use `predict_dca_contacts` with a bounded `top_k`; increase it only when the user needs more pairs. Set `output_csv` or `output_npy` when the complete matrix is needed instead of returning it inline.
6. Use `compare_dca_contacts` when the user wants a side-by-side mean-field/model heatmap. Keep `symmetrize=true` unless raw device-level numerical asymmetry is itself under investigation.
7. Treat `sample_dca_sequences` and especially `train_dca_model` as computationally expensive. Preserve user-specified seeds and runtime settings.
8. Use a filename-safe training `label` when multiple models should coexist in one output directory.
9. Report all written output paths and the device selected by model metadata.
10. For scoring or sampling more than 100 sequences, provide `output_csv` or
   `output_fasta`; tool responses contain only a bounded inline preview.
11. Treat `converged=false` as an incomplete fit even when training completed
   successfully, and report the returned `stop_reason`.

## Interpretation

- Lower DCA energies indicate greater probability under the fitted model; they are not universal biological fitness measurements.
- Negative mutation delta energy means the mutant has lower model energy than the supplied wild type.
- Contact scores rank candidate residue pairs; they are not calibrated contact probabilities.
- Sequence indices returned by tools include both zero-based and one-based positions when applicable.

## Safety and reproducibility

- Do not overwrite a user file unless the user explicitly selected that output path.
- Keep `overwrite=false` unless the user explicitly authorized replacement.
- Do not launch training without an explicit request and a specified output directory.
- Prefer `device="auto"` unless the user requests a specific device.
- Surface structured tool errors without hiding the failing path or incompatible sequence length.
