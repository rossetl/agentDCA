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
5. Use `predict_dca_contacts` with a bounded `top_k`; increase it only when the user needs more pairs.
6. Treat `sample_dca_sequences` and especially `train_dca_model` as computationally expensive. Preserve user-specified seeds and runtime settings.
7. Report all written output paths and the device selected by model metadata.

## Interpretation

- Lower DCA energies indicate greater probability under the fitted model; they are not universal biological fitness measurements.
- Negative mutation delta energy means the mutant has lower model energy than the supplied wild type.
- Contact scores rank candidate residue pairs; they are not calibrated contact probabilities.
- Sequence indices returned by tools include both zero-based and one-based positions when applicable.

## Safety and reproducibility

- Do not overwrite a user file unless the user explicitly selected that output path.
- Do not launch training without an explicit request and a specified output directory.
- Prefer `device="auto"` unless the user requests a specific device.
- Surface structured tool errors without hiding the failing path or incompatible sequence length.
