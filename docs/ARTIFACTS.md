# Artifact map

The Git repository contains code and small canonical inputs only. The Zenodo data
record contains seven independently downloadable archives.

| Archive | Contents | Paper use |
|---|---|---|
| `reference_landscape.tar.gz` | Frozen JSONL landscape and manifest | Oracle lookup; gate and offset analyses |
| `freeze_manifests.tar.gz` | Original, extension-r2, condition-C amendment, superseded and identifier-relabel freezes | Protocol and hash verification |
| `campaign_traces.tar.gz` | Adaptive, static, ablation and relabel traces, including six failed C traces | Campaign endpoints, discovery curves, prediction metrics |
| `baseline_outputs.tar.gz` | Original and added-control endpoints | Table 1; Figures 2–3 |
| `sensitivity_outputs.tar.gz` | Imine, offset and non-imine sensitivity analyses | Sensitivity tables and supplementary results |
| `figure_source_data.tar.gz` | CSV/JSON inputs behind every released table and figure | Reproduction without LLM calls |
| `calibration.tar.gz` | Six PBE+D3 calculation folders, xTB values, fitted pairs and transfer function | Calibration methods and supplementary data |

The original manuscript figures use the archived `publication/tables` data as follows:

- Figure 1: deterministic schematic (no empirical source table).
- Figure 2: `endpoint_replicates_and_seeds.csv` and control summaries.
- Figure 3: `discovery_curves_all_runs.csv`.
- Figure 4: `recurrent_scaffold_matrix.csv`.
- Figure 5: `campaign_experiments.csv` and `linkage_outcomes.csv`.
- Figure 6: `falsification_episode_rep04.csv`.

The extension figures use `per_campaign_summary.csv`,
`item3_added_baselines.csv`, and the amended condition-C campaign CSVs.
