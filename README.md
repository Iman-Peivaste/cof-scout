# COF-SCOUT

COF-SCOUT is an auditable language-model-agent workflow for navigating a frozen
covalent-organic-framework design landscape under a 50-evaluation budget. This
repository accompanies *Decomposing what a language-model agent adds to covalent
organic framework discovery* and separates reusable code and small inputs from the
large, immutable study outputs deposited on Zenodo.

## Citation

Please cite the COF-SCOUT paper (DOI: `PAPER_DOI`), this software release (DOI:
`ZENODO_SOFTWARE_DOI`), and the data archive (DOI:
[10.5281/zenodo.23688942](https://doi.org/10.5281/zenodo.23688942)). The related Ara
paper is <https://doi.org/10.1038/s41524-026-02168-w>.

## Install

The frozen environment uses Python 3.12.2, RDKit 2025.9.3, NumPy 1.26.4, SciPy
1.17.1, scikit-learn 1.5.1, pandas 2.3.2, and xTB 6.7.1.

```bash
conda env create -f environment.yml
conda activate cof-scout
pip install -e .
```

## Quick start

Download and verify the Zenodo data, verify the frozen manifests, and reproduce
Figure 2 without any LLM or network call after download:

```bash
python scripts/download_data.py --doi 10.5281/zenodo.23688942
export COF_SCOUT_DATA_DIR="$PWD/data/zenodo"
python scripts/verify_freeze.py
python scripts/reproduce_figures.py --figure 2
```

Expected runtime after extraction is approximately 2 minutes for all tables and
paper figures on a laptop; Figure 2 alone takes under 20 seconds. Baseline reruns are
CPU-only and take substantially longer. Agent reruns require an OpenAI-compatible
endpoint and incur provider costs.

## Repository map

- `src/cof_scout/inventory`: building blocks and deterministic assembly.
- `src/cof_scout/oracle`: geometry, xTB, calibration, four gates, persistence, and
  hash-verified reference lookup.
- `src/cof_scout/agent`: prediction contract, guards, memory, client, adaptive loop,
  and static portfolio.
- `src/cof_scout/baselines`: random, MaxMin, RBF/Tanimoto GP-UCB, and corrected
  linkage elimination.
- `src/cof_scout/analysis`: endpoints, bootstrap intervals, superiority, and
  sensitivity utilities.
- `src/cof_scout/figures`: one entry point per paper figure.
- `configs`: frozen protocol metadata with model names but no endpoints or keys.
- `data/small`: the building-block library, feasible inventory, and calibration
  coefficients.
- `docs/ARTIFACTS.md`: mapping from archived artifacts to paper figures and tables.
- `release/zenodo`: archive manifest and redaction record; binary archives are not
  tracked by Git.

## Reproduction and reruns

See [ARTIFACTS](docs/ARTIFACTS.md), [PROTOCOLS](docs/PROTOCOLS.md), and
[RELEASE](docs/RELEASE.md). Baseline and agent reruns are intentionally separate:

```bash
python scripts/rerun_baselines.py --help
OPENAI_BASE_URL=... OPENAI_API_KEY=... python scripts/rerun_campaigns.py --help
```

## Licences and contact

Code is MIT licensed. The Zenodo data archive is CC BY 4.0. Contact: Iman Peivaste,
Luxembourg Institute of Science and Technology (LIST), <iman.peivaste@list.lu>.
