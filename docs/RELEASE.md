# Release procedure (do not execute from the reproduction scripts)

1. Create the public GitHub repository and push the `public-release` branch.
2. Enable the Zenodo–GitHub integration for that repository.
3. Create a separate Zenodo data record, upload the seven archives plus
   `MANIFEST.csv`, `README.md`, and `REDACTION_LOG.md`, select CC BY 4.0, and mint the
   data DOI.
4. Replace `ZENODO_DATA_DOI` and `ZENODO_DOI` placeholders in the repository.
5. Tag `v1.0.0` on GitHub. Let the integration mint the software DOI.
6. Replace `ZENODO_SOFTWARE_DOI`, `PAPER_DOI`, repository-owner and ORCID
   placeholders in `README.md`, `CITATION.cff`, and `.zenodo.json`.
7. Insert the software and data DOIs in the paper's Code and Data Availability
   statements, then run `python scripts/verify_freeze.py` and the offline CI suite.
