# COF-SCOUT Zenodo data archive

This directory is the upload staging area for the separate COF-SCOUT data record.
The data licence is Creative Commons Attribution 4.0 (CC BY 4.0). Binary archives
are generated locally by `python scripts/build_zenodo_archive.py` and intentionally
excluded from Git. `MANIFEST.csv` records bytes and SHA-256 for every upload file.

Traces are copied from immutable frozen artifacts and redacted only for gateway
URLs/hostnames, usernames, and local absolute paths. Model prompts, responses,
predictions, candidate selections, feedback and outcomes are not altered.
`REDACTION_LOG.md` records the original and redacted hashes of every changed copy.
