#!/usr/bin/env python3
"""Build the redacted, deterministic COF-SCOUT Zenodo data archives."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "release" / "zenodo"
EXTERNAL_CALIBRATION = Path(os.getenv("COF_SCOUT_CALIBRATION_SOURCE", "."))
EXTERNAL_RESULTS = Path(os.getenv("COF_SCOUT_CALIBRATION_RESULTS", "."))

TEXT_SUFFIXES = {
    ".csv", ".diff", ".in", ".json", ".jsonl", ".log", ".md", ".out",
    ".py", ".sha256", ".sh", ".txt", ".xyz", "",
}
_LOCAL_USER = os.getenv("USER", "")
REPLACEMENTS = [
    (
        re.compile(rb"gateway\.pia\.private\.list\.lu", re.IGNORECASE),
        b"<INSTITUTIONAL_GATEWAY>",
        "institutional gateway hostname",
    ),
    (re.compile(re.escape(str(ROOT).encode())), b"${REPOSITORY_ROOT}", "repository path"),
    (
        re.compile(b"/" + rb"home/[^/\s]+/[^\s\"'<>]*"),
        b"${HOME}/[REDACTED_LOCAL_PATH]",
        "home path",
    ),
]
if _LOCAL_USER:
    REPLACEMENTS.append(
        (
            re.compile(
                rb"(?<![A-Za-z0-9_])"
                + re.escape(_LOCAL_USER.encode())
                + rb"(?![A-Za-z0-9_])"
            ),
            b"[REDACTED_USERNAME]",
            "username",
        )
    )


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Source:
    path: Path
    archive_path: str
    logical_source: str


@dataclass(frozen=True)
class Redaction:
    archive: str
    file: str
    source: str
    original_sha256: str
    redacted_sha256: str
    replacements: str


@dataclass(frozen=True)
class CompiledCacheExclusion:
    file: str
    source: str
    original_sha256: str
    source_file: str
    appears_in_hash_list: bool


def redact(source: Source, archive: str) -> tuple[bytes, Redaction | None]:
    raw = source.path.read_bytes()
    if source.path.suffix.lower() not in TEXT_SUFFIXES or b"\0" in raw[:8192]:
        return raw, None
    changed = raw
    counts: list[str] = []
    for pattern, replacement, label in REPLACEMENTS:
        changed, count = pattern.subn(replacement, changed)
        if count:
            counts.append(f"{label}={count}")
    if changed == raw:
        return raw, None
    return changed, Redaction(
        archive=archive,
        file=source.archive_path,
        source=source.logical_source,
        original_sha256=digest_bytes(raw),
        redacted_sha256=digest_bytes(changed),
        replacements="; ".join(counts),
    )


def repo_source(path: Path, archive_path: str | None = None) -> Source:
    relative = path.relative_to(ROOT).as_posix()
    return Source(path, archive_path or relative, relative)


def tree_sources(root: Path, prefix: str, *, exclude: set[str] | None = None) -> list[Source]:
    excluded = exclude or set()
    sources: list[Source] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        logical = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else f"external/{relative}"
        sources.append(Source(path, f"{prefix}/{relative}", logical))
    return sources


def freeze_sources() -> tuple[list[Source], list[CompiledCacheExclusion]]:
    study_root = ROOT / "output" / "studies"
    sources: list[Source] = []
    exclusions: list[CompiledCacheExclusion] = []
    for study in sorted(path for path in study_root.iterdir() if path.is_dir()):
        freeze = study / "freeze"
        if not freeze.is_dir():
            continue
        hash_lists = sorted(freeze.rglob("*.sha256"))
        hash_list_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace") for path in hash_lists
        )
        for path in sorted(freeze.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(freeze)
            archive_path = f"freeze_manifests/{study.name}/freeze/{relative.as_posix()}"
            if path.suffix.lower() == ".pyc" or "__pycache__" in relative.parts:
                source_name = path.name.split(".cpython-", 1)[0] + ".py"
                source_path = path.parent.parent / source_name
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"compiled cache has no retained source file: {path}"
                    )
                digest = digest_file(path)
                exclusions.append(
                    CompiledCacheExclusion(
                        file=archive_path,
                        source=path.relative_to(ROOT).as_posix(),
                        original_sha256=digest,
                        source_file=source_path.relative_to(ROOT).as_posix(),
                        appears_in_hash_list=(
                            digest in hash_list_text
                            or relative.as_posix() in hash_list_text
                            or path.name in hash_list_text
                        ),
                    )
                )
                continue
            sources.append(Source(path, archive_path, path.relative_to(ROOT).as_posix()))
    return sources, exclusions


def campaign_sources() -> list[Source]:
    run_root = ROOT / "output" / "runs"
    prefixes = ("ara5_mcap_", "cof_scout_r2_", "cof_scout_idrelabel_")
    include_suffixes = {".csv", ".diff", ".json", ".jsonl", ".md", ".txt"}
    sources: list[Source] = []
    for run in sorted(path for path in run_root.iterdir() if path.is_dir() and path.name.startswith(prefixes)):
        for path in sorted(run.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in include_suffixes:
                continue
            if "xtb" in path.relative_to(run).parts or path.name == "plotly.min.js":
                continue
            relative = path.relative_to(run).as_posix()
            sources.append(repo_source(path, f"campaign_traces/{run.name}/{relative}"))
    return sources


def calibration_sources() -> list[Source]:
    names = ("COF-LZU1", "TAPB-PDA", "COF-5", "COF-1", "CTF-1", "sp2c-COF-Py")
    sources: list[Source] = []
    for name in names:
        for base, prefix in (
            (EXTERNAL_CALIBRATION / name, f"calibration/dft/{name}"),
            (EXTERNAL_CALIBRATION / "gfn2_runs" / name, f"calibration/xtb/{name}"),
        ):
            if not base.is_dir():
                raise FileNotFoundError(base)
            for source in tree_sources(base, prefix, exclude={"out", "save"}):
                sources.append(Source(source.path, source.archive_path, f"external_calibration/{prefix}/{source.path.name}"))
    for name in ("week2_pbe_results.json", "week2_pbe_results_clean6.json", "week3v2_clean6_transfer_function.json"):
        path = EXTERNAL_RESULTS / name
        if path.is_file():
            sources.append(Source(path, f"calibration/results/{name}", f"external_calibration/results/{name}"))
    for path in (ROOT / "data" / "cbm_calibration.json", ROOT / "data" / "week3v2_clean6_transfer_function.json"):
        sources.append(repo_source(path, f"calibration/frozen/{path.name}"))
    return sources


def archive_definitions() -> dict[str, list[Source]]:
    study_root = ROOT / "output" / "studies"
    original = study_root / "ara5_reasoning_mcap_v1"
    sensitivities = ROOT / "output" / "analysis" / "reference_sensitivities"
    freezes, _ = freeze_sources()

    figure_sources = tree_sources(original / "publication" / "tables", "figure_source_data/publication/tables")
    for path in (
        original / "publication" / "publication_results.json",
        original / "analysis" / "study_results.json",
    ):
        figure_sources.append(repo_source(path, f"figure_source_data/{path.name}"))
    for study_name in (
        "cof_scout_v2_extension_r2",
        "cof_scout_v2_extension_r2_c_amendment",
        "cof_scout_v2_identifier_relabel_control",
    ):
        analysis = study_root / study_name / "analysis"
        if analysis.is_dir():
            figure_sources.extend(tree_sources(analysis, f"figure_source_data/{study_name}/analysis"))
    for path in sorted((ROOT / "figures").glob("*.csv")):
        figure_sources.append(repo_source(path, f"figure_source_data/extension_figures/{path.name}"))

    return {
        "reference_landscape.tar.gz": [
            repo_source(original / "reference" / "ground_truth_mcap_v1.jsonl", "reference_landscape/ground_truth_mcap_v1.jsonl"),
            repo_source(original / "reference" / "reference_manifest.json", "reference_landscape/reference_manifest.json"),
        ],
        "freeze_manifests.tar.gz": freezes,
        "campaign_traces.tar.gz": campaign_sources(),
        "baseline_outputs.tar.gz": [
            repo_source(original / "baselines.json", "baseline_outputs/baselines.json"),
            repo_source(sensitivities / "item2_offset_sensitivity_and_baselines.csv", "baseline_outputs/item2_offset_sensitivity_and_baselines.csv"),
            repo_source(sensitivities / "item3_added_baselines.csv", "baseline_outputs/item3_added_baselines.csv"),
        ],
        "sensitivity_outputs.tar.gz": tree_sources(sensitivities, "sensitivity_outputs/reference_sensitivities") + [
            repo_source(original / "publication" / "non_imine_sensitivity.json", "sensitivity_outputs/non_imine_sensitivity.json")
        ],
        "figure_source_data.tar.gz": figure_sources,
        "calibration.tar.gz": calibration_sources(),
    }


def write_archive(
    destination: Path,
    sources: list[Source],
    *,
    archive_name: str | None = None,
) -> list[Redaction]:
    redactions: list[Redaction] = []
    with destination.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with tarfile.open(mode="w", fileobj=gzip_handle, format=tarfile.PAX_FORMAT) as archive:
                for source in sorted(sources, key=lambda item: item.archive_path):
                    data, record = redact(source, archive_name or destination.name)
                    if record:
                        redactions.append(record)
                    info = tarfile.TarInfo(source.archive_path)
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(data))
    return redactions


def write_redaction_log(
    records: list[Redaction],
    compiled_cache_exclusions: list[CompiledCacheExclusion] | None = None,
) -> None:
    exclusions = compiled_cache_exclusions or []
    lines = [
        "# Redaction log", "",
        "Only the approved institutional gateway hostname, local paths and the local username were replaced in copied files.",
        "Prompts, responses, predictions, candidate selections, feedback and outcomes were otherwise unchanged.", "",
        f"Compiled-cache exclusions: {len(exclusions)} `.pyc` files and their `__pycache__` directories were excluded; source `.py` files were retained.",
        f"Excluded compiled caches referenced by a frozen `.sha256` list: {sum(item.appears_in_hash_list for item in exclusions)}.", "",
        "| Archive | File | Logical source | Original SHA-256 | Redacted SHA-256 | Replacements |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r.archive} | `{r.file}` | `{r.source}` | `{r.original_sha256}` | `{r.redacted_sha256}` | {r.replacements} |"
        for r in records
    )
    hash_listed = [item for item in exclusions if item.appears_in_hash_list]
    if hash_listed:
        lines.extend([
            "", "## Hash-listed compiled-cache exclusions", "",
            "| File | Logical source | Original SHA-256 | Retained source | Reason |",
            "|---|---|---|---|---|",
        ])
        lines.extend(
            f"| `{item.file}` | `{item.source}` | `{item.original_sha256}` | `{item.source_file}` | excluded compiled cache; source `.py` retained and verifiable |"
            for item in hash_listed
        )
    (DEST / "REDACTION_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_redaction_log_records() -> list[Redaction]:
    path = DEST / "REDACTION_LOG.md"
    if not path.is_file():
        return []
    records: list[Redaction] = []
    pattern = re.compile(
        r"^\| ([^|]+) \| `([^`]*)` \| `([^`]*)` \| `([^`]*)` \| `([^`]*)` \| ([^|]+) \|$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        records.append(
            Redaction(
                archive=match.group(1).strip(),
                file=match.group(2),
                source=match.group(3),
                original_sha256=match.group(4),
                redacted_sha256=match.group(5),
                replacements=match.group(6).strip(),
            )
        )
    return records


def rebuild_freeze_archive() -> None:
    archive_name = "freeze_manifests.tar.gz"
    target = DEST / archive_name
    temporary = DEST / f".{archive_name}.tmp"
    sources, exclusions = freeze_sources()
    print(
        f"rebuilding {archive_name}: {len(sources)} files; "
        f"excluding {len(exclusions)} compiled caches",
        flush=True,
    )
    records = write_archive(temporary, sources, archive_name=archive_name)
    os.replace(temporary, target)
    retained = [
        record for record in read_redaction_log_records()
        if record.archive != archive_name
    ]
    write_redaction_log(retained + records, exclusions)
    write_manifest()
    print(f"rebuilt {archive_name}; redacted {len(records)} copied files")


def write_manifest() -> None:
    files = sorted(
        path for path in DEST.iterdir()
        if path.is_file() and path.name != "MANIFEST.csv"
    )
    with (DEST / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("file", "bytes", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        for path in files:
            writer.writerow({"file": path.name, "bytes": path.stat().st_size, "sha256": digest_file(path)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace existing generated archives")
    parser.add_argument(
        "--only-freeze-manifests",
        action="store_true",
        help="rebuild only freeze_manifests.tar.gz and merge its redaction records",
    )
    args = parser.parse_args()
    if args.only_freeze_manifests:
        rebuild_freeze_archive()
        return
    if EXTERNAL_CALIBRATION == Path(".") or EXTERNAL_RESULTS == Path("."):
        parser.error(
            "set COF_SCOUT_CALIBRATION_SOURCE and COF_SCOUT_CALIBRATION_RESULTS "
            "to the six-COF calculation and result directories"
        )
    definitions = archive_definitions()
    existing = [DEST / name for name in definitions if (DEST / name).exists()]
    if existing and not args.force:
        parser.error("generated archives already exist; review them before rerunning with --force")
    DEST.mkdir(parents=True, exist_ok=True)
    records: list[Redaction] = []
    for name, sources in definitions.items():
        target = DEST / name
        if target.exists():
            target.unlink()
        print(f"building {name}: {len(sources)} files", flush=True)
        records.extend(write_archive(target, sources))
    _, exclusions = freeze_sources()
    write_redaction_log(records, exclusions)
    write_manifest()
    print(f"built {len(definitions)} archives; redacted {len(records)} copied files")


if __name__ == "__main__":
    main()
