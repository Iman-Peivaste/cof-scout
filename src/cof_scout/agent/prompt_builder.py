"""Generate the leakage-audited Ara-3 COF-design system prompt."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from cof_scout.paths import data_path


# Scores, implementation labels, and campaign-specific findings must not appear
# in the prompt seen by the proposing model.
FORBIDDEN = (
    "0.95", "0.85", "0.80", "0.75", "0.50", "0.10",
    "s_csi", "s_linkage", "reward", "oracle_class", "near_miss",
    "class A", "class B", "class C", "class D",
    "vinylene wins", "feasible_v3", "feasible_v4", "ara2", "week7",
    "comparative_study",
)


def _build_feasible_index(feasible_csv: str | Path | None = None) -> tuple[dict, dict]:
    """Map each node/linker ID to linkages proven feasible in the CSV.

    This deliberately does not trust ``compatible_linkages`` from the library:
    feasibility is defined by rows where ``is_feasible`` is true.
    """
    node_linkages, linker_linkages = defaultdict(set), defaultdict(set)
    target = Path(feasible_csv) if feasible_csv is not None else data_path("search_space_v2_feasible.csv")
    with target.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("is_feasible", "")).strip().lower() != "true":
                continue
            linkage = row["linkage_norm"].strip()
            node_linkages[row["node_id"].strip()].add(linkage)
            linker_linkages[row["linker_id"].strip()].add(linkage)
    return ({identifier: sorted(linkages) for identifier, linkages in node_linkages.items()},
            {identifier: sorted(linkages) for identifier, linkages in linker_linkages.items()})


def _load_feasible_set(feasible_csv: str | Path | None = None) -> frozenset:
    """Return exact (node_id, linker_id, linkage_norm, r_group_id) tuples from the feasible CSV.

    This is the authoritative inventory for the Ara-4 guard: a proposal is valid iff its
    candidate key is a member of this set.
    """
    keys = set()
    target = Path(feasible_csv) if feasible_csv is not None else data_path("search_space_v2_feasible.csv")
    with target.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("is_feasible", "")).strip().lower() != "true":
                continue
            keys.add((
                row["node_id"].strip(),
                row["linker_id"].strip(),
                row["linkage_norm"].strip(),
                row["r_group_id"].strip(),
            ))
    return frozenset(keys)


def build_prompt(library_path: str | Path | None = None,
                 feasible_csv: str | Path | None = None) -> tuple[str, dict]:
    """Build a leakage-free system prompt and return it with audit statistics."""
    library_target = Path(library_path) if library_path is not None else data_path("building_blocks_library_v2.json")
    library = json.loads(library_target.read_text(encoding="utf-8"))
    node_linkages, linker_linkages = _build_feasible_index(feasible_csv)
    lines = [
        "You are an expert computational chemist designing covalent organic framework (COF)",
        "photocatalysts for OVERALL WATER SPLITTING. You will propose candidates one at a time.",
        "Each will be evaluated by simulation, and you will see the complete results before",
        "proposing the next.", "",
        "TARGET CRITERIA (a candidate is a HIT only if ALL FOUR hold simultaneously):",
        "  1. optical gap (PBE-corrected) between 1.8 and 2.2 eV",
        "  2. CBM  <  0.00 V vs NHE       (drives proton reduction: H+ -> H2)",
        "  3. VBM  >= +1.23 V vs NHE      (drives water oxidation: H2O -> O2)",
        "  4. hydrolytic stability of the linkage under aqueous photocatalytic conditions", "",
        "Note: VBM = CBM + gap. Satisfying criteria 2 and 3 simultaneously constrains both",
        "the gap magnitude and the absolute position of the CBM.", "",
        "On criterion 4: stability is a hard gate. You are NOT given the stability values in",
        "advance. Reason about the bond chemistry yourself — you will see the computed score",
        "after each evaluation.", "",
        "AVAILABLE BUILDING BLOCKS:", "", "NODES:",
    ]

    excluded_nodes = []
    for node in library["nodes"]:
        identifier = node["id"]
        linkages = node_linkages.get(identifier, [])
        if not linkages:
            excluded_nodes.append(identifier)
            continue
        lines.append(
            f"  {identifier} | {node['name']} | SMILES: {node['smiles']}"
            f" | role: {node.get('reactive_group', 'unknown')}"
            f" | compatible_linkages: {', '.join(linkages)}"
        )

    lines += ["", "LINKERS:"]
    excluded_linkers = []
    for linker in library["linkers"]:
        identifier = linker["id"]
        linkages = linker_linkages.get(identifier, [])
        if not linkages:
            excluded_linkers.append(identifier)
            continue
        lines.append(
            f"  {identifier} | {linker['name']} | SMILES: {linker['smiles']}"
            f" | role: {linker.get('reactive_group', 'unknown')}"
            f" | compatible_linkages: {', '.join(linkages)}"
        )

    lines += ["", "R-GROUPS (substituents applied to the assembled fragment; R001 = unsubstituted):"]
    for r_group in library["r_groups"]:
        # Do not disclose the library's donor/acceptor electronic-effect label.
        lines.append(f"  {r_group['id']} | {r_group['name']} | SMILES: {r_group['smiles']}")

    lines += [
        "", "AVAILABLE LINKAGES (names only — no scores are provided here):",
        "  beta_ketoenamine", "  hydrazone", "  imine", "  vinylene", "",
        "ASSEMBLY RULES:",
        "The compatible_linkages fields are block-level summaries of the approved inventory.",
        "Your exact NODE/LINKER/LINKAGE/R_GROUP tuple is checked against that inventory and",
        "then checked by RDKit fragment assembly before simulation. If it is absent from the",
        "inventory or cannot assemble, you will be asked to propose again immediately.", "",
        "R-GROUP NOTE: each R-group is a single substituent installed at ONE aromatic C-H",
        "position on the assembled fragment. Reason about one substitution site, not several.", "",
        "YOUR TASK EACH ITERATION:",
        "  1. Reason mechanistically from the structures — electronic structure, conjugation,",
        "     band edge positions, reactivity, stability mechanism. Think out loud.",
        "  2. Propose exactly ONE candidate, stating clearly on separate lines:",
        "        NODE: <id>   LINKER: <id>   LINKAGE: <name>   R_GROUP: <id>",
        "  3. End with a falsifiable directional CLAIM about how this candidate will compare",
        "     to a previously evaluated one, or which criterion it will pass or fail, and why.",
        '     State it on a single line starting with "CLAIM:", tagged with the relevant axis:',
        "     [gap] [cbm] [vbm] [stability] [overall].",
        "     Predict directions and comparisons — do NOT predict numeric values.", "",
        "You will then be shown the complete simulation results. Use them to refine your",
        "understanding before proposing the next candidate. You are building knowledge from",
        "scratch; no information about which families work has been given to you.", "",
        "SCORING: you are evaluated solely on DISTINCT SCAFFOLDS hit (node + linker + linkage).",
        "R-group variants of an already-discovered scaffold do not count as new discoveries.",
    ]
    prompt = "\n".join(lines)
    return prompt, {
        "chars": len(prompt), "lines": len(lines),
        "excluded_nodes": excluded_nodes, "excluded_linkers": excluded_linkers,
        "n_nodes_shown": len(library["nodes"]) - len(excluded_nodes),
        "n_linkers_shown": len(library["linkers"]) - len(excluded_linkers),
    }


def run_leakage_audit(prompt: str) -> None:
    """Raise an AssertionError if known score/taxonomy leakage enters the prompt."""
    failures = []
    for token in FORBIDDEN:
        for line_number, line in enumerate(prompt.splitlines(), 1):
            if token.lower() in line.lower():
                failures.append(f"line {line_number}: {token!r} found in {line.strip()[:100]!r}")
    if failures:
        raise AssertionError("LEAKAGE AUDIT FAILED:\n" + "\n".join(failures))


def build_and_save(library_path: str | Path = "data/building_blocks_library_v2.json",
                   feasible_csv: str | Path = "data/search_space_v2_feasible.csv",
                   output_path: str | Path = "output/system_prompt.txt") -> tuple[str, str]:
    """Generate, audit, save, and return ``(prompt, sha256)``."""
    prompt, stats = build_prompt(library_path, feasible_csv)
    run_leakage_audit(prompt)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    print(f"Prompt written: {output_path} ({stats['chars']:,} chars, {stats['lines']} lines)")
    print(f"SHA-256: {digest}")
    print("Leakage audit: PASSED")
    print(f"Nodes shown: {stats['n_nodes_shown']} excluded: {stats['excluded_nodes']}")
    print(f"Linkers shown: {stats['n_linkers_shown']} excluded: {stats['excluded_linkers']}")
    return prompt, digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/system_prompt.txt")
    parser.add_argument("--library", default="data/building_blocks_library_v2.json")
    parser.add_argument("--feasible-csv", default="data/search_space_v2_feasible.csv")
    args = parser.parse_args()
    build_and_save(args.library, args.feasible_csv, args.out)
