#!/usr/bin/env python3
"""Objective-matched lookup baselines for the frozen COF-SCOUT publication study."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from cof_scout.analysis.ground_truth import feasible_keys, key, read_ground_truth, validate_complete
from cof_scout.inventory.assembly import assemble_fragment, load_library


Candidate = tuple[str, str, str, str]
Scaffold = tuple[str, str, str]


def load_landscape(path: Path, expected: set[Candidate]) -> tuple[
    list[Candidate], np.ndarray, list[str], list[str]
]:
    """Load every feasible candidate; failed oracle evaluations remain budgetable misses."""
    records = read_ground_truth(path)
    validate_complete(records, expected)
    by_key = {key(record): record for record in records}
    library = load_library()
    candidates: list[Candidate] = []
    labels: list[int] = []
    smiles: list[str] = []
    statuses: list[str] = []
    for candidate in sorted(expected):
        record = by_key[candidate]
        result = record.get("result", {})
        criterion = result.get("criterion") or record.get("criterion") or {}
        status = str(result.get("status", "unknown"))
        fragment = result.get("fragment_smiles")
        if not fragment:
            fragment = assemble_fragment(*candidate, library=library)
        if not fragment:
            raise ValueError(
                "Cannot construct a structural representation for baseline candidate "
                + "/".join(candidate)
            )
        candidates.append(candidate)
        labels.append(int(status == "success" and bool(criterion.get("is_hit", False))))
        smiles.append(fragment)
        statuses.append(status)
    return candidates, np.asarray(labels, dtype=float), smiles, statuses


def morgan_features(smiles: list[str], n_bits: int = 1024) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    matrix = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid fragment SMILES: {value}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), matrix[index])
    return matrix


def selection_metrics(indices: list[int], candidates: list[Candidate],
                      labels: np.ndarray, statuses: list[str]) -> dict[str, Any]:
    hit_scaffolds: set[Scaffold] = set()
    tested_scaffolds: set[Scaffold] = set()
    cumulative = []
    hits = 0
    failures = 0
    for index in indices:
        scaffold = candidates[index][:3]
        tested_scaffolds.add(scaffold)
        hits += int(labels[index])
        failures += int(statuses[index] != "success")
        if labels[index] == 1:
            hit_scaffolds.add(scaffold)
        cumulative.append(len(hit_scaffolds))
    budget = len(indices)
    return {
        "hits": hits,
        "scaffolds": len(hit_scaffolds),
        "scaffolds_tested": len(tested_scaffolds),
        "decoration_rate": 1.0 - len(tested_scaffolds) / budget if budget else 0.0,
        "raw_hits_per_hit_scaffold": hits / len(hit_scaffolds) if hit_scaffolds else 0.0,
        "scaffold_discovery_auc": sum(cumulative) / budget if budget else 0.0,
        "oracle_failures": failures,
        "selected_candidates": ["/".join(candidates[index]) for index in indices],
    }


def candidate_random_search(candidates: list[Candidate], labels: np.ndarray,
                            statuses: list[str], seeds: int,
                            budget: int) -> list[dict[str, Any]]:
    outcomes = []
    for seed in range(seeds):
        indices = list(range(len(candidates)))
        random.Random(seed).shuffle(indices)
        outcome = selection_metrics(indices[:budget], candidates, labels, statuses)
        outcome["seed"] = seed
        outcomes.append(outcome)
    return outcomes


def _scaffold_groups(candidates: list[Candidate]) -> dict[Scaffold, list[int]]:
    groups: dict[Scaffold, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        groups[candidate[:3]].append(index)
    return dict(groups)


def scaffold_first_random_search(candidates: list[Candidate], labels: np.ndarray,
                                 statuses: list[str], seeds: int,
                                 budget: int) -> list[dict[str, Any]]:
    groups = _scaffold_groups(candidates)
    scaffolds = sorted(groups)
    if budget > len(scaffolds):
        raise ValueError("scaffold-first budget exceeds the number of scaffolds")
    outcomes = []
    for seed in range(seeds):
        rng = random.Random(seed)
        order = list(scaffolds)
        rng.shuffle(order)
        selected = [rng.choice(groups[scaffold]) for scaffold in order[:budget]]
        outcome = selection_metrics(selected, candidates, labels, statuses)
        outcome["seed"] = seed
        outcomes.append(outcome)
    return outcomes


def _scaffold_representatives(candidates: list[Candidate]) -> tuple[
    list[Scaffold], list[int]
]:
    groups = _scaffold_groups(candidates)
    scaffolds = sorted(groups)
    representatives = []
    for scaffold in scaffolds:
        indices = groups[scaffold]
        representatives.append(next(
            (index for index in indices if candidates[index][3] == "R001"), indices[0]
        ))
    return scaffolds, representatives


def _maxmin_indices(features: np.ndarray, budget: int, seed: int) -> list[int]:
    if budget > len(features):
        raise ValueError("MaxMin budget exceeds the feature pool")
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(features)))]
    available = np.ones(len(features), dtype=bool)
    available[selected[0]] = False
    bit_counts = features.sum(axis=1)
    min_distances = np.full(len(features), np.inf, dtype=float)
    while len(selected) < budget:
        last = selected[-1]
        intersections = features @ features[last]
        unions = bit_counts + bit_counts[last] - intersections
        similarities = np.divide(
            intersections, unions, out=np.zeros_like(intersections), where=unions > 0
        )
        min_distances = np.minimum(min_distances, 1.0 - similarities)
        choices = np.flatnonzero(available)
        best = min_distances[choices].max()
        ties = choices[np.isclose(min_distances[choices], best)]
        next_index = int(rng.choice(ties))
        selected.append(next_index)
        available[next_index] = False
    return selected


def scaffold_maxmin_search(candidates: list[Candidate], labels: np.ndarray,
                           statuses: list[str], features: np.ndarray,
                           seeds: int, budget: int) -> list[dict[str, Any]]:
    _, representative_indices = _scaffold_representatives(candidates)
    scaffold_features = features[np.asarray(representative_indices, dtype=int)]
    outcomes = []
    for seed in range(seeds):
        local_indices = _maxmin_indices(scaffold_features, budget, seed)
        selected = [representative_indices[index] for index in local_indices]
        outcome = selection_metrics(selected, candidates, labels, statuses)
        outcome["seed"] = seed
        outcomes.append(outcome)
    return outcomes


def gp_ucb_search(candidates: list[Candidate], labels: np.ndarray,
                  statuses: list[str], features: np.ndarray, seeds: int,
                  budget: int, initial: int, kappa: float,
                  scaffold_aware: bool) -> list[dict[str, Any]]:
    if initial >= budget:
        raise ValueError("initial batch must be smaller than the budget")
    outcomes = []
    all_indices = np.arange(len(candidates))
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        selected = list(rng.choice(all_indices, size=initial, replace=False).astype(int))
        remaining = np.ones(len(candidates), dtype=bool)
        remaining[selected] = False
        discovered = {
            candidates[index][:3] for index in selected if labels[index] == 1
        }
        if scaffold_aware and discovered:
            for index, candidate in enumerate(candidates):
                if candidate[:3] in discovered:
                    remaining[index] = False
        while len(selected) < budget:
            train = np.asarray(selected, dtype=int)
            kernel = (
                ConstantKernel(1.0, constant_value_bounds="fixed")
                * RBF(1.0, length_scale_bounds="fixed")
                + WhiteKernel(1e-6, noise_level_bounds="fixed")
            )
            model = GaussianProcessRegressor(
                kernel=kernel, optimizer=None, normalize_y=False, random_state=seed
            )
            model.fit(features[train], labels[train])
            choices = np.flatnonzero(remaining)
            if not len(choices):
                raise RuntimeError("GP-UCB exhausted its candidate pool")
            mean, std = model.predict(features[choices], return_std=True)
            next_index = int(choices[np.argmax(mean + kappa * std)])
            selected.append(next_index)
            remaining[next_index] = False
            if scaffold_aware and labels[next_index] == 1:
                scaffold = candidates[next_index][:3]
                for index, candidate in enumerate(candidates):
                    if candidate[:3] == scaffold:
                        remaining[index] = False
        outcome = selection_metrics(selected, candidates, labels, statuses)
        outcome["seed"] = seed
        outcomes.append(outcome)
    return outcomes


def distribution(outcomes: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = np.asarray([row[field] for row in outcomes], dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize(method: str, outcomes: list[dict[str, Any]],
              budget: int) -> dict[str, Any]:
    fields = (
        "hits", "scaffolds", "scaffolds_tested", "decoration_rate",
        "raw_hits_per_hit_scaffold", "scaffold_discovery_auc", "oracle_failures",
    )
    return {
        "method": method,
        "evaluations": budget,
        "runs": len(outcomes),
        **{field: distribution(outcomes, field) for field in fields},
        "per_seed": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--feasible-csv", type=Path, default=Path("data/search_space_v2_feasible.csv")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--random-runs", type=int, default=1000)
    parser.add_argument("--diversity-runs", type=int, default=1000)
    parser.add_argument("--bo-runs", type=int, default=200)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--kappa", type=float, default=1.0)
    args = parser.parse_args()

    expected = feasible_keys(args.feasible_csv)
    candidates, labels, smiles, statuses = load_landscape(args.ground_truth, expected)
    if args.budget > len({candidate[:3] for candidate in candidates}):
        parser.error("budget exceeds the scaffold pool")
    features = morgan_features(smiles)
    methods = {
        "candidate_random": candidate_random_search(
            candidates, labels, statuses, args.random_runs, args.budget
        ),
        "scaffold_first_random": scaffold_first_random_search(
            candidates, labels, statuses, args.random_runs, args.budget
        ),
        "scaffold_maxmin": scaffold_maxmin_search(
            candidates, labels, statuses, features, args.diversity_runs, args.budget
        ),
        "candidate_morgan_gp_ucb": gp_ucb_search(
            candidates, labels, statuses, features, args.bo_runs, args.budget,
            args.initial, args.kappa, scaffold_aware=False,
        ),
        "scaffold_aware_morgan_gp_ucb": gp_ucb_search(
            candidates, labels, statuses, features, args.bo_runs, args.budget,
            args.initial, args.kappa, scaffold_aware=True,
        ),
    }
    result = {
        "candidate_pool": len(candidates),
        "scaffold_pool": len({candidate[:3] for candidate in candidates}),
        "positive_candidates": int(labels.sum()),
        "positive_scaffolds": len({
            candidates[index][:3] for index in range(len(candidates)) if labels[index] == 1
        }),
        "failed_oracle_records": sum(status != "success" for status in statuses),
        "budget": args.budget,
        "methods": {
            name: summarize(name, outcomes, args.budget)
            for name, outcomes in methods.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for method in result["methods"].values():
        print(
            f"{method['method']}: hit scaffolds="
            f"{method['scaffolds']['mean']:.2f}±{method['scaffolds']['std']:.2f}; "
            f"raw hits={method['hits']['mean']:.2f}±{method['hits']['std']:.2f}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
