#!/usr/bin/env python3
"""Post-freeze electronic-offset and baseline sensitivity analyses.

The script reads the immutable MCAP reference landscape and never modifies frozen
study artifacts.  It writes one CSV for each requested analysis item plus a short
Markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import sparse
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from cof_scout.analysis.ground_truth import feasible_keys, key, read_ground_truth
from cof_scout.baselines.search import (
    Candidate,
    Scaffold,
    _scaffold_representatives,
    candidate_random_search,
    distribution,
    load_landscape,
    morgan_features,
    scaffold_first_random_search,
    selection_metrics,
    summarize,
)


# Thousands of <=50 x 50 GP factorizations are much faster without per-fit BLAS
# thread teams.  Keep the limiter alive for the lifetime of the process.
_THREADPOOL_LIMITER = threadpool_limits(limits=1)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE = (
    ROOT / "output/studies/ara5_reasoning_mcap_v1/reference/ground_truth_mcap_v1.jsonl"
)
DEFAULT_RECURRENT = (
    ROOT
    / "output/studies/ara5_reasoning_mcap_v1/publication/tables"
    / "recurrent_scaffold_representatives.csv"
)
DEFAULT_OUT = ROOT / "output/analysis/reference_sensitivities"

GAP_MIN = 1.8
GAP_MAX = 2.2
CBM_MAX = 0.0
VBM_MIN = 1.23
STABILITY_MIN = 0.70
OFFSETS = (2.42, 2.49, 2.62, 2.94)
BASELINE_OFFSETS = (2.42, 2.94)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _result_for(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    return result if isinstance(result, dict) else record


def _candidate_key(candidate: Candidate) -> str:
    return "/".join(candidate)


def _scaffold_key(scaffold: Scaffold) -> str:
    return "/".join(scaffold)


def recalculated_verdict(result: dict[str, Any], c_offset: float) -> dict[str, Any]:
    """Rebuild the frozen four gates using a replacement EA-to-NHE offset."""
    status = str(result.get("status", "unknown"))
    if status != "success":
        return {
            "status": status,
            "gap_corrected_eV": None,
            "cbm_vs_nhe": None,
            "vbm_vs_nhe": None,
            "stability_score": None,
            "gap_pass": False,
            "cbm_pass": False,
            "vbm_pass": False,
            "stability_pass": False,
            "is_hit": False,
        }
    gap = float(result["gap_corrected_eV"])
    ea = float(result["ea_eV"])
    stability = float(result.get("stability_score", result["s_linkage"]))
    cbm = -ea + c_offset
    vbm = cbm + gap
    gap_pass = GAP_MIN <= gap <= GAP_MAX
    cbm_pass = cbm < CBM_MAX
    vbm_pass = vbm >= VBM_MIN
    stability_pass = stability >= STABILITY_MIN
    return {
        "status": status,
        "gap_corrected_eV": gap,
        "cbm_vs_nhe": cbm,
        "vbm_vs_nhe": vbm,
        "stability_score": stability,
        "gap_pass": gap_pass,
        "cbm_pass": cbm_pass,
        "vbm_pass": vbm_pass,
        "stability_pass": stability_pass,
        "is_hit": gap_pass and cbm_pass and vbm_pass and stability_pass,
    }


def labels_for_offset(results: list[dict[str, Any]], c_offset: float) -> np.ndarray:
    return np.asarray(
        [int(recalculated_verdict(result, c_offset)["is_hit"]) for result in results],
        dtype=float,
    )


class SimilarityColumns:
    """Lazy full-landscape similarity columns backed by sparse Morgan bits."""

    def __init__(self, features: np.ndarray, kind: str):
        if kind not in {"rbf", "tanimoto"}:
            raise ValueError(f"Unsupported similarity: {kind}")
        self.kind = kind
        self.features = sparse.csr_matrix(features, dtype=np.float64)
        self.counts = np.asarray(self.features.sum(axis=1)).ravel()
        self.cache: dict[int, np.ndarray] = {}

    def column(self, index: int) -> np.ndarray:
        cached = self.cache.get(index)
        if cached is not None:
            return cached
        intersections = np.asarray(
            (self.features @ self.features[index].T).toarray()
        ).ravel()
        if self.kind == "rbf":
            squared_distances = self.counts + self.counts[index] - 2.0 * intersections
            values = np.exp(-0.5 * np.maximum(squared_distances, 0.0))
        else:
            unions = self.counts + self.counts[index] - intersections
            values = np.divide(
                intersections,
                unions,
                out=np.zeros_like(intersections),
                where=unions > 0,
            )
        values[index] = 1.0
        self.cache[index] = values
        return values

    def matrix(self, rows: np.ndarray, columns: list[int]) -> np.ndarray:
        return np.column_stack([self.column(column)[rows] for column in columns])


def _fixed_rbf_acquisition(
    cache: SimilarityColumns,
    selected: list[int],
    labels: np.ndarray,
    choices: np.ndarray,
    kappa: float,
) -> np.ndarray:
    train = np.asarray(selected, dtype=int)
    kernel = cache.matrix(train, selected)
    kernel.flat[:: len(train) + 1] += 1e-6 + 1e-10
    cholesky = np.linalg.cholesky(kernel)
    alpha = cho_solve((cholesky, True), labels[train], check_finite=False)
    cross = cache.matrix(choices, selected)
    mean = cross @ alpha
    projected = solve_triangular(cholesky, cross.T, lower=True, check_finite=False)
    variance = 1.0 + 1e-6 - np.einsum("ij,ij->j", projected, projected)
    return mean + kappa * np.sqrt(np.maximum(variance, 0.0))


def _tanimoto_mle(
    similarity: np.ndarray,
    targets: np.ndarray,
    start: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    identity = np.eye(len(targets), dtype=float)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        signal = math.exp(float(theta[0]))
        noise = math.exp(float(theta[1]))
        covariance = signal * similarity + (noise + 1e-10) * identity
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            return 1e30, np.zeros(2, dtype=float)
        alpha = cho_solve((cholesky, True), targets, check_finite=False)
        value = (
            0.5 * float(targets @ alpha)
            + float(np.log(np.diag(cholesky)).sum())
            + 0.5 * len(targets) * math.log(2.0 * math.pi)
        )
        inverse = cho_solve((cholesky, True), identity, check_finite=False)
        common = np.outer(alpha, alpha) - inverse
        log_likelihood_gradient = np.asarray([
            0.5 * np.sum(common * (signal * similarity)),
            0.5 * np.sum(common * (noise * identity)),
        ])
        return value, -log_likelihood_gradient

    fit = minimize(
        objective,
        start,
        method="L-BFGS-B",
        jac=True,
        bounds=((math.log(1e-3), math.log(1e3)), (math.log(1e-6), math.log(1.0))),
        options={"maxiter": 100, "ftol": 1e-10},
    )
    theta = np.asarray(fit.x if fit.success else start, dtype=float)
    return theta, math.exp(float(theta[0])), math.exp(float(theta[1]))


def _tanimoto_acquisition(
    cache: SimilarityColumns,
    selected: list[int],
    labels: np.ndarray,
    choices: np.ndarray,
    kappa: float,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    train = np.asarray(selected, dtype=int)
    similarity = cache.matrix(train, selected)
    theta, signal, noise = _tanimoto_mle(similarity, labels[train], theta)
    covariance = signal * similarity
    covariance.flat[:: len(train) + 1] += noise + 1e-10
    cholesky = np.linalg.cholesky(covariance)
    alpha = cho_solve((cholesky, True), labels[train], check_finite=False)
    cross = signal * cache.matrix(choices, selected)
    mean = cross @ alpha
    projected = solve_triangular(cholesky, cross.T, lower=True, check_finite=False)
    variance = signal + noise - np.einsum("ij,ij->j", projected, projected)
    acquisition = mean + kappa * np.sqrt(np.maximum(variance, 0.0))
    return acquisition, theta, signal, noise


def cached_gp_ucb_search(
    candidates: list[Candidate],
    labels: np.ndarray,
    statuses: list[str],
    cache: SimilarityColumns,
    seeds: int,
    budget: int,
    initial: int,
    kappa: float,
    scaffold_aware: bool,
    fitted_tanimoto: bool = False,
    progress: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Equivalent fixed-RBF GP-UCB or fitted-Tanimoto sensitivity variant."""
    all_indices = np.arange(len(candidates))
    scaffold_members: dict[Scaffold, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        scaffold_members[candidate[:3]].append(index)
    outcomes: list[dict[str, Any]] = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        selected = list(rng.choice(all_indices, size=initial, replace=False).astype(int))
        remaining = np.ones(len(candidates), dtype=bool)
        remaining[selected] = False
        if scaffold_aware:
            for index in selected:
                if labels[index] == 1:
                    remaining[scaffold_members[candidates[index][:3]]] = False
        theta = np.asarray([0.0, math.log(1e-3)], dtype=float)
        final_signal = 1.0
        final_noise = 1e-3
        while len(selected) < budget:
            choices = np.flatnonzero(remaining)
            if not len(choices):
                raise RuntimeError("GP-UCB exhausted its candidate pool")
            if fitted_tanimoto:
                acquisition, theta, final_signal, final_noise = _tanimoto_acquisition(
                    cache, selected, labels, choices, kappa, theta
                )
            else:
                acquisition = _fixed_rbf_acquisition(
                    cache, selected, labels, choices, kappa
                )
            next_index = int(choices[int(np.argmax(acquisition))])
            selected.append(next_index)
            remaining[next_index] = False
            if scaffold_aware and labels[next_index] == 1:
                remaining[scaffold_members[candidates[next_index][:3]]] = False
        outcome = selection_metrics(selected, candidates, labels, statuses)
        outcome["seed"] = seed
        if fitted_tanimoto:
            outcome["fitted_signal_variance"] = final_signal
            outcome["fitted_noise_variance"] = final_noise
        outcomes.append(outcome)
        if progress is not None and ((seed + 1) % 20 == 0 or seed + 1 == seeds):
            progress(seed + 1)
    return outcomes


def linkage_elimination_maxmin(
    candidates: list[Candidate],
    labels: np.ndarray,
    statuses: list[str],
    features: np.ndarray,
    seeds: int,
    budget: int,
    miss_limit: int,
) -> list[dict[str, Any]]:
    scaffolds, representatives = _scaffold_representatives(candidates)
    representative_array = np.asarray(representatives, dtype=int)
    scaffold_features = features[representative_array]
    bit_counts = scaffold_features.sum(axis=1)
    intersections = scaffold_features @ scaffold_features.T
    unions = bit_counts[:, None] + bit_counts[None, :] - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions > 0,
    )
    distances = 1.0 - similarities
    linkages = np.asarray([scaffold[2] for scaffold in scaffolds], dtype=object)
    outcomes: list[dict[str, Any]] = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        available = np.ones(len(scaffolds), dtype=bool)
        eliminated: set[str] = set()
        miss_streak = defaultdict(int)
        linkage_hits = defaultdict(int)
        selected_local: list[int] = []
        min_distances = np.full(len(scaffolds), np.inf, dtype=float)
        fallback_selections = 0
        while len(selected_local) < budget:
            eligible = available & np.asarray(
                [linkage not in eliminated for linkage in linkages], dtype=bool
            )
            choices = np.flatnonzero(eligible)
            if not len(choices):
                # Fixed-horizon safeguard: if protected/active classes are exhausted,
                # continue MaxMin over every unselected scaffold.
                choices = np.flatnonzero(available)
                fallback_selections += 1
            if not len(choices):
                raise RuntimeError("Linkage-elimination MaxMin exhausted all scaffolds")
            if not selected_local:
                next_local = int(rng.choice(choices))
            else:
                best = min_distances[choices].max()
                ties = choices[np.isclose(min_distances[choices], best)]
                next_local = int(rng.choice(ties))
            selected_local.append(next_local)
            available[next_local] = False
            min_distances = np.minimum(min_distances, distances[:, next_local])
            global_index = representatives[next_local]
            linkage = str(linkages[next_local])
            if labels[global_index] == 1:
                linkage_hits[linkage] += 1
                miss_streak[linkage] = 0
                # A fallback observation can rescue a previously eliminated,
                # formerly hitless class.  From its first hit onward it is protected.
                eliminated.discard(linkage)
            else:
                miss_streak[linkage] += 1
                other_active_with_candidates = {
                    str(linkages[index])
                    for index in np.flatnonzero(available)
                    if str(linkages[index]) not in eliminated
                    and str(linkages[index]) != linkage
                }
                if (
                    miss_streak[linkage] >= miss_limit
                    and linkage_hits[linkage] == 0
                    and other_active_with_candidates
                ):
                    eliminated.add(linkage)
        selected = [representatives[index] for index in selected_local]
        if any(linkage_hits[linkage] for linkage in eliminated):
            raise AssertionError("A linkage class with an observed hit remained eliminated")
        outcome = selection_metrics(selected, candidates, labels, statuses)
        outcome.update(
            seed=seed,
            evaluations=len(selected),
            early_stop=False,
            eliminated_linkages=";".join(sorted(eliminated)),
            fallback_selections=fallback_selections,
        )
        outcomes.append(outcome)
    return outcomes


def cached_scaffold_maxmin_search(
    candidates: list[Candidate],
    labels: np.ndarray,
    statuses: list[str],
    features: np.ndarray,
    seeds: int,
    budget: int,
) -> list[dict[str, Any]]:
    """Exact scaffold-MaxMin policy with one precomputed distance matrix."""
    _, representatives = _scaffold_representatives(candidates)
    representative_array = np.asarray(representatives, dtype=int)
    scaffold_features = features[representative_array]
    bit_counts = scaffold_features.sum(axis=1)
    intersections = scaffold_features @ scaffold_features.T
    unions = bit_counts[:, None] + bit_counts[None, :] - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions > 0,
    )
    distances = 1.0 - similarities
    outcomes: list[dict[str, Any]] = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        selected_local = [int(rng.integers(len(representatives)))]
        available = np.ones(len(representatives), dtype=bool)
        available[selected_local[0]] = False
        min_distances = np.full(len(representatives), np.inf, dtype=float)
        while len(selected_local) < budget:
            last = selected_local[-1]
            min_distances = np.minimum(min_distances, distances[:, last])
            choices = np.flatnonzero(available)
            best = min_distances[choices].max()
            ties = choices[np.isclose(min_distances[choices], best)]
            next_local = int(rng.choice(ties))
            selected_local.append(next_local)
            available[next_local] = False
        selected = [representatives[index] for index in selected_local]
        outcome = selection_metrics(selected, candidates, labels, statuses)
        outcome["seed"] = seed
        outcomes.append(outcome)
    return outcomes


def _summary_row(
    method: str,
    outcomes: list[dict[str, Any]],
    budget: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "record_type": "summary",
        "method": method,
        "runs": len(outcomes),
        "budget": budget,
        "mean_evaluations": float(np.mean([len(row["selected_candidates"]) for row in outcomes])),
        "early_stop_rate": float(np.mean([bool(row.get("early_stop")) for row in outcomes])),
        "mean_hits": distribution(outcomes, "hits")["mean"],
        "std_hits": distribution(outcomes, "hits")["std"],
        "mean_hit_scaffolds": distribution(outcomes, "scaffolds")["mean"],
        "std_hit_scaffolds": distribution(outcomes, "scaffolds")["std"],
        "median_hit_scaffolds": distribution(outcomes, "scaffolds")["median"],
        "p5_hit_scaffolds": distribution(outcomes, "scaffolds")["p5"],
        "p95_hit_scaffolds": distribution(outcomes, "scaffolds")["p95"],
        "mean_scaffold_discovery_auc": distribution(
            outcomes, "scaffold_discovery_auc"
        )["mean"],
        **extra,
    }


def _seed_rows(
    method: str,
    outcomes: list[dict[str, Any]],
    budget: int,
    include_selected_candidates: bool = False,
    **extra: Any,
) -> list[dict[str, Any]]:
    rows = []
    for outcome in outcomes:
        row = {
            "record_type": "seed",
            "method": method,
            "seed": outcome["seed"],
            "budget": budget,
            "evaluations": len(outcome["selected_candidates"]),
            "hits": outcome["hits"],
            "hit_scaffolds": outcome["scaffolds"],
            "scaffolds_tested": outcome["scaffolds_tested"],
            "decoration_rate": outcome["decoration_rate"],
            "scaffold_discovery_auc": outcome["scaffold_discovery_auc"],
            "oracle_failures": outcome["oracle_failures"],
            "early_stop": outcome.get("early_stop", False),
            "eliminated_linkages": outcome.get("eliminated_linkages", ""),
            "fitted_signal_variance": outcome.get("fitted_signal_variance", ""),
            "fitted_noise_variance": outcome.get("fitted_noise_variance", ""),
            "fallback_selections": outcome.get("fallback_selections", 0),
            **extra,
        }
        if include_selected_candidates:
            row["selected_candidates"] = ";".join(outcome["selected_candidates"])
        rows.append(row)
    return rows


def item1_imines(
    candidates: list[Candidate], results: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = defaultdict(int)
    electronic_pass_scaffolds: set[Scaffold] = set()
    for candidate, result in zip(candidates, results):
        if candidate[2] != "imine":
            continue
        verdict = recalculated_verdict(result, 2.62)
        counts["feasible_imines"] += 1
        if verdict["status"] != "success":
            counts["oracle_failure"] += 1
            continue
        counts["evaluated_imines"] += 1
        gap_pass = bool(verdict["gap_pass"])
        cbm_pass = bool(verdict["cbm_pass"])
        vbm_pass = bool(verdict["vbm_pass"])
        counts["gap_pass"] += int(gap_pass)
        counts["cbm_pass"] += int(cbm_pass)
        counts["vbm_pass"] += int(vbm_pass)
        if gap_pass and cbm_pass and vbm_pass:
            counts["all_three_pass"] += 1
            electronic_pass_scaffolds.add(candidate[:3])
        elif not gap_pass and cbm_pass and vbm_pass:
            counts["gap_only_failure"] += 1
        elif gap_pass and cbm_pass and not vbm_pass:
            counts["vbm_only_failure"] += 1
        elif not gap_pass and cbm_pass and not vbm_pass:
            counts["gap_and_vbm_failure"] += 1
        elif not cbm_pass:
            counts["includes_cbm_failure"] += 1
        else:
            counts["other_failure"] += 1
    frozen_hit_scaffolds: set[Scaffold] = set()
    relaxed_hit_scaffolds: set[Scaffold] = set()
    for candidate, result in zip(candidates, results):
        verdict = recalculated_verdict(result, 2.62)
        if verdict["is_hit"]:
            frozen_hit_scaffolds.add(candidate[:3])
            relaxed_hit_scaffolds.add(candidate[:3])
        elif (
            candidate[2] == "imine"
            and verdict["status"] == "success"
            and verdict["gap_pass"]
            and verdict["cbm_pass"]
            and verdict["vbm_pass"]
        ):
            relaxed_hit_scaffolds.add(candidate[:3])
    counts["electronic_pass_scaffolds"] = len(electronic_pass_scaffolds)
    counts["frozen_four_gate_hit_scaffolds"] = len(frozen_hit_scaffolds)
    counts["hit_scaffolds_if_imine_persistence_dropped"] = len(relaxed_hit_scaffolds)

    definitions = {
        "evaluated_imines": "Successful/evaluated imines used for the electronic-gate breakdown.",
        "feasible_imines": "All feasible imine rows, including frozen oracle failures.",
        "gap_pass": "Passes 1.8 <= corrected gap <= 2.2 eV.",
        "cbm_pass": "Passes CBM < 0.00 V at C_offset=2.62 V.",
        "vbm_pass": "Passes VBM >= 1.23 V at C_offset=2.62 V.",
        "all_three_pass": "Passes gap, CBM and VBM; persistence ignored.",
        "electronic_pass_scaffolds": "Distinct imine (node, linker, linkage) scaffolds represented by all-three-pass candidates.",
        "frozen_four_gate_hit_scaffolds": "Distinct four-gate hit scaffolds at C_offset=2.62 V.",
        "hit_scaffolds_if_imine_persistence_dropped": "Distinct hit scaffolds if electronic-pass imines are accepted regardless of persistence.",
        "gap_only_failure": "Fails gap only; CBM and VBM pass.",
        "vbm_only_failure": "Fails VBM only; gap and CBM pass.",
        "gap_and_vbm_failure": "Fails gap and VBM; CBM passes.",
        "includes_cbm_failure": "Fails CBM, with any status of gap/VBM.",
        "oracle_failure": "Frozen oracle status is not success.",
        "other_failure": "Any remaining electronic-gate combination.",
    }
    rows = []
    total = counts["evaluated_imines"]
    no_percentage_metrics = {
        "electronic_pass_scaffolds",
        "frozen_four_gate_hit_scaffolds",
        "hit_scaffolds_if_imine_persistence_dropped",
        "feasible_imines",
        "oracle_failure",
    }
    for category in definitions:
        count = counts[category]
        rows.append({
            "category": category,
            "count": count,
            "percent_of_evaluated_imines": (
                "" if category in no_percentage_metrics
                else 100.0 * count / total if total else 0.0
            ),
            "definition": definitions[category],
        })
    return rows, dict(counts)


def _load_recurrent(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def item2_offsets_and_baselines(
    candidates: list[Candidate],
    results: list[dict[str, Any]],
    statuses: list[str],
    features: np.ndarray,
    recurrent_rows: list[dict[str, str]],
    budget: int,
    random_runs: int,
    diversity_runs: int,
    bo_runs: int,
    initial: int,
    kappa: float,
) -> tuple[list[dict[str, Any]], dict[float, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[float, dict[str, Any]] = {}
    result_by_candidate = {candidate: result for candidate, result in zip(candidates, results)}
    candidate_indices_by_scaffold: dict[Scaffold, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        candidate_indices_by_scaffold[candidate[:3]].append(index)
    labels_by_offset: dict[float, np.ndarray] = {}
    for offset in OFFSETS:
        labels = labels_for_offset(results, offset)
        labels_by_offset[offset] = labels
        linkage_summary: dict[str, dict[str, int]] = {}
        for linkage in sorted({candidate[2] for candidate in candidates}):
            indices = [index for index, candidate in enumerate(candidates) if candidate[2] == linkage]
            hit_indices = [index for index in indices if labels[index] == 1]
            hit_scaffolds = {candidates[index][:3] for index in hit_indices}
            linkage_summary[linkage] = {
                "hits": len(hit_indices), "hit_scaffolds": len(hit_scaffolds)
            }
            rows.append({
                "record_type": "offset_linkage_summary",
                "c_offset_V": offset,
                "linkage": linkage,
                "candidate_hits": len(hit_indices),
                "hit_scaffolds": len(hit_scaffolds),
                "candidate_pool": len(indices),
            })
        all_hit_indices = np.flatnonzero(labels == 1)
        all_hit_scaffolds = {candidates[index][:3] for index in all_hit_indices}
        rows.append({
            "record_type": "offset_linkage_summary",
            "c_offset_V": offset,
            "linkage": "ALL",
            "candidate_hits": len(all_hit_indices),
            "hit_scaffolds": len(all_hit_scaffolds),
            "candidate_pool": len(candidates),
        })
        survivors = []
        for recurrent in recurrent_rows:
            scaffold = tuple(recurrent["scaffold"].split("/"))
            indices = candidate_indices_by_scaffold[scaffold]  # type: ignore[index]
            survives = any(labels[index] == 1 for index in indices)
            representative = (*scaffold, recurrent["representative_r_group"])
            representative_result = result_by_candidate[representative]  # type: ignore[index]
            representative_hit = recalculated_verdict(representative_result, offset)["is_hit"]
            if survives:
                survivors.append(recurrent["scaffold"])
            rows.append({
                "record_type": "recurrent_scaffold",
                "c_offset_V": offset,
                "recurrent_scaffold": recurrent["scaffold"],
                "linkage": recurrent["linkage"],
                "survives_any_candidate": survives,
                "representative_r_group": recurrent["representative_r_group"],
                "representative_survives": representative_hit,
            })
        summaries[offset] = {
            "candidate_hits": int(labels.sum()),
            "hit_scaffolds": len(all_hit_scaffolds),
            "linkages": linkage_summary,
            "recurrent_survivors": survivors,
        }
        successful = np.asarray(
            [str(result.get("status", "unknown")) == "success" for result in results],
            dtype=bool,
        )
        successful_count = int(successful.sum())
        gate_failures: dict[str, int] = {}
        for gate in ("gap", "cbm", "vbm"):
            failure_count = sum(
                1
                for result in results
                if str(result.get("status", "unknown")) == "success"
                and not bool(recalculated_verdict(result, offset)[f"{gate}_pass"])
            )
            gate_failures[gate] = failure_count
            rows.append({
                "record_type": "gate_failure_share",
                "c_offset_V": offset,
                "gate": gate,
                "failure_count": failure_count,
                "successful_candidate_denominator": successful_count,
                "failure_fraction": failure_count / successful_count,
                "failure_percent": 100.0 * failure_count / successful_count,
                "note": "Gate failures overlap; 12 non-success records are excluded.",
            })
        summaries[offset]["gate_failures"] = gate_failures
        summaries[offset]["successful_candidate_denominator"] = successful_count

    rbf_cache = SimilarityColumns(features, "rbf")
    for offset in BASELINE_OFFSETS:
        labels = labels_by_offset[offset]
        print(f"offset {offset:.2f}: random and MaxMin baselines", flush=True)
        methods = {
            "candidate_random": candidate_random_search(
                candidates, labels, statuses, random_runs, budget
            ),
            "scaffold_first_random": scaffold_first_random_search(
                candidates, labels, statuses, random_runs, budget
            ),
            "scaffold_maxmin": cached_scaffold_maxmin_search(
                candidates, labels, statuses, features, diversity_runs, budget
            ),
        }
        for scaffold_aware, name in (
            (False, "candidate_morgan_gp_ucb"),
            (True, "scaffold_aware_morgan_gp_ucb"),
        ):
            print(f"offset {offset:.2f}: {name}", flush=True)
            methods[name] = cached_gp_ucb_search(
                candidates,
                labels,
                statuses,
                rbf_cache,
                bo_runs,
                budget,
                initial,
                kappa,
                scaffold_aware=scaffold_aware,
                progress=lambda completed, prefix=name: print(
                    f"  {prefix}: {completed}/{bo_runs} seeds", flush=True
                ),
            )
        for name, outcomes in methods.items():
            rows.append(_summary_row(name, outcomes, budget, c_offset_V=offset))
            rows.extend(_seed_rows(name, outcomes, budget, c_offset_V=offset))
    return rows, summaries


def item3_new_baselines(
    candidates: list[Candidate],
    labels: np.ndarray,
    statuses: list[str],
    features: np.ndarray,
    runs: int,
    budget: int,
    initial: int,
    kappa: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for miss_limit in (3, 5, 8):
        print(f"linkage-elimination MaxMin k={miss_limit}", flush=True)
        outcomes = linkage_elimination_maxmin(
            candidates, labels, statuses, features, runs, budget, miss_limit
        )
        method = "scaffold_maxmin_linkage_elimination"
        summary = _summary_row(method, outcomes, budget, miss_limit=miss_limit)
        summary_rows.append(summary)
        rows.append(summary)
        rows.extend(_seed_rows(
            method,
            outcomes,
            budget,
            include_selected_candidates=True,
            miss_limit=miss_limit,
        ))

    print("scaffold-aware fitted-Tanimoto GP-UCB", flush=True)
    cache = SimilarityColumns(features, "tanimoto")
    tanimoto_outcomes = cached_gp_ucb_search(
        candidates,
        labels,
        statuses,
        cache,
        runs,
        budget,
        initial,
        kappa,
        scaffold_aware=True,
        fitted_tanimoto=True,
        progress=lambda completed: print(
            f"  fitted-Tanimoto GP-UCB: {completed}/{runs} seeds", flush=True
        ),
    )
    method = "scaffold_aware_tanimoto_gp_ucb_mle"
    summary = _summary_row(
        method,
        tanimoto_outcomes,
        budget,
        mean_final_signal_variance=float(np.mean([
            row["fitted_signal_variance"] for row in tanimoto_outcomes
        ])),
        mean_final_noise_variance=float(np.mean([
            row["fitted_noise_variance"] for row in tanimoto_outcomes
        ])),
    )
    summary_rows.append(summary)
    rows.append(summary)
    rows.extend(_seed_rows(
        method,
        tanimoto_outcomes,
        budget,
        include_selected_candidates=True,
    ))
    return rows, summary_rows


def item4_candidates(
    candidates: list[Candidate], results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_candidate = {candidate: result for candidate, result in zip(candidates, results)}
    requested: list[Candidate] = [
        ("N020", "L018", "vinylene", "R001"),
        ("N015", "L018", "vinylene", "R001"),
    ]
    rows = []
    for candidate in requested:
        result = by_candidate[candidate]
        reference = recalculated_verdict(result, 2.62)
        criterion = result.get("criterion") or {}
        row: dict[str, Any] = {
            "candidate": _candidate_key(candidate),
            "node_id": candidate[0],
            "linker_id": candidate[1],
            "linkage": candidate[2],
            "r_group_id": candidate[3],
            "status": result.get("status"),
            "fragment_smiles": result.get("fragment_smiles"),
            "ip_eV": result.get("ip_eV"),
            "ea_eV": result.get("ea_eV"),
            "gap_xtb_eV": result.get("gap_xtb_eV"),
            "gap_corrected_eV": reference["gap_corrected_eV"],
            "stability_score": reference["stability_score"],
            "gap_pass": reference["gap_pass"],
            "stability_pass": reference["stability_pass"],
            "gap_margin_floor_eV": criterion.get("gap_margin_floor"),
            "gap_margin_ceiling_eV": criterion.get("gap_margin_ceiling"),
            "stability_margin": criterion.get("stability_margin"),
        }
        for offset in OFFSETS:
            verdict = recalculated_verdict(result, offset)
            suffix = str(offset).replace(".", "_")
            row.update({
                f"cbm_{suffix}_V": verdict["cbm_vs_nhe"],
                f"vbm_{suffix}_V": verdict["vbm_vs_nhe"],
                f"cbm_pass_{suffix}": verdict["cbm_pass"],
                f"vbm_pass_{suffix}": verdict["vbm_pass"],
                f"four_gate_hit_{suffix}": verdict["is_hit"],
            })
        rows.append(row)
    return rows


def _format_baseline(summary: dict[str, Any]) -> str:
    return (
        f"{summary['mean_hit_scaffolds']:.3f} +/- "
        f"{summary['std_hit_scaffolds']:.3f} hit scaffolds"
    )


def write_summary(
    path: Path,
    reference_path: Path,
    item1_counts: dict[str, int],
    offset_summaries: dict[float, dict[str, Any]],
    item2_rows: list[dict[str, Any]],
    item3_summaries: list[dict[str, Any]],
    item4_rows: list[dict[str, Any]],
) -> None:
    baseline_summaries = [
        row for row in item2_rows if row.get("record_type") == "summary"
    ]
    lines = [
        "# Frozen-reference sensitivity summary",
        "",
        f"Reference: `{reference_path}`",
        f"SHA-256: `{_sha256(reference_path)}`",
        "",
        "## 1. Imine electronic gates",
        "",
        (
            f"Of {item1_counts['evaluated_imines']:,} successfully evaluated imines, "
            f"{item1_counts.get('all_three_pass', 0):,} pass gap, CBM, and VBM when "
            "persistence is ignored."
        ),
        "",
        (
            f"Failures: gap only {item1_counts.get('gap_only_failure', 0):,}; "
            f"VBM only {item1_counts.get('vbm_only_failure', 0):,}; gap and VBM "
            f"{item1_counts.get('gap_and_vbm_failure', 0):,}; cases including a CBM "
            f"failure {item1_counts.get('includes_cbm_failure', 0):,}."
        ),
        "",
        (
            f"The feasible inventory contains {item1_counts['feasible_imines']:,} imine rows; "
            f"{item1_counts.get('oracle_failure', 0):,} frozen oracle failures are excluded "
            "from the requested 5,658-candidate electronic breakdown."
        ),
        "",
        (
            f"The 693 electronic-pass candidates cover "
            f"{item1_counts['electronic_pass_scaffolds']:,} distinct imine scaffolds. "
            f"Dropping persistence only for those imines increases the landscape from "
            f"{item1_counts['frozen_four_gate_hit_scaffolds']:,} to "
            f"{item1_counts['hit_scaffolds_if_imine_persistence_dropped']:,} hit scaffolds."
        ),
        "",
        "## 2. Offset sensitivity",
        "",
        "| C_offset (V) | Candidate hits | Hit scaffolds | Recurrent survivors |",
        "|---:|---:|---:|---:|",
    ]
    for offset in OFFSETS:
        summary = offset_summaries[offset]
        lines.append(
            f"| {offset:.2f} | {summary['candidate_hits']} | {summary['hit_scaffolds']} | "
            f"{len(summary['recurrent_survivors'])}/12 |"
        )
    lines.extend([
        "",
        "Overlapping electronic-gate failure shares among 7,197 successful records:",
        "",
        "| C_offset (V) | Gap fail | CBM fail | VBM fail |",
        "|---:|---:|---:|---:|",
    ])
    for offset in OFFSETS:
        summary = offset_summaries[offset]
        denominator = summary["successful_candidate_denominator"]
        failures = summary["gate_failures"]
        lines.append(
            f"| {offset:.2f} | {failures['gap'] / denominator:.1%} | "
            f"{failures['cbm'] / denominator:.1%} | "
            f"{failures['vbm'] / denominator:.1%} |"
        )
    lines.extend(["", "Recurrent survivors by offset:", ""])
    for offset in OFFSETS:
        survivors = offset_summaries[offset]["recurrent_survivors"]
        lines.append(f"- {offset:.2f} V: " + (", ".join(survivors) if survivors else "none"))
    lines.extend(["", "Extreme-offset baseline reruns:", ""])
    for offset in BASELINE_OFFSETS:
        lines.append(f"- C_offset={offset:.2f} V")
        for summary in baseline_summaries:
            if float(summary["c_offset_V"]) == offset:
                lines.append(f"  - {summary['method']}: {_format_baseline(summary)}")
    lines.extend(["", "## 3. Added baselines", ""])
    for summary in item3_summaries:
        suffix = (
            f" (k={summary['miss_limit']})" if summary.get("miss_limit") else ""
        )
        lines.append(
            f"- {summary['method']}{suffix}: {_format_baseline(summary)}; "
            f"mean evaluations {summary['mean_evaluations']:.2f}; "
            f"early-stop rate {summary['early_stop_rate']:.3f}."
        )
    lines.extend(["", "## 4. Requested candidates", ""])
    for row in item4_rows:
        lines.append(
            f"- {row['candidate']}: gap={float(row['gap_corrected_eV']):.6f} eV, "
            f"CBM={float(row['cbm_2_62_V']):+.6f} V, "
            f"VBM={float(row['vbm_2_62_V']):+.6f} V, "
            f"stability={float(row['stability_score']):.6f}, "
            f"four-gate hit={row['four_gate_hit_2_62']}."
        )
    lines.extend([
        "",
        "All baseline endpoints use a 50-evaluation budget. The linkage-elimination "
        "variant protects every class after its first hit, never eliminates the last "
        "active class, and falls back to all remaining scaffolds if necessary. Random "
        "and MaxMin controls use the "
        "same seed numbering and selection rules as the frozen baseline code. The "
        "fixed-RBF GP implementation is an algebraically equivalent cached-kernel "
        "version of the existing scikit-learn GP, used to make the offset reruns tractable.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--feasible-csv", type=Path, default=ROOT / "data/search_space_v2_feasible.csv"
    )
    parser.add_argument("--recurrent-csv", type=Path, default=DEFAULT_RECURRENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--random-runs", type=int, default=1000)
    parser.add_argument("--diversity-runs", type=int, default=1000)
    parser.add_argument("--bo-runs", type=int, default=200)
    parser.add_argument("--new-baseline-runs", type=int, default=200)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--kappa", type=float, default=1.0)
    args = parser.parse_args()

    expected = feasible_keys(args.feasible_csv)
    candidates, _, smiles, statuses = load_landscape(args.ground_truth, expected)
    records = read_ground_truth(args.ground_truth)
    by_key = {key(record): record for record in records}
    results = [_result_for(by_key[candidate]) for candidate in candidates]
    features = morgan_features(smiles)
    recurrent = _load_recurrent(args.recurrent_csv)

    item1_rows, item1_counts = item1_imines(candidates, results)
    item1_path = args.out_dir / "item1_imine_electronic_gates.csv"
    _write_csv(item1_path, item1_rows)
    print(f"Wrote {item1_path}", flush=True)

    item2_rows, offset_summaries = item2_offsets_and_baselines(
        candidates,
        results,
        statuses,
        features,
        recurrent,
        args.budget,
        args.random_runs,
        args.diversity_runs,
        args.bo_runs,
        args.initial,
        args.kappa,
    )
    item2_path = args.out_dir / "item2_offset_sensitivity_and_baselines.csv"
    _write_csv(item2_path, item2_rows)
    print(f"Wrote {item2_path}", flush=True)

    reference_labels = labels_for_offset(results, 2.62)
    item3_rows, item3_summaries = item3_new_baselines(
        candidates,
        reference_labels,
        statuses,
        features,
        args.new_baseline_runs,
        args.budget,
        args.initial,
        args.kappa,
    )
    item3_path = args.out_dir / "item3_added_baselines.csv"
    _write_csv(item3_path, item3_rows)
    print(f"Wrote {item3_path}", flush=True)

    item4_rows = item4_candidates(candidates, results)
    item4_path = args.out_dir / "item4_requested_candidates.csv"
    _write_csv(item4_path, item4_rows)
    print(f"Wrote {item4_path}", flush=True)

    summary_path = args.out_dir / "summary.md"
    write_summary(
        summary_path,
        args.ground_truth,
        item1_counts,
        offset_summaries,
        item2_rows,
        item3_summaries,
        item4_rows,
    )
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
