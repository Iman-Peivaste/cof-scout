"""Safe, deterministic analysis tools available to the COF-SCOUT scientist.

The toolbox is intentionally constructed without a ground-truth path.  It can
inspect public inventory metadata and observations produced during the current
campaign, but it cannot query unevaluated outcomes.
"""

from __future__ import annotations

import itertools
import statistics
from collections import defaultdict
from typing import Any

from cof_scout.agent.memory import build_deterministic_facts


TOOL_CATALOG = {
    "inventory_summary": (
        "Count the exact active candidate inventory by node, linker, linkage, "
        "R-group, and scaffold. This contains no outcome information. Arguments: {}."
    ),
    "inspect_components": (
        "Inspect names, SMILES, roles, and metadata for specified component IDs, "
        "and optionally check exact candidate membership. Arguments: any of node_id, "
        "linker_id, linkage, and r_group_id."
    ),
    "campaign_summary": (
        "Summarize only observations already generated in this campaign: gates, "
        "scaffolds, hits, failures, and linkage coverage. Arguments: {}."
    ),
    "compare_observations": (
        "Compare observed iterations and calculate property differences relative "
        "to the first requested observation. Arguments: {iterations: [1, 2, ...]}."
    ),
    "matched_pair_analysis": (
        "Find observed candidate pairs differing in exactly one chosen factor and "
        "summarize the corresponding gap, CBM, VBM, or stability deltas. Arguments: "
        "{factor: node|linker|linkage|r_group, axis: gap|cbm|vbm|stability}."
    ),
    "scaffold_frontier": (
        "List deterministic, unevaluated scaffold representatives from the active "
        "inventory, optionally restricted to a linkage. No outcomes are exposed. "
        "Arguments: optional linkage, preferred_r_group, and limit."
    ),
    "candidate_neighborhood": (
        "List active candidates one component change away from a supplied candidate, "
        "marking which have already been evaluated. Arguments: {candidate: {node_id, "
        "linker_id, linkage, r_group_id}, limit?: 30}; flat candidate fields are also "
        "accepted."
    ),
    "resolve_candidate": (
        "Check an exact proposed candidate and, when it is unavailable, return the nearest "
        "feasible alternatives with changed factors and component linkage compatibility. "
        "Arguments: {candidate: {node_id, linker_id, linkage, r_group_id}, limit?: 8}."
    ),
}


def _entry_index(library: dict[str, Any], collection: str) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("id")): entry
        for entry in library.get(collection, [])
        if isinstance(entry, dict) and entry.get("id")
    }


def _experiment_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") == "experiment"]


def _event_candidate(event: dict[str, Any]) -> tuple[str, str, str, str]:
    experiment = event.get("action", {}).get("experiment", {})
    return tuple(str(experiment.get(name, "?")) for name in (
        "node_id", "linker_id", "linkage", "r_group_id"
    ))  # type: ignore[return-value]


def _value(event: dict[str, Any], axis: str) -> float | None:
    result = event.get("result", {})
    if not isinstance(result, dict) or result.get("status") != "success":
        return None
    fields = {
        "gap": "gap_corrected_eV",
        "cbm": "cbm_vs_nhe",
        "vbm": "vbm_vs_nhe",
        "stability": "s_linkage",
    }
    value = result.get(fields[axis])
    return float(value) if isinstance(value, (int, float)) else None


class ScientistToolbox:
    """Execute allowlisted analysis over public inventory and current observations."""

    def __init__(self, library: dict[str, Any],
                 feasible_set: frozenset[tuple[str, str, str, str]]):
        self.library = library
        self.feasible_set = feasible_set
        self.nodes = _entry_index(library, "nodes")
        self.linkers = _entry_index(library, "linkers")
        self.r_groups = _entry_index(library, "r_groups")

    @staticmethod
    def catalog() -> dict[str, str]:
        return dict(TOOL_CATALOG)

    def run(self, name: str, arguments: dict[str, Any],
            events: list[dict[str, Any]]) -> dict[str, Any]:
        """Run one allowlisted tool and return JSON-serializable evidence."""
        if name not in TOOL_CATALOG:
            raise ValueError(f"Unknown scientist tool: {name!r}")
        normalized_arguments = dict(arguments)
        nested_candidate = arguments.get("candidate")
        if name in {"candidate_neighborhood", "resolve_candidate"} and isinstance(
            nested_candidate, dict
        ):
            normalized_arguments = {
                **nested_candidate,
                **{key: value for key, value in arguments.items() if key != "candidate"},
            }
        handler = getattr(self, f"_{name}")
        result = handler(normalized_arguments, events)
        return {
            "tool": name,
            "normalized_arguments": normalized_arguments,
            "evidence_scope": "public_inventory_and_current_run",
            "result": result,
        }

    def _inventory_summary(self, _: dict[str, Any], __: list[dict[str, Any]]) -> dict[str, Any]:
        scaffolds = {candidate[:3] for candidate in self.feasible_set}
        by_linkage: dict[str, dict[str, int]] = {}
        for linkage in sorted({candidate[2] for candidate in self.feasible_set}):
            candidates = [candidate for candidate in self.feasible_set if candidate[2] == linkage]
            by_linkage[linkage] = {
                "candidates": len(candidates),
                "scaffolds": len({candidate[:3] for candidate in candidates}),
            }
        return {
            "candidates": len(self.feasible_set),
            "scaffolds": len(scaffolds),
            "nodes": len({candidate[0] for candidate in self.feasible_set}),
            "linkers": len({candidate[1] for candidate in self.feasible_set}),
            "linkages": len({candidate[2] for candidate in self.feasible_set}),
            "r_groups": len({candidate[3] for candidate in self.feasible_set}),
            "by_linkage": by_linkage,
        }

    def _inspect_components(self, arguments: dict[str, Any],
                            _: list[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for field, index, position in (
            ("node_id", self.nodes, 0), ("linker_id", self.linkers, 1),
            ("r_group_id", self.r_groups, None),
        ):
            identifier = arguments.get(field)
            if identifier is not None:
                normalized = str(identifier).upper()
                entry = index.get(normalized)
                if entry is None:
                    output[field] = None
                    continue
                public_fields = (
                    "id", "name", "smiles", "n_connections", "reactive_group",
                    "attachment", "tier", "source_dois", "notes",
                )
                public = {key: entry[key] for key in public_fields if key in entry}
                if position is not None:
                    public["active_inventory_linkages"] = sorted({
                        candidate[2] for candidate in self.feasible_set
                        if candidate[position] == normalized
                    })
                output[field] = public
        linkage = arguments.get("linkage")
        if linkage is not None:
            output["linkage"] = str(linkage).strip().lower().replace("-", "_")
        required = ("node_id", "linker_id", "linkage", "r_group_id")
        if all(arguments.get(field) is not None for field in required):
            key = (
                str(arguments["node_id"]).upper(), str(arguments["linker_id"]).upper(),
                str(arguments["linkage"]).lower().replace("-", "_").replace(" ", "_"),
                str(arguments["r_group_id"]).upper(),
            )
            output["candidate"] = list(key)
            output["in_active_inventory"] = key in self.feasible_set
        return output

    def _campaign_summary(self, _: dict[str, Any],
                          events: list[dict[str, Any]]) -> dict[str, Any]:
        return build_deterministic_facts(events)

    def _compare_observations(self, arguments: dict[str, Any],
                              events: list[dict[str, Any]]) -> dict[str, Any]:
        iterations = arguments.get("iterations")
        if not isinstance(iterations, list) or not iterations:
            raise ValueError("compare_observations requires a nonempty iterations list")
        if len(iterations) > 10:
            raise ValueError("compare_observations accepts at most 10 iterations")
        by_iteration = {
            int(event.get("iteration")): event
            for event in _experiment_events(events)
            if isinstance(event.get("iteration"), int)
        }
        rows = []
        for value in iterations:
            if not isinstance(value, int) or value not in by_iteration:
                raise ValueError(f"Unknown observed iteration: {value!r}")
            event = by_iteration[value]
            result = event.get("result", {})
            criterion = result.get("criterion", {}) if isinstance(result, dict) else {}
            rows.append({
                "iteration": value,
                "candidate": list(_event_candidate(event)),
                "status": result.get("status"),
                "gap_corrected_eV": result.get("gap_corrected_eV"),
                "cbm_vs_nhe": result.get("cbm_vs_nhe"),
                "vbm_vs_nhe": result.get("vbm_vs_nhe"),
                "s_linkage": result.get("s_linkage"),
                "is_hit": criterion.get("is_hit") if isinstance(criterion, dict) else None,
            })
        baseline = rows[0]
        for row in rows:
            row["delta_from_first"] = {
                field: (
                    round(float(row[field]) - float(baseline[field]), 6)
                    if isinstance(row.get(field), (int, float))
                    and isinstance(baseline.get(field), (int, float)) else None
                )
                for field in ("gap_corrected_eV", "cbm_vs_nhe", "vbm_vs_nhe", "s_linkage")
            }
        return {"observations": rows}

    def _matched_pair_analysis(self, arguments: dict[str, Any],
                               events: list[dict[str, Any]]) -> dict[str, Any]:
        factor = str(arguments.get("factor", "")).strip().lower()
        axis = str(arguments.get("axis", "")).strip().lower()
        factors = {"node": 0, "linker": 1, "linkage": 2, "r_group": 3}
        if factor not in factors:
            raise ValueError("factor must be node, linker, linkage, or r_group")
        if axis not in {"gap", "cbm", "vbm", "stability"}:
            raise ValueError("axis must be gap, cbm, vbm, or stability")
        index = factors[factor]
        observations = []
        for event in _experiment_events(events):
            value = _value(event, axis)
            if value is not None:
                observations.append((_event_candidate(event), value, event.get("iteration")))

        transitions: dict[str, list[tuple[float, int | None, int | None]]] = defaultdict(list)
        for left, right in itertools.combinations(observations, 2):
            key_left, value_left, iteration_left = left
            key_right, value_right, iteration_right = right
            differing = [position for position in range(4) if key_left[position] != key_right[position]]
            if differing != [index]:
                continue
            if key_left[index] <= key_right[index]:
                label = f"{key_left[index]}->{key_right[index]}"
                delta = value_right - value_left
                example = (iteration_left, iteration_right)
            else:
                label = f"{key_right[index]}->{key_left[index]}"
                delta = value_left - value_right
                example = (iteration_right, iteration_left)
            transitions[label].append((delta, *example))

        summary = []
        for transition, values in sorted(transitions.items()):
            deltas = [item[0] for item in values]
            summary.append({
                "transition": transition,
                "n": len(deltas),
                "mean_delta": round(statistics.mean(deltas), 6),
                "stdev_delta": round(statistics.pstdev(deltas), 6) if len(deltas) > 1 else 0.0,
                "examples": [list(item[1:]) for item in values[:5]],
            })
        return {"factor": factor, "axis": axis, "matched_transitions": summary}

    def _scaffold_frontier(self, arguments: dict[str, Any],
                           events: list[dict[str, Any]]) -> dict[str, Any]:
        linkage = arguments.get("linkage")
        if linkage is not None:
            linkage = str(linkage).strip().lower().replace("-", "_").replace(" ", "_")
        preferred_r_group = str(arguments.get("preferred_r_group", "R001")).upper()
        limit = int(arguments.get("limit", 20))
        if limit < 1 or limit > 50:
            raise ValueError("scaffold_frontier limit must be between 1 and 50")
        tested_scaffolds = {_event_candidate(event)[:3] for event in _experiment_events(events)}
        candidates_by_scaffold: dict[tuple[str, str, str], list[tuple[str, str, str, str]]] = defaultdict(list)
        for candidate in sorted(self.feasible_set):
            if linkage is not None and candidate[2] != linkage:
                continue
            if candidate[:3] in tested_scaffolds:
                continue
            candidates_by_scaffold[candidate[:3]].append(candidate)
        rows = []
        for scaffold, candidates in sorted(candidates_by_scaffold.items())[:limit]:
            representative = next(
                (candidate for candidate in candidates if candidate[3] == preferred_r_group),
                candidates[0],
            )
            rows.append({
                "scaffold": list(scaffold),
                "representative_candidate": list(representative),
                "node_name": self.nodes.get(scaffold[0], {}).get("name"),
                "linker_name": self.linkers.get(scaffold[1], {}).get("name"),
                "available_r_groups": len(candidates),
            })
        return {
            "untested_scaffolds_matching_filter": len(candidates_by_scaffold),
            "returned": rows,
        }

    def _candidate_neighborhood(self, arguments: dict[str, Any],
                                events: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            query = (
                str(arguments["node_id"]).upper(), str(arguments["linker_id"]).upper(),
                str(arguments["linkage"]).lower().replace("-", "_").replace(" ", "_"),
                str(arguments["r_group_id"]).upper(),
            )
        except KeyError as error:
            raise ValueError("candidate_neighborhood requires all four candidate fields") from error
        limit = int(arguments.get("limit", 30))
        if limit < 1 or limit > 100:
            raise ValueError("candidate_neighborhood limit must be between 1 and 100")
        evaluated = {_event_candidate(event) for event in _experiment_events(events)}
        neighbors = []
        for candidate in sorted(self.feasible_set):
            differing = [index for index in range(4) if candidate[index] != query[index]]
            if len(differing) != 1:
                continue
            neighbors.append({
                "candidate": list(candidate),
                "changed_factor": ("node", "linker", "linkage", "r_group")[differing[0]],
                "already_evaluated": candidate in evaluated,
            })
            if len(neighbors) >= limit:
                break
        return {"query": list(query), "in_active_inventory": query in self.feasible_set,
                "neighbors": neighbors}

    def _resolve_candidate(self, arguments: dict[str, Any],
                           events: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            query = (
                str(arguments["node_id"]).upper(), str(arguments["linker_id"]).upper(),
                str(arguments["linkage"]).lower().replace("-", "_").replace(" ", "_"),
                str(arguments["r_group_id"]).upper(),
            )
        except KeyError as error:
            raise ValueError("resolve_candidate requires all four candidate fields") from error
        limit = int(arguments.get("limit", 8))
        if limit < 1 or limit > 20:
            raise ValueError("resolve_candidate limit must be between 1 and 20")
        evaluated = {_event_candidate(event) for event in _experiment_events(events)}
        factor_names = ("node", "linker", "linkage", "r_group")
        ranked = []
        for candidate in self.feasible_set:
            differing = [index for index in range(4) if candidate[index] != query[index]]
            ranked.append((
                len(differing),
                candidate[3] != query[3],
                candidate,
                differing,
            ))
        nearest = []
        for distance, _, candidate, differing in sorted(ranked)[:limit]:
            nearest.append({
                "candidate": list(candidate),
                "distance": distance,
                "changed_factors": [factor_names[index] for index in differing],
                "already_evaluated": candidate in evaluated,
                "node_name": self.nodes.get(candidate[0], {}).get("name"),
                "linker_name": self.linkers.get(candidate[1], {}).get("name"),
            })
        return {
            "query": list(query),
            "in_active_inventory": query in self.feasible_set,
            "component_compatibility": {
                "node_active_linkages": sorted({
                    candidate[2] for candidate in self.feasible_set
                    if candidate[0] == query[0]
                }),
                "linker_active_linkages": sorted({
                    candidate[2] for candidate in self.feasible_set
                    if candidate[1] == query[1]
                }),
                "pair_active_linkages": sorted({
                    candidate[2] for candidate in self.feasible_set
                    if candidate[0] == query[0] and candidate[1] == query[1]
                }),
            },
            "nearest_feasible": nearest,
        }
