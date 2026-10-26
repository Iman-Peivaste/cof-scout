from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MM = 1 / 25.4
COLORS = {
    "adaptive_ara5": "#4477AA",
    "one_shot_llm_portfolio": "#66CCEE",
    "candidate_random": "#999999",
    "scaffold_first_random": "#BBBBBB",
    "candidate_morgan_gp_ucb": "#CC6677",
    "scaffold_maxmin": "#228833",
    "scaffold_aware_morgan_gp_ucb": "#AA3377",
    "beta_ketoenamine": "#4477AA",
    "hydrazone": "#EE7733",
    "imine": "#CC3311",
    "vinylene": "#228833",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def endpoint_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["method", "method_label"], sort=False)["hit_scaffolds_at_50"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )


def figure_1(_: Path, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(180 * MM, 68 * MM))
    axis.axis("off")
    boxes = [
        (0.03, "Frozen\ninventory", "#DDEBF7"),
        (0.23, "Scientist\naction + prediction", "#E2F0D9"),
        (0.47, "Guards and\nreference oracle", "#FFF2CC"),
        (0.71, "Outcome, score\nand reflection", "#FCE4D6"),
    ]
    for x, label, color in boxes:
        axis.add_patch(plt.Rectangle((x, 0.35), 0.17, 0.3, facecolor=color, edgecolor="#333333", lw=0.8))
        axis.text(x + 0.085, 0.5, label, ha="center", va="center", fontweight="bold")
    for x in (0.20, 0.44, 0.68):
        axis.annotate("", xy=(x + 0.025, 0.5), xytext=(x - 0.015, 0.5), arrowprops={"arrowstyle": "->", "lw": 1})
    axis.annotate("50-evaluation adaptive loop", xy=(0.31, 0.28), xytext=(0.79, 0.28), ha="center", arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=.35"})
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    _save(figure, output, "figure_1_framework_architecture")


def figure_2(source: Path, output: Path) -> None:
    frame = pd.read_csv(source / "endpoint_replicates_and_seeds.csv")
    methods = list(dict.fromkeys(frame["method"]))
    labels = [frame.loc[frame["method"] == method, "method_label"].iloc[0] for method in methods]
    figure, axis = plt.subplots(figsize=(180 * MM, 90 * MM))
    rng = np.random.default_rng(12)
    for index, method in enumerate(methods):
        values = frame.loc[frame["method"] == method, "hit_scaffolds_at_50"].to_numpy(float)
        x = index + rng.uniform(-0.16, 0.16, len(values))
        axis.scatter(x, values, s=10, alpha=0.45, color=COLORS.get(method, "#777777"), edgecolors="none")
        axis.plot([index - 0.22, index + 0.22], [np.mean(values)] * 2, color="#111111", lw=1.5)
    axis.set_xticks(range(len(labels)), [label.replace(" ", "\n") for label in labels], rotation=0)
    axis.set_ylabel("Distinct hit scaffolds after 50 evaluations")
    axis.grid(axis="y", color="#DDDDDD", lw=0.5)
    _save(figure, output, "figure_2_primary_endpoint")


def figure_3(source: Path, output: Path) -> None:
    frame = pd.read_csv(source / "discovery_curves_all_runs.csv")
    figure, axis = plt.subplots(figsize=(180 * MM, 82 * MM))
    for method, group in frame.groupby("method", sort=False):
        pivot = group.pivot(index="iteration", columns="run", values="cumulative_hit_scaffolds")
        x = pivot.index.to_numpy()
        mean = pivot.mean(axis=1).to_numpy()
        low = pivot.quantile(0.25, axis=1).to_numpy()
        high = pivot.quantile(0.75, axis=1).to_numpy()
        label = method.replace("_", " ").title().replace("Ara5", "COF-SCOUT")
        color = COLORS.get(method, "#777777")
        axis.plot(x, mean, label=label, color=color, lw=1.5)
        axis.fill_between(x, low, high, color=color, alpha=0.16, lw=0)
    axis.set(xlabel="Evaluations", ylabel="Cumulative distinct hit scaffolds", xlim=(1, 50))
    axis.grid(color="#E4E4E4", lw=0.5)
    axis.legend(frameon=False, ncol=2)
    _save(figure, output, "figure_3_discovery_curves")


def figure_4(source: Path, output: Path) -> None:
    frame = pd.read_csv(source / "recurrent_scaffold_matrix.csv")
    matrix = frame.pivot(index="scaffold", columns="run_id", values="hit_scaffold").fillna(0)
    figure, axis = plt.subplots(figsize=(180 * MM, max(65, 4 * len(matrix)) * MM))
    image = axis.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), [item.replace("ara5_mcap_full_", "") for item in matrix.columns], rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xlabel("COF-SCOUT campaign")
    axis.set_ylabel("Recurrent scaffold")
    figure.colorbar(image, ax=axis, label="Discovered")
    _save(figure, output, "figure_4_recurrent_scaffolds")


def figure_5(source: Path, output: Path) -> None:
    frame = pd.read_csv(source / "campaign_experiments.csv")
    frame = frame[frame["status"] == "success"]
    figure, axis = plt.subplots(figsize=(180 * MM, 82 * MM))
    for linkage, group in frame.groupby("linkage", sort=False):
        axis.scatter(group["gap_corrected_eV"], group["vbm_vs_nhe"], s=9, alpha=0.5, color=COLORS.get(linkage, "#777777"), label=linkage.replace("_", " "))
    axis.axvspan(1.8, 2.2, color="#BBBBBB", alpha=0.12)
    axis.axhline(1.23, color="#444444", ls="--", lw=0.8)
    axis.set(xlabel="Corrected gap (eV)", ylabel="VBM vs NHE (V)")
    axis.legend(frameon=False, ncol=2)
    axis.grid(color="#E4E4E4", lw=0.5)
    _save(figure, output, "figure_5_electronic_feasibility")


def figure_6(source: Path, output: Path) -> None:
    frame = pd.read_csv(source / "falsification_episode_rep04.csv")
    figure, axis = plt.subplots(figsize=(180 * MM, 76 * MM))
    colors = ["#228833" if value else "#CC3311" for value in frame["is_hit"]]
    axis.plot(frame["iteration"], frame["vbm_vs_nhe"], color="#777777", lw=0.8, zorder=1)
    axis.scatter(frame["iteration"], frame["vbm_vs_nhe"], c=colors, s=30, zorder=2)
    for row in frame.itertuples():
        axis.annotate(row.candidate, (row.iteration, row.vbm_vs_nhe), xytext=(3, 4), textcoords="offset points", fontsize=6)
    axis.axhline(1.23, color="#333333", ls="--", lw=0.8, label="VBM gate")
    axis.set(xlabel="Campaign iteration", ylabel="VBM vs NHE (V)")
    axis.legend(frameon=False)
    axis.grid(color="#E4E4E4", lw=0.5)
    _save(figure, output, "figure_6_falsification_episode")


RENDERERS = {1: figure_1, 2: figure_2, 3: figure_3, 4: figure_4, 5: figure_5, 6: figure_6}


def render_figure(number: int, source: Path, output: Path) -> None:
    set_style()
    RENDERERS[number](source, output)
