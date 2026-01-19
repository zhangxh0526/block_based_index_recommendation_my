import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch


def _find_latest_results(base_dir: Path, experiment_id: str) -> Path:
    pattern = re.compile(r"ID_(?P<id>.+)_timetamps_(?P<ts>\d+)$")
    latest_ts = -1
    latest_path = None
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        if match.group("id") != experiment_id:
            continue
        ts = int(match.group("ts"))
        if ts > latest_ts:
            latest_ts = ts
            latest_path = child / "comparison_results.json"
    if latest_path is None:
        raise FileNotFoundError(f"No comparison_results.json found for {experiment_id} under {base_dir}")
    return latest_path


def _load_results(results_path: Path) -> dict:
    with results_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_budget_series(results: dict, algo_key: str) -> dict:
    series = {}
    entries = results.get("comparison_indexes_by_budget", {}).get("validation", {}).get(algo_key, [])
    for entry in entries:
        budget = entry.get("budget")
        value = entry.get("average_final_cost_proportion")
        if budget is None:
            continue
        series[budget] = value
    return series


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _plot_line(budgets, data, output_path: Path) -> None:
    plt.rcParams["hatch.linewidth"] = 0.9
    plt.figure(figsize=(9, 5))
    for label, values in data.items():
        plt.plot(budgets, values, marker="o", label=label)
    plt.xlabel("Budget (MB)")
    plt.ylabel("Relative workload cost (% of without index)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_bar(budgets, data, output_path: Path) -> None:
    plt.rcParams["hatch.linewidth"] = 0.9
    plt.figure(figsize=(11, 5))
    ax = plt.gca()
    labels = list(data.keys())
    width = 0.8 / max(len(labels), 1)
    x_positions = list(range(len(budgets)))

    palette = ["#7FB3D5", "#F5B041", "#E67E73"]
    bar_hatches = ["/", "-", None]
    legend_hatches = ["///", "||", None]

    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        values = data[label]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=label,
            color=palette[idx % len(palette)],
            edgecolor="#2C3E50",
            linewidth=0.9,
            hatch=bar_hatches[idx % len(bar_hatches)],
            alpha=0.9,
            zorder=3,
        )
        for rect in bars:
            rect.set_path_effects(
                [pe.SimplePatchShadow(offset=(1.5, -1.5), alpha=0.25), pe.Normal()]
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Budget (MB)")
    ax.set_ylabel("Relative workload cost (% of without index)")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        Patch(
            facecolor=palette[0],
            edgecolor="#2C3E50",
            linewidth=0.9,
            hatch=legend_hatches[0],
            label=labels[0],
        ),
        Patch(
            facecolor=palette[1],
            edgecolor="#2C3E50",
            linewidth=0.9,
            hatch=legend_hatches[1],
            label=labels[1],
        ),
        Patch(
            facecolor=palette[2],
            edgecolor="#333333",
            linewidth=0.7,
            hatch=legend_hatches[2] or "",
            label=labels[2],
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=True,
        handlelength=1.4,
        handleheight=0.9,
        borderaxespad=0.6,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _write_csv(budgets, data, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        header = ["budget"] + list(data.keys())
        f.write(",".join(header) + "\n")
        for idx, budget in enumerate(budgets):
            row = [str(budget)]
            for label in data.keys():
                value = data[label][idx]
                row.append("" if value is None else f"{value:.2f}")
            f.write(",".join(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot workload performance results")
    parser.add_argument("--results", type=str, default=None, help="Path to comparison_results.json")
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="TPCHskew_Test_Experiment",
        help="Experiment id used in results folder name",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="free-origin/index/rl_index_selection/experiment_results",
        help="Base directory of experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/plots",
        help="Directory to write plots and CSV",
    )
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else _find_latest_results(
        Path(args.results_dir), args.experiment_id
    )
    results = _load_results(results_path)

    budgets = results.get("validation_budgets") or []
    if not budgets:
        raise ValueError("No validation_budgets found in results.")

    algo_map = {
        "Extend": "Extend",
        "Extend global": "Extend_global",
        "Extend partition SA": "Extend_partition_sa",
    }

    data = {}
    for label, key in algo_map.items():
        series = _extract_budget_series(results, key)
        data[label] = [series.get(budget) for budget in budgets]

    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    _write_csv(budgets, data, output_dir / "workload_performance.csv")
    _plot_line(budgets, data, output_dir / "workload_performance_line.png")
    _plot_bar(budgets, data, output_dir / "workload_performance_bar.png")

    print(f"Results loaded from: {results_path}")
    print(f"Saved CSV and plots to: {output_dir}")


if __name__ == "__main__":
    main()
