import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt


PARTITION_REGEX = re.compile(r"prt_p(\d+)")


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


def _extract_table_name(column_repr: str) -> str:
    # Format: "C table.column"
    parts = column_repr.split()
    if len(parts) < 2:
        return ""
    table_and_col = parts[1]
    if "." not in table_and_col:
        return table_and_col
    return table_and_col.split(".")[0]


def _partition_key_from_index(index_entry: dict) -> str:
    columns = index_entry.get("columns") or []
    if not columns:
        return "unknown"
    table = _extract_table_name(columns[0])
    match = PARTITION_REGEX.search(table)
    if match:
        return f"p{match.group(1)}"
    return "unknown"


def _size_mb(index_entry: dict) -> float:
    size_mb = index_entry.get("size_mb")
    if size_mb is not None:
        return float(size_mb)
    size_bytes = index_entry.get("size_bytes")
    if size_bytes is None:
        return 0.0
    return float(size_bytes) / (1024 * 1024)


def _aggregate_partition_sizes(entries, budgets) -> Tuple[List[str], Dict[int, Dict[str, float]]]:
    budget_to_partition_sizes = {budget: defaultdict(float) for budget in budgets}
    budget_counts = defaultdict(int)

    for entry in entries:
        budget = entry.get("budget")
        if budget not in budget_to_partition_sizes:
            continue
        budget_counts[budget] += 1
        for index_entry in entry.get("indexes", []):
            partition_key = _partition_key_from_index(index_entry)
            budget_to_partition_sizes[budget][partition_key] += _size_mb(index_entry)

    partition_keys = set()
    for budget, sizes in budget_to_partition_sizes.items():
        count = budget_counts.get(budget, 0)
        if count > 1:
            for key in list(sizes.keys()):
                sizes[key] = sizes[key] / count
        partition_keys.update(sizes.keys())

    def _partition_sort_key(label: str) -> int:
        match = re.match(r"p(\d+)", label)
        return int(match.group(1)) if match else 999

    ordered_partitions = sorted(partition_keys, key=_partition_sort_key)
    return ordered_partitions, budget_to_partition_sizes


def _write_csv(budgets, partitions, sizes_by_budget, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        header = ["budget"] + partitions
        f.write(",".join(header) + "\n")
        for budget in budgets:
            row = [str(budget)]
            size_map = sizes_by_budget.get(budget, {})
            for partition in partitions:
                value = size_map.get(partition)
                row.append("" if value is None else f"{value:.2f}")
            f.write(",".join(row) + "\n")


def _plot_stacked_bar(budgets, partitions, sizes_by_budget, output_path: Path, ylabel: str) -> None:
    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    x_positions = list(range(len(budgets)))
    bottom = [0.0 for _ in budgets]
    palette = plt.get_cmap("tab20").colors

    for idx, partition in enumerate(partitions):
        values = [sizes_by_budget.get(budget, {}).get(partition, 0.0) for budget in budgets]
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=partition,
            color=palette[idx % len(palette)],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Budget (MB)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Partition", ncol=3, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot partition allocation by budget")
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
    parser.add_argument(
        "--algo",
        type=str,
        default="Extend_partition_sa",
        help="Algorithm key in comparison_results.json",
    )
    parser.add_argument(
        "--run-type",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Run type to plot",
    )
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else _find_latest_results(
        Path(args.results_dir), args.experiment_id
    )
    results = _load_results(results_path)

    budgets = results.get("validation_budgets") or []
    if args.run_type == "test":
        budgets = results.get("test_budgets") or budgets
    if not budgets:
        raise ValueError("No budgets found in results.")

    entries = (
        results.get("comparison_index_details", {})
        .get(args.run_type, {})
        .get(args.algo, [])
    )
    if not entries:
        raise ValueError(f"No entries found for {args.algo} in {args.run_type}.")

    partitions, sizes_by_budget = _aggregate_partition_sizes(entries, budgets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_algo = args.algo.replace("/", "_")

    csv_path = output_dir / f"partition_budget_allocation_{safe_algo}.csv"
    plot_path = output_dir / f"partition_budget_allocation_{safe_algo}.png"

    _write_csv(budgets, partitions, sizes_by_budget, csv_path)
    _plot_stacked_bar(
        budgets,
        partitions,
        sizes_by_budget,
        plot_path,
        ylabel="Allocated index size (MB)",
    )

    print(f"Results loaded from: {results_path}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
