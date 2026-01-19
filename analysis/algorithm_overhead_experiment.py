import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


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


def _extract_runtime_series(results: dict, algo_key: str, metric: str) -> dict:
    series = {}
    entries = results.get("comparison_index_details", {}).get("validation", {}).get(algo_key, [])
    for entry in entries:
        budget = entry.get("budget")
        value = entry.get(metric)
        if budget is None:
            continue
        series.setdefault(budget, []).append(value)
    averaged = {}
    for budget, values in series.items():
        clean = [v for v in values if v is not None]
        averaged[budget] = sum(clean) / len(clean) if clean else None
    return averaged


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _plot_line(budgets, data, output_path: Path, ylabel: str, title: str) -> None:
    plt.figure(figsize=(9, 5))
    for label, values in data.items():
        plt.plot(budgets, values, marker="o", label=label)
    plt.xlabel("Budget Constraint (MB)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
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
                row.append("" if value is None else f"{value:.4f}")
            f.write(",".join(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot algorithm overhead metrics")
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
        "Extend_global": "Extend_global",
        "Extend_partition_sa": "Extend_partition_sa",
    }

    metrics = {
        "calculation_time": ("Selection Time (s)", "Algorithm Selection Time vs Budget"),
        "cost_evaluation_time": ("Cost Evaluation Time (s)", "Cost Evaluation Time vs Budget"),
    }

    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    for metric, (ylabel, title) in metrics.items():
        data = {}
        for label, key in algo_map.items():
            series = _extract_runtime_series(results, key, metric)
            data[label] = [series.get(budget) for budget in budgets]

        _write_csv(budgets, data, output_dir / f"overhead_{metric}.csv")
        _plot_line(
            budgets,
            data,
            output_dir / f"overhead_{metric}_line.png",
            ylabel,
            title,
        )

    print(f"Results loaded from: {results_path}")
    print(f"Saved overhead plots to: {output_dir}")


if __name__ == "__main__":
    main()
