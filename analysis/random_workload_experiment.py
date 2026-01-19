import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _run_export(config_path: Path, root: Path) -> None:
    export_script = root / "export_results.py"
    subprocess.run(
        ["python3", str(export_script), str(config_path)],
        check=True,
        cwd=str(root),
    )


def _find_latest_result(results_dir: Path, experiment_id: str) -> Path:
    candidates = sorted(results_dir.glob(f"ID_{experiment_id}_timetamps_*/comparison_results.json"))
    if not candidates:
        raise FileNotFoundError(f"No results found for {experiment_id} under {results_dir}")
    return candidates[-1]


def _load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_budget_value(results: dict, algo_key: str, budget: int):
    details = results.get("comparison_index_details", {}).get("validation", {}).get(algo_key, [])
    for entry in details:
        if entry.get("budget") == budget:
            return entry.get("final_cost_proportion")
    return None


def _extract_budget_time(results: dict, algo_key: str, budget: int):
    details = results.get("comparison_index_details", {}).get("validation", {}).get(algo_key, [])
    for entry in details:
        if entry.get("budget") == budget:
            return entry.get("calculation_time")
    return None


def _plot_box(data, output_path: Path, title: str, ylabel: str) -> None:
    labels = list(data.keys())
    values = [data[label] for label in labels]
    plt.figure(figsize=(8, 4.5))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Random workload robustness experiment")
    parser.add_argument(
        "--base-config",
        type=str,
        default="free-origin/index/rl_index_selection/experiments/tpchskew.json",
        help="Base experiment config path",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Random seeds to sweep",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2000,
        help="Budget to compare across seeds",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run experiments (otherwise only generate configs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/plots",
        help="Output directory for plots and CSV",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_config_path = root / args.base_config
    base_config = _load_config(base_config_path)

    results_dir = root / "free-origin" / "index" / "rl_index_selection" / "experiment_results"
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_map = {
        "Extend": "Extend",
        "Extend_global": "Extend_global",
        "Extend_partition_sa": "Extend_partition_sa",
    }
    series = {label: [] for label in algo_map}
    time_series = {label: [] for label in algo_map}

    for seed in args.seeds:
        exp_id = f"{base_config['id']}_seed_{seed}"
        config = dict(base_config)
        config["id"] = exp_id
        config["random_seed"] = seed
        config["comparison_algorithms"] = ["extend", "extend_global", "extend_partition_sa"]
        config.setdefault("sa_allocation", {})
        config["sa_allocation"] = dict(config["sa_allocation"])
        config["sa_allocation"]["log_top_n"] = 0

        cfg_path = root / "analysis" / "tmp_configs" / f"tpchskew_seed_{seed}.json"
        _write_config(config, cfg_path)

        if args.run:
            _run_export(cfg_path, root)
            result_path = _find_latest_result(results_dir, exp_id)
            results = _load_results(result_path)
            for label, key in algo_map.items():
                value = _extract_budget_value(results, key, args.budget)
                time_value = _extract_budget_time(results, key, args.budget)
                series[label].append(value)
                time_series[label].append(time_value)

    if args.run:
        csv_path = output_dir / "random_workload_seed_sweep.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("seed," + ",".join(series.keys()) + "," + ",".join(f"{k}_time" for k in series.keys()) + "\n")
            for idx, seed in enumerate(args.seeds):
                row = [str(seed)]
                for label in series.keys():
                    value = series[label][idx]
                    row.append("" if value is None else f"{value:.2f}")
                for label in time_series.keys():
                    value = time_series[label][idx]
                    row.append("" if value is None else f"{value:.4f}")
                f.write(",".join(row) + "\n")

        _plot_box(
            series,
            output_dir / "random_workload_box.png",
            f"Random Seed Robustness (Budget {args.budget} MB)",
            "Final Cost Proportion (%)",
        )
        _plot_box(
            time_series,
            output_dir / "random_workload_time_box.png",
            f"Random Seed Runtime (Budget {args.budget} MB)",
            "Selection Time (s)",
        )
        print(f"Saved robustness results to: {csv_path}")


if __name__ == "__main__":
    main()
