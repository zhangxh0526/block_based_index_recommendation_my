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


def _load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_result(results_dir: Path, experiment_id: str) -> Path:
    candidates = sorted(results_dir.glob(f"ID_{experiment_id}_timetamps_*/comparison_results.json"))
    if not candidates:
        raise FileNotFoundError(f"No results found for {experiment_id} under {results_dir}")
    return candidates[-1]


def _average_metric(results: dict, algo_key: str, budgets, metric: str) -> float:
    details = results.get("comparison_index_details", {}).get("validation", {}).get(algo_key, [])
    by_budget = {entry.get("budget"): entry.get(metric) for entry in details}
    values = [by_budget.get(b) for b in budgets if by_budget.get(b) is not None]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _plot(alphas, values, output_path: Path, ylabel: str, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(alphas, values, marker="o")
    plt.xlabel("Anneal Alpha")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_tradeoff(costs, times, alphas, output_path: Path) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.scatter(times, costs)
    for alpha, x, y in zip(alphas, times, costs):
        plt.annotate(str(alpha), (x, y), textcoords="offset points", xytext=(4, 4))
    plt.xlabel("Avg Selection Time (s)")
    plt.ylabel("Avg Final Cost Proportion (%)")
    plt.title("SA Alpha Trade-off (Quality vs Time)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep anneal_alpha for SA algorithm")
    parser.add_argument(
        "--base-config",
        type=str,
        default="free-origin/index/rl_index_selection/experiments/tpchskew.json",
        help="Base experiment config path",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 4.0, 5.0],
        help="Anneal alpha values to sweep",
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

    cost_values = []
    time_values = []
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for alpha in args.alphas:
        exp_id = f"{base_config['id']}_alpha_{alpha}".replace(".", "p")
        config = dict(base_config)
        config["id"] = exp_id
        config["comparison_algorithms"] = ["extend_partition_sa"]
        config.setdefault("sa_allocation", {})
        config["sa_allocation"] = dict(config["sa_allocation"])
        config["sa_allocation"]["anneal_alpha"] = float(alpha)
        config["sa_allocation"]["log_top_n"] = 0

        cfg_path = root / "analysis" / "tmp_configs" / f"tpchskew_alpha_{alpha}.json"
        _write_config(config, cfg_path)

        if args.run:
            _run_export(cfg_path, root)
            result_path = _find_latest_result(results_dir, exp_id)
            results = _load_results(result_path)
            budgets = results.get("validation_budgets", [])
            avg_cost = _average_metric(results, "Extend_partition_sa", budgets, "final_cost_proportion")
            avg_time = _average_metric(results, "Extend_partition_sa", budgets, "calculation_time")
            cost_values.append(avg_cost)
            time_values.append(avg_time)

    if args.run:
        csv_path = output_dir / "anneal_alpha_sweep.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("alpha,avg_final_cost,avg_selection_time\n")
            for alpha, cost, runtime in zip(args.alphas, cost_values, time_values):
                f.write(f"{alpha},{cost:.4f},{runtime:.4f}\n")

        _plot(
            args.alphas,
            cost_values,
            output_dir / "anneal_alpha_sweep_cost.png",
            "Average Final Cost Proportion (%)",
            "SA Anneal Alpha Sensitivity (Quality)",
        )
        _plot(
            args.alphas,
            time_values,
            output_dir / "anneal_alpha_sweep_time.png",
            "Average Selection Time (s)",
            "SA Anneal Alpha Sensitivity (Time)",
        )
        _plot_tradeoff(cost_values, time_values, args.alphas, output_dir / "anneal_alpha_tradeoff.png")
        print(f"Saved sweep results to: {csv_path}")


if __name__ == "__main__":
    main()
