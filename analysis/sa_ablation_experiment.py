import argparse
import json
import subprocess
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


VARIANTS = [
    ("Full", "full", "full"),
    ("w/o OCW", "w_o_ocw", "wo_ocw"),
    ("w/o DynamicK", "w_o_dynamick", "wo_dynamick"),
    ("w/o GapFilling", "w_o_gapfilling", "wo_gapfilling"),
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run_export(root: Path, config_path: Path) -> None:
    subprocess.run(
        ["python3", str(root / "export_results.py"), str(config_path)],
        cwd=str(root),
        check=True,
    )


def _find_latest_result(results_dir: Path, experiment_id: str) -> Path:
    candidates = sorted(results_dir.glob(f"ID_{experiment_id}_timetamps_*/comparison_results.json"))
    if not candidates:
        raise FileNotFoundError(f"No comparison results found for {experiment_id}")
    return candidates[-1]


def _extract_metrics(result_json: Dict) -> Tuple[float, float]:
    entries = result_json.get("comparison_index_details", {}).get("validation", {}).get("Extend_partition_sa", [])
    if not entries:
        raise ValueError("No validation entries for Extend_partition_sa")

    costs = []
    runtimes = []
    for entry in entries:
        cost = entry.get("final_cost_proportion")
        runtime = entry.get("calculation_time")
        if runtime is None:
            runtime = entry.get("cost_evaluation_time")
        if cost is None or runtime is None:
            continue
        costs.append(float(cost))
        runtimes.append(float(runtime))

    if not costs or not runtimes:
        raise ValueError("No valid final_cost_proportion/runtime found in comparison_index_details")

    return statistics.mean(costs), statistics.mean(runtimes)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _write_raw_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "variant,variant_slug,sa_variant,budget_mb,seed,final_cost_proportion,runtime_seconds,result_path\n"
        )
        for row in rows:
            f.write(
                f"{row['variant']},{row['variant_slug']},{row['sa_variant']},{row['budget_mb']},{row['seed']},"
                f"{row['final_cost_proportion']:.6f},{row['runtime_seconds']:.6f},"
                f"{row['result_path']}\n"
            )


def _aggregate(rows: List[Dict], budgets: List[int]) -> List[Dict]:
    grouped = {}
    for row in rows:
        key = (row["variant"], row["sa_variant"], row["variant_slug"], row["budget_mb"])
        if key not in grouped:
            grouped[key] = {"costs": [], "times": []}
        grouped[key]["costs"].append(float(row["final_cost_proportion"]))
        grouped[key]["times"].append(float(row["runtime_seconds"]))

    summary_rows = []
    for variant, sa_variant, variant_slug, budget in [
        (v, s, slug, b) for b in budgets for (v, s, slug) in VARIANTS
    ]:
        values = grouped.get((variant, sa_variant, variant_slug, budget), {"costs": [], "times": []})
        cost_mean, cost_std = _mean_std(values["costs"])
        time_mean, time_std = _mean_std(values["times"])
        summary_rows.append(
            {
                "variant": variant,
                "variant_slug": variant_slug,
                "sa_variant": sa_variant,
                "budget_mb": int(budget),
                "runs": len(values["costs"]),
                "cost_mean": cost_mean,
                "cost_std": cost_std,
                "time_mean": time_mean,
                "time_std": time_std,
            }
        )

    return summary_rows


def _write_summary_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("variant,variant_slug,sa_variant,budget_mb,runs,cost_mean,cost_std,time_mean,time_std\n")
        for row in rows:
            f.write(
                f"{row['variant']},{row['variant_slug']},{row['sa_variant']},{row['budget_mb']},{row['runs']},"
                f"{row['cost_mean']:.6f},{row['cost_std']:.6f},{row['time_mean']:.6f},{row['time_std']:.6f}\n"
            )


def _plot(summary_rows: List[Dict], budgets: List[int], png_path: Path, pdf_path: Path) -> None:
    variant_to_color = {
        "full": "#4E79A7",
        "wo_ocw": "#F28E2B",
        "wo_dynamick": "#E15759",
        "wo_gapfilling": "#76B7B2",
    }

    by_variant = {}
    for row in summary_rows:
        by_variant.setdefault(row["variant_slug"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for variant, sa_variant, variant_slug in VARIANTS:
        rows = sorted(by_variant.get(variant_slug, []), key=lambda x: x["budget_mb"])
        x = [r["budget_mb"] for r in rows]
        y = [r["cost_mean"] for r in rows]
        yerr = [r["cost_std"] for r in rows]
        axes[0].errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.6,
            color=variant_to_color.get(variant_slug, "#333333"),
            label=variant,
        )
    axes[0].set_title("Final Cost Proportion")
    axes[0].set_ylabel("Cost (%)")
    axes[0].set_xlabel("Budget (MB)")
    axes[0].set_xticks(budgets)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    for variant, sa_variant, variant_slug in VARIANTS:
        rows = sorted(by_variant.get(variant_slug, []), key=lambda x: x["budget_mb"])
        x = [r["budget_mb"] for r in rows]
        y = [r["time_mean"] for r in rows]
        yerr = [r["time_std"] for r in rows]
        axes[1].errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.6,
            color=variant_to_color.get(variant_slug, "#333333"),
            label=variant,
        )
    axes[1].set_title("Recommendation Runtime")
    axes[1].set_ylabel("Time (s)")
    axes[1].set_xlabel("Budget (MB)")
    axes[1].set_xticks(budgets)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MS-SA ablation study and export CSV/plots")
    parser.add_argument(
        "--base-config",
        type=str,
        default="free-origin/index/rl_index_selection/experiments/tpchskew.json",
        help="Base TPCHSkew config path",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1000, 1500, 2000],
        help="Budget list in MB",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Random seed list",
    )
    parser.add_argument(
        "--num-workloads",
        type=int,
        default=10,
        help="validation_testing.number_of_workloads per run",
    )
    parser.add_argument(
        "--skew-factor",
        type=float,
        default=None,
        help="Optional TPCHskew factor override",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/plots",
        help="Output directory for CSV and figures",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="analysis/tmp_configs",
        help="Temporary config output directory",
    )
    parser.add_argument("--run", action="store_true", help="Execute experiments")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_cfg_path = root / args.base_config
    base_cfg = _load_json(base_cfg_path)

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dir = root / args.configs_dir
    cfg_dir.mkdir(parents=True, exist_ok=True)

    results_dir = root / "free-origin" / "index" / "rl_index_selection" / "experiment_results"

    budgets = sorted(set(int(b) for b in args.budgets))
    seeds = sorted(set(int(s) for s in args.seeds))

    rows = []
    for budget in budgets:
        for seed in seeds:
            for label, sa_variant, variant_slug in VARIANTS:
                exp_id = f"{base_cfg['id']}_ablation_{variant_slug}_b{budget}_s{seed}"
                cfg = dict(base_cfg)
                cfg["id"] = exp_id
                cfg["random_seed"] = seed
                cfg["comparison_algorithms"] = ["extend_partition_sa"]
                cfg["sa_variant"] = sa_variant
                cfg["sa_fixed_budget_mb"] = int(budget)

                if args.skew_factor is not None:
                    cfg["skew_factor"] = args.skew_factor

                cfg["sa_allocation"] = dict(cfg.get("sa_allocation", {}))
                cfg["sa_allocation"]["log_top_n"] = 0

                cfg["budgets"] = dict(cfg.get("budgets", {}))
                cfg["budgets"]["validation"] = [int(budget)]
                cfg["budgets"]["validation_and_testing"] = [int(budget)]

                cfg["workload"] = dict(cfg.get("workload", {}))
                cfg["workload"]["benchmark"] = "TPCHskew"
                cfg["workload"]["test_number_of_workloads"] = 0
                cfg["workload"]["validation_testing"] = dict(cfg["workload"].get("validation_testing", {}))
                cfg["workload"]["validation_testing"]["number_of_workloads"] = int(args.num_workloads)

                cfg_path = cfg_dir / f"tpchskew_ablation_{variant_slug}_b{budget}_s{seed}.json"
                _save_json(cfg_path, cfg)

                if args.run:
                    _run_export(root, cfg_path)
                    result_path = _find_latest_result(results_dir, exp_id)
                    result_json = _load_json(result_path)
                    cost, runtime = _extract_metrics(result_json)
                    rows.append(
                        {
                            "variant": label,
                            "variant_slug": variant_slug,
                            "sa_variant": sa_variant,
                            "budget_mb": int(budget),
                            "seed": int(seed),
                            "final_cost_proportion": cost,
                            "runtime_seconds": runtime,
                            "result_path": str(result_path),
                        }
                    )

    if not args.run:
        total_cfg = len(budgets) * len(seeds) * len(VARIANTS)
        print(f"Generated {total_cfg} config files in: {cfg_dir}")
        print("Add --run to execute all variants and export CSV/plots.")
        return

    raw_csv_path = output_dir / "ablation_core_mechanisms_raw.csv"
    summary_csv_path = output_dir / "ablation_core_mechanisms_summary.csv"
    png_path = output_dir / "ablation_core_mechanisms_errorbar.png"
    pdf_path = output_dir / "ablation_core_mechanisms_errorbar.pdf"

    summary_rows = _aggregate(rows, budgets)
    _write_raw_csv(raw_csv_path, rows)
    _write_summary_csv(summary_csv_path, summary_rows)
    _plot(summary_rows, budgets, png_path, pdf_path)

    print(f"Saved raw CSV: {raw_csv_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
