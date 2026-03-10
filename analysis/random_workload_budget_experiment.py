import argparse
import json
import subprocess
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def _run_export(config_path: Path, root: Path) -> None:
    export_script = root / "export_results.py"
    print(f"Running export for config {config_path.name}...")
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

def plot_line_with_confidence(ax, x_vals, y_means, y_stds, label, color, marker):
    ax.plot(x_vals, y_means, label=label, color=color, marker=marker, linewidth=2, 
            path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])
    ax.fill_between(x_vals, np.array(y_means) - np.array(y_stds), np.array(y_means) + np.array(y_stds), 
                    color=color, alpha=0.2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive Random Workload vs Budget Experiment")
    parser.add_argument(
        "--base-config",
        type=str,
        default="free-origin/index/rl_index_selection/experiments/tpch.json",
        help="Base experiment config path. Should be uniform random workload like tpch.json without skew.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Random seeds to sweep for generating query randomness",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
        help="List of budgets to test",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run experiments instead of just generating / plotting existing data",
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
    if not base_config_path.exists():
        print(f"Config path {base_config_path} not found. Please verify.")
        return

    base_config = _load_config(base_config_path)

    # Get benchmark string for result path correctly
    benchmark_name = base_config.get("workload", {}).get("benchmark", "TPCH")
    
    results_dir = root / "free-origin" / "index" / "rl_index_selection" / "experiment_results" / benchmark_name
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_map = {
        "Extend": "Extend_partition",
        "Extend(Global)": "Extend_global",
        "MS-SA (Ours)": "Extend_partition_sa",
    }
    
    # Store aggregated data: label -> budget -> [values from different seeds]
    cost_data = {label: {b: [] for b in args.budgets} for label in algo_map}
    time_data = {label: {b: [] for b in args.budgets} for label in algo_map}

    for seed in args.seeds:
        # Use a distinguishable string format for random base vs skew
        exp_id = f"random_{Path(args.base_config).stem}_seed_{seed}_budget_sweep"
        
        if args.run:
            config = dict(base_config)
            config["id"] = exp_id
            config["random_seed"] = seed
            # Use lowercase for the input config, because experiment.py parses lowercase
            config["comparison_algorithms"] = ["extend_partition", "extend_global", "extend_partition_sa"]
            
            # Setup SA constraints explicitly
            config.setdefault("sa_allocation", {})
            config["sa_allocation"] = dict(config["sa_allocation"])
            config["sa_allocation"]["log_top_n"] = 0
            
            # Setup Budgets
            config.setdefault("budgets", {})
            config["budgets"]["validation"] = sorted(args.budgets)
            
            # Ensure the result_path directory exists so export_results.py doesn't crash
            result_path_dir = root / "free-origin" / "index" / "rl_index_selection" / "swirl" / config["result_path"]
            result_path_dir.mkdir(parents=True, exist_ok=True)
            
            cfg_path = root / "analysis" / "tmp_configs" / f"{exp_id}.json"
            _write_config(config, cfg_path)
            
            _run_export(cfg_path, root)
        
        # Pull data (whether --run was passed or not, assume data exists if not run)
        try:
            result_path = _find_latest_result(results_dir, exp_id)
            results = _load_results(result_path)
            
            for label, key in algo_map.items():
                for budget in args.budgets:
                    val = _extract_budget_value(results, key, budget)
                    t_val = _extract_budget_time(results, key, budget)
                    if val is not None:
                        cost_data[label][budget].append(val)
                    if t_val is not None:
                        time_data[label][budget].append(t_val)
        except Exception as e:
            print(f"Skipping seed {seed} data due to error: {e}")

    # Process and Plot
    # Pre-calculate means and stds for plotting and saving
    labels = list(algo_map.keys())
    plot_budgets = sorted(args.budgets)
    
    cost_means = {label: [] for label in labels}
    cost_stds = {label: [] for label in labels}
    time_means = {label: [] for label in labels}
    time_stds = {label: [] for label in labels}
    
    for label in labels:
        for budget in plot_budgets:
            c_vals = cost_data[label][budget]
            t_vals = time_data[label][budget]
            
            cost_means[label].append(np.mean(c_vals) if c_vals else 0)
            cost_stds[label].append(np.std(c_vals) if c_vals else 0)
            time_means[label].append(np.mean(t_vals) if t_vals else 0)
            time_stds[label].append(np.std(t_vals) if t_vals else 0)

    # Output CSV summary
    csv_path = output_dir / "random_workload_comprehensive.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        header = ["Budget_MB"]
        for label in labels:
            header.extend([f"{label}_Cost_Mean", f"{label}_Cost_Std", f"{label}_Time_Mean", f"{label}_Time_Std"])
        f.write(",".join(header) + "\n")
        
        for i, budget in enumerate(plot_budgets):
            row = [str(budget)]
            for label in labels:
                row.extend([
                    f"{cost_means[label][i]:.2f}",
                    f"{cost_stds[label][i]:.2f}",
                    f"{time_means[label][i]:.4f}",
                    f"{time_stds[label][i]:.4f}",
                ])
            f.write(",".join(row) + "\n")
            
    print(f"Saved numerical comprehensive results to: {csv_path}")

    # Visualizations (Paper-level confidence interval line plots)
    # Styles and markers
    colors = {"Extend": "#1f77b4", "Extend(Global)": "#ff7f0e", "MS-SA (Ours)": "#d62728"}
    markers = {"Extend": "o", "Extend(Global)": "^", "MS-SA (Ours)": "s"}

    # 1. Cost Proportion vs Budget
    fig_cost, ax_cost = plt.subplots(figsize=(8, 5))
    for label in labels:
        plot_line_with_confidence(ax_cost, plot_budgets, cost_means[label], cost_stds[label], label, colors[label], markers[label])
    
    ax_cost.set_xlabel("Storage Budget (MB)", fontsize=12)
    ax_cost.set_ylabel("Final Relative Cost vs. Empty Index (%)", fontsize=12)
    ax_cost.set_title("Performance Under Random Workload Across Budgets", fontsize=14)
    ax_cost.grid(True, linestyle="--", alpha=0.6)
    ax_cost.legend(loc='upper right', fontsize=10)
    fig_cost.tight_layout()
    cost_plot_path = output_dir / "random_workload_budget_cost.png"
    fig_cost.savefig(cost_plot_path, dpi=300)
    plt.close(fig_cost)

    # 2. Calculation Time vs Budget
    fig_time, ax_time = plt.subplots(figsize=(8, 5))
    for label in labels:
        plot_line_with_confidence(ax_time, plot_budgets, time_means[label], time_stds[label], label, colors[label], markers[label])
    
    ax_time.set_xlabel("Storage Budget (MB)", fontsize=12)
    ax_time.set_yscale("log") # log scale for time is often necessary for paper level figures given Extend_global scales terribly
    ax_time.set_ylabel("Index Selection Time (s) [Log Scale]", fontsize=12)
    ax_time.set_title("Calculation Time Under Random Workload", fontsize=14)
    ax_time.grid(True, linestyle="--", alpha=0.6, which="both")
    ax_time.legend(loc='lower right', fontsize=10)
    fig_time.tight_layout()
    time_plot_path = output_dir / "random_workload_budget_time.png"
    fig_time.savefig(time_plot_path, dpi=300)
    plt.close(fig_time)
    
    print(f"Saved plots to {cost_plot_path} and {time_plot_path}!")

if __name__ == "__main__":
    main()
