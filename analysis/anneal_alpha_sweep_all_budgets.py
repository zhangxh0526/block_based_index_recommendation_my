import argparse
import json
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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
    candidates = sorted(results_dir.rglob(f"ID_{experiment_id}_timetamps_*/comparison_results.json"))
    if not candidates:
        raise FileNotFoundError(f"No results found for {experiment_id} under {results_dir}")
    return candidates[-1]

def _get_metric_for_budget(results: dict, algo_key: str, budget: int, metric: str) -> float:
    details = results.get("comparison_index_details", {}).get("validation", {}).get(algo_key, [])
    for entry in details:
        if entry.get("budget") == budget:
            return float(entry.get(metric))
    return float('nan')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, default="free-origin/index/rl_index_selection/experiments/tpchskew1_5.json")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.5, 1.0, 3.0, 10.0, 100.0])
    parser.add_argument("--output-dir", type=str, default="analysis/plots_alpha_sweep")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_config_path = root / args.base_config
    base_config = _load_config(base_config_path)

    results_dir = root / "free-origin" / "index" / "rl_index_selection" / "experiment_results"

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_budgets = [500, 1000, 1500, 2000, 3000, 5000, 7000, 10000]
    results_data = {b: {"costs": [], "times": []} for b in target_budgets}

    for alpha in args.alphas:
        exp_id = f"{base_config['id']}_alpha_ablation_all_{alpha}".replace(".", "p")
        config = dict(base_config)
        config["id"] = exp_id
        config["comparison_algorithms"] = ["extend_partition_sa"]
        config.setdefault("sa_allocation", {})
        config["sa_allocation"] = dict(config["sa_allocation"])
        config["sa_allocation"]["anneal_alpha"] = float(alpha)
        
        config["validation_budgets"] = target_budgets
        config["test_budgets"] = []
        # No speed run restriction to make sure we got reliable values
        config["queries"] = 2000 
        
        cfg_path = root / "analysis" / "tmp_configs" / f"exp_alpha_all_{alpha}.json"
        _write_config(config, cfg_path)

        # Execute
        print(f"Running Full Budget Alpha Sweep for alpha = {alpha}")
        _run_export(cfg_path, root)
        
        # Load
        result_path = _find_latest_result(results_dir, exp_id)
        results = _load_results(result_path)
        
        for b in target_budgets:
            cost = _get_metric_for_budget(results, "Extend_partition_sa", b, "final_cost_proportion")
            time_val = _get_metric_for_budget(results, "Extend_partition_sa", b, "calculation_time")
            results_data[b]["costs"].append(cost)
            results_data[b]["times"].append(time_val)

    # Plot Costs (All Budgets)
    # Using a colormap to distinguish budgets gracefully
    colors = plt.cm.viridis(np.linspace(0, 1, len(target_budgets)))
    
    plt.figure(figsize=(10, 6))
    for i, b in enumerate(target_budgets):
        plt.plot(args.alphas, results_data[b]["costs"], marker="o", label=f"Budget {b}MB", color=colors[i], linewidth=2)
        
    plt.xscale('symlog', linthresh=1.0)
    plt.xlabel("Anneal Alpha (Log Scale)")
    plt.ylabel("Cost Proportion (%)")
    plt.title("SA Anneal Alpha Sensitivity Across All Budgets")
    
    # Put legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "anneal_alpha_sweep_cost_all_budgets.png", dpi=300)
    plt.close()
    
    print(f"Done plotting! Saved to {output_dir / 'anneal_alpha_sweep_cost_all_budgets.png'}")

if __name__ == "__main__":
    main()
