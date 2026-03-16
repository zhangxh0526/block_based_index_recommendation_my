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
    parser.add_argument("--run", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="analysis/plots_alpha_sweep")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    base_config_path = root / args.base_config
    base_config = _load_config(base_config_path)

    results_dir = root / "free-origin" / "index" / "rl_index_selection" / "experiment_results"

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_budgets = [1500, 10000]
    results_data = {b: {"costs": [], "times": []} for b in target_budgets}

    for alpha in args.alphas:
        exp_id = f"{base_config['id']}_alpha_ablation_{alpha}".replace(".", "p")
        config = dict(base_config)
        config["id"] = exp_id
        config["comparison_algorithms"] = ["extend_partition_sa"]
        config.setdefault("sa_allocation", {})
        config["sa_allocation"] = dict(config["sa_allocation"])
        config["sa_allocation"]["anneal_alpha"] = float(alpha)
        
        # Override budgets to save time, only run the ones we care about
        # Wait, if we change the config's test/validation budgets, it will only evaluate those!
        config["validation_budgets"] = target_budgets
        config["test_budgets"] = []
        config["queries"] = 1000 # speed run
        
        cfg_path = root / "analysis" / "tmp_configs" / f"exp_alpha_{alpha}.json"
        _write_config(config, cfg_path)

        _run_export(cfg_path, root)
        result_path = _find_latest_result(results_dir, exp_id)
        results = _load_results(result_path)
        
        for b in target_budgets:
            cost = _get_metric_for_budget(results, "Extend_partition_sa", b, "final_cost_proportion")
            time_val = _get_metric_for_budget(results, "Extend_partition_sa", b, "calculation_time")
            results_data[b]["costs"].append(cost)
            results_data[b]["times"].append(time_val)

    # Plot Costs
    plt.figure(figsize=(8, 4.5))
    for b in target_budgets:
        plt.plot(args.alphas, results_data[b]["costs"], marker="o", label=f"Budget {b}MB")
    plt.xscale('symlog', linthresh=1.0)
    plt.xlabel("Anneal Alpha (Log Scale)")
    plt.ylabel("Cost Proportion (%)")
    plt.title("SA Anneal Alpha Sensitivity")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "anneal_alpha_sweep_cost_custom.png", dpi=200)
    plt.close()
    
    print("Done plotting!")

if __name__ == "__main__":
    main()
