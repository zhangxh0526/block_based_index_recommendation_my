import argparse
import json
import statistics
from pathlib import Path
import matplotlib.pyplot as plt

def _load_files(results_dir: Path, exp_prefix: str) -> list:
    return list(results_dir.rglob(f"ID_{exp_prefix}*_timetamps_*/comparison_results.json"))

def _extract_overhead(results: dict, algo_key: str, budget: int):
    details = results.get("comparison_index_details", {}).get("validation", {}).get(algo_key, [])
    for entry in details:
        if entry.get("budget") == budget:
            return {
                "evaluation_time": float(entry.get("cost_evaluation_time", 0)),
                "cost_requests": int(entry.get("cost_requests", 0)),
                "cache_hits": int(entry.get("cache_hits", 0))
            }
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-prefix", type=str, default="random_tpchskew1_5")
    parser.add_argument("--output-dir", type=str, default="analysis/plots")
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "free-origin" / "index" / "rl_index_selection" / "experiment_results"
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = _load_files(results_dir, args.exp_prefix)
    if not files:
        print(f"No files found for prefix {args.exp_prefix}")
        return
        
    algos = {
        "Extend": "Extend", 
        "Extend(Global)": "Extend_global", 
        "MS-SA (Ours)": "Extend_partition_sa"
    }
    budgets = [500, 1000, 1500, 2000, 3000, 5000, 7000, 10000]
    
    # Store aggregated data
    eval_times = {label: {b: [] for b in budgets} for label in algos}
    requests = {label: {b: [] for b in budgets} for label in algos}
    
    for fpath in files:
        with open(fpath, "r") as f:
            res = json.load(f)
        for label, key in algos.items():
            for b in budgets:
                metrics = _extract_overhead(res, key, b)
                if metrics:
                    eval_times[label][b].append(metrics["evaluation_time"])
                    requests[label][b].append(metrics["cost_requests"])
    
    # Plot Evaluation Time
    plt.figure(figsize=(8, 5))
    for label in algos:
        y_means = [statistics.mean(eval_times[label][b]) if eval_times[label][b] else 0 for b in budgets]
        plt.plot(budgets, y_means, marker="o", label=label, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Storage Budget (MB)")
    plt.ylabel("What-If Cost Evaluation Time (s) [Log Scale]")
    plt.title("Database Optimizer Overhead (Time)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "overhead_cost_evaluation_time.png", dpi=200)
    plt.close()

    # Plot Cost Requests (What-If API Calls)
    plt.figure(figsize=(8, 5))
    for label in algos:
        y_means = [statistics.mean(requests[label][b]) if requests[label][b] else 0 for b in budgets]
        plt.plot(budgets, y_means, marker="s", label=label, linewidth=2)
    plt.yscale("log")
    plt.xlabel("Storage Budget (MB)")
    plt.ylabel("Number of What-If API Calls [Log Scale]")
    plt.title("Database Optimizer Overhead (API Calls)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "overhead_whatif_calls.png", dpi=200)
    plt.close()
    
    print("Exported overhead plots to", output_dir)

if __name__ == "__main__":
    main()
