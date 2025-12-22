import argparse
import json
import sys
import os
import logging
from pathlib import Path

# Add module paths
ROOT = Path(__file__).resolve().parent / "free-origin" / "index" / "rl_index_selection"
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "index_selection_evaluation"))
sys.path.append(str(ROOT / "swirl"))

# Enable INFO logs so histogram budget messages surface
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from swirl.experiment import Experiment  # noqa: E402


def index_to_str(index_obj):
    try:
        return str(index_obj)
    except Exception:
        return repr(index_obj)


def serialize_performances(perf_dict):
    out = {}
    for split, algos in perf_dict.items():
        out[split] = {}
        for algo, values in algos.items():
            out[split][algo] = values
    return out


def serialize_indexes(index_dict):
    out = {}
    for algo, idx_set in index_dict.items():
        out[algo] = [index_to_str(idx) for idx in idx_set]
    return out


def serialize_index_details(index_details):
    out = {}
    for split, algos in index_details.items():
        out[split] = {}
        for algo, entries in algos.items():
            out[split][algo] = []
            for entry in entries:
                entry_copy = dict(entry)
                entry_copy["indexes"] = [dict(idx) for idx in entry.get("indexes", [])]
                out[split][algo].append(entry_copy)
    return out


def aggregate_index_details_by_budget(index_details):
    aggregated = {}
    for split, algos in index_details.items():
        aggregated[split] = {}
        for algo, entries in algos.items():
            budget_map = {}
            for entry in entries:
                budget = entry.get("budget")
                if budget is None:
                    continue
                bucket = budget_map.setdefault(
                    budget,
                    {"budget": budget, "runs": 0, "final_cost_proportions": [], "indexes": {}},
                )
                bucket["runs"] += 1
                if entry.get("final_cost_proportion") is not None:
                    bucket["final_cost_proportions"].append(entry["final_cost_proportion"])
                for idx in entry.get("indexes", []):
                    bucket["indexes"].setdefault(idx["repr"], idx)

            aggregated[split][algo] = []
            for budget_value in sorted(budget_map.keys()):
                bucket = budget_map[budget_value]
                indexes = list(bucket["indexes"].values())
                total_size_bytes = sum(
                    idx["size_bytes"] for idx in indexes if idx.get("size_bytes") is not None
                )
                total_size_mb = round(total_size_bytes / (1024 * 1024), 2) if indexes else None
                avg_cost = None
                if bucket["final_cost_proportions"]:
                    avg_cost = sum(bucket["final_cost_proportions"]) / len(bucket["final_cost_proportions"])

                aggregated[split][algo].append(
                    {
                        "budget": budget_value,
                        "runs": bucket["runs"],
                        "average_final_cost_proportion": avg_cost,
                        "final_cost_proportions": bucket["final_cost_proportions"],
                        "index_count": len(indexes),
                        "total_index_size_bytes": total_size_bytes,
                        "total_index_size_mb": total_size_mb,
                        "indexes": indexes,
                    }
                )
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run Experiment.compare and export results")
    parser.add_argument("config", nargs="?", default="free-origin/index/rl_index_selection/experiments/tpchskew.json",
                        help="Path to experiment configuration JSON")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    # Experiment expects cwd so that ../experiment_results exists relative to swirl
    os.chdir(ROOT / "swirl")

    exp = Experiment(str(config_path))
    exp.prepare()
    exp.compare()

    result = {
        "experiment_id": exp.id,
        "config_path": str(config_path),
        "comparison_performances": serialize_performances(exp.comparison_performances),
        "comparison_indexes": serialize_indexes(exp.comparison_indexes),
        "comparison_index_details": serialize_index_details(exp.comparison_index_details),
        "comparison_indexes_by_budget": aggregate_index_details_by_budget(exp.comparison_index_details),
        "validation_workloads_count": sum(len(wl_list) for wl_list in exp.workload_generator.wl_validation),
        "testing_workloads_count": sum(len(wl_list) for wl_list in exp.workload_generator.wl_testing),
        "validation_budgets": exp.config.get("budgets", {}).get("validation", []),
        "test_budgets": exp.config.get("budgets", {}).get("validation_and_testing", []),
    }

    out_path = Path(exp.experiment_folder_path) / "comparison_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
