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
