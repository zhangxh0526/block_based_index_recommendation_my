import json
import os
import sys

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("swirl"))

from swirl.schema import Schema
from swirl.workload_generator import WorkloadGenerator

with open("experiments/tpchskew1_5.json", "r") as f:
    config = json.load(f)

print("Starting schema generation for skew_factor 1.5...")
schema = Schema(
    benchmark_name=config["workload"]["benchmark"],
    scale_factor=config["workload"]["scale_factor"],
    partition_num=config["partition_num"],
    used_tables=config["used_tables"],
    config=config
)

print("Starting workload generation for skew_factor 1.5...")
wg = WorkloadGenerator(
    config=config["workload"],
    workload_columns=schema.columns,
    random_seed=config["random_seed"],
    database_name=schema.database_name,
    experiment_id=config["id"],
    filter_utilized_columns=config["filter_utilized_columns"],
    partition_num=config["partition_num"],
    violation_queries=set()
)
print("Done for 1.5!")
