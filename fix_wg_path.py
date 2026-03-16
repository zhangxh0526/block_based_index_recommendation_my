import re
file_path = "/home/zzzxh/projects/block_based_index_recommendation_my/free-origin/index/rl_index_selection/swirl/workload_generator.py"
with open(file_path, "r") as f:
    text = f.read()

text = text.replace('f"../query_files/{benchmark_name}/{benchmark_name}_{query_class}.txt"',
                    'f"query_files/{benchmark_name}/{benchmark_name}_{query_class}.txt"')

with open(file_path, "w") as f:
    f.write(text)
