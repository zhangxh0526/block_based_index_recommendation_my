import re
file_path = "/home/zzzxh/projects/block_based_index_recommendation_my/free-origin/index/rl_index_selection/swirl/workload_generator.py"
with open(file_path, "r") as f:
    text = f.read()

text = text.replace('open(f"../{QUERY_PATH}/{self.benchmark}/{self.benchmark}_{file_number}.txt", "r")',
                    'open(f"{QUERY_PATH}/{self.benchmark}/{self.benchmark}_{file_number}.txt", "r")')

with open(file_path, "w") as f:
    f.write(text)
