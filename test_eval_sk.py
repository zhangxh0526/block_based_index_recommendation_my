import urllib
file_path = "/home/zzzxh/projects/block_based_index_recommendation_my/free-origin/index/rl_index_selection/swirl/workload_generator.py"
with open(file_path, "r") as f:
    text = f.read()

text = text.replace('f"{QUERY_PATH}/TPCHskew/{self.benchmark}_{file_number}.txt" if self.benchmark == "TPCHskew" else f"{QUERY_PATH}/{self.benchmark}/{self.benchmark}_{file_number}.txt"', 
                    'f"{QUERY_PATH}/TPCHskew/TPCHskew_{file_number}.txt" if self.benchmark.lower() == "tpchskew" else f"{QUERY_PATH}/{self.benchmark}/{self.benchmark}_{file_number}.txt"')

with open(file_path, "w") as f:
    f.write(text)
