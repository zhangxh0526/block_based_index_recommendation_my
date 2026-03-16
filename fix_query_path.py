import os 
file_path = "/home/zzzxh/projects/block_based_index_recommendation_my/free-origin/index/rl_index_selection/swirl/workload_generator.py"
with open(file_path, "r") as f:
    text = f.read()

text = text.replace('QUERY_PATH = "query_files"', 
                    'import os\nQUERY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "query_files")')

with open(file_path, "w") as f:
    f.write(text)

