import re
file_path = "/home/zzzxh/projects/block_based_index_recommendation_my/free-origin/index/rl_index_selection/index_selection_evaluation/selection/table_generator.py"
with open(file_path, "r") as f:
    text = f.read()

# remove all the added parts
text = re.sub(r' +if self\.benchmark_name == "tpchskew".*?replace\("\.", "_"\)\n', '', text, flags=re.DOTALL)

# add exactly once
text = text.replace('        name += str(self.scale_factor).replace(".", "_")\n', 
                    '        name += str(self.scale_factor).replace(".", "_")\n'
                    '        if self.benchmark_name == "tpchskew" and self.config and "skew_factor" in self.config:\n'
                    '            name += "_z" + str(self.config["skew_factor"]).replace(".", "_")\n')

with open(file_path, "w") as f:
    f.write(text)
