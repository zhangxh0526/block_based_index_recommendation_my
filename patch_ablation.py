import re

file_path = "analysis/sa_ablation_experiment.py"
with open(file_path, "r") as f:
    text = f.read()

text = re.sub(r' +?\("w/o OCW", "w_o_ocw", "wo_ocw"\),\n', '', text)
text = re.sub(r' +?\("w/o DynamicK", "w_o_dynamick", "wo_dynamick"\),\n', '', text)

with open(file_path, "w") as f:
    f.write(text)
