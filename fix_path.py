import re
import os

file_path = "/home/zzzxh/projects/block_based_index_recommendation_my/free-origin/index/rl_index_selection/index_selection_evaluation/selection/table_generator.py"
with open(file_path, "r") as f:
    text = f.read()

# Replace relative directories with robust absolute path
text = text.replace('        elif self.benchmark_name == "tpchskew":\n            self.make_command = ["make", "-f", "makefile_linux.original"]\n            if platform.system() == "Darwin":\n                self.make_command = ["make", "-f", "makefile_MacSolaris"]\n            self.directory = "../index_selection_evaluation/tpchskew-kit"',
'''        elif self.benchmark_name == "tpchskew":
            self.make_command = ["make", "-f", "makefile_linux.original"]
            if platform.system() == "Darwin":
                self.make_command = ["make", "-f", "makefile_MacSolaris"]
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.directory = os.path.join(base_dir, "tpchskew-kit")''')

with open(file_path, "w") as f:
    f.write(text)
