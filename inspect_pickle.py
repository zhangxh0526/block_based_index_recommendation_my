import os
import sys
import pickle
import json
from pathlib import Path

# 加入项目根路径，确保能反序列化对象
ROOT = Path(__file__).resolve().parent / "free-origin" / "index" / "rl_index_selection"
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "index_selection_evaluation"))

pickle_path = ROOT / "experiment_results" / "ID_TPCHskew_Test_Experiment_timetamps_1765784198" / "validation_workloads.pickle"

print("File exists:", pickle_path.exists())

with open(pickle_path, "rb") as f:
    data = pickle.load(f)

print("Top-level type:", type(data))

# 简单结构探测
if isinstance(data, dict):
    print("Dict keys:", list(data.keys())[:10])
elif isinstance(data, (list, tuple)):
    print("Length:", len(data))
    print("First element type:", type(data[0]))
    if len(data) > 0:
        print("Inner length of first element:", len(data[0]) if hasattr(data[0], "__len__") else None)
        # 打印一个 workload 的基本属性
        try:
            wl = data[0][0]
            print("Workload type:", type(wl))
            print("Queries count in workload:", len(getattr(wl, "queries", [])))
            if wl.queries:
                q0 = wl.queries[0]
                print("First query id:", getattr(q0, "nr", None))
                print("First query text (truncated):", getattr(q0, "text", "")[:200])
                print("First query columns count:", len(getattr(q0, "columns", [])))
        except Exception as e:
            print("Sample inspect failed:", e)
else:
    print("Data:", data)


# 导出为 JSON（仅包含查询文本、id、预算），方便查看
out_path = Path("validation_workloads.json")
export = []
if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple)):
    for workload in data[0]:
        entry = {
            "budget": getattr(workload, "budget", None),
            "queries": [
                {
                    "id": getattr(q, "nr", None),
                    "text": getattr(q, "text", ""),
                    "columns": [str(c) for c in getattr(q, "columns", [])],
                }
                for q in getattr(workload, "queries", [])
            ],
        }
        export.append(entry)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"Exported JSON -> {out_path} ({len(export)} workloads)")
else:
    print("Skip JSON export: unexpected structure")
