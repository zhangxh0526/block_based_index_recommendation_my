import random

import numpy as np
# 非线性率和捕获率log的处理脚本
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

def draw_mask_pic(benchmark, file_name):
    # 画图相关storage budget H=10GB, maximum index width ]max=3, |G| = 819 candidates
    plt.figure(figsize=(6, 3))

    # valid_actions_percent = [random.uniform(0, _/90) for _ in range(1, 91)]
    # valid_actions_percent = [0.07, 0.07, 0.071, 0.072, 0.075, 0.078, 0.078, 0.08, 0.082, 0.082, 0.083, 0.085, 0.088, 0.088,
    #                          0.088, 0.088, 0.09, 0.091, 0.091, 0.092, 0.092, 0.092, 0.094, 0.093, 0.096, 0.097, 0.097, 0.097,
    #                          0.1, 0.11, 0.12, 0.13, 0.131, 0.129, 0.13, 0.129, 0.128, 0.127, 0.127, 0.12, 0.11, 0.105, 0.1,
    #                          0.09, 0.09, 0.085, 0.08, 0.079, 0.078, 0.07, 0.067, 0.057, 0.05, 0.047, 0.045, 0.04, 0.03, 0.025, 0.02, 0.01]
    valid_actions_percent = []
    storage_consumption = []
    f = open(file_name, "r+", encoding="utf-8")
    for line in f:
        if "ction Mask Dim" in line:
            result = line.split(":")[-1].replace("[", "").replace("]", "")
            total = int(result.split(",")[0])
            valid = int(result.split(",")[1])
            valid_actions_percent.append(valid / total * 100)
        elif "Total budget" in line:
            total_storage = line.strip().split(",")[0].split(":")[-1].strip()
            consumed_storage = float(line.strip().split(",")[-1].split(":")[-1].strip())
            total_storage = 1024 * 1024 * 1024 * 10
            if total_storage == "None":
                # 10GB
                total_storage = 1024 * 1024 * 1024 * 10
            else:
                total_storage = int(total_storage)
            storage_consumption.append((total_storage - consumed_storage) / total_storage * 100)
    f.close()
    steps = [_ for _ in range(len(valid_actions_percent))]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # fig mask
    #ax1.plot([_ for _ in range(100)], [random.randint(0, 100) for _ in range(100)], alpha=1, linewidth=2, label='ZoneMap', color="#FC8002")
    # ax1.plot(steps, valid_actions_percent, alpha=1, linewidth=2, label='Valid actions', color="#D89C7A")
    ax1.bar(steps, valid_actions_percent, alpha=1, width=0.5, label='Valid actions', color="#686789")
    ax1.set_ylabel('Valid actions \n (% of all actions)', fontsize=12, fontweight='bold')  # accuracy
    ax1.legend(ncol=1, fontsize=12, frameon=False, loc=(0.56, 0.8))

    print("Benchmark {}, begining vaild action : {}, max valid action : {}".format(benchmark, valid_actions_percent[0], max(valid_actions_percent)))

    # fig storage
    ax2.plot(steps, storage_consumption, alpha=1, linewidth=2,  color='#D89C7A', label='Remaining budget')
    #ax2.bar([_ for _ in range(100)], [random.randint(0, 100) for _ in range(100)], alpha=1, width=0.5, color='#B0B1B6')
    ax2.set_ylabel('Remaining budget \n (% of total budget)', fontsize=12, fontweight='bold')  # accuracy


    ax2.legend(ncol=1, fontsize=12, frameon=False, loc=(0.56, 0.9))
    # plt.legend(fontsize=16,
    #            frameon=False)
    ax1.set_xlabel('Agent index selection decisions (Steps)', fontsize=12, fontweight='bold')
    # plt.ylim((0, 0.16))
    # y_tickets = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    # y_names = ["0", "1", "2", "3", "4", "5"]
    # plt.yticks(y_tickets, y_names, fontsize=12)
    plt.grid(linestyle='--', alpha=0.6)  # 设置网格线段
    plt.minorticks_off()  # 去掉最小的刻度

    # plt.xlabel('Records Num Per Block', fontsize=16, fontweight='bold')  # accuracy
    plt.savefig("{}_action_mask.pdf".format(benchmark), bbox_inches='tight')
    plt.show()

   # print("Beginning Per {}, Max Per {}".format(valid_actions_percent[0], max(valid_actions_percent)))



if __name__ == "__main__":
    draw_mask_pic("TPCH", "tpch_mask.log")
    # draw_mask_pic("TPCHSkew15", "tpchskew15_mask.log")
    # draw_mask_pic("TPCHSkew2", "tpchskew2_mask.log")
    draw_mask_pic("SSB", "ssb_mask.log")
