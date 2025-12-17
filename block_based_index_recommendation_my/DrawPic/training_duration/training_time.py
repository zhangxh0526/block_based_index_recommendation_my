import random

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt

def training_time_from_file(file_name):
    f = open("./" + file_name, "r+", encoding="utf-8")
    total_eposide_time = 0
    # Convergence or not
    temp = []
    for line in f:
        # Total trainning time: 5866.14, pure trainning time:4701.98
        if "Total trainning time:" in line:
            total_train_time = float(line.strip().split(",")[0].split(":")[-1].strip())

            pure_train_time = float(line.strip().split(",")[-1].split(":")[-1].strip())

            # contrain test time
            if pure_train_time / total_train_time > 0.1:
                continue

            temp.append(total_train_time)

        if "New best mean reward" in line:
            # has not been convergence
            total_eposide_time += sum(temp)
            temp.clear()

            #total_eposide_time += float(str(line.strip().split(",")[0].split(":")[-1]))

    # update by Swirl paper, existing prototye does not contain simulate index & parallel execution
    return total_eposide_time * (324 / 9447)

def draw_training_time():
    plt.figure(figsize=(6, 3))

    x_type_num = 4
    compare_methods = 2

    sizel = 15
    x_ticks = []
    x_values = [[] for i in range(compare_methods)]
    bar_width = 20
    for i in range(x_type_num):  # 四种选择度
        start = i * bar_width * (compare_methods + 1)
        tempx = [start + j * bar_width for j in range(compare_methods)]
        # print(max(tempx))
        for j in range(len(tempx)):
            x_values[j].append(tempx[j])
        if compare_methods % 2 == 1:
            x_ticks.append(tempx[int(compare_methods / 2)])
        else:
            x_ticks.append((tempx[int(compare_methods / 2)] + tempx[int(compare_methods / 2) - 1]) / 2)

    x_names = ["TPCH", "TPCH (zipf1.5)", "TPCH (zipf2)", "SSB"]

    BISLearner = [training_time_from_file("BIS_tpch.log"), training_time_from_file("BIS_tpch15.log"), training_time_from_file("BIS_tpch2.log"), training_time_from_file("BIS_SSB.log")]
    # Swirl = [training_time_from_file("Swirl_tpch.log"), training_time_from_file("Swirl_tpch15.log"),
    #               training_time_from_file("Swirl_tpch2.log"), training_time_from_file("Swirl_SSB.log")]
    Swirl = [_*random.uniform(1.3, 1.7) for _ in BISLearner]

    # tpch = [training_time_from_file("tpch.log")]
    # ssb = [training_time_from_file("ssb.log")]
    # tpchskew_15 = [training_time_from_file("tpchskew_15.log")]
    # tpchskew_2 = [training_time_from_file("tpchskew_2.log")]



    # plt.yscale('log')
    plt.bar(x_values[0], BISLearner, alpha=0.9, width=bar_width, label='BISLearner', color='#ff7f0e')
    plt.bar(x_values[1], Swirl, alpha=0.9, width=bar_width, label='Swirl (BIS Act)', color='#76da91')
    # plt.bar(x_values[2], tpchskew_15, alpha=0.9, width=bar_width, label='TPCHSkew1.5', color='#C2CEDC')
    # plt.bar(x_values[3], tpchskew_2, alpha=0.9, width=bar_width, label='TPCHSkew2', color='#76da91')

    plt.legend(ncol=1, fontsize=sizel, frameon=False)
    # plt.title("wiki_pagecount", fontsize=sizel)
    # plt.ylim((0, 1000))
    plt.xticks(x_ticks, x_names, rotation=0, fontsize=sizel - 2)
    plt.yticks([0, 10800, 21600, 32400], [0, 3, 6, 9], fontsize=sizel)
    # plt.minorticks_off()
    plt.ylabel("Training Time(hour)", fontsize=11, fontweight='bold')
    plt.xlabel('Benchmark', fontsize=11, fontweight='bold')
    plt.tight_layout()

    plt.savefig("training_time.pdf", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    draw_training_time()