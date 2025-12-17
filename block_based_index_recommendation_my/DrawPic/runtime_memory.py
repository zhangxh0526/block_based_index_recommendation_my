import numpy as np
# 非线性率和捕获率log的处理脚本
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

def draw_runtime_pic(benchmark, results):
    plt.figure(figsize=(6, 4))
    x_axis_data = [1, 2, 3, 4, 5, 6, 7, 8]
    x_names = ["0.5G", "1G", "1.5G", "2G", "3G", "5G", "7G", "10G"]

    fsize = 15
    y_extend = results[0]
    y_slalom = results[1]
    y_bislearner = results[2]
    y_swirl = results[3]


    max_improvement = 0
    for _ in range(len(y_extend)):
        max_improvement = max(max_improvement, (y_extend[_] - y_bislearner[_]))
    print("Bislearner than Extend Max improvement:{}".format(max_improvement))

    print("Extend than Slalom Max improvement:{}".format(
        max([(y_slalom[_] - y_extend[_]) for _ in range(len(y_extend))])))

    print("Bislearner than Swirl improvement:{}".format(
       max([(y_swirl[_] - y_bislearner[_]) / 100 for _ in range(len(y_extend))]) ))

    plt.plot(x_axis_data, y_extend, alpha=1, linewidth=2, label='Extend', color="#696969")
    plt.plot(x_axis_data, y_slalom, alpha=1, linewidth=2, label='Slalom', color="#1f77b4")
    plt.plot(x_axis_data, y_bislearner, alpha=1, linewidth=2, label='BISlearner', color="#ff7f0e")
    plt.plot(x_axis_data, y_swirl, alpha=1, linewidth=2, label='Swirl (BIS Act)', color="#76da91")

    plt.legend(fontsize=15,
               frameon=False)
    plt.xlabel('Memory Budget', fontsize=16, fontweight='bold')

    plt.xticks(x_axis_data, x_names, fontsize=16)
    y_tickets = [40, 60, 80, 100]
    y_names = ["40", "60", "80", "100"]
    plt.yticks(y_tickets, y_names, fontsize=16)
    plt.grid(linestyle='--', alpha=0.6)  # 设置网格线段
    plt.minorticks_off()  # 去掉最小的刻度
    plt.ylabel('Relative workload cost \n (% of without indexes)', fontsize=19, fontweight='bold')  # accuracy
    plt.savefig("{}.pdf".format(benchmark), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 画图相关
    # extend, slalom, bislearner, swirl

    ###############################TPCH######################################
    # results = [[100.39, 99.73, 86.64, 86.03, 75.75, 46.18, 43.05, 42.25],
    #            [97.79, 97.0, 96.44, 92.95, 88.94, 83.84, 75.56, 66.64],
    #            [89.91, 76.85, 69.86, 57.85, 55.18, 47.93, 43.62, 39.7],
    #            [91.21, 90.02, 77.06, 75.27, 64.44, 58.91, 54.82, 44.3]
    #            ]
    # draw_runtime_pic("tpch_runtime", results)

    ###############################SSB######################################
    # results = [[100.53, 100.74, 74.84, 74.52, 45.67, 41.21, 41.32, 40.87],
    #            [98.76, 99.44, 97.51, 99.25, 95.1, 83.08, 67.16, 45.87],
    #            [87.81, 67.27, 55.49, 50.24, 41.64, 37.36, 36.58, 36.63],
    #            ]
    # Swirl = [86.16, 78.81, 65.24, 54.27, 51.77, 50.19, 42.12, 37.58]
    # Swirl = [_ * (64.96 / 64.07) for _ in Swirl]
    # results.append(Swirl)
    # draw_runtime_pic("ssb_runtime", results)


    ##############################TPCHSkew-1.5######################################
    # results = [[99.52, 102.07, 93.68, 89.93, 87.16, 55.92, 41.9, 40.87],
    #            [95.04, 96.96, 93.76, 91.74, 81.92, 80.56, 66.44, 56.6],
    #            [94.99, 88.5, 73.83, 68.97, 63.82, 53.32, 43.8, 42.73],
    #            ]
    # Swirl = [99.89, 84.96, 82.5, 76.1, 68.27, 56.92, 51.42, 47.84]
    # Swirl = [_ * (76.38 / 75.77) for _ in Swirl]
    # results.append(Swirl)
    # draw_runtime_pic("tpch15_runtime", results)

    ###############################TPCHSkew-2######################################
    results = [[100.65, 98.75, 82.26, 68.75, 67.98, 51.39, 14.83, 10.74],
               [101.48, 98.34, 90.7, 93.18, 87.35, 63.94, 45.94, 34.8],
               [97.59, 73.59, 63.88, 54.42, 32.54, 27.67, 12.22, 10.9],
               ]
    Swirl = [93.06, 87.96, 84.58, 76.6, 73.58, 43.16, 29.23, 20.08]
    Swirl = [_ * (61.92 /  61.38) for _ in Swirl]
    results.append(Swirl)
    draw_runtime_pic("tpch2_runtime", results)