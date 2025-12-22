import random

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt


def draw_execution_time(benchmark, execution_time, y0, y1, y2, y3):
    print("Benchmark is {}".format(benchmark))
    datasetnum = 8
    indexnum = 4

    fig = plt.figure(figsize=(6, 4))
    sizel = 12
    n = 4
    x_ticks = []
    xs1 = [[] for i in range(indexnum)]
    bar_width = 20
    for i in range(datasetnum):#四种选择度
        start = i * bar_width * (indexnum + 1)
        tempx = [start + j * bar_width for j in range(indexnum)]
        # print(max(tempx))
        for j in range(len(tempx)):
            xs1[j].append(tempx[j])
        if indexnum % 2 == 1:
            x_ticks.append(tempx[int(indexnum/2)])
        else:
            x_ticks.append((tempx[int(indexnum/2)] + tempx[int(indexnum/2)-1])/2)

    x_names = ["0.5G", "1G", "1.5G", "2G", "3G", "5G", "7G", "10G"]

    bislearner = execution_time[2]
    Extend = execution_time[0]
    Slalom = execution_time[1]
    Swirl = execution_time[3]

    print("Bislearner is {} times fast than Extend".format(max([Extend[_]/bislearner[_] for _ in range(len(bislearner))])))
    print("Bislearner is {} times fast than Swirl".format(
        max([Swirl[_] / bislearner[_] for _ in range(len(bislearner))])))

    plt.bar(xs1[0], Extend, alpha=0.9, width=bar_width, label='Extend', color='#696969')
    plt.bar(xs1[1], Slalom, alpha=0.9, width=bar_width, label='Slalom', color='#1f77b4')
    plt.bar(xs1[2], Swirl, alpha=0.9, width=bar_width, label='Swirl (using BIS Act)', color='#76da91')
    plt.bar(xs1[3], bislearner, alpha=0.9, width=bar_width, label='BISlearner', color='#ff7f0e')





    # ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.35])
    # ax2 = fig.add_axes([0.15, 0.55, 0.8, 0.35])
    # plt.yscale('log')
    # ax1.bar(xs1[0], Slalom, alpha=0.9, width=bar_width, label='Slalom', color='#1f77b4')
    # ax1.bar(xs1[1], bislearner, alpha=0.9, width=bar_width, label='Bislearner', color='#ff7f0e')
    # ax1.bar(xs1[2], Swirl, alpha=0.9, width=bar_width, label='Swirl (BIS Act)', color='#76da91')
    # ax1.bar(xs1[3], Extend, alpha=0.9, width = bar_width, label='Extend', color='#696969')
    #
    # ax2.bar(xs1[0], Slalom, alpha=0.9, width=bar_width, label='Slalom', color='#1f77b4')
    # ax2.bar(xs1[1], bislearner, alpha=0.9, width=bar_width, label='Bislearner', color='#ff7f0e')
    # ax2.bar(xs1[2], Swirl, alpha=0.9, width=bar_width, label='Swirl (BIS Act)', color='#76da91')
    # ax2.bar(xs1[3], Extend, alpha=0.9, width = bar_width, label='Extend', color='#696969')
    #
    # ax1.set_ylim(y0, y1)
    # ax2.set_ylim(y2, y3)
    #
    # # 绘制断裂处的标记
    # d = .85  # 设置倾斜度
    # kwargs = dict(marker=[(-1, -d), (1, d)], markersize=5,
    #               linestyle='none', color='k', mec='k', mew=1, clip_on=False)
    # ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
    # ax1.plot([0, 1], [1, 1], transform=ax1.transAxes, **kwargs)
    # ax2.spines['bottom'].set_visible(False)  # 关闭子图2中底部脊
    # ax1.spines['top'].set_visible(False)  ##关闭子图1中顶部脊
    # ax2.set_xticks([])


    plt.legend(fontsize=15,frameon=False, loc='best')
    # plt.title("wiki_pagecount", fontsize=15)
    # plt.ylim((0, 1000))
    plt.xticks(x_ticks, x_names, fontsize=15)
    plt.yticks(fontsize=15)
    # ax1.set_xticks(x_ticks)
    # ax1.set_xticklabels(x_names)
    # # ax1.yticks(fontsize=sizel)
    # plt.minorticks_off()
    plt.ylabel("Selection Runtime(Sec)", fontsize=15, fontweight='bold')
    plt.xlabel('Memory Budget', fontsize=15, fontweight='bold')
    plt.tight_layout()

    plt.savefig("{}.pdf".format(benchmark), dpi=300, bbox_inches='tight')
    plt.show()


def trans_results_by_swirl(result):
    _result = []
    for _ in result:
        tmp = []
        for data in _:
            tmp.append(data * (690 / 7288.85))
        _result.append(tmp)
    return _result

if __name__ == "__main__":
    # 画图相关
    # extend, slalom, bislearner, swirl


    # Greenplum暂时没有提供假设索引的服务（以及多线程实现等优化操作），因此按照Swirl的Extend执行开销对其进行调整，否则Extend开销不合理
    ###############################TPCH######################################
    # results = [[663.9, 665.8, 672.46, 676.79, 675, 669.14, 677, 690],
    #            [0.01, 0.02, 0.01, 0.02, 0.02, 0.04, 0.04, 0.05],
    #            [1.92, 0.58, 0.94, 1.29, 1.8, 2.95, 4.23, 5.2],
    #            [0.2, 0.2, 0.3, 0.4, 0.6, 1.1, 2, 3]
    #            ]
    results = [[434.54, 439.23, 1292.54,  1256.88,  2532.26, 4479.57, 7392.7, 7288.85],
               [882.87, 877.92, 874.25, 874.16, 879.97, 879.46, 878.93, 876.9],
               [20.73, 55.12, 69.15, 105.1, 166.46, 366.3, 569.4, 811.0],
               [16, 43, 60, 100, 100, 300, 500, 800]
               ]
    draw_execution_time("tpch_executiontime", trans_results_by_swirl(results), 0, 10, 650, 700)

    ###############################SSB######################################
    # results = [[324.56, 320.75, 318.97, 322.77, 321.91, 321.99, 315.72, 321.65],
    #            [0.001, 0.02, 0.001, 0.09, 0.05, 0.03, 0.05, 0.07],
    #            [0.41, 0.21, 0.33, 0.41, 0.79, 1.34, 2.58, 2.86],
    #            [0.17, 0.04, 0.09, 0.14, 0.18, 0.32, 0.48, 0.8]
    #            ]
    results = [[330.72, 320.75, 1050.29, 1250.7,  3894.16, 4952.75, 8213, 8721],
               [614.16, 321.38, 320.18, 323.89, 322.91, 320.93,  321.88 , 320.79],
               [12.51, 38.09,  68.21, 87.01, 193.3, 327.76, 464.65, 733.46],
               [19.05, 48.59,  66.81, 70.05, 139.49,  284.38, 373.77, 511.24]
               ]
    draw_execution_time("ssb_executiontime", trans_results_by_swirl(results), 0, 10, 650, 700)


    ##############################TPCHSkew-1.5######################################
    # results = [[295.71,  297.34,  297.04, 294.77, 297.18,  295.83,  294.5, 293.2],
    #            [ 0.05, 0.03, 0.05, 0.08, 0.06, 0.02, 0.1, 0.08],
    #            [1.8, 1.78, 1.88, 1.91, 2.28, 3.18, 4.03, 4.9],
    #            [0.24, 0.33,  0.35, 0.43, 0.48, 0.79, 0.98, 1.3],
    #            ]
    results = [[ 298.54 , 297.35,  988.59,  1069.47, 1200.01, 2860.89,  4892.83,  5968.2],
               [ 537.42, 298.18,  299.57, 297.22, 300.21, 298.76,  297.36, 299.29],
               [32.66, 24.31, 3.65, 76.73, 121.02, 271.34, 353.43, 471.78],
               [26.89, 41.78,  55.5, 71.37, 161.45, 250.76, 370.22, 549.21],
               ]
    draw_execution_time("tpch15_executiontime", trans_results_by_swirl(results), 0, 10, 650, 700)

    ###############################TPCHSkew-2######################################
    # results = [[274.93, 275.55, 273.65, 274.53,  275.73, 275.63,  275.46, 275.71],
    #            [0.1,  0.09, 0.06, 0.06, 0.08, 0.09,  0.1, 0.12],
    #            [3.39, 0.91, 2.75, 3.09, 3.17, 2.98, 3.9, 4.87],
    #            [1.52, 1.3,1.94,1.49,2.15,2.51,3.41,3.84],
    #            ]
    results = [[276.48, 275.56, 926.98, 1020.16,  1087.86, 2677.02,  4462.71, 9447.81],
               [524.25,  276.82, 275.72, 279.44, 277.6, 276.17,  277.22, 277.27],
               [33.76, 38.85, 52.62, 90.99, 141.25, 262.96, 495.21, 675.59],
               [ 51.59, 71.47, 81.93, 133.39, 250.33, 261.25, 472.59, 734.86],
               ]
    draw_execution_time("tpch2_executiontime",  trans_results_by_swirl(results), 0, 10, 650, 700)