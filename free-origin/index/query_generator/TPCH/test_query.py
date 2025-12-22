import sys
sys.path.append('../..')
import math
import random
import psycopg2

from rl_index_selection.index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector


if __name__ == "__main__":

    database_name = "indexselection_tpch___10"
    connector = PostgresDatabaseConnector(database_name, autocommit=True)
    statement = ("select count(*) from lineitem;")
    all_count = connector.exec_fetch(statement,False)[0][0]   # lineitem表的总记录数
    selectivity = [0.00001, 0.0001, 0.001, 0.01, 0.05]  # 5个选择度
    
    # 计算选择度浮动±0.1得到的记录数
    bound = []  
    for item in selectivity:
        bound.append((item, math.floor(all_count * item * 0.9), math.ceil(all_count * item * 1.1)))

    for i in range(len(selectivity)):
        print(f"selectivity={bound[i][0]}")
        # 计数器
        query_path = "query_1_{:.5f}.txt".format(selectivity[i])
        with open(query_path, "r") as query_file:
            for line in query_file:
                query = line.strip()
                tmp_count = connector.exec_fetch(query,False)[0][0] 
                flag = bound[i][1] <= tmp_count <= bound[i][2]
                print(f"flag={flag}, upper_bound={bound[i][1]}, ceil_bound={bound[i][2]}, tmp_count={tmp_count}")
        print("------------------------------------------------------------------------")