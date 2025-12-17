import sys
sys.path.append('../..')
import math
import random
import psycopg2

from rl_index_selection.index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector


if __name__ == "__main__":

    database_name = "indexselection_tpch___0_01"
    connector = PostgresDatabaseConnector(database_name, autocommit=True)
    statement = ("select count(*) from lineitem;")
    all_count = connector.exec_fetch(statement,False)[0][0]   # lineitem表的总记录数
    selectivity = [0.00001, 0.0001, 0.001, 0.01, 0.05]  # 5个选择度
    
    # 计算选择度浮动±0.1得到的记录数
    bound = []  
    for item in selectivity:
        bound.append((item, math.floor(all_count * item * 0.9), math.ceil(all_count * item * 1.1)))

    # 查询模板
    query_template = "select count(*) from lineitem where l_receiptdate >= date '{x1}-01-01' and l_receiptdate < date '{x2}-01-01' + interval '{x3}' day and l_tax between {x4} and {x5} and l_extendedprice < {x6};"

    for i in range(len(selectivity)):
        # 计数器
        query_path = "query_1_{:.5f}.txt".format(selectivity[i])
        with open(query_path, "w") as query_file:
            count = 0
            while count < 10:
                # 生成随机参数值
                x1 = random.randint(1992, 1998)
                x2 = random.randint(x1, 1998)
                x3 = random.randint(0, 30)
                x4 = round(random.uniform(0, 0.1), 2)
                x5 = round(random.uniform(x4, 0.1), 2)
                x6 = random.randint(900, 90000)
                
                # 构建并执行查询
                query = query_template.format(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6)
                tmp_count = connector.exec_fetch(query,False)[0][0]   
                
                # 检查是否在指定范围内
                if bound[i][1] <= tmp_count <= bound[i][2]:
                    query_file.write(query.replace("count(*)", "sum(l_extendedprice * l_discount) as revenue") + "\n")
                    print(f"Found parameters (x1={x1}, x2={x2}, x3={x3}, x4={x4}, x5={x5}, x6={x6}) with tmp_count: {tmp_count}")
                    count += 1

