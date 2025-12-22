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
    all_count = connector.exec_fetch(statement, False)[0][0]  # lineitem表的总记录数
    selectivity = [0.00001, 0.0001, 0.001, 0.01, 0.05]  # 5个选择度
    _columns = ["l_shipdate", "l_orderkey", "l_suppkey", "l_discount", "l_quantity", "l_commitdate", "l_partkey", "l_receiptdate"]

    # 计算选择度浮动±0.1得到的记录数
    bound = []
    queries = []
    for item in selectivity:
        bound.append((item, math.floor(all_count * item * 0.9), math.ceil(all_count * item * 1.1)))
        queries.append([])

    # 查询模板
    cache_bound = {}

    random.seed(10)

    while True:
        try:
            count = 0
            # select column
            query_template = " select count(*) from lineitem where "
            columns = random.sample(_columns, 3)
            columns.sort()
            _col_i = 0
            for col in columns:
                if "date" in col:
                    month = random.randint(1, 12)
                    day = random.randint(1, 30)
                    if day < 10:
                        day = "0" + str(day)
                    if columns.index(col) != 0:
                        query_template += "and "
                    query_template += col
                    query_template += "  >= date \'{x"
                    query_template += str(_col_i)
                    query_template += "}"
                    query_template += "-{}-{}\' and ".format(month, day)
                    query_template += col
                    query_template += " < date \'{x"
                    query_template += str(_col_i + 1)
                    query_template += "}"
                    query_template += "-{}-{}\' ".format(month, day)
                    query_template += "+ interval \'{x"
                    query_template += str(_col_i + 2)
                    query_template += "}\' day "
                    _col_i += 3
                else:
                    if columns.index(col) != 0:
                        query_template += "and "
                    query_template += col
                    query_template += " between {x"
                    query_template += str(_col_i)
                    query_template += "} and {x"
                    query_template += str(_col_i + 1)
                    query_template += "} "
                    _col_i += 2
            # range for x1 - x6
            _ranges = []
            for col in columns:
                if "date" in col:
                    start = random.randint(1992, 1998)
                    end = start
                    day = random.randint(0, 365)
                    _ranges.extend([start, end, day])
                else:
                    if col in cache_bound:
                        upper_bound = cache_bound[col][0]
                        lower_bound = cache_bound[col][1]
                    else:
                        _query = "select max({}) from lineitem;".format(col)
                        upper_bound = float(connector.exec_fetch(_query, False)[0][0])
                        _query = "select min({}) from lineitem;".format(col)
                        lower_bound = float(connector.exec_fetch(_query, False)[0][0])
                        cache_bound[col] = [upper_bound, lower_bound]
                    if "key" in col:
                        start = random.randint(lower_bound, upper_bound)
                        end = random.randint(start, upper_bound)
                    else:
                        start = round(float(random.uniform(lower_bound, upper_bound)), 2)
                        end = round(float(random.uniform(start, upper_bound)), 2)
                    _ranges.extend([start, end])
            # 构建并执行查询
            if len(_ranges) == 6:
                query = query_template.format(x0=_ranges[0], x1=_ranges[1], x2=_ranges[2], x3=_ranges[3], x4=_ranges[4], x5=_ranges[5])
            elif len(_ranges) == 7:
                query = query_template.format(x0=_ranges[0], x1=_ranges[1], x2=_ranges[2], x3=_ranges[3],
                                              x4=_ranges[4], x5=_ranges[5], x6=_ranges[6])
            elif len(_ranges) == 8:
                query = query_template.format(x0=_ranges[0], x1=_ranges[1], x2=_ranges[2], x3=_ranges[3],
                                          x4=_ranges[4], x5=_ranges[5], x6=_ranges[6], x7=_ranges[7])
            elif len(_ranges) == 9:
                query = query_template.format(x0=_ranges[0], x1=_ranges[1], x2=_ranges[2], x3=_ranges[3],
                                          x4=_ranges[4], x5=_ranges[5], x6=_ranges[6], x7=_ranges[7], x8=_ranges[8])

            print("Check query is {}".format(query))

            tmp_count = connector.exec_fetch(query, False)[0][0]

            #print("Query: {}, Count: {}".format(query, tmp_count))

            # 查找满足范围
            for _ in bound:
                if _[1] <= tmp_count <= _[2]:
                    if len(queries[bound.index(_)]) >= 30:
                        continue
                    query_path = "query_1_{:.5f}_tyl.txt".format(_[0])
                    # print(query)
                    with open(query_path, "a+") as query_file:
                        print("Query: {}".format(query))
                        query_file.write(
                            query.replace("count(*)", "sum(l_quantity * l_discount) as revenue") + ";\n")
                        print(
                            f"Found parameters (x0={_ranges[0]}, x1={_ranges[1]}, x2={_ranges[2]}, x3={_ranges[3]}, x4={_ranges[4]}, x5={_ranges[5]}) with tmp_count: {tmp_count}")

                    queries[bound.index(_)].append(query)
                    break

            # 是否足够
            all_satisfy = True
            for _ in queries:
                if len(_) < 30:
                    all_satisfy = False
                    break
            if all_satisfy:
                print("Generate all queries!")
                break
        except:
            pass





