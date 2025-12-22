import sys

sys.path.append('../..')
import math
import random
import psycopg2

from rl_index_selection.index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector

scale = 50

table = "store_sales"

pre_query = "ss_sales_price * ss_ext_discount_amt"

if __name__ == "__main__":

    database_name = "indexselection_tpcds___10"
    connector = PostgresDatabaseConnector(database_name, autocommit=True)
    statement = ("select count(*) from {};".format(table))
    all_count = connector.exec_fetch(statement, False)[0][0]  # lineitem表的总记录数
    print("Total number of records is {}".format(all_count))
    selectivity = [0.00001, 0.0001, 0.001, 0.01, 0.05]  # 5个选择度
    _columns = ["ss_item_sk", "ss_customer_sk", "ss_cdemo_sk", "ss_hdemo_sk",
                "ss_ticket_number", "ss_wholesale_cost", " ss_list_price", "ss_net_profit"]
    # frequently used colums
    # _web_sales = ["ss_item_sk", "ss_sold_date_sk", "ss_store_sk", "ss_quantity", "ss_customer_sk", "ss_sales_price",
    #             "ss_list_price", "ss_ticket_number", "ss_ext_sales_price"]
    # _columns = ["cs_sold_date_sk", "cs_ship_date_sk", "cs_sold_time_sk", "cs_bill_customer_sk", "cs_order_number", "cs_quantity",
    #             "cs_warehouse_sk", "cs_item_sk"]



    # 计算选择度浮动±0.1得到的记录数
    bound = []
    queries = []
    for item in selectivity:
        bound.append((item, math.floor(all_count * item * 0.9), math.ceil(all_count * item * 1.1)))
        queries.append([])

    print("Bounds are {}".format(bound))

    # 查询模板
    cache_bound = {}
    random.seed(10)

    while True:
        # try:
        count = 0
        # select column
        query_template = " select count(*) from {} where ".format(table)
        columns = random.sample(_columns, 3)
        columns.sort()
        _col_i = 0
        for col in columns:
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
            if col in cache_bound:
                upper_bound = cache_bound[col][0]
                lower_bound = cache_bound[col][1]
            else:
                _query = "select max({}) from {};".format(col, table)
                upper_bound = float(connector.exec_fetch(_query, False)[0][0])
                _query = "select min({}) from {};".format(col, table)
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
        assert len(_ranges) == 6
        query = query_template.format(x0=_ranges[0], x1=_ranges[1], x2=_ranges[2], x3=_ranges[3], x4=_ranges[4], x5=_ranges[5])
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
                        query.replace("count(*)", "sum({}) as revenue".format(pre_query)) + ";\n")
                    print(
                        f"Found parameters (x0={_ranges[0]}, x1={_ranges[1]}, x2={_ranges[2]}, x3={_ranges[3]}, x4={_ranges[4]}, x5={_ranges[5]}) with tmp_count: {tmp_count}")
                queries[bound.index(_)].append(query)
                break

        # 是否足够
        # all_satisfy = True
        # for _ in queries:
        #     if len(_) < 30:
        #         all_satisfy = False
        #         break
        # if all_satisfy:
        #     print("Generate all queries!")
        #         break
        # except:
        #     pass





