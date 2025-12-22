from cmath import inf
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import itertools
import os

def modify_date(date_string):
    # 将原始日期转换为datetime对象
    parts = date_string.split("'")
    original_date = parts[1]
    original_date = datetime.strptime(original_date, "%Y-%m-%d")
    if 'interval' not in date_string:
        return int(original_date.strftime('%Y%m%d'))
    op = parts[2].strip()[0]
    num = int(parts[3].split()[0])
    unit = parts[4].split()[0]

    # 创建一个relativedelta对象用于日期计算
    delta = relativedelta()

    # 根据操作符和单位设置日期计算的参数
    if op == "-":
        num = -num
    if unit == "year":
        delta.years = num
    elif unit == "month":
        delta.months = int(num % 12)
        delta.years = int(num / 12)
    elif unit == "day":
        delta.days = num

    new_date = original_date + delta
    new_date_integer = int(new_date.strftime("%Y%m%d"))
    return new_date_integer


# 输出数字计算结果，并以浮点数的形式返回
def modify_num(num_string):
    parts = num_string.split(" ")
    num_1 = float(parts[0])
    if '+' in num_string:
        num_2 = float(parts[2])
        return num_1 + num_2
    elif '-' in num_string and num_string[0] != '-':
        num_2 = float(parts[2])
        return num_1 - num_2
    else:
        return num_1

def _is_predicate_number(value):
    if value[0].isdigit():
        return True
    if value[0] == "-" and value[1].isdigit():
        return True
    return False

def predicate_splitting(workload):
    workload_inf = []

    for query in workload:
        query_inf = {}
        where_clauses = re.findall(r'where(.*?);', query, flags=re.IGNORECASE)
        for where_clause in where_clauses:

            conditions = where_clause.split('and')
            column = None
            for condition in conditions:
                lower_bound = None
                upper_bound = None

                if 'select' in condition:
                    continue

                # 处理下界
                if '>' in condition or 'between' in condition:
                    if '>=' in condition:
                        column, lower_bound = condition.split('>=')
                    elif '>' in condition:
                        column, lower_bound = condition.split('>')
                    else:
                        column, lower_bound = condition.split('between')
                    column = column.strip().split()[-1]
                    if column not in query_inf:
                        query_inf[column] = [-inf, inf]
                    lower_bound = lower_bound.strip()
                    if 'date' in lower_bound:
                        lower_bound = modify_date(lower_bound)
                    elif _is_predicate_number(lower_bound):
                        lower_bound = modify_num(lower_bound)
                    else:
                        lower_bound = lower_bound.split()[0]
                    query_inf[column][0] = lower_bound

                # 处理上界
                elif '<' in condition or _is_predicate_number(condition.strip()):
                    if '<=' in condition:
                        column, upper_bound = condition.split('<=')
                    elif '<' in condition:
                        column, upper_bound = condition.split('<')
                    else:
                        upper_bound = condition.strip()
                    column = column.strip().split()[-1]
                    if column not in query_inf:
                        query_inf[column] = [-inf, inf]
                    upper_bound = upper_bound.strip()
                    if 'date' in upper_bound:
                        upper_bound = modify_date(upper_bound)
                    elif _is_predicate_number(upper_bound):
                        upper_bound = modify_num(upper_bound)
                    else:
                        upper_bound = upper_bound.split()[0]
                    query_inf[column][1] = upper_bound

                elif '=' in condition:
                    column, lower_bound = condition.split('=')
                    column = column.strip().split()[-1]
                    if 'date' in lower_bound:
                        lower_bound = modify_date(lower_bound)
                    elif _is_predicate_number(lower_bound):
                        lower_bound = modify_num(lower_bound)
                    else:
                        lower_bound = lower_bound.split()[0]
                    query_inf[column] = [lower_bound, lower_bound]

        workload_inf.append(query_inf)

    return workload_inf

query1 = " select sum(ss_sales_price * ss_ext_discount_amt) as revenue from store_sales where  ss_list_price between 75.02 and 112.11 and ss_hdemo_sk between 3715.52 and 6930.01 and ss_net_profit between -8181.14 and -1084.41 ;"
query2 = "select sum(ss_sales_price * ss_ext_discount_amt) as revenue from store_sales where ss_customer_sk between 100060.23 and 294492.69 and ss_item_sk between 21163.7 and 51021.38 and ss_net_profit between -430.22 and 4070.22 ;"
query3 = " select sum(ss_sales_price * ss_ext_discount_amt) as revenue from store_sales where ss_hdemo_sk between 2830.45 and 4118.68 and ss_item_sk between 44355.71 and 75824.54 and ss_net_profit between -3777.71 and 3582.23 ;"

value = "-1084.41"
print(value[0] == "-")
print(value[1].isdigit())


predicate_splitting([query1, query2, query3])