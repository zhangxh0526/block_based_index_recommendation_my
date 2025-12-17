import copy
import itertools
import os
import functools

from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.index import Index

from calendar import month
from cmath import inf
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

def compare_indexes(x, y):
    return 0

# Todo: This could be improved by passing index candidates as input
def predict_index_sizes(column_combinations, database_name, partition_num):
    """
    预测给定列组合的索引大小。

    :param column_combinations: 列组合的列表，每个组合是一个列对象的元组
    :param database_name: 数据库名称
    :param partition_num: 分区数量
    :return: 包含每个列组合预测索引大小的列表
    """
    # 创建PostgreSQL数据库连接器，自动提交事务
    connector = PostgresDatabaseConnector(database_name, autocommit=True)
    # 删除数据库中的所有索引
    connector.drop_indexes()

    # 创建成本评估对象，用于评估索引的成本
    cost_evaluation = CostEvaluation(connector)

    # 存储每个列组合的预测索引大小
    predicted_index_sizes = []

    # 存储每个列组合的完整索引大小
    parent_index_size_map = {}

    print("Start obtaining index sizes...")
    # 检查是否之前已经生成过索引大小信息
    file_name = "../experiments/back_up_index_size/{}-{}".format(database_name, partition_num)
    # 存储从备份文件中读取的索引大小信息
    index_size_bak = {}
    # 如果备份文件存在
    if os.path.exists(file_name):
        # 以读写模式打开备份文件
        index_size_f = open(file_name, "r+", encoding="utf-8")
        # 逐行读取备份文件
        for line in index_size_f:
            # 解析索引名称和索引大小
            index_name = line.strip().split("|")[0]
            index_size = float(line.strip().split("|")[1])
            # 将索引名称和索引大小存储到字典中
            index_size_bak[str(index_name)] = index_size
        # 关闭备份文件
        index_size_f.close()
    else:
        # 如果备份文件不存在，则创建该文件
        os.mknod(file_name)

    # 以追加模式打开备份文件
    index_size_f = open(file_name, "a", encoding="utf-8")
    # 遍历每个列组合
    for column_combination in column_combinations:
        # 如果该列组合的索引大小已经在备份文件中
        if str(column_combination) in index_size_bak.keys():
            # 从备份文件中获取该列组合的完整索引大小
            full_index_size = float(index_size_bak[str(column_combination)])
            # 将该列组合的索引大小添加到预测索引大小列表中
            predicted_index_sizes.append(full_index_size)
            # 将该列组合的完整索引大小存储到父索引大小映射中
            parent_index_size_map[column_combination] = full_index_size
        else:
            # 创建一个潜在的索引对象
            potential_index = Index(column_combination)
            # 改动
            cost_evaluation.what_if.simulate_index(potential_index, True)
            # 在数据库中创建该索引
            # connector.create_index(potential_index)
            # 获取该索引的估计大小
            full_index_size = potential_index.estimated_size
            # 索引增量大小初始化为完整索引大小
            index_delta_size = full_index_size
            # 改动 ,greenplum这样获得的复合索引带下不同减去之前的
            # if len(column_combination) > 1:
            #     index_delta_size -= parent_index_size_map[column_combination[:-1]]
            # 将索引增量大小添加到预测索引大小列表中
            predicted_index_sizes.append(index_delta_size)
            # 改动
            cost_evaluation.what_if.drop_simulated_index(potential_index)
            # 在数据库中删除该索引
            # connector.drop_index(potential_index)
            # 将该列组合的完整索引大小存储到父索引大小映射中
            parent_index_size_map[column_combination] = full_index_size
            # 将该列组合的索引名称和索引大小写入备份文件
            index_size_f.write("{}|{}\n".format(str(column_combination), full_index_size))
            print("Finish obtaining index size for index {}".format(column_combination))
            print("Already generating {} indexes".format(len(parent_index_size_map)))

    # 关闭备份文件
    index_size_f.close()
    # 返回预测索引大小列表
    return predicted_index_sizes, parent_index_size_map



def create_column_permutation_indexes(columns, max_index_width):
    """
    生成指定列集合的所有可能的列组合索引，索引宽度从1到最大索引宽度。

    :param columns: 列对象的列表，用于生成组合索引
    :param max_index_width: 最大索引宽度，即组合索引中列的最大数量
    :return: 包含不同宽度列组合索引的列表，每个子列表对应一个特定宽度的索引组合
    """
    result_column_combinations = []

    # 创建一个字典，用于按表对列进行分组
    table_column_dict = {}
    for column in columns:
        # 如果表不在字典中，初始化一个空集合
        if column.table not in table_column_dict:
            table_column_dict[column.table] = set()
        # 将列添加到对应表的集合中
        table_column_dict[column.table].add(column)

    # 遍历从1到最大索引宽度的每个长度
    for length in range(1, max_index_width + 1):
        # 用于存储唯一的列组合
        unique = set()
        # 记录当前长度的索引组合数量
        count = 0
        # 遍历每个表及其对应的列集合
        for key, columns_per_table in table_column_dict.items():
            # 生成当前表中列的指定长度的所有排列组合
            permutations = set(itertools.permutations(columns_per_table, length))
            # 将生成的排列组合添加到唯一集合中
            unique |= permutations
            # 累加当前表生成的排列组合数量
            count += len(permutations)
        # 打印当前长度的索引组合数量
        print(f"{length}-column indexes: {count}")

        # 将当前长度的唯一列组合添加到结果列表中
        result_column_combinations.append(list(unique))

    return result_column_combinations






# 输出日期计算结果，并以整数的形式返回
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


def sort_columns(columns):
    _column_dict = {}
    for column in columns:
        _column_dict[str(column)] = column
    _sorted_columns = []
    column_names = list(_column_dict.keys())
    column_names.sort()
    for column_name in column_names:
        _sorted_columns.append(_column_dict[column_name])
    return _sorted_columns


def classfy_colCombines_by_partition(columns):
    """
    根据分区 ID 对列组合进行分类。

    :param columns: 列组合的列表，每个元素是一个列组合的列表
    :return: 分类后的列组合列表，每个元素是一个字典，键为分区 ID，值为该分区的列组合列表
    """
    result_columns = []
    # 遍历每一组列组合
    for n, n_column_combinations in enumerate(columns):
        # 用于存储当前组按分区 ID 分类的列组合
        _columnCombine_By_Partition = {}
        # 遍历当前组的每个列组合
        for _ in n_column_combinations:
            # 从列组合的第一个列的表名中提取分区 ID
            partition_id = int(_[0].table.name.split("_")[-1].replace("p", ""))
            # 如果分区 ID 不在字典中，初始化一个空列表
            if partition_id not in _columnCombine_By_Partition:
                _columnCombine_By_Partition[partition_id] = []
            # 将当前列组合添加到对应分区 ID 的列表中
            _columnCombine_By_Partition[partition_id].append(_)
        # 将当前组按分区 ID 分类的列组合添加到结果列表中
        result_columns.append(copy.deepcopy(_columnCombine_By_Partition))
    # # 打印分类后的列组合列表
    # print(result_columns)
    # # 遍历分类后的列组合列表并打印每个元素
    # for _ in result_columns:
    #     print(_)
    return result_columns

