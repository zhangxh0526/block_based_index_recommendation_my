import copy
import os
import json
import re
import psycopg2
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
import datetime
import decimal

# todo: assume only one table, can be more general
class HistogramsManager(object):
    """
    直方图管理器类，用于管理数据库中表和分区的直方图信息。

    参数:
    - indexed_columns (dict): 包含索引列的字典，格式为 {table_name: [column1, column2, ...]}
    - database_name (str): 数据库名称
    - histgrams_num (int, 可选): 直方图的数量，默认为100
    - query_limit (int, 可选): 查询的最大数量，默认为3
    - partition_num (int, 可选): 分区的数量，默认为2
    """
    #{"1":{columna:[], columnb:[]....}}
    def __init__(self, indexed_columns, database_name, histgrams_num=100, query_limit=3, partition_num=2):
        # 初始化直方图数量
        self.histgrams_num = histgrams_num
        # 初始化查询限制
        self.query_limit = query_limit
        # 初始化索引列
        self.indexed_columns = indexed_columns
        # 初始化数据库名称
        self.database_name = database_name
        # 初始化数据库连接器
        self.connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
        # 初始化备份直方图的路径
        self.bak_histograms_path = "../experiments/histograms-{}-{}-{}".format(database_name, histgrams_num, len(indexed_columns))
        #{partition_id:{cloumn a: [value1, value2]}}
        # 初始化每个分区的记录
        self.records_per_partition = None
        # 检查备份文件是否存在
        if os.path.exists(self.bak_histograms_path):
            # 如果存在，初始化数据
            self.init(self.records_per_partition)
        else:
            # 如果不存在，获取每个分区的记录
            self.records_per_partition = self.get_records_per_partition(self.connector)
            # 初始化数据
            self.init(self.records_per_partition)
        # 初始化缓存的直方图
        self.cache_histograms = {}
        # 重置每个分区的记录
        self.records_per_partition = None

    def init(self, records_per_partition):
        """
        初始化直方图数据，如果备份文件存在则从文件中加载，否则生成新的直方图。

        参数:
        - records_per_partition (dict): 每个分区的记录
        """
        # 检查备份文件是否存在
        if os.path.exists(self.bak_histograms_path):
            # 从文件中加载分区信息
            with open(self.bak_histograms_path + "/partitions.json", 'r') as f:
                self.partitions = json.loads(json.load(f))
            # 从文件中加载列直方图信息
            with open(self.bak_histograms_path + "/column_histgrams.json", 'r') as f:
                self.column_histgrams = json.loads(json.load(f))
            # 从文件中加载分区直方图信息
            with open(self.bak_histograms_path + "/partition_histgrams.json", 'r') as f:
                self.partition_histgrams = json.loads(json.load(f))
        else:
            # 创建备份目录
            os.mkdir(self.bak_histograms_path)
            # 初始化缓存的直方图
            self.cache_histograms = {}
            # 生成列直方图
            self.generate_histgrams(records_per_partition)
            # 生成每个分区的直方图
            self.generate_histgrams_per_partition(records_per_partition)
            # 将分区信息写入文件
            with open(self.bak_histograms_path + "/partitions.json", 'w') as f:
                json.dump(json.dumps(self.partitions), f)
            # 将列直方图信息写入文件
            with open(self.bak_histograms_path + "/column_histgrams.json", 'w') as f:
                json.dump(json.dumps(self.column_histgrams), f)
            # 将分区直方图信息写入文件
            with open(self.bak_histograms_path + "/partition_histgrams.json", 'w') as f:
                json.dump(json.dumps(self.partition_histgrams), f)

    def _compare_partition_column(self, records_per_partition):
        """
        比较每个分区中列的值范围。

        参数:
        - records_per_partition (dict): 每个分区的记录

        返回:
        - dict: 每个表中列的值范围
        """
        # 初始化每个分区的列信息
        _column_per_partition = {}
        # 遍历每个表
        for table in records_per_partition:
            # 初始化表的列信息
            _column_per_partition[table] = {}
            # 遍历每个分区
            for partition in records_per_partition[table]:
                # 遍历每个列
                for col in records_per_partition[table][partition]:
                    # 如果列不在列信息中，初始化列信息
                    if col not in _column_per_partition[table]:
                        _column_per_partition[table][col] = []
                    # 添加列的值范围
                    _column_per_partition[table][col].append((min(records_per_partition[table][partition][col]), max(records_per_partition[table][partition][col])))
        return _column_per_partition

    def generate_histgrams(self, records_per_partition):
        """
        生成列直方图。

        参数:
        - records_per_partition (dict): 每个分区的记录
        """
        # 初始化所有记录
        all_records = {}
        # 初始化分区信息
        self.partitions = {}
        # 遍历每个表
        for table in records_per_partition.keys():
            # 初始化表的分区信息
            self.partitions[table] = list()
            # 初始化表的所有记录
            all_records[table] = {}
            # 遍历每个分区
            for partition in records_per_partition[table].keys():
                # 添加分区信息
                self.partitions[table].append(partition)
                # 遍历每个列
                for column in records_per_partition[table][partition]:
                    # 如果列不在所有记录中，初始化列的所有记录
                    if column not in all_records[table]:
                        all_records[table][column] = []
                    # 添加列的所有记录
                    for key in records_per_partition[table][partition][column]:
                        all_records[table][column].append(key)
            # 对分区信息进行排序
            self.partitions[table].sort()

        # 比较每个分区中列的值范围
        #self._compare_partition_column(records_per_partition)

        # 初始化列直方图
        self.column_histgrams = {}
        # 遍历每个索引列
        for table in self.indexed_columns:
            # 初始化表的列直方图
            self.column_histgrams[table] = {}
            # 遍历每个索引列
            for column in self.indexed_columns[table]:
                # 对列的所有记录进行排序
                all_records[table][column].sort()
                # 计算每个直方图的大小
                histgram_size = len(all_records[table][column]) / self.histgrams_num
                # 初始化直方图
                histgrams = []
                # 遍历每个直方图
                for i in range(0, self.histgrams_num):
                    # 计算直方图的起始位置
                    start = int(i * histgram_size)
                    # 计算直方图的结束位置
                    end = int((i+1) * histgram_size)
                    # 如果起始位置和结束位置相同，添加单个值
                    if start == end:
                        histgrams.append([all_records[table][column][start]])
                    else:
                        # 否则，添加值范围
                        histgrams.append(all_records[table][column][start:end])
                # 计算每个直方图的起始和结束值
                self.column_histgrams[table][column] = [[histgram[0], histgram[-1]] for histgram in histgrams]

    def generate_histgrams_per_partition(self, records_per_partition):
        """
        生成每个分区的直方图。

        参数:
        - records_per_partition (dict): 每个分区的记录
        """
        # 初始化分区直方图
        self.partition_histgrams = {}
        # 打印开始生成直方图的信息
        print("Start generating histograms ...")
        # 遍历每个索引列
        for table in self.indexed_columns.keys():
            # 初始化表的分区直方图
            self.partition_histgrams[table] = {}
            # 遍历每个分区
            for partition in records_per_partition[table].keys():
                # 初始化分区的列直方图
                self.partition_histgrams[table][partition] = {}
                # 遍历每个索引列
                for column in self.indexed_columns[table]:
                    # 打印正在生成直方图的信息
                    print("Generating histograms for table {} partition {} column {}...".format(table, partition, column))
                    # 初始化分区的列直方图
                    self.partition_histgrams[table][partition][column] = [0 for _ in range(self.histgrams_num)]
                    # 获取列的唯一值
                    records = list(set(records_per_partition[table][partition][column]))
                    # 对列的唯一值进行排序
                    records.sort()
                    # 复制列的直方图
                    buckets = copy.deepcopy(self.column_histgrams[table][column])
                    # 遍历每个唯一值
                    for r in records:
                        # 标记是否已经找到
                        already_finded = False
                        # 遍历每个直方图
                        for _ in buckets:
                            # 如果唯一值在直方图范围内
                            if r >= _[0] and r <= _[1]:
                                # 标记已经找到
                                already_finded = True
                                # 更新分区的列直方图
                                self.partition_histgrams[table][partition][column][self.column_histgrams[table][column].index(_)] = 1
                            # 如果唯一值小于直方图的起始值
                            elif r < _[0]:
                                # 移除冗余的直方图
                                buckets = self.remove_redundant_buckets(buckets, r)
                                break
                            # 如果唯一值大于直方图的结束值且已经找到
                            elif r > _[1] and already_finded:
                                # 移除冗余的直方图
                                buckets = self.remove_redundant_buckets(buckets, r)
                                break
        # 打印成功生成直方图的信息
        print("Successfully generating histograms!")

    def columns_per_histogram(self):
        """
        计算每个直方图的列数。

        返回:
        - int: 每个直方图的列数
        """
        # 初始化列数
        column_num = 0
        # 遍历每个索引列
        for table in self.indexed_columns:
            # 遍历每个索引列
            for column in self.indexed_columns[table]:
                # 增加列数
                column_num += 1
        return column_num

    def histogram_num(self):
        """
        计算直方图的总数。

        返回:
        - int: 直方图的总数
        """
        # 初始化直方图数量
        histogram_num = 0
        # 遍历每个分区直方图
        for table in self.partition_histgrams:
            # 遍历每个分区
            for partition in self.partition_histgrams[table]:
                # 增加直方图数量
                histogram_num += 1
        return histogram_num

    # workload_inf:[query1_inf, query2_inf]
    # query1_inf:{table:{column:[start, end]...}}
    # select * from lineitem where a > 1 {a:[1, inf], [1, 1]} env->agent env : workload[query1, query2....]
    def block_based_workload_histgrams(self, workload_inf):
        """
        基于块的工作负载直方图。

        参数:
        - workload_inf (list): 工作负载信息

        返回:
        - list: 基于块的工作负载直方图
        """
        # 打印工作负载信息
        #print("Workload inf:{}".format(workload_inf))
        # 如果工作负载信息已经缓存，返回缓存的结果
        # if str(workload_inf) in self.cache_histograms:
        #     return self.cache_histograms[str(workload_inf)]
        # else:
        # 初始化查询直方图
        queries_histgrams = []
        # 遍历每个查询信息
        for query_inf in workload_inf:
            # 生成查询直方图
            queries_histgrams.append(self.query_histgrams(query_inf))
        # 如果查询直方图的数量超过查询限制，截取前query_limit个
        if len(queries_histgrams) > self.query_limit:
            queries_histgrams = queries_histgrams[:self.query_limit]
        # 如果查询直方图的数量小于查询限制，补充空查询直方图
        elif len(queries_histgrams) < self.query_limit:
            for _ in range(self.query_limit - len(queries_histgrams)):
                queries_histgrams.append(self.empty_query_histgrams())

        # 初始化基于块的表示
        block_based_representation = []
        # 遍历每个索引列
        for table in self.indexed_columns:
            # 遍历每个分区
            for partition in self.partitions[table]:
                # 初始化分区表示
                partition_representation = []
                # 遍历每个查询直方图
                for query in queries_histgrams:
                    # 遍历每个索引列
                    for column in self.indexed_columns[table]:
                        # 初始化查询列分区直方图
                        query_column_partition_histgram = [0 for _ in range(self.histgrams_num)]
                        # 遍历每个直方图
                        for _ in range(self.histgrams_num):
                            # 如果分区直方图和查询直方图都为1
                            if self.partition_histgrams[table][partition][column][_] == 1 and query[column][_] == 1:
                                # 更新查询列分区直方图
                                query_column_partition_histgram[_] = 1
                        # 将查询列分区直方图转换为特征
                        partition_representation.extend(self.trans_histgrams_2_feature(query_column_partition_histgram, self.partition_histgrams[table][partition][column]))
                # 添加分区表示
                block_based_representation.append(copy.deepcopy(partition_representation))

            # 缓存工作负载信息
            #self.cache_histograms[str(workload_inf)] = copy.deepcopy(block_based_representation)
        return block_based_representation

    def query_histgrams(self, query_inf):
        """
        生成查询直方图。

        参数:
        - query_inf (dict): 查询信息

        返回:
        - dict: 查询直方图
        """
        # 初始化查询直方图
        query_histgrams = {}
        # 遍历每个索引列
        for table in self.indexed_columns:
            # 遍历每个索引列
            for column in self.indexed_columns[table]:
                # 如果列不在查询信息中，初始化查询直方图为0
                if column not in query_inf:
                    query_histgrams[column] = [0 for _ in range(self.histgrams_num)]
                else:
                    # 初始化临时查询直方图为0
                    temp_histgrams = [0 for _ in range(self.histgrams_num)]
                    # 打印列和工作负载信息
                    # print("Column : {}, workload_inf : {}".format(column, query_inf[column]))
                    # 打印直方图信息
                    # print("Histogram : {}".format(self.column_histgrams[table][column]))
                    # 遍历每个直方图
                    for _ in self.column_histgrams[table][column]:
                        # 如果直方图和查询范围有交集
                        if self.bucket_union(_, query_inf[column]):
                            # 更新临时查询直方图
                            temp_histgrams[self.column_histgrams[table][column].index(_)] = 1
                    # 复制临时查询直方图
                    query_histgrams[column] = copy.deepcopy(temp_histgrams)
                    # 打印查询直方图信息
                    # print("Query histogram : {}".format(temp_histgrams))
        return query_histgrams

    def empty_query_histgrams(self):
        """
        生成空的查询直方图。

        返回:
        - dict: 空的查询直方图
        """
        # 初始化查询直方图
        query_histgrams = {}
        # 遍历每个索引列
        for table in self.indexed_columns:
            # 遍历每个索引列
            for column in self.indexed_columns[table]:
                # 初始化查询直方图为0
                query_histgrams[column] = [0 for _ in range(self.histgrams_num)]
        return query_histgrams

    def bucket_union(self, range1, range2):
        """
        判断两个范围是否有交集。

        参数:
        - range1 (list): 第一个范围
        - range2 (list): 第二个范围

        返回:
        - bool: 如果有交集返回True，否则返回False
        """
        # 如果第一个范围的起始值大于第二个范围的结束值
        if range1[0] > range2[1]:
            return False
        # 如果第一个范围的结束值小于第二个范围的起始值
        if range1[1] < range2[0]:
            return False
        return True

    def trans_histgrams_2_feature(self, histgrams, block_histgrams):
        """
        将直方图转换为特征。

        参数:
        - histgrams (list): 直方图
        - block_histgrams (list): 块直方图

        返回:
        - list: 特征
        """
        # 初始化起始位置
        start = -1
        # 初始化结束位置
        end = -1
        # 初始化直方图数量
        histgram_num = 0
        # 遍历每个直方图
        for _ in range(len(histgrams)):
            # 如果直方图为1
            if histgrams[_] == 1:
                # 如果起始位置未初始化，初始化起始位置
                if start == -1:
                    start = _
                # 更新结束位置
                end = max(end, _)
                # 增加直方图数量
                histgram_num += 1

        # 如果块直方图的总和为0
        if sum(block_histgrams) == 0:
            return [max(start, 0), max(end, 0), 0]
        return [max(start, 0), max(end, 0), histgram_num/sum(block_histgrams)]


    def remove_redundant_buckets(self, buckets, record):
        """
        移除冗余的直方图。

        参数:
        - buckets (list): 直方图
        - record (int): 记录

        返回:
        - list: 移除冗余后的直方图
        """
        # 初始化保存索引
        save_index = -1
        # 遍历每个直方图
        for _ in buckets:
            # 如果直方图的结束值大于等于记录
            if _[1] >= record:
                # 保存索引
                save_index = buckets.index(_)
                break
        return buckets[save_index:]

    #从表名中提取分区id，例如 "customer_1_prt_p1" 分区为1
    def extract_partition_id(self,table_name):
        """
        从表名中提取分区ID。

        参数:
        - table_name (str): 表名

        返回:
        - tuple: (表名, 分区ID)，如果未匹配到则返回 (None, None)
        """
        # 定义正则表达式模式
        pattern = r'^(.*?)_\d+_prt_p(\d+)$'
        # 匹配表名
        match = re.match(pattern, table_name)
        # 如果匹配成功
        if match:
            # 获取表名
            table = match.group(1)
            # 获取分区ID
            partition_id = match.group(2)
            return table, partition_id
        else:
            return None, None

    #获取数据库中每个分区的信息
    def get_records_per_partition(self,connector):
        """
        获取数据库中每个分区的信息。

        参数:
        - connector (object): 数据库连接器

        返回:
        - dict: 每个分区的信息
        """
        # 获取数据库中所有表的信息
        statement = ("select table_name from information_schema.tables where table_schema = 'public';")
        tables = connector.exec_fetch(statement,False)

        # 遍历每个表并收集相关信息
        partition_info = {}

        # 打印生成直方图时有用的表
        print("Useful tables when generating histgrams : {}".format(self.indexed_columns.keys()))

        # 遍历每个表
        for table in tables:
            # 获取表名
            table_name = table[0]
            # 获取分区id
            table_pre, partition_id = self.extract_partition_id(table_name)
            # 如果表名不在索引列中，跳过
            if table_pre not in self.indexed_columns.keys():
                continue
            # 如果表名不在指定表中，跳过
            # if table_pre not in ["store_sales", "web_sales"]:
            #     continue
            # 如果分区id为空，跳过
            if partition_id is None:
                continue

            # 收集每个列的值
            statement = f"select column_name from information_schema.columns where table_name = '{table_name}';"
            columns = connector.exec_fetch(statement,False)
            column_values = {}
            # 遍历每个列
            for column in columns:
                # 获取列名
                column_name = column[0]
                # 查询列的唯一值
                statement = f"select {column_name} from {table_name} group by {column_name};"
                values = connector.exec_fetch(statement, False)
                # 如果查询结果不为空
                if len(values) > 0:
                    # 如果值为日期类型
                    if isinstance(values[0][0], datetime.date):
                        individual_keys = list(set([self.trans_datetime(value[0]) for value in values if value[0] is not None]))
                    # 如果值为小数类型
                    elif isinstance(values[0][0], decimal.Decimal):
                        individual_keys = list(set([float(value[0]) for value in values if value[0] is not None]))
                    else:
                        individual_keys = list(set([value[0] for value in values if value[0] is not None]))
                else:
                    individual_keys = list(set([value[0] for value in values if value[0] is not None]))
                # 对唯一值进行排序
                individual_keys.sort()
                # 保存列的唯一值
                column_values[column_name] = copy.deepcopy(individual_keys)

                # 打印列的基数
                print("Column {}'s cardinality is {}".format(column_name, len(individual_keys)))

            # 将分区id和对应的列值添加到字典中
            if table_pre not in partition_info:
                partition_info[table_pre] = {}
            partition_info[table_pre][partition_id] = column_values

        return partition_info

    def trans_datetime(self, time):
        """
        将日期时间转换为整数表示。

        参数:
        - time (datetime.date): 日期时间

        返回:
        - int: 整数表示的日期时间
        """
        # 获取年份
        year = time.year
        # 获取月份
        month = time.month
        # 获取日期
        day = time.day
        # 计算日期时间的整数表示
        time_result = year * 10000 + month * 100 + day
        return time_result


# records = {1:{"a":[_ for _ in range(3000)], "b":[int(_/100) for _ in range(5000)], "c":[1, 2, 3]}}
# indexed_columns = ["a", "b"]
#
# his = HistgramsManager(records, indexed_columns)
# his.block_based_workload_histgrams([{"a":[17, 30]}])