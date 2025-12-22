import importlib
import logging

from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.table_generator import TableGenerator


class Schema(object):
    """
    Schema 类用于构建数据库模式，包含表和列的信息，并可根据过滤器对列进行筛选。

    :param benchmark_name: 基准测试的名称
    :param scale_factor: 规模因子
    :param filters: 过滤器字典，默认为空字典
    :param partition_num: 分区数量，默认为 0
    :param used_tables: 使用的表列表，默认为空列表
    :param config: 配置信息，默认为 None
    """
    def __init__(self, benchmark_name, scale_factor, filters={}, partition_num=0, used_tables=[], config=None):
        # 创建一个 PostgresDatabaseConnector 实例，用于连接 PostgreSQL 数据库
        generating_connector = PostgresDatabaseConnector(None, autocommit=True)
        # 将分区数量赋值给类的属性
        self.partition_num = partition_num
        # 将使用的表列表赋值给类的属性
        self.used_tables = used_tables
        # 将配置信息赋值给类的属性
        self.config = config
        # 确保分区数量大于 0，否则抛出异常
        assert self.partition_num > 0, "In Partition envs, you need specify the number of block"
        # 创建一个 TableGenerator 实例，用于生成表
        table_generator = TableGenerator(
            benchmark_name=benchmark_name.lower(), scale_factor=scale_factor, database_connector=generating_connector,partition_num = self.partition_num, config=self.config
        )
        #assert ()
        # 获取数据库名称
        self.database_name = table_generator.database_name()
        # 获取表信息
        self.tables = table_generator.tables
        # 获取所有表的信息
        self.total_tables = table_generator.total_tables
        # 再次将分区数量赋值给类的属性
        self.partition_num = partition_num
        # 再次确保分区数量大于 0，否则抛出异常
        assert self.partition_num > 0, "In Partition envs, you need specify the number of block"
        # 初始化存储表列信息的列表
        self.columns = []
        # 初始化存储所有表列信息的列表
        self.total_columns = []
        # 遍历表信息，将表中的列添加到 columns 列表中
        for table in self.tables:
            for column in table.columns:
                self.columns.append(column)
        # 遍历所有表的信息，将表中的列添加到 total_columns 列表中
        for total_table in self.total_tables:
            for column in total_table.columns:
                self.total_columns.append(column)

        # 遍历过滤器字典的键
        for filter_name in filters.keys():
            # 动态导入 swirl.schema 模块，并获取过滤器类
            filter_class = getattr(importlib.import_module("swirl.schema"), filter_name)
            # 创建过滤器实例
            filter_instance = filter_class(filters[filter_name], self.database_name, self.used_tables)
            # 调用过滤器实例的 apply_filter 方法对列进行筛选，并更新 columns 列表
            self.columns = filter_instance.apply_filter(self.columns, self.total_columns)
        # 记录模式构建完成的信息
        logging.info("Schema build Finished")



class TableNumRowsFilter(object):
    """
    TableNumRowsFilter 类用于根据表的行数对列进行过滤。

    :param threshold: 行数阈值，用于判断表是否满足过滤条件
    :param database_name: 数据库名称
    :param used_tables: 使用的表列表
    """
    def __init__(self, threshold, database_name, used_tables):
        # 初始化行数阈值
        self.threshold = threshold
        # 初始化使用的表列表
        self.used_tables = used_tables
        # 创建一个 PostgresDatabaseConnector 实例，用于连接数据库
        self.connector = PostgresDatabaseConnector(database_name, autocommit=True)
        # 创建数据库统计信息
        self.connector.create_statistics()

    def _check_table_satisfy(self, table_name, satisfy_tables):
        """
        检查给定的表名是否满足过滤条件。

        :param table_name: 要检查的表名
        :param satisfy_tables: 满足过滤条件的表名集合
        :return: 如果表名满足条件返回 True，否则返回 False
        """
        for _ in satisfy_tables:
            if _ in table_name:
                return True
        return False

    def apply_filter(self, columns, total_columns):
        """
        应用过滤器，根据表的行数过滤列。

        :param columns: 要过滤的列列表
        :param total_columns: 所有列的列表
        :return: 过滤后的列列表
        """
        # 初始化过滤后的列列表
        output_columns = []
        # 初始化满足过滤条件的表名集合
        filtered_tables = set()
        # 遍历所有列
        for column in total_columns:
            # 获取列所属的表名
            table_name = column.table.name
            # 如果表名不在使用的表列表中，则跳过
            if table_name not in self.used_tables:
                continue
            # 执行 SQL 查询，获取表的估计行数
            table_num_rows = self.connector.exec_fetch(
                f"SELECT reltuples::bigint AS estimate FROM pg_class where relname='{table_name}_bak'"
            )[0]
            # 如果表的行数大于阈值，则将表名添加到满足过滤条件的表名集合中
            if table_num_rows > self.threshold:
                filtered_tables.add(table_name)
                logging.info(f"filtered_table : {table_name}")
        # 遍历要过滤的列
        for column in columns:
            # 获取列所属的表名
            table_name = column.table.name
            # 检查表名是否满足过滤条件
            if self._check_table_satisfy(table_name, filtered_tables):
                # 如果满足条件，则将列添加到过滤后的列列表中
                output_columns.append(column)

        # 记录过滤前后列的数量变化信息
        logging.warning(f"Reduced columns from {len(columns)} to {len(output_columns)}.")

        return output_columns


