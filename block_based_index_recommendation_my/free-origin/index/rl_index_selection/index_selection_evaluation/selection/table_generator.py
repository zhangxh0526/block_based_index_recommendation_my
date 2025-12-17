import logging
import os
import platform
import re
import subprocess
import random
import math
import numpy as np

from datetime import timedelta
import datetime

from .utils import b_to_mb
from .workload import Column, Table


class TableGenerator:
    def __init__(
        self,
        benchmark_name,
        scale_factor,
        database_connector,
        partition_num = None,
        explicit_database_name=None,
        config=None,
    ):
        """
        初始化 TableGenerator 类的实例。

        :param benchmark_name: 基准测试的名称，如 "tpch", "tpcds" 等
        :param scale_factor: 数据生成的缩放因子
        :param database_connector: 数据库连接对象
        :param partition_num: 分区数量，默认为 None
        :param explicit_database_name: 显式指定的数据库名称，默认为 None
        :param config: 配置信息，默认为 None
        """
        self.partition_num = partition_num
        # 确保分区数量大于 0
        assert self.partition_num > 0, "In Partition envs, you need specify the number of block"
        self.scale_factor = scale_factor
        self.benchmark_name = benchmark_name
        self.db_connector = database_connector
        self.explicit_database_name = explicit_database_name
        # 存储表和列的数据类型信息
        self.table_types = {}   #table->column->datatype :numeric,date,character varying,character,integer
        # 获取数据库名称列表
        self.database_names = self.db_connector.database_names()
        # 存储分区表对象
        self.tables = []
        # 存储全量表对象
        self.total_tables = []
        # 存储列对象
        self.columns = []
        self.config = config
        # 准备工作，设置相关命令和路径
        self._prepare()
        # 如果指定名称的数据库不存在，则生成数据并创建数据库
        if self.database_name() not in self.database_names:
            self._generate()
            self.create_database()
        else:
            # 如果数据库已存在，则连接到该数据库
            self.db_connector.db_name = self.database_name()
            self.db_connector.create_connection()
            logging.debug("Database with given scale factor already " "existing")
        # 读取表和列的名称
        self._read_column_names()
        print("------------")

    def database_name(self):
        """
        生成数据库的名称。

        :return: 数据库的名称
        """
        if self.explicit_database_name:
            return self.explicit_database_name

        name = "indexselection_" + self.benchmark_name + "___"
        name += str(self.scale_factor).replace(".", "_")
        return name

    def _read_column_names(self):
        """
        从 'create table' 语句中读取表和列的名称，并创建 Table 和 Column 对象。
        """
        # 拼接 'create table' 语句文件的路径
        filename = self.directory + "/" + self.create_table_statements_file
        with open(filename, "r") as file:
            # 读取文件内容并转换为小写
            data = file.read().lower()
        # 分割 'create table' 语句
        create_tables = data.split("create table ")[1:]
        for create_table in create_tables:
            # 分割出表名和列定义部分
            splitted = create_table.split("(", 1)
            table_name = splitted[0].strip()
            # 构建 SQL 语句，查询表的列名和数据类型
            statement = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';"
            # 执行 SQL 查询，获取列名和数据类型列表
            colum_datatype_list = self.db_connector.exec_fetch(statement, False)
            # 初始化表的数据类型字典
            self.table_types[table_name] = {}
            # 处理分区表
            for partition_id in range(self.partition_num):
                # 创建分区表对象
                table = Table(table_name + f"_1_prt_p{partition_id}")
                self.tables.append(table)
                for (colum, datatype) in colum_datatype_list:
                    # 存储列的数据类型
                    self.table_types[table_name][colum] = datatype
                    # 创建列对象
                    column_object = Column(colum)
                    # 将列添加到表中
                    table.add_column(column_object)
                    # 存储列对象
                    self.columns.append(column_object)
            # 处理全量表
            table = Table(table_name)
            self.total_tables.append(table)
            for (colum, datatype) in colum_datatype_list:
                # 创建列对象
                column_object = Column(colum)
                # 将列添加到表中
                table.add_column(column_object)

    def _modify_tpcds_origin_data(self):
        """
        修改 TPCDS 原始数据文件，替换文件中的换行符。
        """
        # 构建 Perl 命令，用于替换文件中的换行符
        command = f"cd {self.directory} && perl -i -pe 's/\|\n/\n/g' *.dat && perl -i -pe 's/\|\n/\n/g' *.tbl"
        try:
            # 执行命令并获取输出结果
            result = subprocess.check_output(command, shell=True)
            print("Success update data origin files！")
            print(result.decode('utf-8'))
        except subprocess.CalledProcessError as e:
            print("Fail to update data origin files！")
            print(e.returncode)
            print(e.output.decode('utf-8'))
            # 若执行失败则退出程序
            exit(0)

    def _generate(self):
        """
        生成数据文件。
        """
        logging.info("Generating {} data".format(self.benchmark_name))
        logging.info("scale factor: {}".format(self.scale_factor))
        # 执行 make 命令
        self._run_make()
        # 执行数据生成命令
        self._run_command(self.cmd)
        if self.benchmark_name == "tpcds" or self.benchmark_name == "tpchskew" or self.benchmark_name == "ssb":
            # 修改 TPCDS 原始数据文件
            self._modify_tpcds_origin_data()
        logging.info("[Generate command] " + " ".join(self.cmd))
        if self.benchmark_name == "tpchskew":
            # 重命名 TPC-H Skew 数据文件
            os.rename(self.directory + "/" + "order.tbl", self.directory + "/" + "orders.tbl")
        # 获取表文件列表
        self._table_files()
        logging.info("Files generated: {}".format(self.table_files))

    def create_database(self):
        """
        创建数据库，并执行表创建和数据加载操作。
        """
        # 创建数据库
        self.db_connector.create_database(self.database_name())
        # 拼接 'create table' 语句文件的路径
        filename = self.directory + "/" + self.create_table_statements_file
        with open(filename, "r") as file:
            # 读取 'create table' 语句
            create_statements = file.read()
        # 移除主键定义
        create_statements = re.sub(r",\s*primary key (.*)", "", create_statements)
        # 设置数据库连接的数据库名称
        self.db_connector.db_name = self.database_name()
        # 创建数据库连接
        self.db_connector.create_connection()
        # 创建表
        self.create_tables(create_statements)
        # 加载数据到表中
        self._load_table_data(self.db_connector)
        # 如果指定了分区数量，则创建分区表
        if self.partition_num is not None:
            print("Creating {} partitions".format(self.partition_num))
            self.create_partitions(self.partition_num)  # number of partition

    def create_tables(self, create_statements):
        """
        执行表创建语句。

        :param create_statements: 表创建语句
        """
        logging.info("Creating tables")
        # 分割表创建语句，逐个执行
        for create_statement in create_statements.split(";")[:-1]:
            self.db_connector.exec_only(create_statement)
        # 提交事务
        self.db_connector.commit()

    def _database_tables(self):
        """
        返回不同基准测试对应的表和分区键。

        :return: 一个字典，包含表名和分区键
        """
        if "tpch" in self.database_name():
            return {
                'customer': 'c_custkey',
                # 'lineitem': 'l_orderkey',
                'lineitem': 'l_shipdate',
                'nation': 'n_nationkey',
                'orders': 'o_orderkey',
                'part': 'p_partkey',
                'partsupp': 'ps_partkey',
                'region': 'r_regionkey',
                'supplier': 's_suppkey'
            }
        elif "ssb" in self.database_name():
            return {
                'lineorder': 'lo_orderdate',
                'supplier': 's_suppkey',
                'part': 'p_partkey',
                'customer': 'c_custkey',
                'date': 'd_datekey'

            }
        elif "tpcds" in self.database_name():
            return {
                'dbgen_version': 'dv_create_date',
                'customer_address': 'ca_address_sk',
                'customer_demographics': 'cd_demo_sk',
                'date_dim': 'd_date_sk',
                'warehouse': 'w_warehouse_sk',
                'ship_mode': 'sm_ship_mode_sk',
                'time_dim': 't_time_sk',
                'reason': 'r_reason_sk',
                'income_band': 'ib_income_band_sk',
                'item': 'i_item_sk',
                'store': 's_store_sk',
                'call_center': 'cc_call_center_sk',
                'customer': 'c_customer_sk',
                'web_site': 'web_site_sk',
                'store_returns': 'sr_item_sk',
                'household_demographics': 'hd_demo_sk',
                'web_page': 'wp_web_page_sk',
                'promotion': 'p_promo_sk',
                'catalog_page': 'cp_catalog_page_sk',
                'inventory': 'inv_date_sk',
                'catalog_returns': 'cr_item_sk',
                'web_returns': 'wr_item_sk',
                'web_sales': 'ws_sold_date_sk',
                'catalog_sales': 'cs_sold_date_sk',
                'store_sales': 'ss_ticket_number'
            }
        else:
            return None


    def create_partitions(self,partition_num=10):
        """
        创建分区表。

        :param partition_num: 分区数量，默认为 10
        """
        logging.info("Creating partition tables")
        # assert "tpch" in self.database_name(), "You need to specify this datasets instead of default_value 'tpch'"
        tables = self._database_tables()
        # 确保支持当前数据库
        assert tables is not None, "Present supported database is TPCH and TPCDS"
        # 创建分区表
        for key, value in tables.items():
            # 分区表的 SQL 语句列表
            statements = []
            # 将原表重命名为备份表
            statements.append(f"ALTER TABLE {key} RENAME TO {key}_bak;")
            # 创建新表，结构与备份表相同，并指定分布和分区方式
            # statement = f"create table {key} (LIKE {key}_bak) WITH (appendonly=true, orientation=column) DISTRIBUTED BY ({value}) PARTITION BY RANGE({value})("
            #statement = f"create table {key} (LIKE {key}_bak) PARTITION BY RANGE({value});"
            statement = f"create table {key} (LIKE {key}_bak) PARTITION BY HASH ({value});"
            # if "date" not in value or ("date" in value and "sk" in value):
            #     s_max_value = f"select max({value}) from {key}"
            #     s_min_value = f"select min({value}) from {key}"
            #     max_value = self.db_connector.exec_fetch(s_max_value)
            #     min_value = self.db_connector.exec_fetch(s_min_value)
            #     value_range = list(np.linspace(min_value, max_value, partition_num+1))
            #     value_range = [math.ceil(v) for v in value_range]
            #     # deal with the range cannot be divided into partition_num fields
            #     value_range = list(set(value_range))
            #     while len(value_range) < partition_num+1:
            #         value_range.append(value_range[-1]+1)
            #     value_range.sort()
            #     # todo: 均分->部分切分
            #     for i in range(len(value_range) - 1):
            #         if i+1 == len(value_range)-1:
            #             op = 'inclusive'
            #         else:
            #             op = 'exclusive,'
            #         statement += f"PARTITION p{i} start ({value_range[i]})inclusive end ({value_range[i+1]}) {op}"
            #         # statement = f"create table {key}_{i} as select * from {key} where {value} >= {value_range[i]} and {value} {op} {value_range[i+1]}"
            #         # self.db_connector.exec_only(s)
            #     statement += ')'
            #     statements.append(statement)
            # else:
            # 生成值范围的 SQL 语句
            statements.append(statement)
            #self._generate_value_ranges(partition_num, key, value, statements)
            self._generate_hash(partition_num, key, value, statements)
            # 将备份表的数据插入到新表中
            statements.append(f"INSERT INTO {key} SELECT * FROM {key}_bak;")
            # 执行所有 SQL 语句
            for s in statements:
                self.db_connector.exec_only(s)
        # 提交事务
        self.db_connector.commit()

    def _generate_hash(self, partition_num, table, partition_key, statements):
        for i in range(partition_num):
            statement = f"CREATE TABLE {table}_1_prt_p{i} PARTITION OF {table} FOR  VALUES WITH (MODULUS {partition_num}, REMAINDER {i});"
            statements.append(statement)
        return statements

    def _fix_partition_ranges(self, partition_ranges):
        """
        修复分区范围，确保每个分区的起始值小于结束值。

        :param partition_ranges: 分区范围列表
        :return: 修复后的分区范围列表
        """
        for _ in range(len(partition_ranges)):
            if partition_ranges[_][0] >= partition_ranges[_][1]:
                if isinstance(partition_ranges[_][0], datetime.date):
                    partition_ranges[_][1] = partition_ranges[_][0] + timedelta(days=1)
                else:
                    partition_ranges[_][1] = partition_ranges[_][0] + 1
                for i in range(_+1, len(partition_ranges)):
                    partition_ranges[i][0] = max(partition_ranges[i][0], partition_ranges[_][1])

        return partition_ranges

    def _generate_value_ranges(self, partition_num, table, partition_key, statements):
        """
        生成表的分区值范围。

        :param partition_num: 分区数量
        :param table: 表名
        :param partition_key: 分区键
        :param statement: 初始的 SQL 语句
        :return: 包含分区值范围的 SQL 语句
        """
        # 获取表的总行数
        total_value = self.db_connector.exec_fetch(f"select count(*) from {table}")
        # 获取所有分区键的值
        all_keys = self.db_connector.exec_fetch(f"select {partition_key} from {table}", False)
        # 过滤掉空值
        all_keys = [_[0] for _ in all_keys if _[0] is not None]

        if len(all_keys) < partition_num:
            # 如果键的数量少于分区数量，扩展键值
            extend_value = all_keys[-1]
            for _ in range(partition_num - len(all_keys) + 1):
                if isinstance(extend_value, datetime.date):
                    extend_value = extend_value + timedelta(days=1)
                else:
                    extend_value = extend_value + 1
                all_keys.append(extend_value)

        # 对键值进行排序
        all_keys.sort()

        # 获取最大和最小键值
        _max_key = all_keys[-1]
        _min_key = all_keys[0]

        # 计算每个分区的大小
        partition_size = int(len(all_keys) / partition_num)
        # 计算分区的分割点
        partition_split = [[_*partition_size, min((_+1)*partition_size, len(all_keys)-1)] for _ in range(partition_num)]
        # 确保最后一个分区包含所有剩余的键值
        partition_split[-1][-1] = len(all_keys) - 1
        # 计算每个分区的范围
        partition_ranges = [[all_keys[_[0]], all_keys[_[1]]] for _ in partition_split]

        # 修复分区范围
        partition_ranges = self._fix_partition_ranges(partition_ranges)

        # 遍历每个分区，生成对应的分区定义语句
        for i in range(partition_num):
            ranges = partition_ranges[i]
            if i == partition_num - 1:
                if isinstance(ranges[1], datetime.date):
                    ranges[1] = ranges[1] + timedelta(days=1)
                else:
                    #print(f"ranges[1] : {ranges[1]}  => {type(ranges[1])}")
                    ranges[1] = int(ranges[1]) + 1

        if isinstance(partition_ranges[0][0], datetime.date):
            # 如果分区范围的起始值是日期类型，将日期转换为字符串格式
            for _ in partition_ranges:
                _[0] = _[0].strftime("%Y-%m-%d")
                _[1] = _[1].strftime("%Y-%m-%d")
        # # 遍历每个分区，生成对应的分区定义语句
        # for _ in range(partition_num):
        #     ranges = partition_ranges[_]
        #     op = 'inclusive'
        #     # 如果不是第一个分区，检查前一个分区的结束值是否等于当前分区的起始值
        #     if _ > 0:
        #         if partition_ranges[_-1][-1] == ranges[0]:
        #             op = 'exclusive'
        #     # 拼接分区定义语句
        #     statement += f"PARTITION p{_} start (\'{ranges[0]}\') {op} end (\'{ranges[1]}\') inclusive"
        #     # 除了最后一个分区，每个分区定义后添加逗号
        #     statement += ","
        #
        # # 添加默认分区
        # statement += f"default partition other"
        # # 结束分区定义语句
        # statement += ')'
        #
        # return statement

        # 遍历每个分区，生成对应的分区定义语句
        for i in range(partition_num):
            ranges = partition_ranges[i]
            statement = f"CREATE TABLE {table}_1_prt_p{i} PARTITION OF {table} FOR  VALUES FROM (\'{ranges[0]}\') TO (\'{ranges[1]}\');"
            statements.append(statement)
        return statements

    def _load_table_data(self, database_connector):
        """
        将数据加载到表中。

        :param database_connector: 数据库连接对象
        """
        logging.info("Loading data into the tables")
        # 遍历每个表文件
        for filename in self.table_files:
            logging.debug("    Loading file {}".format(filename))
            # 获取表名
            table = filename.replace(".tbl", "").replace(".dat", "")
            # 构建文件路径
            path = self.directory + "/" + filename
            # 获取文件大小
            size = os.path.getsize(path)
            # 将文件大小转换为 MB 格式
            size_string = f"{b_to_mb(size):,.4f} MB"
            logging.debug(f"    Import data of size {size_string}")
            # 导入数据到表中
            database_connector.import_data(table, path)
            # 删除已导入的文件
            os.remove(os.path.join(self.directory, filename))
        # 提交事务
        database_connector.commit()
        # statement = "Insert into lineitem Select * from lineitem;"
        # res = database_connector.execute(statement)
        # print(res)
        # res = database_connector.execute(statement)
        # print(res)
        # res = database_connector.execute(statement)
        # print(res)


    def _run_make(self):
        """
        执行 make 命令来生成必要的文件。
        """
        if "dbgen" not in self._files() and "dsdgen" not in self._files():
            logging.info("Running make in {}".format(self.directory))
            self._run_command(self.make_command)
        else:
            logging.info("No need to run make")

    def _table_files(self):
        """
        获取所有表文件的列表。
        """
        self.table_files = [x for x in self._files() if ".tbl" in x or ".dat" in x]

    def _run_command(self, command):
        """
        执行外部命令并记录输出。

        :param command: 要执行的命令
        """
        cmd_out = "[SUBPROCESS OUTPUT] "
        p = subprocess.Popen(
            command,
            cwd=self.directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with p.stdout:
            for line in p.stdout:
                logging.info(cmd_out + line.decode("utf-8").replace("\n", ""))
        p.wait()

    def _files(self):
        """
        获取指定目录下的所有文件列表。

        :return: 文件列表
        """
        return os.listdir(self.directory)

    def _prepare(self):
        """
        根据不同的基准测试名称，准备相应的命令和文件路径。
        """
        if self.benchmark_name == "tpch":
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("MACHINE=MACOS")
            self.directory = "../index_selection_evaluation/tpch-kit/dbgen"
            self.create_table_statements_file = "dss.ddl"
            self.cmd = ["./dbgen", "-s", str(self.scale_factor), "-f"]
            #self.cmd = ["/mydata/tyl/workspace/remote/block_based_index_recommendation/free-origin/index/rl_index_selection/index_selection_evaluation/tpch-kit/dbgen", "-s", str(self.scale_factor), "-f"]
        elif self.benchmark_name == "tpcds":
            self.make_command = ["make"]
            if platform.system() == "Darwin":
                self.make_command.append("OS=MACOS")
            self.directory = "../index_selection_evaluation/tpcds-kit/tools"
            self.create_table_statements_file = "tpcds.sql"
            self.cmd = ["./dsdgen", "-SCALE", str(self.scale_factor), "-FORCE"]
            # 0.001 is allowed for testing
            if (
                int(self.scale_factor) - self.scale_factor != 0
                and self.scale_factor != 0.01
            ):
                raise Exception("Wrong TPCDS scale factor")
        elif self.benchmark_name == "tpchskew":
            self.make_command = ["make", "-f", "makefile_linux.original"]
            if platform.system() == "Darwin":
                self.make_command = ["make", "-f", "makefile_MacSolaris"]
            self.directory = "../index_selection_evaluation/tpchskew-kit"
            self.create_table_statements_file = "dss.ddl"
            self.cmd = ["./dbgen", "-s", str(self.scale_factor), "-f", "-z", str(self.config["skew_factor"])]
        elif self.benchmark_name == "ssb":
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("MACHINE=MACOS")
            self.directory = "../index_selection_evaluation/ssb-kit/dbgen"
            self.create_table_statements_file = "dss.ddl"
            self.cmd = ["./dbgen", "-s", str(self.scale_factor), "-T", "a", "-f"]
        else:
            raise NotImplementedError("only tpch/ds implemented.")

