import logging
import re

import psycopg2

from ..database_connector import DatabaseConnector


class PostgresDatabaseConnector(DatabaseConnector):
    def __init__(self, db_name, autocommit=False):
        """
        初始化 PostgresDatabaseConnector 类的实例。

        :param db_name: 要连接的数据库名称
        :param autocommit: 是否自动提交事务，默认为 False
        """
        DatabaseConnector.__init__(self, db_name, autocommit=autocommit)
        # 设置数据库系统为 postgres
        self.db_system = "postgres"
        # 初始化数据库连接对象
        self._connection = None

        # 如果没有提供数据库名称，则默认使用 'postgres'
        if not self.db_name:
            self.db_name = "postgres"
        # 创建数据库连接
        self.create_connection()
        # 开启模拟索引功能
        self.enable_simulation()
        # 设置随机种子
        self.set_random_seed()

        # 禁用位图扫描
        # self.exec_only("SET max_parallel_workers_per_gather = 0;")
        self.exec_only("SET enable_bitmapscan TO off;")

        # 记录调试信息，表明 Postgres 连接器已创建
        logging.debug("Postgres connector created: {}".format(db_name))

    def create_connection(self):
        """
        创建与 PostgreSQL 数据库的连接。
        如果已经存在连接，则先关闭该连接。
        """
        if self._connection:
            self.close()
        # 使用 psycopg2 库连接到指定的数据库
        # 优先使用常见的 Unix socket 目录 /var/run/postgresql，失败则回退到 /tmp
        # 如需 TCP/指定用户密码，可改为 host=localhost port=5432 user=... password=...
        socket_dir = "/var/run/postgresql"
        try:
            self._connection = psycopg2.connect("dbname={} host='{}'".format(self.db_name, socket_dir))
        except Exception:
            self._connection = psycopg2.connect("dbname={} host='/tmp/'".format(self.db_name))
        # 设置是否自动提交事务
        self._connection.autocommit = self.autocommit
        # 创建游标对象，用于执行 SQL 语句
        self._cursor = self._connection.cursor()

    def enable_simulation(self):
        """
        启用索引模拟功能。
        创建 hypopg 扩展并提交事务。
        """
        #self.exec_only("create extension hypopg")
        self.exec_only("CREATE EXTENSION IF NOT EXISTS hypopg;")
        self.commit()

    def database_names(self):
        """
        获取所有数据库的名称。

        :return: 包含所有数据库名称的列表
        """
        result = self.exec_fetch("select datname from pg_database", False)
        return [x[0] for x in result]

    # Updates query syntax to work in PostgreSQL
    def update_query_text(self, text):
        """
        更新查询文本，使其符合 PostgreSQL 的语法要求。

        :param text: 原始查询文本
        :return: 更新后的查询文本
        """
        # 替换查询文本中的特定字符串
        text = text.replace(";\nlimit ", " limit ").replace("limit -1", "")
        # 使用正则表达式替换日期格式
        text = re.sub(r" ([0-9]+) days\)", r" interval '\1 days')", text)
        # 为子查询添加别名
        text = self._add_alias_subquery(text)
        return text

    # PostgreSQL requires an alias for subqueries
    def _add_alias_subquery(self, query_text):
        """
        为 PostgreSQL 查询中的子查询添加别名。

        :param query_text: 原始查询文本
        :return: 添加别名后的查询文本
        """
        # 将查询文本转换为小写
        text = query_text.lower()
        # 存储需要添加别名的位置
        positions = []
        # 查找所有 from 或逗号后面紧跟着左括号的位置
        for match in re.finditer(r"((from)|,)[  \n]*\(", text):
            # 初始化括号计数器
            counter = 1
            # 获取匹配位置的结束位置
            pos = match.span()[1]
            # 遍历查询文本，直到找到对应的右括号
            while counter > 0:
                char = text[pos]
                if char == "(":
                    counter += 1
                elif char == ")":
                    counter -= 1
                pos += 1
            # 获取右括号后面的第一个单词
            next_word = query_text[pos:].lstrip().split(" ")[0].split("\n")[0]
            # 如果下一个单词是特定关键字或符号，则记录该位置
            if next_word[0] in [")", ","] or next_word in [
                "limit",
                "group",
                "order",
                "where",
            ]:
                positions.append(pos)
        # 从后往前遍历位置列表，为子查询添加别名
        for pos in sorted(positions, reverse=True):
            query_text = query_text[:pos] + " as alias123 " + query_text[pos:]
        return query_text

    def create_database(self, database_name):
        """
        创建一个新的数据库。

        :param database_name: 要创建的数据库名称
        """
        self.exec_only("create database {}".format(database_name))
        # 记录信息日志，表明数据库已创建
        logging.info("Database {} created".format(database_name))

    def import_data(self, table, path, delimiter="|", encoding=None):
        """
        从文件中导入数据到指定的表中。

        :param table: 要导入数据的表名
        :param path: 数据文件的路径
        :param delimiter: 数据文件中的分隔符，默认为 '|'
        :param encoding: 数据文件的编码，默认为 None
        """
        if encoding:
            # 如果指定了编码，则使用指定的编码打开文件
            with open(path, "r", encoding=encoding) as file:
                self._cursor.copy_expert(
                    (
                        f"COPY {table} FROM STDIN WITH DELIMITER AS '{delimiter}' NULL "
                        f"AS 'NULL' CSV QUOTE AS '\"' ENCODING '{encoding}'"
                    ),
                    file,
                )
        else:
            # 如果未指定编码，则使用默认编码打开文件
            with open(path, "r") as file:
                self._cursor.copy_from(file, table, sep=delimiter, null="")

    def indexes_size(self):
        """
        获取所有索引的总大小（以字节为单位）。

        :return: 所有索引的总大小
        """
        # 构建 SQL 查询语句
        statement = (
            "select sum(pg_indexes_size(table_name::text)) from "
            "(select table_name from information_schema.tables "
            "where table_schema='public') as all_tables"
        )
        result = self.exec_fetch(statement)
        return result[0]

    def drop_database(self, database_name):
        """
        删除指定的数据库。

        :param database_name: 要删除的数据库名称
        """
        statement = f"DROP DATABASE {database_name};"
        self.exec_only(statement)

        # 记录信息日志，表明数据库已删除
        logging.info(f"Database {database_name} dropped")

    def create_statistics(self):
        """
        运行 `analyze` 命令来更新数据库的统计信息。
        """
        # 记录信息日志，表明正在运行 `analyze` 命令
        logging.info("Postgres: Run `analyze`")
        self.commit()
        # 设置自动提交为 True
        self._connection.autocommit = True
        self.exec_only("analyze")
        # 恢复自动提交设置
        self._connection.autocommit = self.autocommit

    def set_random_seed(self, value=0.17):
        """
        设置 PostgreSQL 数据库的随机种子。

        :param value: 随机种子的值，默认为 0.17
        """
        # 记录信息日志，表明正在设置随机种子
        logging.info(f"Postgres: Set random seed `SELECT setseed({value})`")
        self.exec_only(f"SELECT setseed({value})")

    def supports_index_simulation(self):
        """
        检查数据库系统是否支持索引模拟。

        :return: 如果数据库系统是 postgres，则返回 True，否则返回 False
        """
        if self.db_system == "postgres":
            return True
        return False

    def _simulate_index(self, index):
        """
        模拟创建一个索引。

        :param index: 要模拟创建的索引对象
        :return: 模拟创建索引的结果
        """
        table_name = index.table()
        # 构建模拟创建索引的 SQL 语句
        statement = (
            "select * from hypopg_create_index( "
            f"'create index on {table_name} "
            f"({index.joined_column_names()})')"
        )
        result = self.exec_fetch(statement)
        return result

    def _drop_simulated_index(self, oid):
        """
        删除模拟创建的索引。

        :param oid: 要删除的模拟索引的 OID
        """
        # 构建删除模拟索引的 SQL 语句
        statement = f"select * from hypopg_drop_index({oid})"
        # 执行 SQL 语句并获取结果
        result = self.exec_fetch(statement)

        # 断言删除操作是否成功，如果失败则抛出异常
        assert result[0] is True, f"Could not drop simulated index with oid = {oid}."

    def get_index_size(self, index):
        """
        获取指定索引的大小。
        先创建索引，然后删除索引。

        :param index: 要获取大小的索引对象
        """
        # 创建索引
        self.create_index(index)
        # 删除索引
        self.drop_index(index)

    def create_index(self, index):
        """
        创建一个索引并估计其大小。

        :param index: 要创建的索引对象
        """
        # 获取索引所在的表名
        table_name = index.table()
        # 构建创建索引的 SQL 语句
        statement = (
            f"create index {index.index_idx()} "
            f"on {table_name} ({index.joined_column_names()})"
        )
        print("创建索引 Execute Statement is: {}".format(statement))
        # 执行创建索引的 SQL 语句
        self.exec_only(statement)
        # 查询索引的页面数
        size = self.exec_fetch(
            f"select relpages from pg_class c " f"where c.relname = '{index.index_idx()}'"  # 获得包括(a,b)+a+b的大小
        )
        # 改动：查询索引的大小（以可读格式表示）
        size_kb = self.exec_fetch(
            f"select pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size FROM pg_indexes WHERE indexname = '{index.index_idx()}'"
        )
        # 如果查询到的页面数不为空且不为 0
        if size is not None and size[0] != 0:
            # 获取页面数
            size = size[0]
            # 计算索引的估计大小（单位：字节）
            index.estimated_size = size * 8 * 1024  # size 单位是8KB
        else:
            # 如果查询结果包含 'kB'
            if 'kB' in size_kb[0]:
                # 提取大小数值并转换为浮点数
                size = float(size_kb[0].split('kB')[0].replace(" ",""))
                # 计算索引的估计大小（单位：字节）
                index.estimated_size = size * 1024
            # 如果查询结果包含 'MB'
            elif 'MB' in size_kb[0]:
                # 提取大小数值并转换为浮点数
                size = float(size_kb[0].split('MB')[0].replace(" ",""))
                # 计算索引的估计大小（单位：字节）
                index.estimated_size = size * 1024 * 1024
            else:
                # 如果无法解析大小信息，抛出断言异常
                assert()

    def drop_indexes(self):
        """
        删除所有公共模式下的索引。
        """
        # 记录信息日志，表明正在删除索引
        logging.info("Dropping indexes")
        # 构建查询公共模式下所有索引名称的 SQL 语句
        stmt = "select indexname from pg_indexes where schemaname='public'"
        # 执行 SQL 语句并获取所有索引名称
        indexes = self.exec_fetch(stmt, one=False)
        # 遍历所有索引名称
        for index in indexes:
            # 获取当前索引的名称
            index_name = index[0]
            # 构建删除当前索引的 SQL 语句
            drop_stmt = "drop index {}".format(index_name)
            # 记录调试信息，表明正在删除特定索引
            logging.debug("Dropping index {}".format(index_name))
            # 执行删除索引的 SQL 语句
            self.exec_only(drop_stmt)

    def drop_simulated_indexes(self):
        logging.info("Dropping simulated indexes")
        statement = "SELECT hypopg_reset()"
        self.exec_fetch(statement)

    # PostgreSQL expects the timeout in milliseconds
    def exec_query(self, query, timeout=None, cost_evaluation=False):
        """
        执行一个查询，并可以设置超时时间。

        :param query: 要执行的查询对象
        :param timeout: 查询的超时时间（以毫秒为单位），默认为 None
        :param cost_evaluation: 是否进行成本评估，默认为 False
        :return: 查询的执行时间和执行计划
        """
        # 如果不进行成本评估，则提交事务
        if not cost_evaluation:
            self._connection.commit()
        # 预处理查询文本
        query_text = self._prepare_query(query)
        # 如果设置了超时时间
        if timeout:
            # 构建设置语句超时的 SQL 语句
            set_timeout = f"set statement_timeout={timeout}"
            # 执行设置超时的 SQL 语句
            self.exec_only(set_timeout)
        # 构建包含执行计划的 SQL 语句
        statement = f"explain (analyze, buffers, format json) {query_text}"
        try:
            # 执行查询并获取执行计划
            plan = self.exec_fetch(statement, one=True)[0][0]["Plan"]
            # 获取查询的实际总时间和执行计划
            result = plan["Actual Total Time"], plan
        except Exception as e:
            # 记录错误信息
            logging.error(f"{query.nr}, {e}")
            # 回滚事务
            self._connection.rollback()
            # 获取查询的执行计划
            result = None, self._get_plan(query)
        # 禁用超时设置
        self._cursor.execute("set statement_timeout = 0")
        # 清理查询
        self._cleanup_query(query)
        return result

    def _cleanup_query(self, query):
        """
        清理查询，删除查询中创建的视图。

        :param query: 要清理的查询对象
        """
        # 遍历查询语句中的每个子语句
        for query_statement in query.text.split(";"):
            # 如果子语句包含 'drop view'
            if "drop view" in query_statement:
                # 执行删除视图的 SQL 语句
                self.exec_only(query_statement)
                # 提交事务
                self.commit()

    def _get_cost(self, query):
        """
        获取查询的成本。

        :param query: 要获取成本的查询对象
        :return: 查询的总成本
        """
        # 获取查询的执行计划
        query_plan = self._get_plan(query)
        # 获取执行计划中的总成本
        total_cost = query_plan["Total Cost"]
        return total_cost

    def get_raw_plan(self, query):
        """
        获取查询的原始执行计划。

        :param query: 要获取执行计划的查询对象
        :return: 查询的原始执行计划
        """
        # 预处理查询文本
        query_text = self._prepare_query(query)
        # 构建获取执行计划的 SQL 语句
        statement = f"explain (format json) {query_text}"
        # 执行 SQL 语句并获取执行计划
        query_plan = self.exec_fetch(statement)[0]
        # 清理查询
        self._cleanup_query(query)
        return query_plan

    def _get_plan(self, query):
        """
        获取查询的执行计划。

        :param query: 要获取执行计划的查询对象
        :return: 查询的执行计划
        """
        # 预处理查询文本
        query_text = self._prepare_query(query)
        # 构建获取执行计划的 SQL 语句
        statement = f"explain (format json) {query_text}"
        # 执行 SQL 语句并获取执行计划
        query_plan = self.exec_fetch(statement)[0][0]["Plan"]
        # 清理查询
        self._cleanup_query(query)
        return query_plan

    def number_of_indexes(self):
        """
        获取公共模式下的索引数量。

        :return: 公共模式下的索引数量
        """
        # 构建查询公共模式下索引数量的 SQL 语句
        statement = """select count(*) from pg_indexes
                       where schemaname = 'public'"""
        # 执行 SQL 语句并获取结果
        result = self.exec_fetch(statement)
        return result[0]

    def table_exists(self, table_name):
        """
        检查指定的表是否存在。

        :param table_name: 要检查的表名
        :return: 如果表存在返回 True，否则返回 False
        """
        # 构建检查表是否存在的 SQL 语句
        # 使用 f-string 插入表名，查询 pg_tables 系统表，判断指定表名是否存在
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_tables
            WHERE tablename = '{table_name}');"""
        # 执行 SQL 语句并获取结果
        result = self.exec_fetch(statement)
        # 返回查询结果的第一个元素
        return result[0]

    def database_exists(self, database_name):
        # 构建检查数据库是否存在的 SQL 语句
        # 使用 f-string 插入数据库名，查询 pg_database 系统表，判断指定数据库名是否存在
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_database
            WHERE datname = '{database_name}');"""
        # 执行 SQL 语句并获取结果
        result = self.exec_fetch(statement)
        # 返回查询结果的第一个元素
        return result[0]

