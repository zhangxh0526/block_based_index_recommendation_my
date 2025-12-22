import logging
import time


class DatabaseConnector:
    """
    数据库连接器类，用于与数据库进行交互。

    该类提供了一系列方法来执行数据库操作，如执行 SQL 语句、获取查询结果、
    提交事务、回滚事务等。部分方法为抽象方法，需要在子类中实现。

    :param db_name: 数据库名称
    :param autocommit: 是否自动提交事务，默认为 False
    """
    def __init__(self, db_name, autocommit=False):
        # 存储数据库名称
        self.db_name = db_name
        # 存储是否自动提交事务的标志
        self.autocommit = autocommit
        # 记录数据库连接器创建的调试信息
        logging.debug("Database connector created: {}".format(db_name))

        # 这并不反映唯一模拟索引的数量，而是 simulate_index 方法的调用次数
        self.simulated_indexes = 0
        # 记录成本估计的次数
        self.cost_estimations = 0
        # 记录成本估计的总时长
        self.cost_estimation_duration = 0
        # 记录索引模拟的总时长
        self.index_simulation_duration = 0

    def exec_only(self, statement):
        """
        仅执行 SQL 语句，不返回结果。

        :param statement: 要执行的 SQL 语句
        """
        self._cursor.execute(statement)

    def exec_fetch(self, statement, one=True):
        """
        执行 SQL 语句并返回结果。

        :param statement: 要执行的 SQL 语句
        :param one: 是否只返回一条结果，默认为 True
        :return: 如果 one 为 True，返回单条结果；否则返回所有结果
        """
        self._cursor.execute(statement)
        if one:
            return self._cursor.fetchone()
        return self._cursor.fetchall()

    def enable_simulation(self):
        """
        启用索引模拟功能。

        此方法为抽象方法，需要在子类中实现。
        """
        raise NotImplementedError

    def commit(self):
        """
        提交当前事务。
        """
        self._connection.commit()

    def close(self):
        """
        关闭数据库连接。

        记录数据库连接器关闭的调试信息。
        """
        self._connection.close()
        logging.debug("Database connector closed: {}".format(self.db_name))

    def rollback(self):
        """
        回滚当前事务。
        """
        self._connection.rollback()

    def drop_index(self, index):
        """
        删除指定的索引。

        :param index: 要删除的索引对象
        """
        statement = f"drop index {index.index_idx()}"
        self.exec_only(statement)

    def _prepare_query(self, query):
        """
        预处理查询语句。

        执行查询语句中的创建视图语句，并返回第一个 SELECT 语句。

        :param query: 查询对象
        :return: 第一个 SELECT 语句
        """
        for query_statement in query.text.split(";"):
            if "create view" in query_statement:
                try:
                    # 执行创建视图的语句
                    self.exec_only(query_statement)
                except Exception as e:
                    # 记录创建视图语句执行失败的错误信息
                    logging.error(e)
            elif "select" in query_statement or "SELECT" in query_statement:
                return query_statement

    def simulate_index(self, index):
        """
        模拟创建索引。

        记录模拟索引的次数和耗时。

        :param index: 要模拟创建的索引对象
        :return: 模拟创建索引的结果
        """
        self.simulated_indexes += 1

        start_time = time.time()
        result = self._simulate_index(index)
        end_time = time.time()
        self.index_simulation_duration += end_time - start_time

        return result

    def drop_simulated_index(self, identifier):
        """
        删除模拟的索引。

        记录删除模拟索引的耗时。

        :param identifier: 模拟索引的标识符
        """
        start_time = time.time()
        self._drop_simulated_index(identifier)
        end_time = time.time()
        self.index_simulation_duration += end_time - start_time

    def get_cost(self, query):
        """
        获取查询的成本。

        记录成本估计的次数和耗时。

        :param query: 查询对象
        :return: 查询的成本
        """
        self.cost_estimations += 1

        start_time = time.time()
        cost = self._get_cost(query)
        end_time = time.time()
        self.cost_estimation_duration += end_time - start_time

        return cost

    # This is very similar to get_cost() above. Some algorithms need to directly access
    # get_plan. To not exclude it from costing, we add the instrumentation here.
    def get_plan(self, query):
        """
        获取查询的执行计划。

        记录成本估计的次数和耗时。

        :param query: 查询对象
        :return: 查询的执行计划
        """
        self.cost_estimations += 1

        start_time = time.time()
        plan = self._get_plan(query)
        end_time = time.time()
        self.cost_estimation_duration += end_time - start_time

        return plan

    def table_exists(self, table_name):
        """
        检查指定的表是否存在。

        此方法为抽象方法，需要在子类中实现。

        :param table_name: 表名
        """
        raise NotImplementedError

    def database_exists(self, database_name):
        """
        检查指定的数据库是否存在。

        此方法为抽象方法，需要在子类中实现。

        :param database_name: 数据库名
        """
        raise NotImplementedError

    def drop_database(self, database_name):
        """
        删除指定的数据库。

        此方法为抽象方法，需要在子类中实现。

        :param database_name: 数据库名
        """
        raise NotImplementedError

    def create_statistics(self):
        """
        创建数据库统计信息。

        此方法为抽象方法，需要在子类中实现。
        """
        raise NotImplementedError

    def set_random_seed(self, value):
        """
        设置数据库的随机种子。

        此方法为抽象方法，需要在子类中实现。

        :param value: 随机种子的值
        """
        raise NotImplementedError

    def _get_cost(self, query):
        """
        获取查询的成本。

        此方法为抽象方法，需要在子类中实现。

        :param query: 查询对象
        """
        raise NotImplementedError

    def _get_plan(self, query):
        """
        获取查询的执行计划。

        此方法为抽象方法，需要在子类中实现。

        :param query: 查询对象
        """
        raise NotImplementedError

    def _simulate_index(self, index):
        """
        模拟创建索引。

        此方法为抽象方法，需要在子类中实现。

        :param index: 要模拟创建的索引对象
        """
        raise NotImplementedError

    def _drop_simulated_index(self, identifier):
        """
        删除模拟的索引。

        此方法为抽象方法，需要在子类中实现。

        :param identifier: 模拟索引的标识符
        """
        raise NotImplementedError

