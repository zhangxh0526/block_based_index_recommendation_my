import logging


# Class that encapsulates simulated/WhatIf-Indexes.
# This is usually used by the CostEvaluation class and there should be no need
# to use it manually.
# Uses hypopg for postgreSQL
class WhatIfIndexCreation:
    def __init__(self, db_connector):
        """
        初始化 WhatIfIndexCreation 类的实例。

        :param db_connector: 数据库连接器对象，用于与数据库进行交互。
        """
        # 记录调试信息，表明开始初始化 WhatIfIndexCreation
        logging.debug("Init WhatIfIndexCreation")

        # 用于存储模拟索引的字典，键为索引的 OID，值为索引的名称
        self.simulated_indexes = {}
        # 存储数据库连接器对象，用于后续与数据库的交互
        self.db_connector = db_connector

    def simulate_index(self, potential_index, store_size=False):
        """
        模拟创建一个索引。

        :param potential_index: 潜在的索引对象，包含索引的相关信息。
        :param store_size: 是否存储索引的估计大小，默认为 False。
        """
        # 调用数据库连接器的 simulate_index 方法模拟创建索引，并获取结果
        result = self.db_connector.simulate_index(potential_index)
        # 从结果中提取索引的 OID
        index_oid = result[0]
        # 从结果中提取索引的名称
        index_name = result[1]
        # 将索引的 OID 和名称存储到 simulated_indexes 字典中
        self.simulated_indexes[index_oid] = index_name
        # 为潜在索引对象设置 hypopg_name 属性
        potential_index.hypopg_name = index_name
        # 为潜在索引对象设置 hypopg_oid 属性
        potential_index.hypopg_oid = index_oid

        # 如果 store_size 为 True，则估计并存储索引的大小
        if store_size:
            # 调用 estimate_index_size 方法估计索引的大小，并赋值给潜在索引对象的 estimated_size 属性
            potential_index.estimated_size = self.estimate_index_size(index_oid)

    def drop_simulated_index(self, index):
        """
        删除一个模拟创建的索引。

        :param index: 要删除的索引对象。
        """
        # 从索引对象中获取索引的 OID
        oid = index.hypopg_oid
        # 调用数据库连接器的 drop_simulated_index 方法删除模拟索引
        self.db_connector.drop_simulated_index(oid)
        # 从 simulated_indexes 字典中删除该索引的记录
        del self.simulated_indexes[oid]

    def all_simulated_indexes(self):
        """
        获取所有模拟创建的索引信息。

        :return: 包含所有模拟索引信息的列表。
        """
        # 定义 SQL 语句，用于查询所有模拟索引的信息
        statement = "select * from hypopg_list_indexes"
        # 调用数据库连接器的 exec_fetch 方法执行 SQL 语句，并获取结果
        indexes = self.db_connector.exec_fetch(statement, one=False)
        # 返回查询到的所有模拟索引信息
        return indexes

    def estimate_index_size(self, index_oid):
        """
        估计指定索引的大小。

        :param index_oid: 要估计大小的索引的 OID。
        :return: 索引的估计大小。
        """
        # 定义 SQL 语句，用于查询指定索引的大小
        statement = f"select hypopg_relation_size({index_oid})"
        # 调用数据库连接器的 exec_fetch 方法执行 SQL 语句，并获取结果的第一个元素
        result = self.db_connector.exec_fetch(statement)[0]
        # 断言结果大于 0，确保假设索引存在
        assert result > 0, "Hypothetical index does not exist."
        # 返回索引的估计大小
        return result

    # TODO: refactoring
    # This is never used, we keep it for debugging reasons.
    def index_names(self):
        """
        获取所有模拟索引的名称。

        :return: 包含所有模拟索引名称的列表。
        """
        # 调用 all_simulated_indexes 方法获取所有模拟索引的信息
        indexes = self.all_simulated_indexes()

        # 从索引信息中提取索引的名称，并存储在列表中返回
        # Apparently, x[1] is the index' name
        return [x[1] for x in indexes]

    def drop_all_simulated_indexes(self):
        """
        删除所有模拟创建的索引。
        """
        # 遍历 simulated_indexes 字典中的所有键（即索引的 OID）
        for key in self.simulated_indexes:
            # 调用数据库连接器的 drop_simulated_index 方法删除模拟索引
            self.db_connector.drop_simulated_index(key)
        # 清空 simulated_indexes 字典
        self.simulated_indexes = {}
