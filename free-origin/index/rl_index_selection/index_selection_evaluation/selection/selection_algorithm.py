import logging
import time

from .cost_evaluation import CostEvaluation

# If not specified by the user, algorithms should use these default parameter values to
# avoid diverging values for different algorithms.
DEFAULT_PARAMETER_VALUES = {
    "budget_MB": 500,
    "max_indexes": 15,
    "max_index_width": 2,
}


class SelectionAlgorithm:
    def __init__(self, database_connector, parameters=None, default_parameters=None):
        """
        初始化 SelectionAlgorithm 类的实例。

        :param database_connector: 用于与数据库交互的连接器。
        :param parameters: 算法的参数，默认为 None。
        :param default_parameters: 默认参数，默认为 None。
        """
        # 如果默认参数为空，则初始化为空字典
        if default_parameters is None:
            default_parameters = {}
        # 记录初始化日志
        logging.debug("Init selection algorithm")
        # 标记算法是否已经运行
        self.did_run = False
        # 存储算法的参数
        self.parameters = parameters
        # 为缺失的参数存储默认值
        for key, value in default_parameters.items():
            if key not in self.parameters:
                self.parameters[key] = value

        # 存储数据库连接器
        self.database_connector = database_connector
        # 删除数据库中的所有索引
        self.database_connector.drop_indexes()
        # 初始化成本评估对象
        self.cost_evaluation = CostEvaluation(database_connector)
        # 如果参数中指定了成本估计方法，则设置成本评估对象的成本估计方法
        if "cost_estimation" in self.parameters:
            estimation = self.parameters["cost_estimation"]
            self.cost_evaluation.cost_estimation = estimation

    def calculate_best_indexes(self, workload):
        """
        计算最佳索引。

        :param workload: 数据库工作负载。
        :return: 计算得到的最佳索引列表。
        """
        # 确保算法只运行一次
        assert self.did_run is False, "Selection algorithm can only run once."
        # 记录开始时间
        start_time = time.time()
        # 标记算法已经运行
        self.did_run = True
        # 调用子类实现的方法计算最佳索引
        indexes = self._calculate_best_indexes(workload)
        # 计算计算时间
        self.calculation_time = round(time.time() - start_time, 2)
        # 记录成本缓存命中信息
        self._log_cache_hits()
        # 计算最终成本比例
        self.final_cost_proportion = self._calculate_final_cost_proportion(
            workload, indexes
        )
        # 完成成本评估
        self.cost_evaluation.complete_cost_estimation()

        return indexes

    def _calculate_final_cost_proportion(self, workload, indexes):
        """
        计算最终成本比例。

        :param workload: 数据库工作负载。
        :param indexes: 计算得到的最佳索引列表。
        :return: 最终成本比例。
        """
        # 打印开始测试索引的信息
        print("Start testing index ...")
        # 设置成本估计方法为实际运行时间
        # self.cost_evaluation.cost_estimation = "actual_runtimes"
        # 计算没有索引时的成本和扫描行数
        start_cost, start_rows = self.cost_evaluation.calculate_cost_and_rows(workload, [])
        # 计算使用索引时的成本和扫描行数
        final_cost, final_rows = self.cost_evaluation.calculate_cost_and_rows(workload, indexes)
        # 计算索引组合的总大小
        index_combination_size = 0
        for index in indexes:
            index_combination_size += index.estimated_size
        # 记录日志，包含初始成本、最终成本、成本比例、索引信息、总扫描行数和索引组合大小
        logging.info(
            (
                f"Initial cost: {start_cost:,.2f}, now: {final_cost:,.2f} "
                f"({final_cost / (start_cost+1):.2f}) {indexes},"
                f"Total rows: {start_rows:,.2f}, Scanned rows: {final_rows:,.2f},"
                f"{index_combination_size / 1000 / 1000:.2f} MB "
                f"(of {self.parameters['budget_MB']} MB) for workload\n{workload}"
            )
        )
        # 设置成本估计方法为假设分析
        self.cost_evaluation.cost_estimation = "whatif"
        # 打印完成测试索引的信息
        print("Finish testing index ...")
        # 计算并返回最终成本比例
        return round(final_cost / (start_cost+1) * 100, 2)

    def _calculate_best_indexes(self, workload):
        """
        抽象方法，需要在子类中实现，用于计算最佳索引。

        :param workload: 数据库工作负载。
        :raises NotImplementedError: 如果子类没有实现该方法。
        """
        raise NotImplementedError("_calculate_best_indexes(self, " "workload) missing")

    def _log_cache_hits(self):
        """
        记录成本缓存命中信息。
        """
        # 获取成本缓存命中次数
        hits = self.cost_evaluation.cache_hits
        # 获取成本请求次数
        requests = self.cost_evaluation.cost_requests
        # 记录成本缓存命中次数
        logging.debug(f"Total cost cache hits:\t{hits}")
        # 记录成本请求次数
        logging.debug(f"Total cost requests:\t\t{requests}")
        # 如果没有成本请求，则直接返回
        if requests == 0:
            return
        # 计算成本缓存命中率
        ratio = round(hits * 100 / requests, 2)
        # 记录成本缓存命中率
        logging.debug(f"Cost cache hit ratio:\t{ratio}%")



class NoIndexAlgorithm(SelectionAlgorithm):
    """
    NoIndexAlgorithm 类继承自 SelectionAlgorithm 类，用于实现不使用任何索引的算法。
    该算法返回一个空的索引列表，意味着不推荐任何索引。
    """
    def __init__(self, database_connector, parameters=None):
        """
        初始化 NoIndexAlgorithm 类的实例。

        :param database_connector: 用于与数据库交互的连接器。
        :param parameters: 算法的参数，默认为 None。
        """
        # 如果参数为空，则初始化为空字典
        if parameters is None:
            parameters = {}
        # 调用父类的构造函数进行初始化
        SelectionAlgorithm.__init__(self, database_connector, parameters)

    def _calculate_best_indexes(self, workload):
        """
        计算最佳索引，该算法返回空列表，表示不使用任何索引。

        :param workload: 数据库工作负载。
        :return: 空的索引列表。
        """
        return []



class AllIndexesAlgorithm(SelectionAlgorithm):
    """
    AllIndexesAlgorithm 类继承自 SelectionAlgorithm 类，用于实现返回所有可索引列的单索引的算法。
    该算法会为每个可索引的列生成一个单索引。
    """
    def __init__(self, database_connector, parameters=None):
        """
        初始化 AllIndexesAlgorithm 类的实例。

        :param database_connector: 用于与数据库交互的连接器。
        :param parameters: 算法的参数，默认为 None。
        """
        # 如果参数为空，则初始化为空字典
        if parameters is None:
            parameters = {}
        # 调用父类的构造函数进行初始化
        SelectionAlgorithm.__init__(self, database_connector, parameters)

    # Returns single column index for each indexable column
    def _calculate_best_indexes(self, workload):
        """
        计算最佳索引，该算法返回工作负载中每个可索引列的单索引。

        :param workload: 数据库工作负载。
        :return: 包含所有可索引列单索引的列表。
        """
        return workload.potential_indexes()

