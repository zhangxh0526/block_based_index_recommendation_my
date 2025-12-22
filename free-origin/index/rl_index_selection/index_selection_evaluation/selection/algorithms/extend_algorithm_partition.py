import logging
import time
from copy import deepcopy

from ..index import Index
from ..selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
import functools
from ..utils import b_to_mb, mb_to_b
from ..workload import Workload
import  numpy as np

# budget_MB: The algorithm can utilize the specified storage budget in MB.
# max_index_width: The number of columns an index can contain at maximum.
# min_cost_improvement: The value of the relative improvement that must be realized by a
#                       new configuration to be selected.
# The algorithm stops if either the budget is exceeded or no further beneficial
# configurations can be found.
DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "min_cost_improvement": 1.003,
    "max_runtime_minutes": 60 * 24,
}

def indexsort(x, y):
    if x.columns[0].table.name > y.columns[0].table.name:
        return -1
    return 1


# This algorithm is a reimplementation of the Extend heuristic published by Schlosser,
# Kossmann, and Boissier in 2019.
# Details can be found in the original paper:
# Rainer Schlosser, Jan Kossmann, Martin Boissier: Efficient Scalable
# Multi-attribute Index Selection Using Recursive Strategies. ICDE 2019: 1238-1249
class ExtendAlgorithmPartition(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        """
        初始化 ExtendAlgorithm 类的实例。

        :param database_connector: 用于与数据库交互的连接器，用于执行数据库操作，如创建、删除索引等。
        :param parameters: 算法的参数，是一个字典类型，默认为 None。如果为 None，则会初始化为空字典。
        """
        # 如果没有传入参数，则将参数初始化为空字典
        if parameters is None:
            parameters = {}
        # 调用父类 SelectionAlgorithm 的构造函数进行初始化
        # 传递数据库连接器、参数和默认参数
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS
        )
        # 初始化工作负载为 None，后续会在计算最佳索引时进行赋值
        self.workload = None
        # 初始化最大运行时间为 24 小时（以分钟为单位）
        self.max_runtime_minutes = 60 * 24
        # 初始化成本评估时间为 0，用于记录成本评估所花费的总时间
        self.cost_evaluation_time = 0


    def reset(self, parameters):
        """
        重置算法的参数和状态，将算法恢复到初始状态，以便重新开始索引选择过程。

        :param parameters: 新的参数，包含了算法运行所需的各种配置信息，如存储预算、最大索引宽度等。
        """
        # 标记算法未运行，确保在重置后可以重新开始运行
        self.did_run = False
        # 更新参数为传入的新参数
        self.parameters = parameters
        # 存储缺失参数的默认值，确保所有必要的参数都有值
        # 删除数据库中的所有索引，为新的索引选择过程做准备
        self.database_connector.drop_indexes()
        self.cost_evaluation.what_if.drop_all_simulated_indexes()
        # 如果参数中包含成本估计方法
        if "cost_estimation" in self.parameters:
            # 获取成本估计方法
            estimation = self.parameters["cost_estimation"]
            # 设置成本评估的估计方法
            self.cost_evaluation.cost_estimation = estimation
        # 获取存储预算，单位为 MB
        self.budget = (self.parameters["budget_MB"])
        # 获取最大索引宽度，即一个索引最多可以包含的列数
        self.max_index_width = self.parameters["max_index_width"]
        # 重置工作负载为 None，确保后续可以重新加载工作负载
        self.workload = None
        # 获取最小成本改进值，用于判断新的索引组合是否值得选择
        self.min_cost_improvement = self.parameters["min_cost_improvement"]
        # 重置成本评估时间为 0，以便重新记录成本评估所花费的时间
        self.cost_evaluation_time = 0
        self.partition_num = self.parameters["partition_num"]




    def _should_stop_due_to_time(self):
        """
        判断是否由于时间限制而停止算法运行。

        :return: 如果超过最大运行时间则返回 True，否则返回 False。
        """
        # 获取当前时间戳
        current_time = time.time()
        # 计算从算法开始到现在所消耗的时间（秒）
        consumed_time = current_time - self.start_time
        # 判断消耗的时间是否超过了最大运行时间（将最大运行时间从分钟转换为秒）
        if consumed_time > self.max_runtime_minutes * 60:
            # 记录日志，提示由于时间限制停止算法，并输出已消耗的时间（分钟）
            logging.debug(
                (
                    "Stopping because of timing constraints. "
                    f"Time: {consumed_time / 60:.2f} minutes."
                )
            )
            # 如果超过最大运行时间，返回 True 表示停止算法
            return True
        # 如果未超过最大运行时间，返回 False 表示继续算法
        return False


    def _group_partition_index(self, single_attribute_index_candidates):
        """
        对单属性索引候选进行分组，将具有相同列名的索引分组在一起。

        :param single_attribute_index_candidates: 单属性索引候选列表，列表中的每个元素代表一个单属性索引。
        :return: 分组后的索引候选列表，每个子列表包含具有相同列名的索引。
        """

        index_candidates_with_partition = [[] for _ in range(self.partition_num)]
        for index in single_attribute_index_candidates:
            index_candidates_with_partition[int(index.table().name.split("_1_prt_p")[-1])].append(index)
        # 遍历所有分组后的索引候选
        for _ in index_candidates_with_partition:
            # 计算当前组中所有索引的总大小
            self._total_index_size(_)
            # 对当前组中的索引进行排序
            _.sort(key=functools.cmp_to_key(indexsort))
        # 返回分组并排序后的索引候选列表
        return index_candidates_with_partition

    
    def _calculate_best_indexes(self, workload):
        """
        计算最佳索引组合。

        :param workload: 工作负载对象，包含了数据库查询的相关信息。
        :return: 最佳索引组合，是一个包含 Index 对象的列表。
        """
        self.cost_evaluation.what_if.drop_all_simulated_indexes()
        # 重置成本评估时间为 0，用于记录本次计算中成本评估所花费的总时间
        self.cost_evaluation_time = 0
        # 记录日志，提示开始计算最佳索引
        logging.info("Calculating best indexes Extend_partition")
        # 将传入的工作负载赋值给类的属性，以便后续使用
        self.workload = workload
        # 获取工作负载中所有可能的单属性索引候选
        single_attribute_index_candidates = workload.potential_indexes()
        single_attribute_index_candidates = self._group_partition_index(single_attribute_index_candidates)
        # for indexs in single_attribute_index_candidates:
        #     for _ in indexs:
        #         print(f"索引 ： {_} , 大小 : {_.estimated_size}")
        #     print("111111111111111111111111111111111")
        # 复制单属性索引候选列表，作为扩展属性候选列表
        extension_attribute_candidates = deepcopy(single_attribute_index_candidates)

        # 记录算法开始的时间戳
        self.start_time = time.time()
        self.workload_partition_list = []
        self.index_combination_list = []
        _index_combination = []
        for i in range(self.partition_num):
            # print(f"分区:{i}")
            queries_partition = workload.queries[i::self.partition_num]
            workload_partition = Workload(queries_partition)
            workload_partition.budget = self.budget / self.partition_num
            self.workload_partition_list.append(workload_partition)

            # 当前的索引组合，初始为空列表
            index_combination = []
            # 当前索引组合的总大小，初始为 0
            index_combination_size = 0
            # 评估步骤中的最佳索引组合，初始为空列表，收益与大小比初始为 0，成本初始为 None
            best = {"combination": [], "benefit_to_size_ratio": 0, "cost": None}

            # 存储 HypoPG 估计的索引大小
            # 记录开始计算当前成本的时间戳
            start_time = time.time()
            # 计算当前索引组合下工作负载的成本
            current_cost = self.cost_evaluation.calculate_cost(
                workload_partition, index_combination, store_size=False
            )
            # 累加成本评估所花费的时间
            self.cost_evaluation_time += round(float(time.time() - start_time), 2)

            # 记录初始成本，用于后续计算成本节省比例
            self.initial_cost = current_cost
            # 进入循环，直到没有成本改进时停止
            while True:
                # 筛选出在预算范围内的单属性索引候选
                single_attribute_index_candidates[i] = self._get_candidates_within_budget(
                    index_combination_size, single_attribute_index_candidates[i]
                )
                # 遍历筛选后的单属性索引候选
                for candidate in single_attribute_index_candidates[i]:
                    # 仅处理不在当前索引组合中的单属性索引
                    if candidate not in index_combination:
                        # 评估将当前候选索引添加到当前索引组合后的效果
                        self._evaluate_combination(
                            index_combination + [candidate], best, current_cost, workload=workload_partition
                        )
                    # 判断是否由于时间限制而停止算法
                    if self._should_stop_due_to_time() is True:
                        # 如果超过时间限制，返回当前索引组合
                        return index_combination

                # 遍历扩展属性候选列表
                for attribute in extension_attribute_candidates[i]:
                    # 通过将列附加到现有索引上来生成多列索引
                    self._attach_to_indexes(index_combination, attribute, best, current_cost, workload=workload_partition)

                    # 判断是否由于时间限制而停止算法
                    if self._should_stop_due_to_time() is True:
                        # 如果超过时间限制，返回当前索引组合
                        return index_combination

                # 如果最佳索引组合的收益与大小比小于等于 0，说明没有成本改进，跳出循环
                if best["benefit_to_size_ratio"] <= 0:
                    break

                # 更新当前索引组合为最佳索引组合
                index_combination = best["combination"]
                # 计算当前索引组合的总大小
                index_combination_size = self._total_index_size(index_combination)

                # 记录日志，输出当前成本节省比例、初始成本节省比例和当前存储大小
                logging.info(
                    "Add index. Current cost savings: "
                    f"{(1 - best['cost'] / current_cost) * 100:.3f}, "
                    f"initial {(1 - best['cost'] / self.initial_cost) * 100:.3f}. "
                    f"Current storage: {index_combination_size:.2f}"
                )

                # 重置最佳索引组合的收益与大小比为 0
                best["benefit_to_size_ratio"] = 0
                # 更新当前成本为最佳索引组合的成本
                current_cost = best["cost"]



            # 扁平化最佳索引组合，将嵌套列表转换为一维列表
            self.index_combination_list.append(index_combination)
            for _ in index_combination:
                # print(f"索引 ： {_} , 大小 : {_.estimated_size}")
                _index_combination.append(_)
        # 计算算法的总执行时间
        total_execution_time = round(float(time.time() - self.start_time), 2)
        # 输出算法找到索引所花费的总时间和纯计算时间
        print("Extend_partition takes {} time to find indexes, and pure time is {}\n".format(total_execution_time,
                                                                                   total_execution_time - self.cost_evaluation_time))
        # 返回扁平化后的最佳索引组合
        return _index_combination

    def _attach_to_indexes(self, index_combination, attribute, best, current_cost, workload=None):
        """
        通过将给定的属性列附加到现有索引上来生成多列索引，并评估新的索引组合。

        :param index_combination: 当前的索引组合，是一个包含 Index 对象列表的列表。
        :param attribute: 要附加到现有索引的属性，是一个包含 Index 对象的列表，且第一个索引必须是单属性索引。
        :param best: 评估步骤中的最佳索引组合，是一个字典，包含组合、收益与大小比和成本。
        :param current_cost: 当前索引组合下工作负载的成本。
        """
        # 断言传入的属性的第一个索引是单属性索引，确保后续操作的正确性
        assert (
            attribute.is_single_column() is True
        ), "Attach to indexes called with multi column index"

        # 遍历当前索引组合中的每个索引
        for position, index in enumerate(index_combination):
            # 如果当前索引的列数已经达到最大索引宽度，跳过该索引
            if len(index.columns) >= self.max_index_width:
                continue
            # 检查当前索引是否可以通过传入的属性进行扩展
            if index.appendable_by(attribute):
                # 将当前索引和属性组合成新的索引
                new_index = Index(index.columns + attribute.columns)
                # 如果新的索引已经存在于当前索引组合中，跳过该索引
                if new_index in index_combination:
                    continue
                # 复制当前索引组合，避免修改原始组合
                new_combination = index_combination.copy()
                # 删除当前位置的索引
                del new_combination[position]
                # 将新的索引添加到索引组合的末尾
                new_combination.append(new_index)
                # 调用 _evaluate_combination 方法，评估新的索引组合
                self._evaluate_combination(
                    new_combination,
                    best,
                    current_cost,
                    self._total_index_size(index_combination),
                    workload = workload
                )


    def _get_candidates_within_budget(self, index_combination_size, candidates):
        """
        筛选出在预算范围内的索引候选。

        :param index_combination_size: 当前索引组合的总大小，单位为字节。
        :param candidates: 索引候选列表，每个候选是一个包含 Index 对象的列表。
        :return: 在预算范围内的索引候选列表。
        """
        # 用于存储在预算范围内的索引候选
        new_candidates = []
        # 遍历所有索引候选
        for candidate in candidates:
            # 如果索引的估计大小为 None
            if candidate.estimated_size is None:
                # 则将该索引的大小视为 0
                candidate.estimated_size = 0
            # 检查当前候选的总大小加上当前索引组合的总大小是否在预算范围内
            if candidate.estimated_size + index_combination_size <= mb_to_b(self.budget / self.partition_num):
                # 如果在预算范围内，则将该候选添加到新的候选列表中
                new_candidates.append(candidate)

        # 返回在预算范围内的索引候选列表
        return new_candidates


    def _total_index_size(self, indexes):
        """
        计算给定索引列表的总大小。

        :param indexes: 索引列表，其中每个元素是一个 Index 对象。
        :return: 索引列表的总大小，单位为字节。
        """
        # 初始化总大小为 0
        total_size = 0
        # 遍历索引列表中的每个索引
        for _ in indexes:
            # 如果索引的估计大小为 None
            if _.estimated_size is None:
                # 调用数据库连接器的方法获取该索引的大小
                self.cost_evaluation.estimate_size(_)
            # 累加该索引的估计大小到总大小中
            total_size += _.estimated_size
        # 返回索引列表的总大小
        return total_size



    def _evaluate_combination(
        self, index_combination, best, current_cost, old_index_size=0, workload=None
    ):
        """
        评估给定的索引组合，根据成本和存储大小确定是否更新最佳索引组合。

        :param index_combination: 待评估的索引组合，是一个嵌套列表，每个子列表包含 Index 对象。
        :param best: 存储当前最佳索引组合的字典，包含组合、收益与大小比和成本。
        :param current_cost: 当前索引组合下工作负载的成本。
        :param old_index_size: 旧索引组合的总大小，默认为 0。
        """

        # 记录开始计算成本的时间戳
        start_time = time.time()
        # 计算当前索引组合下工作负载的成本
        cost = self.cost_evaluation.calculate_cost(
            workload, index_combination, store_size=False
        )
        # 累加成本评估所花费的时间
        self.cost_evaluation_time += round(float(time.time() - start_time), 2)

        # 如果新的成本乘以最小成本改进值大于等于当前成本，则不更新最佳索引组合
        if (cost * self.min_cost_improvement) >= current_cost:
            return
        # 计算成本节省（收益）
        benefit = current_cost - cost

        # 记录开始计算新索引大小的时间戳
        start_time = time.time()
        # 计算新索引组合的总大小
        new_index_size = self._total_index_size(index_combination)
        # 累加计算新索引大小所花费的时间
        self.cost_evaluation_time += round(float(time.time() - start_time), 2)

        # 计算新索引大小与旧索引大小的差值
        new_index_size_difference = new_index_size - old_index_size
        # 如果新索引大小有变化但差值为 0，使用当前最佳索引组合的收益与大小比
        if new_index_size_difference == 0 and new_index_size != 0:
            ratio = best["benefit_to_size_ratio"]
        else:
            # 确保新索引大小差值不为 0
            assert new_index_size_difference != 0, "Index size difference should not be 0!"
            # 计算收益与大小比
            ratio = benefit / new_index_size_difference

        # 计算新索引组合的总大小
        total_size = sum(index.estimated_size for index in index_combination)

        # 如果新的收益与大小比大于等于当前最佳值，且总大小在预算范围内
        if ratio >= best["benefit_to_size_ratio"] and total_size <= mb_to_b(self.budget / self.partition_num):
            # 记录日志，输出新的最佳成本和总大小
            logging.debug(
                f"new best cost and size: {cost}\t" f"{(total_size):.2f}MB"
            )
            # 更新最佳索引组合
            best["combination"] = index_combination
            # 更新最佳收益与大小比
            best["benefit_to_size_ratio"] = ratio
            # 更新最佳成本
            best["cost"] = cost
