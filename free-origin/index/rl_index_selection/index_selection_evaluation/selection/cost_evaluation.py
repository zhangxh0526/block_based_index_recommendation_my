import datetime
import time
import logging
from logging import exception

from .what_if_index_creation import WhatIfIndexCreation
from .param import *

from dateutil.relativedelta import relativedelta

import datetime
import decimal
import re
from cmath import inf


class CostEvaluation:
    def __init__(self, db_connector, cost_estimation="whatif"):
        """
        初始化 CostEvaluation 类的实例。

        :param db_connector: 数据库连接器对象，用于与数据库进行交互。
        :param cost_estimation: 成本估计方法，默认为 "whatif"。
        """
        # 记录调试信息，表明开始初始化成本评估
        logging.debug("Init cost evaluation")
        # 存储数据库连接器对象，用于后续与数据库的交互
        self.db_connector = db_connector
        # 存储成本估计方法
        self.cost_estimation = cost_estimation
        # 注释掉的代码，原本可能用于重新设置成本估计方法
        # self.cost_estimation = "whatif"
        # 记录信息日志，显示当前使用的成本估计方法
        logging.info("Cost estimation with " + self.cost_estimation)
        # 创建 WhatIfIndexCreation 类的实例，用于模拟索引的创建和删除
        self.what_if = WhatIfIndexCreation(db_connector)
        # 初始化当前索引集合，用于存储当前存在的索引
        self.current_indexes = set()

        # 注释掉的断言语句，原本用于确保模拟索引的数量和当前索引的数量一致
        # assert len(self.what_if.all_simulated_indexes()) == len(self.current_indexes)

        # 初始化成本请求计数器，记录成本请求的次数
        self.cost_requests = 0
        # 初始化缓存命中计数器，记录缓存命中的次数
        self.cache_hits = 0
        # 初始化查询成本缓存，结构为 {(query_object, relevant_indexes): cost}
        # Cache structure:
        # {(query_object, relevant_indexes): cost}
        self.cache = {}

        # 初始化查询成本和计划缓存，结构为 {(query_object, relevant_indexes): (cost, plan)}
        # Cache structure:
        # {(query_object, relevant_indexes): (cost, plan)}
        self.cache_plans = {}

        # 初始化完成标志，用于表示成本评估是否完成
        self.completed = False
        # 注释说明在初始化时不需要删除假设的索引，因为它们是每个连接单独创建的
        # It is not necessary to drop hypothetical indexes during __init__().
        # These are only created per connection. Hence, non should be present.

        # 初始化相关索引缓存，用于存储查询和相关索引的映射
        self.relevant_indexes_cache = {}

        # 初始化成本计算时间，使用 timedelta 对象记录成本计算所花费的总时间
        self.costing_time = datetime.timedelta(0)

        # 调用方法获取每个分区的区域映射信息
        self.get_zonemaps_per_partition()

        # 初始化查询 ID 映射，用于减少缓存大小
        self.query_id = {}
        # 初始化索引 ID 映射，用于减少缓存大小
        self.index_id = {}


    def reset(self):
        """
        重置 CostEvaluation 实例的状态。

        该方法用于清除当前索引集合，并删除数据库中所有的索引。
        通常在需要重新开始成本评估或者重新初始化索引状态时调用。
        """
        # 清空当前索引集合，将其重置为空集合
        self.current_indexes = set()
        # 调用数据库连接器的 drop_indexes 方法，删除数据库中所有的索引
        self.db_connector.drop_indexes()
        self.db_connector.drop_simulated_indexes()


    def estimate_size(self, index):
        """
        估计指定索引的大小。

        如果索引已经存在于当前索引集合中，则尝试查询其大小；
        如果索引不存在，则模拟或创建该索引以进行后续大小估计。

        :param index: 要估计大小的索引对象。
        """
        # TODO: Refactor: It is currently too complicated to compute
        # We must search in current indexes to get an index object with .hypopg_oid
        # 初始化结果变量，用于存储在当前索引集合中找到的匹配索引对象
        result = None
        # 遍历当前索引集合中的每个索引
        for i in self.current_indexes:
            # 检查当前索引是否与传入的索引相等
            if index == i:
                # 如果相等，则将该索引赋值给结果变量
                result = i
                # 找到匹配索引后，跳出循环
                break
        # 如果在当前索引集合中找到了匹配的索引
        if result:
            # Index does currently exist and size can be queried
            # 检查传入的索引是否已经有估计的大小
            if not index.estimated_size:
                # 如果没有估计的大小，则调用 what_if 实例的 estimate_index_size 方法
                # 传入匹配索引的 hypopg_oid 属性，以获取索引的估计大小，并赋值给传入的索引
                index.estimated_size = self.what_if.estimate_index_size(result.hypopg_oid)
        # 如果在当前索引集合中没有找到匹配的索引
        else:
            # 调用 _simulate_or_create_index 方法，模拟或创建该索引，并设置 store_size 为 True
            self._simulate_or_create_index(index, store_size=True)



    def which_indexes_utilized_and_cost(self, query, indexes):
        """
        确定查询中使用了哪些索引，并计算查询成本。

        :param query: 要执行的查询对象。
        :param indexes: 可能被查询使用的索引列表。
        :return: 一个元组，包含被查询使用的索引集合和查询的总成本。
        """
        # 准备成本计算，模拟或创建缺失的索引，删除不需要的索引
        self._prepare_cost_calculation(indexes, store_size=True)

        # 获取查询的执行计划
        plan = self.db_connector.get_plan(query)
        # 从执行计划中提取查询的总成本
        cost = plan["Total Cost"]
        # 将执行计划转换为字符串，方便后续查找索引
        plan_str = str(plan)

        # 初始化一个集合，用于存储被查询使用的索引
        recommended_indexes = set()

        # 遍历当前索引集合，而不是传入的 `indexes` 列表
        # 因为不能保证 `indexes` 中的所有项都设置了 hypopg_name
        # 这是由于 _prepare_cost_calculation 方法只会创建尚未存在的索引
        # 如果没有为某个索引对象创建假设索引，则不会为其分配 hypopg_name
        # 但是，当前索引集合中的所有项必须在 `indexes` 中有对应的项
        for index in self.current_indexes:
            # 断言当前索引必须存在于传入的 `indexes` 列表中
            # 确保 _prepare_cost_calculation 方法正常工作
            assert (
                index in indexes
            ), "Something went wrong with _prepare_cost_calculation."

            # 检查执行计划字符串中是否包含当前索引的 hypopg_name
            # 如果不包含，则跳过该索引
            if index.hypopg_name not in plan_str:
                continue
            # 如果包含，则将该索引添加到推荐索引集合中
            recommended_indexes.add(index)

        # 返回被查询使用的索引集合和查询的总成本
        return recommended_indexes, cost


    def calculate_cost(self, workload, indexes, store_size=False):
        """
        计算给定工作负载在指定索引下的总成本。

        :param workload: 工作负载对象，包含多个查询。
        :param indexes: 索引列表，用于评估查询成本。
        :param store_size: 是否存储索引大小，默认为 False。
        :return: 工作负载的总成本。
        """
        # 断言成本评估未完成，确保可以继续使用该评估器
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        # 暂时注释掉的代码，原本用于准备成本计算
        # self._prepare_cost_calculation(indexes, store_size=store_size)
        # 初始化总成本为 0
        total_cost = 0

        # TODO: 对于频繁运行的查询，提高其查询成本
        # 遍历工作负载中的每个查询
        for query in workload.queries:
            # 增加成本请求计数器
            self.cost_requests += 1
            # 从缓存中获取查询成本
            cost = self._request_cache(query, indexes)
            # 检查查询是否应被跳过
            if not self._is_skipped(query):
                # 如果不跳过，则将查询成本乘以查询频率并累加到总成本中
                total_cost += cost * query.frequency
        # 返回总成本
        return total_cost

    def _is_skipped(self, query):
        """
        判断查询是否应被跳过。

        通过分区的区域映射信息和查询谓词进行比较，
        如果查询的谓词范围与分区的区域映射范围不相交，则跳过该查询。

        :param query: 查询对象。
        :return: 如果查询应被跳过，返回 True；否则返回 False。
        """
        # 对查询文本进行谓词拆分，获取查询信息
        workload_inf = self.predicate_splitting(query.text)
        # 遍历所有分区的区域映射信息
        for partition_id in self.zoneMaps:
            # 检查查询文本中是否包含当前分区 ID
            if partition_id in query.text:
                # 遍历查询信息中的每个列
                for _ in workload_inf[0]:
                    # 检查该列是否在当前分区的区域映射信息中
                    if _ in self.zoneMaps[partition_id]:
                        # 获取该列在分区中的最小值
                        min = self.zoneMaps[partition_id][_][0]
                        # 获取该列在分区中的最大值
                        max = self.zoneMaps[partition_id][_][1]
                        # 判断查询谓词范围与分区区域映射范围是否不相交
                        if (min is None or max is None) or (workload_inf[0][_][0] > max or workload_inf[0][_][1] < min):
                            # 如果不相交，则返回 True 表示跳过该查询
                            return True
                # 找到包含分区 ID 的查询后，跳出循环
                break
        # 如果没有找到需要跳过的查询，则返回 False
        return False

    def _obtain_partition(self, query):
        """
        获取查询所属的分区。

        目前该方法直接返回 1，可能需要根据实际情况进行实现。

        :param query: 查询对象。
        :return: 查询所属的分区 ID。
        """
        return 1

    def calculate_cost_and_plans(self, workload, indexes, store_size=False):
        """
        计算给定工作负载在指定索引下的总成本，并返回每个查询的执行计划和成本。

        :param workload: 工作负载对象，包含多个查询。
        :param indexes: 索引列表，用于评估查询成本。
        :param store_size: 是否存储索引大小，默认为 False。
        :return: 一个元组，包含工作负载的总成本、每个查询的执行计划列表和每个查询的成本列表。
        """
        # 断言成本评估未完成，确保可以继续使用该评估器
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        # 记录开始时间
        start_time = datetime.datetime.now()

        # 初始化总成本为 0
        total_cost = 0
        # 初始化执行计划列表
        plans = []
        # 初始化查询成本列表
        costs = []

        # 遍历工作负载中的每个查询
        for query in workload.queries:
            # 增加成本请求计数器
            self.cost_requests += 1
            # 从缓存中获取查询成本和执行计划
            cost, plan = self._request_cache_plans(query, indexes)
            # 检查查询是否应被跳过
            if self._is_skipped(query):
                # 如果跳过，则将查询成本设为 0
                cost = 0
            # 将查询成本乘以查询频率并累加到总成本中
            total_cost += cost * query.frequency
            # 将执行计划添加到执行计划列表中
            plans.append(plan)
            # 将查询成本添加到查询成本列表中
            costs.append(cost)

        # 记录结束时间
        end_time = datetime.datetime.now()
        # 累加成本计算所花费的时间
        self.costing_time += end_time - start_time

        # 返回总成本、执行计划列表和查询成本列表
        return total_cost, plans, costs


    # Creates the current index combination by simulating/creating
    # missing indexes and unsimulating/dropping indexes
    # that exist but are not in the combination.
    def _prepare_cost_calculation(self, indexes, store_size=False):
        """
        准备成本计算，通过模拟/创建缺失的索引和取消模拟/删除不在组合中的现有索引，来创建当前的索引组合。

        :param indexes: 期望的索引列表
        :param store_size: 是否存储索引大小，默认为 False
        """
        # 遍历期望的索引列表中当前不存在的索引
        for index in set(indexes) - self.current_indexes:
            # 模拟或创建缺失的索引
            self._simulate_or_create_index(index, store_size=store_size)
        # 遍历当前存在但不在期望索引列表中的索引
        for index in self.current_indexes - set(indexes):
            # 取消模拟或删除多余的索引
            self._unsimulate_or_drop_index(index)

        # 断言当前索引集合与期望的索引集合一致
        assert self.current_indexes == set(indexes)

    def _simulate_or_create_index(self, index, store_size=False):
        """
        模拟或创建指定的索引。

        :param index: 要模拟或创建的索引
        :param store_size: 是否存储索引大小，默认为 False
        """
        # 改动
        try:
            # 根据成本估计方法选择不同的索引创建方式
            if self.cost_estimation == "actual_runtimes":
                # 实际运行时间模式下，使用数据库连接器创建索引
                #self.what_if.simulate_index(index, store_size=store_size)
                self.db_connector.create_index(index)
            elif self.cost_estimation == "whatif":
                # whatif 模式下，使用数据库连接器创建索引
                # self.db_connector.create_index(index)
                self.what_if.simulate_index(index, store_size=store_size)
        except Exception as e:
            logging.error(f"An unexpected error occurred while simulate index {index}: {e}")
            # 捕获异常，若索引已存在则打印提示信息
            print("Index {} already exist !".format(index))

        # 将创建的索引添加到当前索引集合中
        self.current_indexes.add(index)

    def _unsimulate_or_drop_index(self, index):
        """
        取消模拟或删除指定的索引。

        :param index: 要取消模拟或删除的索引
        """
        # 改动
        try:
            # 根据成本估计方法选择不同的索引删除方式
            if self.cost_estimation == "actual_runtimes":
                # 实际运行时间模式下，使用数据库连接器删除索引
                #self.what_if.drop_simulated_index(index)
                self.db_connector.drop_index(index)
            elif self.cost_estimation == "whatif":
                # whatif 模式下，使用数据库连接器删除索引
                #self.db_connector.drop_index(index)
                self.what_if.drop_simulated_index(index)
        except Exception as e:
            logging.error(f"An unexpected error occurred while dropping index {index}: {e}")
            # 捕获异常，若索引不存在则打印提示信息
            print("Index {} does not exist !".format(index))

        # 若索引存在于当前索引集合中，则将其移除
        if index in self.current_indexes:
            self.current_indexes.remove(index)

    def _get_cost(self, query):
        """
        获取查询的成本。

        :param query: 查询对象
        :return: 查询的成本
        """
        # 根据成本估计方法选择不同的成本获取方式
        if self.cost_estimation == "whatif":
            # whatif 模式下，使用数据库连接器获取查询成本
            return self.db_connector.get_cost(query)
        elif self.cost_estimation == "actual_runtimes":
            # 实际运行时间模式下，使用数据库连接器执行查询并获取运行时间
            runtime = self.db_connector.exec_query(query)[0]
            return runtime

    def _get_cost_plan(self, query):
        """
        获取查询的成本和执行计划。

        :param query: 查询对象
        :return: 一个元组，包含查询的总成本和执行计划
        """
        # 使用数据库连接器获取查询的执行计划
        query_plan = self.db_connector.get_plan(query)
        # 返回查询的总成本和执行计划
        return query_plan["Total Cost"], query_plan

    def complete_cost_estimation(self):
        """
        完成成本估计，清理当前索引集合并打印缓存命中信息。
        """
        # 打印总缓存命中次数
        print("Total cache hit:{}".format(self.cache_hits))
        # 遍历当前索引集合的副本
        for index in self.current_indexes.copy():
            # 取消模拟或删除每个索引
            self._unsimulate_or_drop_index(index)

        # 断言当前索引集合为空
        assert self.current_indexes == set()

    def _request_cache(self, query, indexes):
        """
        从缓存中获取查询成本，如果缓存未命中则从数据库系统请求成本。

        :param query: 查询对象
        :param indexes: 索引列表
        :return: 查询成本
        """
        # 如果成本估计方法为 "actual_runtimes"
        if self.cost_estimation == "actual_runtimes":
            # 准备成本计算，不存储索引大小
            self._prepare_cost_calculation(indexes, store_size=False)
            # 获取查询成本
            cost = self._get_cost(query)
            return cost
        else:
            # 生成查询文本和索引集合的哈希值
            q_i_hash = (query.text, frozenset(indexes))
            # 检查该哈希值是否在相关索引缓存中
            if q_i_hash in self.relevant_indexes_cache:
                # 如果存在，从缓存中获取相关索引
                relevant_indexes = self.relevant_indexes_cache[q_i_hash]
            else:
                # 如果不存在，计算相关索引
                relevant_indexes = self._relevant_indexes(query, indexes)
                # 将相关索引存入缓存
                self.relevant_indexes_cache[q_i_hash] = relevant_indexes

            # 检查查询文本和相关索引集合的哈希值是否在缓存中
            r_i_hash = (query.text, frozenset(relevant_indexes))
            if r_i_hash in self.cache:
                # 如果缓存命中，增加缓存命中计数器
                self.cache_hits += 1
                # 从缓存中返回查询成本
                return self.cache[r_i_hash]
            # 如果缓存未命中，从数据库系统请求成本
            else:
                # 准备成本计算，不存储索引大小
                self._prepare_cost_calculation(indexes, store_size=False)
                # 获取查询成本
                cost = self._get_cost(query)
                # 将查询成本存入缓存
                self.cache[r_i_hash] = cost
                return cost

    def _trans_query_2_id(self, query):
        """
        将查询转换为唯一的 ID。

        :param query: 查询对象
        :return: 查询的唯一 ID
        """
        # 检查查询文本是否已经存在于查询 ID 映射中
        if query.text not in self.query_id:
            # 如果不存在，为其分配一个新的 ID
            self.query_id[query.text] = len(self.query_id)
        # 返回查询的 ID
        return self.query_id[query.text]

    def _trans_index_2_id(self, index):
        """
        将索引转换为唯一的 ID。

        :param index: 索引对象
        :return: 索引的唯一 ID
        """
        # 检查索引列是否已经存在于索引 ID 映射中
        if index.columns not in self.index_id:
            # 如果不存在，为其分配一个新的 ID
            self.index_id[index.columns] = len(self.index_id)
        # 返回索引的 ID
        return self.index_id[index.columns]

    def _trans_indexes(self, indexes):
        """
        将索引列表转换为对应的 ID 列表。

        :param indexes: 索引列表
        :return: 索引 ID 列表
        """
        # 初始化索引 ID 列表
        indexes_ids = []
        # 遍历索引列表
        for _ in indexes:
            # 将每个索引转换为 ID 并添加到列表中
            indexes_ids.append(self._trans_index_2_id(_))
        return indexes_ids


    def _request_cache_plans(self, query, indexes):
        """
        从缓存中获取查询的成本和执行计划，如果缓存未命中则从数据库系统请求。

        :param query: 查询对象
        :param indexes: 索引列表
        :return: 一个元组，包含查询的成本和执行计划
        """
        if self.cost_estimation == "actual_runtimes":
            # 若成本估计方法为实际运行时间，准备成本计算并存储索引大小
            self._prepare_cost_calculation(indexes, store_size=True)
            # 执行查询并获取成本和执行计划
            cost, plan = self.db_connector.exec_query(query)
            return cost, plan
        else:
            # 将查询转换为唯一 ID
            query_id = self._trans_query_2_id(query)
            #indexes_id = self._trans_indexes(indexes)
            #q_i_hash = (query_id, frozenset(indexes_id))
            # q_i_hash = (frozenset(query.columns), frozenset(indexes))
            #if q_i_hash in self.relevant_indexes_cache:
            #    relevant_indexes = self.relevant_indexes_cache[q_i_hash]
            #else:
            # 计算与查询相关的索引
            relevant_indexes = self._relevant_indexes(query, indexes)
            #    self.relevant_indexes_cache[q_i_hash] = relevant_indexes

            # 检查查询和对应的相关索引是否在缓存中
            # q_cache_hash = (query.text, frozenset(relevant_indexes))

            #relevant_index_ids = self._trans_indexes(relevant_indexes)
            # 生成查询和相关索引的哈希值
            q_cache_hash = (query_id, frozenset(relevant_indexes))

            if q_cache_hash in self.cache:
                # 若缓存命中，增加缓存命中计数器
                self.cache_hits += 1
                # 从缓存中返回成本和执行计划
                return self.cache[q_cache_hash]
            # 若缓存未命中，从数据库系统请求成本和执行计划
            else:
                # 记录开始时间
                start_time = time.time()
                # 准备成本计算，不存储索引大小
                self._prepare_cost_calculation(indexes, store_size=False)
                # 记录结束时间
                end_time = time.time()
                # 累加索引创建时长
                selection_args.index_create_duration += int(end_time - start_time)
                # 获取查询的成本和执行计划
                cost, plan = self._get_cost_plan(query)
                # 将成本和执行计划存入缓存
                self.cache[q_cache_hash] = (cost, plan)
                return cost, plan

    def calculate_cost_slalom(self, workload, indexes, store_size=False):
        """
        计算给定工作负载在指定索引下的总成本（Slalom 模式）。

        :param workload: 工作负载对象，包含多个查询
        :param indexes: 索引列表，用于评估查询成本
        :param store_size: 是否存储索引大小，默认为 False
        :return: 工作负载的总成本
        """
        # 断言索引数量不超过 1
        assert len(indexes) <= 1,"There should be a single index!"
        # 断言成本评估未完成
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        # self._prepare_cost_calculation(indexes, store_size=store_size)
        # 初始化总成本
        total_cost = 0

        # TODO: Make query cost higher for queries which are running often
        # 遍历工作负载中的每个查询
        for query in workload.queries:
            # 增加成本请求计数器
            self.cost_requests += 1
            # 若查询不被跳过
            if not self._is_skipped(query):
                # 累加查询成本
                total_cost += self._request_cache_slalom(query, indexes)
        # 遍历索引列表
        for index in indexes:
            # 取消模拟或删除索引
            self._unsimulate_or_drop_index(index)
        return total_cost

    def _request_cache_slalom(self, query, indexes):
        """
        从缓存中获取查询成本（Slalom 模式），如果缓存未命中则从数据库系统请求。

        :param query: 查询对象
        :param indexes: 索引列表
        :return: 查询成本
        """
        # 生成查询和索引的哈希值
        q_i_hash = (query.text, frozenset(indexes))
        if q_i_hash in self.relevant_indexes_cache:
            # 若哈希值在相关索引缓存中，获取相关索引
            relevant_indexes = self.relevant_indexes_cache[q_i_hash]
        else:
            # 若不在缓存中，计算相关索引
            relevant_indexes = self._relevant_indexes(query, indexes)
            # 将相关索引存入缓存
            self.relevant_indexes_cache[q_i_hash] = relevant_indexes

        # 检查查询和对应的相关索引是否在缓存中
        r_i_hash = (query.text, frozenset(relevant_indexes))
        if r_i_hash in self.cache:
            # 若缓存命中，增加缓存命中计数器
            self.cache_hits += 1
            # 从缓存中返回查询成本
            return self.cache[r_i_hash]
        # 若缓存未命中，从数据库系统请求成本
        else:
            # 准备成本计算，不存储索引大小
            self._prepare_cost_calculation(indexes, store_size=False)
            # 获取查询成本
            cost = self._get_cost_slalom(query)
            # 将查询成本存入缓存
            self.cache[r_i_hash] = cost
            return cost

    def _get_cost_slalom(self, query):
        """
        获取查询的成本（Slalom 模式）。

        :param query: 查询对象
        :return: 查询成本
        """
        if self.cost_estimation == "whatif":
            # 若成本估计方法为 whatif，使用数据库连接器获取成本
            return self.db_connector.get_cost(query)
        elif self.cost_estimation == "actual_runtimes":
            # 若成本估计方法为实际运行时间，执行查询并获取运行时间
            runtime = self.db_connector.exec_query(query)[0]
        return runtime


    # calculate the time of building index
    def get_index_build_cost(self, index, store_size=False):
        """
        计算创建索引所需的时间。

        :param index: 要创建的索引对象
        :param store_size: 是否存储索引大小，默认为 False
        :return: 创建索引所需的时间（秒）
        """
        # 断言索引仅包含一个列，因为 Slalom 只支持单列索引
        assert len(index.columns) == 1, "Slalom only refer to single column index!"
        # 记录开始时间
        time_start = time.time()
        # 使用数据库连接器创建索引
        self.db_connector.create_index(index)
        # 记录结束时间
        time_end = time.time()
        # 使用数据库连接器删除索引
        self.db_connector.drop_index(index)
        # 返回创建索引所需的时间
        return time_end - time_start

    @staticmethod
    def _relevant_indexes(query, indexes):
        """
        确定与查询相关的索引。

        :param query: 查询对象
        :param indexes: 索引列表
        :return: 与查询相关的索引的不可变集合
        """
        # 筛选出至少有一个列在查询列中的索引
        relevant_indexes = [
            x for x in indexes if any(c in query.columns for c in x.columns)
        ]
        # 返回不可变集合
        return frozenset(relevant_indexes)

    def calculate_cost_and_rows(self, workload, indexes, store_size=False):
        """
        计算给定工作负载在指定索引下的总成本和返回的总行数。

        :param workload: 工作负载对象，包含多个查询
        :param indexes: 索引列表，用于评估查询成本
        :param store_size: 是否存储索引大小，默认为 False
        :return: 一个元组，包含工作负载的总成本和返回的总行数
        """
        # 断言成本评估未完成，否则抛出异常
        assert (
                self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        # 记录开始时间
        start_time = datetime.datetime.now()

        # 初始化总成本
        total_cost = 0
        # 初始化返回的总行数
        rows = 0
        # 初始化跳过的查询数
        skipped_queries = 0

        # 遍历工作负载中的每个查询
        for query in workload.queries:
            # 增加成本请求计数器
            self.cost_requests += 1
            # 从缓存中获取查询的成本和执行计划
            cost, plan = self._request_cache_plans(query, indexes)
            # 如果查询不被跳过
            if not self._is_skipped(query):
                # 累加查询成本，考虑查询频率
                total_cost += cost * query.frequency
                if self.cost_estimation == "whatif":
                    # 检查执行计划中是否有子计划
                    if 'Plans' in plan:
                        if 'Plans' in plan['Plans'][0]:
                            if 'Plans' in plan['Plans'][0]['Plans'][0]:
                                if 'Plans' in plan['Plans'][0]['Plans'][0]['Plans'][0]:
                                    # 累加最内层子计划的实际行数
                                    rows += plan['Plans'][0]['Plans'][0]['Plans'][0]['Plans'][0]['Plan Rows']
                                else:
                                    # 累加倒数第二层子计划的实际行数
                                    rows += plan['Plans'][0]['Plans'][0]['Plans'][0]['Plan Rows']
                            else:
                                # 累加第二层子计划的实际行数
                                rows += plan['Plans'][0]['Plans'][0]['Plan Rows']
                        else:
                            # 累加第一层子计划的实际行数
                            rows += plan['Plans'][0]['Plan Rows']
                    else:
                        # 累加执行计划的实际行数
                        rows += plan['Plan Rows']
                elif self.cost_estimation == "actual_runtimes":
                    # 检查执行计划中是否有子计划
                    if 'Plans' in plan:
                        if 'Plans' in plan['Plans'][0]:
                            if 'Plans' in plan['Plans'][0]['Plans'][0]:
                                if 'Plans' in plan['Plans'][0]['Plans'][0]['Plans'][0]:
                                    # 累加最内层子计划的实际行数
                                    rows += plan['Plans'][0]['Plans'][0]['Plans'][0]['Plans'][0]['Actual Rows']
                                else:
                                    # 累加倒数第二层子计划的实际行数
                                    rows += plan['Plans'][0]['Plans'][0]['Plans'][0]['Actual Rows']
                            else:
                                # 累加第二层子计划的实际行数
                                rows += plan['Plans'][0]['Plans'][0]['Actual Rows']
                        else:
                            # 累加第一层子计划的实际行数
                            rows += plan['Plans'][0]['Actual Rows']
                    else:
                        # 累加执行计划的实际行数
                        rows += plan['Actual Rows']
            else:
                # 增加跳过的查询数
                skipped_queries += 1

        # 记录结束时间
        end_time = datetime.datetime.now()
        # 累加成本计算时间
        self.costing_time += end_time - start_time

        # 打印跳过的查询数
        print("****************Skipped queries:{}*****************".format(skipped_queries))

        # 返回总成本和总行数
        return total_cost, rows

    def get_zonemaps_per_partition(self):
        """
        获取每个分区的区域映射信息。

        :return: 分区信息字典
        """
        # 初始化区域映射字典
        self.zoneMaps = {}
        # 获取数据库中所有表的信息
        statement = ("select table_name from information_schema.tables where table_schema = 'public';")
        # 执行 SQL 语句并获取结果
        tables = self.db_connector.exec_fetch(statement, False)
        # 初始化分区信息字典
        partition_info = {}
        # 遍历每个表
        for table in tables:
            # 获取表名
            table_name = table[0]
            # 如果表名不包含 "prt"，则跳过该表
            if "prt" not in table_name:
                continue
            # 初始化该表的区域映射
            self.zoneMaps[table_name] = {}
            # 获取该表的所有列名
            statement = f"select column_name from information_schema.columns where table_name = '{table_name}';"
            # 执行 SQL 语句并获取结果
            columns = self.db_connector.exec_fetch(statement,False)
            # 初始化列值字典
            column_values = {}
            # 遍历每个列
            for column in columns:
                # 获取列名
                column_name = column[0]
                # 获取该列的最大值
                statement = f"select max({column_name}) from {table_name};"
                max = self.db_connector.exec_fetch(statement, False)[0][0]
                # 获取该列的最小值
                statement = f"select min({column_name}) from {table_name};"
                min = self.db_connector.exec_fetch(statement, False)[0][0]
                # 如果最大值是字符串类型，则跳过该列
                if isinstance(max, str):
                    continue
                # 如果最大值是 Decimal 类型，则转换为浮点数
                if isinstance(max, decimal.Decimal):
                    max = float(max)
                    min = float(min)
                # 如果最大值是日期类型，则转换为整数
                elif isinstance(max, datetime.date):
                    max = self.trans_date_to_int(max)
                    min = self.trans_date_to_int(min)
                # 将该列的最小值和最大值存入区域映射字典
                self.zoneMaps[table_name][column_name] = [min, max]

        # 返回分区信息字典
        return partition_info

    def trans_date_to_int(self, date):
        """
        将日期对象转换为整数表示。

        该函数将日期对象的年、月、日信息转换为一个整数，
        年占高位，月和日依次排列，方便进行日期的比较和计算。

        :param date: 要转换的日期对象
        :return: 转换后的整数表示
        """
        # 初始化日期整数为日期的日
        date_int = date.day
        # 加上月份乘以 100
        date_int += date.month * 100
        # 加上年份乘以 10000
        date_int += date.year * 10000
        return date_int

    def predicate_splitting(self, query):
        """
        对查询语句中的谓词进行拆分和解析。

        该函数从查询语句中提取 where 子句，并将其中的条件拆分为
        列名、下界和上界信息，存储在字典中。

        :param query: 要解析的查询语句
        :return: 包含查询信息的列表，列表中的每个元素是一个字典，
                 字典的键为列名，值为一个包含下界和上界的列表
        """
        # 初始化工作负载信息列表
        workload_inf = []
        # 初始化查询信息字典
        query_inf = {}
        # 使用正则表达式查找查询语句中的 where 子句
        where_clauses = re.findall(r'where(.*?);', query, flags=re.IGNORECASE)
        # 遍历每个 where 子句
        for where_clause in where_clauses:
            # 将 where 子句按 'and' 分割成多个条件
            conditions = where_clause.split('and')
            # 初始化列名
            column = None
            # 遍历每个条件
            for condition in conditions:
                # 初始化下界
                lower_bound = None
                # 初始化上界
                upper_bound = None

                # 如果条件中包含 'select'，则跳过该条件
                if 'select' in condition:
                    continue

                # 处理下界
                if '>' in condition or 'between' in condition:
                    if '>=' in condition:
                        # 如果条件中包含 '>='，则按 '>=' 分割条件
                        column, lower_bound = condition.split('>=')
                    elif '>' in condition:
                        # 如果条件中包含 '>'，则按 '>' 分割条件
                        column, lower_bound = condition.split('>')
                    else:
                        # 如果条件中包含 'between'，则按 'between' 分割条件
                        column, lower_bound = condition.split('between')
                    # 去除列名前后的空格，并取最后一个单词作为列名
                    column = column.strip().split()[-1]
                    # 如果列名不在查询信息字典中，则初始化该列的上下界为负无穷和正无穷
                    if column not in query_inf:
                        query_inf[column] = [-inf, inf]
                    # 去除下界前后的空格
                    lower_bound = lower_bound.strip()
                    if 'date' in lower_bound:
                        # 如果下界中包含 'date'，则调用 modify_date 方法处理
                        lower_bound = self.modify_date(lower_bound)
                    elif self._is_predicate_number(lower_bound):
                        # 如果下界是数字，则调用 modify_num 方法处理
                        lower_bound = self.modify_num(lower_bound)
                    else:
                        # 否则，取下界的第一个单词
                        lower_bound = lower_bound.split()[0]
                    # 更新该列的下界
                    query_inf[column][0] = lower_bound

                # 处理上界
                elif '<' in condition or self._is_predicate_number(condition.strip()):
                    if '<=' in condition:
                        # 如果条件中包含 '<='，则按 '<=' 分割条件
                        column, upper_bound = condition.split('<=')
                    elif '<' in condition:
                        # 如果条件中包含 '<'，则按 '<' 分割条件
                        column, upper_bound = condition.split('<')
                    else:
                        # 否则，将条件本身作为上界
                        upper_bound = condition.strip()
                    # 去除列名前后的空格，并取最后一个单词作为列名
                    column = column.strip().split()[-1]
                    # 如果列名不在查询信息字典中，则初始化该列的上下界为负无穷和正无穷
                    if column not in query_inf:
                        query_inf[column] = [-inf, inf]
                    # 去除上界前后的空格
                    upper_bound = upper_bound.strip()
                    if 'date' in upper_bound:
                        # 如果上界中包含 'date'，则调用 modify_date 方法处理
                        upper_bound = self.modify_date(upper_bound)
                    elif self._is_predicate_number(upper_bound):
                        # 如果上界是数字，则调用 modify_num 方法处理
                        upper_bound = self.modify_num(upper_bound)
                    else:
                        # 否则，取上界的第一个单词
                        upper_bound = upper_bound.split()[0]
                    # 更新该列的上界
                    query_inf[column][1] = upper_bound

                elif '=' in condition:
                    # 如果条件中包含 '='，则按 '=' 分割条件
                    column, lower_bound = condition.split('=')
                    # 去除列名前后的空格，并取最后一个单词作为列名
                    column = column.strip().split()[-1]
                    if 'date' in lower_bound:
                        # 如果下界中包含 'date'，则调用 modify_date 方法处理
                        lower_bound = self.modify_date(lower_bound)
                    elif self._is_predicate_number(lower_bound):
                        # 如果下界是数字，则调用 modify_num 方法处理
                        lower_bound = self.modify_num(lower_bound)
                    else:
                        # 否则，取下界的第一个单词
                        lower_bound = lower_bound.split()[0]
                    # 将该列的上下界都设置为下界的值
                    query_inf[column] = [lower_bound, lower_bound]

        # 将查询信息字典添加到工作负载信息列表中
        workload_inf.append(query_inf)
        return workload_inf


    def modify_num(self, num_string):
        """
        处理包含简单数学运算的数字字符串，返回计算结果。

        该函数可以处理包含加法和减法的数字字符串，例如 "1 + 2" 或 "3 - 1"。
        如果字符串中不包含运算符号，则直接返回字符串表示的数字。

        :param num_string: 包含数字和可能的运算符号的字符串
        :return: 计算结果
        """
        # 将字符串按空格分割成多个部分
        parts = num_string.split(" ")
        # 提取第一个数字并转换为浮点数
        num_1 = float(parts[0])
        # 如果字符串中包含加法运算符
        if '+' in num_string:
            # 提取第二个数字并转换为浮点数
            num_2 = float(parts[2])
            # 返回两数相加的结果
            return num_1 + num_2
        # 如果字符串中包含减法运算符且不是以负号开头
        elif '-' in num_string and num_string[0] != '-':
            # 提取第二个数字并转换为浮点数
            num_2 = float(parts[2])
            # 返回两数相减的结果
            return num_1 - num_2
        # 如果字符串中不包含运算符号
        else:
            # 直接返回第一个数字
            return num_1

    def modify_date(self, date_string):
        """
        处理日期字符串，将其转换为整数表示，并处理日期的加减运算。

        该函数可以处理包含日期和日期加减运算的字符串，例如 "date '2023-01-01' + interval '1' day"。
        如果字符串中不包含日期加减运算，则直接返回日期的整数表示。

        :param date_string: 包含日期和可能的日期加减运算的字符串
        :return: 处理后的日期的整数表示
        """
        # 将原始日期转换为datetime对象
        # 按单引号分割字符串
        parts = date_string.split("'")
        # 提取日期部分
        original_date = parts[1]
        # 将日期字符串转换为datetime对象
        original_date = datetime.datetime.strptime(original_date, "%Y-%m-%d")
        # 如果字符串中不包含 'interval' 关键字
        if 'interval' not in date_string:
            # 将日期转换为整数表示并返回
            return int(original_date.strftime('%Y%m%d'))
        # 提取操作符
        op = parts[2].strip()[0]
        # 提取数字部分并转换为整数
        num = int(parts[3].split()[0])
        # 提取时间单位
        unit = parts[4].split()[0]

        # 创建一个relativedelta对象用于日期计算
        delta = relativedelta()

        # 根据操作符和单位设置日期计算的参数
        if op == "-":
            # 如果操作符为减号，将数字取负
            num = -num
        if unit == "year":
            # 如果单位为年，设置 relativedelta 的 years 属性
            delta.years = num
        elif unit == "month":
            # 如果单位为月，分别设置 relativedelta 的 months 和 years 属性
            delta.months = int(num % 12)
            delta.years = int(num / 12)
        elif unit == "day":
            # 如果单位为天，设置 relativedelta 的 days 属性
            delta.days = num

        # 计算新的日期
        new_date = original_date + delta
        # 将新日期转换为整数表示
        new_date_integer = int(new_date.strftime("%Y%m%d"))
        # 返回新日期的整数表示
        return new_date_integer

    def _is_predicate_number(self, value):
        """
        判断一个字符串是否表示一个数字。

        该函数检查字符串是否以数字开头，或者以负号开头且第二个字符是数字。

        :param value: 要判断的字符串
        :return: 如果字符串表示一个数字，返回 True；否则返回 False
        """
        # 如果字符串以数字开头
        if value[0].isdigit():
            return True
        # 如果字符串以负号开头且第二个字符是数字
        if value[0] == "-" and value[1].isdigit():
            return True
        # 否则返回 False
        return False
