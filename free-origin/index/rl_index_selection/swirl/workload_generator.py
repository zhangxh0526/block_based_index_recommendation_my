import copy
import logging
import pickle
import random
import re

import numpy as np

import swirl.embedding_utils as embedding_utils
from index_selection_evaluation.selection.candidate_generation import (
    candidates_per_query,
    syntactically_relevant_indexes,
)
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.utils import get_utilized_indexes
from index_selection_evaluation.selection.workload import Query, Workload

from workload_embedder import WorkloadEmbedder

QUERY_PATH = "query_files"


class WorkloadGenerator(object):
    def __init__(
        self, config, workload_columns, random_seed, database_name, experiment_id=None, filter_utilized_columns=None, partition_num=0, violation_queries=set()
    ):
        """
        初始化 WorkloadGenerator 类的实例。

        :param config: 配置字典，包含基准测试类型、排除的查询类、频率变化等信息。
        :param workload_columns: 工作负载列的列表。
        :param random_seed: 随机数种子，用于确保结果的可重复性。
        :param database_name: 数据库名称。
        :param experiment_id: 实验 ID，用于区分不同的实验，默认为 None。
        :param filter_utilized_columns: 是否过滤使用的列，默认为 None。
        :param partition_num: 分区数量，默认为 0。
        :param violation_queries: 违规查询集合，默认为空集合。
        """
        # 检查配置中的基准测试类型是否支持
        assert config["benchmark"] in [
            "TPCH",
            "TPCDS",
            "JOB",
            "TPCHskew",
            "SSB"
        ], f"Benchmark '{config['benchmark']}' is currently not supported."

        self.config = config
        # 用于创建视图语句的区分
        self.violation_queries = violation_queries
        self.experiment_id = experiment_id
        self.filter_utilized_columns = filter_utilized_columns
        self.partition_num = partition_num
        # 确保分区数量大于 0
        assert self.partition_num > 0, "In Partition envs, you need specify the number of block"
        # 初始化 Python 内置的随机数生成器
        self.rnd = random.Random()
        # 设置随机数种子
        self.rnd.seed(random_seed)
        # 初始化 NumPy 的随机数生成器
        self.np_rnd = np.random.default_rng(seed=random_seed)

        self.workload_columns = workload_columns
        self.database_name = database_name

        self.benchmark = config["benchmark"]
        # 设置查询类的数量
        self.number_of_query_classes = self._set_number_of_query_classes()
        # 获取排除的查询类集合
        self.excluded_query_classes = set(config["excluded_query_classes"])
        # 获取频率是否变化的标志
        self.varying_frequencies = config["varying_frequencies"]

        # self.query_texts 是一个列表的列表。外层列表表示查询类，内层列表表示该类的实例。
        self.query_texts = self._retrieve_query_texts()
        # 获取所有查询类的集合
        self.query_classes = set(range(1, self.number_of_query_classes + 1))
        # 获取可用的查询类集合，排除了配置中指定的查询类
        self.available_query_classes = self.query_classes - self.excluded_query_classes

        # 获取可用查询类对应的查询文本
        self.available_query_texts = [self.query_texts[_ - 1] for _ in self.available_query_classes]

        # 选择可索引的列
        self.globally_indexable_columns = self._select_indexable_columns(self.filter_utilized_columns)

        # 获取验证工作负载的数量
        validation_instances = config["validation_testing"]["number_of_workloads"]
        # 获取测试工作负载的数量
        test_instances = config["test_number_of_workloads"]
        # 初始化验证工作负载列表
        self.wl_validation = []
        # 初始化测试工作负载列表
        self.wl_testing = []

        if config["similar_workloads"] and config["unknown_queries"] == 0:
            # Todo: this branch can probably be removed
            # 确保只有在频率变化时才能创建相似的工作负载
            assert self.varying_frequencies, "Similar workloads can only be created with varying frequencies."
            self.wl_validation = [None]
            self.wl_testing = [None]
            # 生成训练、验证和测试工作负载
            _, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                0, validation_instances, test_instances, config["size"]
            )
            if config["query_class_change_frequency"] is None:
                # 生成相似的工作负载
                self.wl_training = self._generate_similar_workloads(config["training_instances"], config["size"])
            else:
                # 生成具有查询类变化频率的相似工作负载
                self.wl_training = self._generate_similar_workloads_qccf(
                    config["training_instances"], config["size"], config["query_class_change_frequency"]
                )
        elif config["unknown_queries"] > 0:
            # 确保未知查询概率大于 0
            assert (
                config["validation_testing"]["unknown_query_probabilities"][-1] > 0
            ), "Query unknown_query_probabilities should be larger 0."

            # 初始化数据库连接器
            embedder_connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
            # 初始化工作负载嵌入器
            embedder = WorkloadEmbedder(
                # Transform globally_indexable_columns to list of lists.
                self.query_texts,
                0,
                embedder_connector,
                [list(map(lambda x: [x], self.globally_indexable_columns))],
                retrieve_plans=True,
            )
            # 获取要移除的未知查询类
            self.unknown_query_classes = embedding_utils.which_queries_to_remove(
                embedder.plans, config["unknown_queries"], random_seed
            )

            # 排除已排除的查询类
            self.unknown_query_classes = frozenset(self.unknown_query_classes) - self.excluded_query_classes
            # 计算缺失的查询类数量
            missing_classes = config["unknown_queries"] - len(self.unknown_query_classes)
            # 补充缺失的查询类
            self.unknown_query_classes = self.unknown_query_classes | frozenset(
                self.rnd.sample(self.available_query_classes - frozenset(self.unknown_query_classes), missing_classes)
            )
            # 确保未知查询类的数量符合配置
            assert len(self.unknown_query_classes) == config["unknown_queries"]

            # 获取已知查询类集合
            self.known_query_classes = self.available_query_classes - frozenset(self.unknown_query_classes)
            # 释放嵌入器对象
            embedder = None

            # 确保排除的查询类不在未知查询类中
            for query_class in self.excluded_query_classes:
                assert query_class not in self.unknown_query_classes

            # 记录全局未知查询类
            logging.critical(f"Global unknown query classes: {sorted(self.unknown_query_classes)}")
            # 记录全局已知查询类
            logging.critical(f"Global known query classes: {sorted(self.known_query_classes)}")

            # 为不同的未知查询概率生成验证和测试工作负载
            for unknown_query_probability in config["validation_testing"]["unknown_query_probabilities"]:
                _, wl_validation, wl_testing = self._generate_workloads(
                    0,
                    validation_instances,
                    test_instances,
                    config["size"],
                    unknown_query_probability=unknown_query_probability,
                )
                self.wl_validation.append(wl_validation)
                self.wl_testing.append(wl_testing)

            # 确保验证和测试工作负载的长度符合配置
            assert (
                len(self.wl_validation)
                == len(config["validation_testing"]["unknown_query_probabilities"])
                == len(self.wl_testing)
            ), "Validation/Testing workloads length fail"

            # 暂时限制可用的查询类，排除某些类用于训练
            original_available_query_classes = self.available_query_classes
            self.available_query_classes = self.known_query_classes

            if config["similar_workloads"]:
                if config["query_class_change_frequency"] is not None:
                    # 记录具有查询类变化频率的相似工作负载信息
                    logging.critical(
                        f"Similar workloads with query_class_change_frequency: {config['query_class_change_frequency']}"
                    )
                    # 生成具有查询类变化频率的相似工作负载
                    self.wl_training = self._generate_similar_workloads_qccf(
                        config["training_instances"], config["size"], config["query_class_change_frequency"]
                    )
                else:
                    # 生成相似的工作负载
                    self.wl_training = self._generate_similar_workloads(config["training_instances"], config["size"])
            else:
                # 生成训练工作负载
                self.wl_training, _, _ = self._generate_workloads(config["training_instances"], 0, 0, config["size"])
            # 恢复可用的查询类
            self.available_query_classes = original_available_query_classes
        else:
            # 初始化验证和测试工作负载列表
            self.wl_validation = [None]
            self.wl_testing = [None]
            self.wl_validation_sequence = [None]
            self.wl_testing_sequence = [None]
            # 调用 _generate_workloads 方法生成训练、验证和测试工作负载
            self.wl_training, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                config["training_instances"], validation_instances, test_instances, config["size"]
            )
            # with open(f"../wl_validation.pickle4", "wb") as handle:
            #     pickle.dump(self.wl_validation, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #
            # with open(f"../wl_validation.pickle1", "rb") as handle:
            #     wl_validation1 = pickle.load(handle)
            # with open(f"../wl_validation.pickle2", "rb") as handle:
            #     wl_validation2 = pickle.load(handle)
            # with open(f"../wl_validation.pickle3", "rb") as handle:
            #     wl_validation3 = pickle.load(handle)
            # with open(f"../wl_validation.pickle4", "rb") as handle:
            #     wl_validation4 = pickle.load(handle)
            # 取消注释可记录随机抽样的训练工作负载信息
            # logging.critical(f"Sample training workloads: {self.rnd.sample(self.wl_training, 10)}")
            # 记录工作负载生成完成的信息
        logging.info("Finished generating workloads.")

    def _set_number_of_query_classes(self):
        if self.benchmark == "TPCH":
            return 22
        elif self.benchmark == "TPCDS":
            return 20
        elif self.benchmark == "JOB":
            return 113
        elif self.benchmark == "TPCHskew":
            return 25
        elif self.benchmark == "SSB":
            return 5
        else:
            raise ValueError("Unsupported Benchmark type provided, only TPCH, TPCDS, and JOB supported.")

    def _retrieve_query_texts(self):
        """
        从文件中检索查询文本。

        :return: 一个包含所有查询文本的列表，每个元素对应一个查询类的查询文本列表。
        """
        # 打开所有查询文件
        query_files = [
            open(f"../{QUERY_PATH}/{self.benchmark}/{self.benchmark}_{file_number}.txt", "r")
            for file_number in range(1, self.number_of_query_classes + 1)
        ]

        finished_queries = []
        for query_file in query_files:
            # 读取文件中的所有查询
            queries = query_file.readlines()
            # 对查询进行预处理
            queries = self._preprocess_queries(queries)

            finished_queries.append(queries)

            # 关闭文件
            query_file.close()

        # 确保检索到的查询类数量与预期一致
        assert len(finished_queries) == self.number_of_query_classes

        return finished_queries

    def _preprocess_queries(self, queries):
        """
        对查询进行预处理，去除不必要的部分并替换特定的视图名称。

        :param queries: 待处理的查询列表。
        :return: 处理后的查询列表。
        """
        processed_queries = []
        for query in queries:
            # 去除查询中的 limit 子句
            query = query.replace("limit 100", "")
            query = query.replace("limit 20", "")
            query = query.replace("limit 10", "")
            # 去除查询前后的空白字符
            query = query.strip()

            # 如果查询中包含 create view revenue0，则替换为带有实验 ID 的名称
            if "create view revenue0" in query:
                query = query.replace("revenue0", f"revenue0_{self.experiment_id}")

            processed_queries.append(query)

        return processed_queries

    def _store_indexable_columns(self, query):
        """
        存储查询中可索引的列。

        :param query: 查询对象。
        """
        if self.benchmark != "JOB":
            # 从查询文本中提取分区 ID
            partition_id = query.text.split("prt_p")[-1].split(" ")[0]
            partition = f"prt_p{partition_id}"
            for column in self.workload_columns:
                # 如果列名在查询文本中且分区匹配，则将列添加到查询的列列表中
                if column.name in query.text and partition in column.table.name:
                    query.columns.append(column)
        else:
            # 对于 JOB 基准测试，该功能未完成
            assert 0, "not finished"
            query_text = query.text
            # 确保查询中包含 WHERE 子句
            assert "WHERE" in query_text, f"Query without WHERE clause encountered: {query_text} in {query.nr}"

            # 分割查询文本为 WHERE 之前和之后的部分
            split = query_text.split("WHERE")
            # 确保查询分割后不包含子查询
            assert len(split) == 2, "Query split for JOB query contains subquery"
            query_text_before_where = split[0]
            query_text_after_where = split[1]

            for column in self.workload_columns:
                # 如果列名在 WHERE 之后的文本中且表名在 WHERE 之前的文本中，则将列添加到查询的列列表中
                if column.name in query_text_after_where and f"{column.table.name} " in query_text_before_where:
                    query.columns.append(column)

    def _all_available_workloads(self, query_classes, unknown_query_probability=None):
        """
        生成所有可用的工作负载。

        :param query_classes: 查询类列表。
        :param unknown_query_probability: 未知查询的概率，默认为 None。
        :return: 包含所有工作负载的列表。
        """
        workloads = []
        workloads_sequence = []  # workloads_sequence[0] = 10*22
        # 处理未知查询概率为 None 的情况
        unknown_query_probability = "" if unknown_query_probability is None else unknown_query_probability
        queries = []
        queries_sequence = [[] for _ in range(self.partition_num)]
        for query_class in query_classes:
            # 获取当前查询类的查询模型
            query_models = self.query_texts[query_class - 1]
            for query_text in query_models:
                for partition_id in range(self.partition_num):
                    # 对查询文本进行分区处理
                    new_text = self.query_text_parition(query_text, partition_id)
                    # 创建查询对象
                    query = Query(query_class, new_text, frequency=1)
                    # 存储查询中可索引的列
                    self._store_indexable_columns(query)
                    # 确保查询的列列表不为空
                    assert len(query.columns) > 0 , f"Query columns should have length > 0: {query.text}"
                    queries.append(query)
                    # 将查询按分区 ID 分组
                    queries_sequence[partition_id].append(query)
        # 确保查询列表是列表类型
        assert isinstance(queries, list), f"Queries is not of type list but of {type(queries)}"
        # 计算未知查询的数量
        previously_unseen_queries = (
            round(unknown_query_probability * len(queries)) if unknown_query_probability != "" else 0
        )
        # 创建工作负载对象
        workloads.append(
            Workload(queries, description=f"Contains {previously_unseen_queries} previously unseen queries.")
        )
        # 计算每个分区的未知查询数量
        previously_unseen_queries_seq = (
            [round(unknown_query_probability * len(
                queries_sequence[partition_id])) if unknown_query_probability != "" else 0 for partition_id in
             range(self.partition_num)]
        )
        # 创建按分区的工作负载列表
        workloads_sequence.append(
            [Workload(queries_sequence[partition_id],
                      description=f"Contains {previously_unseen_queries_seq[partition_id]} previously unseen queries.")
             for partition_id in range(self.partition_num)]
        )

        return workloads

    def _split_queryTexts_4_training(self, test_tuples, validation_tuples):
        """
        为训练、测试和验证分割查询文本。

        :param test_tuples: 测试元组列表，每个元组包含查询类和频率。
        :param validation_tuples: 验证元组列表，每个元组包含查询类和频率。
        :return: 训练、测试和验证的查询文本列表。
        """
        print("Split queries...")
        print("Test tuples:{}".format(test_tuples))
        print("Validation tuples:{}".format(validation_tuples))

        test_texts = [[] for _ in range(len(self.query_texts))]
        validation_texts = [[] for _ in range(len(self.query_texts))]
        # 复制原始查询文本
        origin_texts = copy.deepcopy(self.query_texts)
        for tupl in test_tuples:
            query_classes, query_class_frequencies = tupl
            for query_class, frequency in zip(query_classes, query_class_frequencies):
                # 随机选择一个查询文本作为测试文本
                _query_text = self.rnd.choice(origin_texts[query_class - 1])
                test_texts[query_class - 1].append(_query_text)
                # 从原始文本中移除已选择的测试文本
                origin_texts[query_class - 1].remove(_query_text)

        for tupl in validation_tuples:
            query_classes, query_class_frequencies = tupl
            for query_class, frequency in zip(query_classes, query_class_frequencies):
                # 随机选择一个查询文本作为验证文本
                _query_text = self.rnd.choice(origin_texts[query_class - 1])
                validation_texts[query_class - 1].append(_query_text)
                # 从原始文本中移除已选择的验证文本
                origin_texts[query_class - 1].remove(_query_text)

        print("test_texts：{}".format(test_texts))
        print("validation_texts：{}".format(validation_texts))
        print("origin_texts：{}".format(origin_texts))

        return origin_texts, test_texts, validation_texts


    def _workloads_from_tuples(self, tuples, unknown_query_probability=None, query_texts=None):
        """
        从元组列表生成工作负载。

        :param tuples: 元组列表，每个元组包含查询类和频率。
        :param unknown_query_probability: 未知查询的概率，默认为 None。
        :param query_texts: 查询文本列表，默认为 None。
        :return: 包含所有工作负载的列表。
        """
        # 如果未提供查询文本，则使用类的查询文本属性
        if query_texts is None:
            query_texts = self.query_texts

        # 存储所有生成的工作负载
        workloads = []  #workloads[0] = 220
        # 存储按分区生成的工作负载序列
        workloads_sequence = [] # workloads_sequence[0] = 10*22
        # 处理未知查询概率为 None 的情况
        unknown_query_probability = "" if unknown_query_probability is None else unknown_query_probability

        # 遍历每个元组
        for tupl in tuples:
            # 解包元组，获取查询类和频率
            query_classes, query_class_frequencies = tupl
            # 存储当前元组生成的所有查询
            queries = []
            # 存储按分区生成的查询序列
            queries_sequence = [[] for _ in range(self.partition_num)]
            # 存储已经获取的查询文本，避免重复
            obtained_texts = set()
            # 遍历查询类和频率
            for query_class, frequency in zip(query_classes, query_class_frequencies):
                while True:
                    # 随机选择一个查询文本
                    query_text = self.rnd.choice(query_texts[query_class - 1])
                    # 如果该查询文本未被获取过
                    if query_text not in obtained_texts:
                        # 将该查询文本添加到已获取集合中
                        obtained_texts.add(query_text)
                        break
                # 遍历每个分区
                for partition_id in range(self.partition_num):
                    # 对查询文本进行分区处理
                    new_text = self.query_text_parition(query_text, partition_id)
                    # 创建查询对象
                    query = Query(query_class, new_text, frequency=frequency)
                    # 存储查询中可索引的列
                    self._store_indexable_columns(query)
                    # 确保查询的列列表不为空
                    assert len(query.columns) > 0 , f"Query columns should have length > 0: {query.text}"
                    # 将查询添加到查询列表中
                    queries.append(query)
                    # 将查询按分区 ID 分组
                    queries_sequence[partition_id].append(query)

            # 确保查询列表是列表类型
            assert isinstance(queries, list), f"Queries is not of type list but of {type(queries)}"
            # 计算未知查询的数量
            previously_unseen_queries = (
                round(unknown_query_probability * len(queries)) if unknown_query_probability != "" else 0
            )
            # 创建工作负载对象并添加到工作负载列表中
            workloads.append(
                Workload(queries, description=f"Contains {previously_unseen_queries} previously unseen queries.")
            )
            # 计算每个分区的未知查询数量
            previously_unseen_queries_seq = (
                [round(unknown_query_probability * len(queries_sequence[partition_id])) if unknown_query_probability != "" else 0 for partition_id in range(self.partition_num)]
            )
            # 创建按分区的工作负载列表并添加到工作负载序列中
            workloads_sequence.append(
                [Workload(queries_sequence[partition_id], description=f"Contains {previously_unseen_queries_seq[partition_id]} previously unseen queries.") for partition_id in range(self.partition_num)]
            )
        return workloads, workloads_sequence   # 改动
        # return workloads

    def _partitioned_table_map(self, partition_id):
        """
        根据基准测试类型和分区 ID 返回分区表映射。

        :param partition_id: 分区 ID。
        :return: 分区表映射字典。
        """
        # 如果基准测试是 TPCH
        if self.benchmark == "TPCH":
            return {
                'customer': f"customer_1_prt_p{partition_id}",
                'lineitem': f"lineitem_1_prt_p{partition_id}",
                'nation': f"nation_1_prt_p{partition_id}",
                'orders': f"orders_1_prt_p{partition_id}",
                'part': f"part_1_prt_p{partition_id}",
                'partsupp': f"partsupp_1_prt_p{partition_id}",
                'region': f"region_1_prt_p{partition_id}",
                'supplier': f"supplier_1_prt_p{partition_id}"
        }
        # 如果数据库名称包含 ssb
        elif "ssb" in self.database_name:
            return {
                'lineorder': f"lineorder_1_prt_p{partition_id}",
                'supplier': f"supplier_1_prt_p{partition_id}",
                'part': f"part_1_prt_p{partition_id}",
                'customer': f"customer_1_prt_p{partition_id}",
                'date': f"dim_date_1_prt_p{partition_id}"
            }
        # 如果基准测试是 TPCHskew
        elif self.benchmark == "TPCHskew":
            return {
                'customer': f"customer_1_prt_p{partition_id}",
                'lineitem': f"lineitem_1_prt_p{partition_id}",
                'nation': f"nation_1_prt_p{partition_id}",
                'orders': f"orders_1_prt_p{partition_id}",
                'part': f"part_1_prt_p{partition_id}",
                'partsupp': f"partsupp_1_prt_p{partition_id}",
                'region': f"region_1_prt_p{partition_id}",
                'supplier': f"supplier_1_prt_p{partition_id}"
        }
        # 如果基准测试是 TPCDS
        elif self.benchmark == "TPCDS":
            return {
                'dbgen_version': f"dbgen_version_1_prt_p{partition_id}",
                'customer_address': f"customer_address_1_prt_p{partition_id}",
                'customer_demographics': f"customer_demographics_1_prt_p{partition_id}",
                'date_dim': f"date_dim_1_prt_p{partition_id}",
                'warehouse': f"warehouse_1_prt_p{partition_id}",
                'ship_mode': f"ship_mode_1_prt_p{partition_id}",
                'time_dim': f"time_dim_1_prt_p{partition_id}",
                'reason': f"reason_1_prt_p{partition_id}",
                'income_band': f"income_band_1_prt_p{partition_id}",
                'item': f"item_1_prt_p{partition_id}",
                'store': f"store_1_prt_p{partition_id}",
                'call_center': f"call_center_1_prt_p{partition_id}",
                'customer': f"customer_1_prt_p{partition_id}",
                'web_site': f"web_site_1_prt_p{partition_id}",
                'store_returns': f"store_returns_1_prt_p{partition_id}",
                'household_demographics': f"household_demographics_1_prt_p{partition_id}",
                'web_page': f"web_page_1_prt_p{partition_id}",
                'promotion': f"promotion_1_prt_p{partition_id}",
                'catalog_page': f"catalog_page_1_prt_p{partition_id}",
                'inventory': f"inventory_1_prt_p{partition_id}",
                'catalog_returns': f"catalog_returns_1_prt_p{partition_id}",
                'web_returns': f"web_returns_1_prt_p{partition_id}",
                'web_sales': f"web_sales_1_prt_p{partition_id}",
                'catalog_sales': f"catalog_sales_1_prt_p{partition_id}",
                'store_sales': f"store_sales_1_prt_p{partition_id}"
            }
        else:
            return None

    def query_text_parition(self, query_text, partition_id):
        """
        根据分区 ID 对查询文本进行分区处理。

        :param query_text: 原始查询文本。
        :param partition_id: 分区 ID。
        :return: 分区处理后的查询文本。
        """
        # 复制原始查询文本，避免修改原始文本
        new_text = copy.deepcopy(query_text)
        # 确保基准测试为 TPCH 或 TPCDS，因为只有这两种基准测试支持分区表映射
        #assert self.benchmark == "TPCH", "you need to specify, this benchmark"
        new_text = copy.deepcopy(query_text)
        # 获取分区表映射，根据分区 ID 得到对应的表名和别名映射
        table_alias_map = self._partitioned_table_map(partition_id)
        # 确保分区表映射不为空，即基准测试是支持的类型
        assert table_alias_map is not None, "benchmark should be TPCH or TPCDS"
        # 将查询中的表名和别名替换为"原表_1"格式
        for table, alias in table_alias_map.items():
            # 匹配表名，使用正则表达式确保匹配的是完整的表名
            table_pattern = re.compile(rf'(?<!\w){table}(?!\w)')
            # 将查询文本中的表名替换为分区后的表名
            new_text = table_pattern.sub(f'{alias}', new_text)

            # 匹配别名，使用正则表达式确保匹配的是完整的别名
            alias_pattern = re.compile(rf'(?<!\w){alias}(?!\w)')
            # 将查询文本中的别名替换为分区后的别名
            new_text = alias_pattern.sub(f'{alias}', new_text)
        return new_text

    def _generate_workloads(
        self, train_instances, validation_instances, test_instances, size, unknown_query_probability=None
    ):
        """
        生成训练、验证和测试工作负载。

        :param train_instances: 训练实例的数量。
        :param validation_instances: 验证实例的数量。
        :param test_instances: 测试实例的数量。
        :param size: 每个工作负载的查询数量。
        :param unknown_query_probability: 未知查询的概率，默认为 None。
        :return: 训练、验证和测试工作负载列表。
        """
        # 计算所需的唯一工作负载元组数量
        required_unique_workloads = train_instances + validation_instances + test_instances

        # 存储唯一的工作负载元组
        unique_workload_tuples = set()

        # 生成唯一的工作负载元组，直到达到所需数量
        while required_unique_workloads > len(unique_workload_tuples):
            # 生成一个随机工作负载元组
            workload_tuple = self._generate_random_workload(size, unknown_query_probability)
            # 将生成的工作负载元组添加到集合中
            unique_workload_tuples.add(workload_tuple)

        # 从唯一工作负载元组中随机选择验证实例
        validation_tuples = self.rnd.sample(unique_workload_tuples, validation_instances)
        # 从唯一工作负载元组中移除已选的验证实例
        unique_workload_tuples = unique_workload_tuples - set(validation_tuples)

        # 从剩余的唯一工作负载元组中随机选择测试实例
        test_workload_tuples = self.rnd.sample(unique_workload_tuples, test_instances)
        # 从唯一工作负载元组中移除已选的测试实例
        unique_workload_tuples = unique_workload_tuples - set(test_workload_tuples)

        # 确保剩余的唯一工作负载元组数量等于训练实例数量
        assert len(unique_workload_tuples) == train_instances
        # 将剩余的唯一工作负载元组作为训练实例
        train_workload_tuples = unique_workload_tuples

        # 确保训练、验证和测试实例的总数等于所需的唯一工作负载元组数量
        assert (
            len(train_workload_tuples) + len(test_workload_tuples) + len(validation_tuples) == required_unique_workloads
        )

        # 如果配置中指定有未知查询
        if self.config["unseen_query"] is True:
            # 为训练、测试和验证分割查询文本
            origin_texts, test_texts, validation_texts = self._split_queryTexts_4_training(test_workload_tuples, validation_tuples)
            # 从验证元组生成验证工作负载
            validation_workloads = self._workloads_from_tuples(validation_tuples, unknown_query_probability, validation_texts)
            # 从测试元组生成测试工作负载
            test_workloads = self._workloads_from_tuples(test_workload_tuples, unknown_query_probability, test_texts)
            # 从训练元组生成训练工作负载
            train_workloads = self._workloads_from_tuples(train_workload_tuples, unknown_query_probability, origin_texts)
        else:
            # 从验证元组生成验证工作负载，不考虑未知查询
            validation_workloads, validation_workloads_sequence = self._workloads_from_tuples(validation_tuples, unknown_query_probability)
            # 从测试元组生成测试工作负载，不考虑未知查询
            test_workloads, test_workloads_sequence = self._workloads_from_tuples(test_workload_tuples, unknown_query_probability)
            # 从训练元组生成训练工作负载，不考虑未知查询
            train_workloads, train_workloads_sequence = self._workloads_from_tuples(train_workload_tuples, unknown_query_probability)

        return train_workloads, validation_workloads, test_workloads

    # The core idea is to create workloads that are similar and only change slightly from one to another.
    # For the following workload, we remove one random element, add another random one with frequency, and
    # randomly change the frequency of one element (including the new one).
    def _generate_similar_workloads(self, instances, size):
        """
        生成相似的工作负载。

        :param instances: 要生成的工作负载实例数量。
        :param size: 每个工作负载的查询数量。
        :return: 相似的工作负载列表。
        """
        # 确保生成的工作负载查询数量不超过可用查询类的数量
        assert size <= len(
            self.available_query_classes
        ), "Cannot generate workload with more queries than query classes"

        # 存储工作负载元组
        workload_tuples = []

        # 随机选择查询类
        query_classes = self.rnd.sample(self.available_query_classes, size)
        # 计算剩余可用的查询类
        available_query_classes = self.available_query_classes - frozenset(query_classes)
        # 生成查询频率，使用 Zipf 分布
        frequencies = list(self.np_rnd.zipf(1.5, size))

        # 将第一个工作负载元组添加到列表中
        workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        # 生成剩余的工作负载实例
        for workload_idx in range(instances - 1):
            # 随机选择一个元素进行移除
            idx_to_remove = self.rnd.randrange(len(query_classes))
            # 移除查询类
            query_classes.pop(idx_to_remove)
            # 移除对应的频率
            frequencies.pop(idx_to_remove)

            # 随机选择一个新的查询类，排除已移除的查询类
            query_classes.append(self.rnd.sample(available_query_classes, 1)[0])
            # 生成新查询类的频率，使用 Zipf 分布
            frequencies.append(self.np_rnd.zipf(1.5, 1)[0])

            # 随机选择一个元素，改变其频率
            frequencies[self.rnd.randrange(len(query_classes))] = self.np_rnd.zipf(1.5, 1)[0]

            # 重新计算剩余可用的查询类
            available_query_classes = self.available_query_classes - frozenset(query_classes)
            # 将新的工作负载元组添加到列表中
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        # 从工作负载元组生成工作负载
        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads

    # This version uses the same query id selction for query_class_change_frequency workloads
    def _generate_similar_workloads_qccf(self, instances, size, query_class_change_frequency):
        """
        生成具有查询类变化频率的相似工作负载。

        :param instances: 要生成的工作负载实例数量。
        :param size: 每个工作负载的查询数量。
        :param query_class_change_frequency: 查询类变化的频率。
        :return: 相似的工作负载列表。
        """
        # 确保每个工作负载的查询数量不超过可用查询类的数量
        assert size <= len(
            self.available_query_classes
        ), "Cannot generate workload with more queries than query classes"

        # 存储工作负载元组的列表
        workload_tuples = []

        # 循环生成指定数量的工作负载元组
        while len(workload_tuples) < instances:
            # 当生成的工作负载元组数量是查询类变化频率的倍数时，重新选择查询类
            if len(workload_tuples) % query_class_change_frequency == 0:
                # 从可用查询类中随机选择指定数量的查询类
                query_classes = self.rnd.sample(self.available_query_classes, size)

            # 生成查询频率，范围在 1 到 10000 之间
            frequencies = list(self.np_rnd.integers(1, 10000, size))
            # 将查询类和频率的副本添加到工作负载元组列表中
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        # 从工作负载元组列表生成工作负载列表
        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads

    def _generate_random_workload(self, size, unknown_query_probability=None):
        # 确保生成的工作负载查询数量不超过查询类的数量
        #assert size <= self.number_of_query_classes, "Cannot generate workload with more queries than query classes"

        # 初始化工作负载查询类
        workload_query_classes = None
        # 如果指定了未知查询的概率
        if unknown_query_probability is not None:
            # 计算未知查询的数量
            number_of_unknown_queries = round(size * unknown_query_probability)
            # 计算已知查询的数量
            number_of_known_queries = size - number_of_unknown_queries
            # 确保已知查询和未知查询的数量之和等于指定的查询数量
            assert number_of_known_queries + number_of_unknown_queries == size

            # 从已知查询类中随机选择指定数量的已知查询类
            known_query_classes = self.rnd.sample(self.known_query_classes, number_of_known_queries)
            # 从未知查询类中随机选择指定数量的未知查询类
            unknown_query_classes = self.rnd.sample(self.unknown_query_classes, number_of_unknown_queries)
            # 合并已知查询类和未知查询类
            query_classes = known_query_classes
            query_classes.extend(unknown_query_classes)
            # 将查询类转换为元组
            workload_query_classes = tuple(query_classes)
            # 确保工作负载查询类的数量等于指定的查询数量
            assert len(workload_query_classes) == size
        else:
            # 初始化工作负载查询类列表
            workload_query_classes = []
            # 循环生成指定数量的查询类
            for _ in range(size):
                # 从可用查询类中排除违规查询类后随机选择一个查询类添加到列表中
                workload_query_classes.extend(self.rnd.sample(self.available_query_classes-self.violation_queries, 1))
            # 将查询类列表转换为元组
            workload_query_classes = tuple(workload_query_classes)

            #workload_query_classes = tuple(self.rnd.sample(self.available_query_classes, size))

        # 创建查询频率
        if self.varying_frequencies:
            # 生成随机的查询频率，范围在 1 到 10000 之间
            query_class_frequencies = tuple(list(self.np_rnd.integers(1, 10000, size)))
        else:
            # 生成固定的查询频率，都为 1
            query_class_frequencies = tuple([1 for frequency in range(size)])

        # 创建工作负载元组
        workload_tuple = (workload_query_classes, query_class_frequencies)

        return workload_tuple

    def _only_utilized_indexes(self, indexable_columns):
        # 为每个可用查询类生成频率为 1 的频率列表
        frequencies = [1 for frequency in range(len(self.available_query_classes))]
        # 创建工作负载元组，包含可用查询类和频率列表
        workload_tuple = (self.available_query_classes, frequencies)
        # 从工作负载元组生成工作负载
        workload = self._workloads_from_tuples([workload_tuple])[0]

        # 为工作负载生成候选索引
        candidates = candidates_per_query(
            workload,
            max_index_width=1,
            candidate_generator=syntactically_relevant_indexes,
        )

        # 连接到 PostgreSQL 数据库
        connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
        # 删除数据库中的所有索引
        connector.drop_indexes()
        # 创建成本评估对象
        cost_evaluation = CostEvaluation(connector)

        # 获取工作负载中实际使用的索引和查询详情
        utilized_indexes, query_details = get_utilized_indexes(workload, candidates, cost_evaluation, True)

        # 存储实际使用索引的列的集合
        columns_of_utilized_indexes = set()
        # 遍历实际使用的索引
        for utilized_index in utilized_indexes:
            # 获取索引的第一列
            column = utilized_index.columns[0]
            # 将列添加到集合中
            columns_of_utilized_indexes.add(column)

        # 计算实际使用的列和可索引列的交集
        output_columns = columns_of_utilized_indexes & set(indexable_columns)
        # 计算可索引列中未被使用的列
        excluded_columns = set(indexable_columns) - output_columns
        # 记录日志，输出排除的列
        logging.critical(f"Excluding columns based on utilization:\n   {excluded_columns}")

        return output_columns

    def _select_indexable_columns(self, only_utilized_indexes=False):
        # 将可用查询类转换为元组
        available_query_classes = tuple(self.available_query_classes)
        # 为每个可用查询类生成频率为 1 的频率列表，并转换为元组
        query_class_frequencies = tuple([1 for frequency in range(len(available_query_classes))])

        # 记录日志，输出正在选择可索引列的查询类数量
        logging.info(f"Selecting indexable columns on {len(available_query_classes)} query classes.")

        # 生成包含所有可用查询类的工作负载
        workload = self._all_available_workloads(available_query_classes)[0]
        #workload = self._workloads_from_tuples([(available_query_classes, query_class_frequencies)])[0]

        # 获取工作负载中的可索引列
        indexable_columns = workload.indexable_columns()
        # 如果只选择实际使用的索引列
        if only_utilized_indexes:
            # 调用 _only_utilized_indexes 方法筛选出实际使用的索引列
            indexable_columns = self._only_utilized_indexes(indexable_columns)
        # 存储最终选择的列的列表
        selected_columns = []

        # 全局列 ID 计数器
        global_column_id = 0
        # 遍历工作负载中的所有列
        for column in self.workload_columns:
            # 如果列在可索引列中
            if column in indexable_columns:
                # 为列分配全局列 ID
                column.global_column_id = global_column_id
                # 全局列 ID 计数器加 1
                global_column_id += 1

                # 将列添加到最终选择的列列表中
                selected_columns.append(column)

        return selected_columns
