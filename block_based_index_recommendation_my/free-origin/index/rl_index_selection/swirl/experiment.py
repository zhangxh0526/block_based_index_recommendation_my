import copy
import datetime
import gzip
import importlib
import json
import logging
import os
import pickle
import random
import subprocess
import sys

from index_selection_evaluation.selection.algorithms.extend_algorithm_partition_histogram import \
    ExtendAlgorithmPartitionHistogram

sys.path.append("..")
import PettingZoo.custom_environment.env.index_environment as eie

import gym
import numpy as np

import time

from gym_db.common import EnvironmentType
from index_selection_evaluation.selection.algorithms.db2advis_algorithm import DB2AdvisAlgorithm
from index_selection_evaluation.selection.algorithms.extend_algorithm import ExtendAlgorithm
from index_selection_evaluation.selection.algorithms.extend_algorithm_partition import ExtendAlgorithmPartition
from index_selection_evaluation.selection.algorithms.slalom_algorithm import SlalomAlgorithm
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.param import *

import utils
from swirl.configuration_parser import ConfigurationParser
from swirl.schema import Schema
from swirl.workload_generator import WorkloadGenerator


class Experiment(object):
    def __init__(self, configuration_file):
        # 初始化实验的时间戳
        self._init_times()

        # 创建配置解析器实例，解析传入的配置文件
        cp = ConfigurationParser(configuration_file)
        # 将解析后的配置信息存储在实例属性中
        self.config = cp.config
        # 根据配置文件中的 Stable Baselines 版本设置特定方法
        self._set_sb_version_specific_methods()

        # 从配置中获取实验的唯一标识符
        self.id = self.config["id"]
        # 初始化模型属性，初始值为 None
        self.model = None

        # 创建一个随机数生成器实例
        self.rnd = random.Random()
        # 使用配置文件中指定的随机种子对随机数生成器进行初始化
        self.rnd.seed(self.config["random_seed"])

        # 初始化比较性能的字典，包含测试和验证阶段不同算法的性能列表
        self.comparison_performances = {
            "test": {"Extend": [], "Extend_partition": [], "DB2Adv": [], "Slalom": [], "Extend_partition_histogram" : []},
            "validation": {"Extend": [], "Extend_partition": [], "DB2Adv": [], "Slalom": [], "Extend_partition_histogram" : []},
        }
        # 初始化比较索引的字典，包含不同算法的索引集合
        self.comparison_indexes = {"Extend": set(), "Extend_partition": set(), "DB2Adv": set(), "Slalom": set(), "Extend_partition_histogram": set()}

        # 初始化特征数量，初始值为 None
        self.number_of_features = None
        # 初始化动作数量，初始值为 None
        self.number_of_actions = None
        # 初始化已评估工作负载的字符串列表
        self.evaluated_workloads_strs = []

        # 从配置中获取实验结果的存储路径
        self.EXPERIMENT_RESULT_PATH = self.config["result_path"]
        # 创建实验结果的存储文件夹
        self._create_experiment_folder()


    def prepare(self):
        """
        准备实验所需的环境和数据，包括创建模式、生成工作负载、处理可索引列等。
        """
        # 创建模式对象，包含数据库模式的相关信息
        self.schema = Schema(
            self.config["workload"]["benchmark"],
            self.config["workload"]["scale_factor"],
            self.config["column_filters"],
            self.config["partition_num"],
            self.config["used_tables"]["names"],
            self.config,
        )

        # 创建工作负载生成器，用于生成实验所需的工作负载
        self.workload_generator = WorkloadGenerator(
            self.config["workload"],
            workload_columns=self.schema.columns,
            random_seed=self.config["random_seed"],
            database_name=self.schema.database_name,
            experiment_id=self.id,
            filter_utilized_columns=self.config["filter_utilized_columns"],
            partition_num=self.config["partition_num"],
        )
        # 为工作负载分配预算
        self._assign_budgets_to_workloads()
        # 序列化工作负载
        self._pickle_workloads()

        # 获取全局可索引列
        self.globally_indexable_columns = self.workload_generator.globally_indexable_columns

        # 生成列组合索引，[[单列索引], [两列组合], [三列组合]...]
        self.globally_indexable_columns = utils.create_column_permutation_indexes(
            self.globally_indexable_columns, self.config["max_index_width"]
        )

        # 创建单列索引的集合
        self.single_column_flat_set = set(map(lambda x: x[0], self.globally_indexable_columns[0]))

        # 将所有可索引列组合展平为一维列表
        self.globally_indexable_columns_flat = [item for sublist in self.globally_indexable_columns for item in sublist]
        # 初始化块索引列表
        block_index = []
        for partition_id in range(self.config["partition_num"]):
            block_index.append([])
        # 根据分区ID将可索引列组合分配到不同的块中
        for indexable_column_combination_flat in self.globally_indexable_columns_flat:
            partition_id = int(indexable_column_combination_flat[0].table.name.split("prt_p")[-1])
            cc = indexable_column_combination_flat
            block_index[partition_id].append(cc)
        # 对每个分区的可索引列组合进行排序
        for partition_id in range(self.config["partition_num"]):
            block_index[partition_id].sort()
        # 将块索引列表展平为一维列表
        self.globally_indexable_columns_flat = [item for sublist in block_index for item in sublist]
        # 记录日志，显示要输入到环境中的候选索引数量
        logging.info(f"Feeding {len(self.globally_indexable_columns_flat)} candidates into the environments.")
        self.index_size_map = {}
        # 初始化动作存储消耗列表
        self.action_storage_consumptions= []
        # 预测每个可索引列组合的索引大小
        self.action_storage_consumptions, self.index_size_map = utils.predict_index_sizes(
            self.globally_indexable_columns_flat, self.schema.database_name, self.config["partition_num"]
        )

        # 如果配置中包含工作负载嵌入器，则创建工作负载嵌入器对象
        if "workload_embedder" in self.config:
            # 动态导入工作负载嵌入器类
            workload_embedder_class = getattr(
                importlib.import_module("swirl.workload_embedder"), self.config["workload_embedder"]["type"]
            )
            # 创建数据库连接器
            workload_embedder_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
            # 创建工作负载嵌入器实例
            self.workload_embedder = workload_embedder_class(
                self.schema.database_name,
                self.workload_generator.available_query_texts,
                self.config["workload_embedder"]["representation_size"],
                workload_embedder_connector,
                self.globally_indexable_columns,
                partition_num=self.config["partition_num"],
            )

        # 初始化多验证工作负载列表
        self.multi_validation_wl = []
        # 如果验证工作负载数量大于1，则从每个验证工作负载中随机选择最多7个工作负载
        if len(self.workload_generator.wl_validation) > 1:
            for workloads in self.workload_generator.wl_validation:
                self.multi_validation_wl.extend(self.rnd.sample(workloads, min(7, len(workloads))))


    def _assign_budgets_to_workloads(self):
        """
        为不同类型的工作负载分配预算。

        此方法会根据配置文件中的预算设置，为测试、验证和训练工作负载分配预算。
        对于测试工作负载，从验证和测试预算列表中随机选择一个预算分配给每个工作负载。
        对于验证工作负载，会为每个工作负载创建多个副本，每个副本分配不同的验证预算。
        如果配置中指定训练使用预算，则为训练工作负载从验证预算列表中随机选择一个预算分配。
        """
        # 为测试工作负载分配预算
        for workload_list in self.workload_generator.wl_testing:
            for workload in workload_list:
                # 从验证和测试预算列表中随机选择一个预算分配给当前工作负载
                workload.budget = self.rnd.choice(self.config["budgets"]["validation_and_testing"])

        # 为验证工作负载分配不同大小的预算
        result_workloads = []
        for workload_list in self.workload_generator.wl_validation:
            for workload in workload_list:
                _tmp_workloads = []
                for _budget in self.config["budgets"]["validation"]:
                    # 复制当前工作负载
                    _workload = copy.deepcopy(workload)
                    # 为复制的工作负载分配当前预算
                    _workload.budget = _budget
                    _tmp_workloads.append(_workload)
                # 将临时工作负载列表添加到结果工作负载列表中
                result_workloads.extend(_tmp_workloads)
        # 更新验证工作负载列表
        self.workload_generator.wl_validation = [copy.deepcopy(result_workloads)]

        # 如果配置中指定训练使用预算，则为训练工作负载分配预算
        if self.config["workload"]["training_with_budget"]:
            for workload in self.workload_generator.wl_training:
                # 从验证预算列表中随机选择一个预算分配给当前训练工作负载
                workload.budget = self.rnd.choice(self.config["budgets"]["validation"])

        # 以下代码被注释掉，未实际执行
        # for workload_list in self.workload_generator.wl_validation:
        #     for workload in workload_list:
        #         # 为验证工作负载从验证和测试预算列表中随机选择一个预算分配
        #         workload.budget = self.rnd.choice(self.config["budgets"]["validation_and_testing"])
                # workload.budget = self.rnd.choice(self.config["budgets"]["validation_and_testing"])


    def _pickle_workloads(self):
        """
        将测试和验证工作负载序列化为pickle文件。

        此方法将工作负载生成器中的测试和验证工作负载分别保存到指定路径的pickle文件中。
        这样做的目的是为了后续可以方便地重新加载这些工作负载，避免重复生成。

        :return: 无
        """
        # 打开用于保存测试工作负载的文件，以二进制写入模式打开
        with open(f"{self.experiment_folder_path}/testing_workloads.pickle", "wb") as handle:
            # 使用pickle模块将测试工作负载序列化为二进制数据，并写入文件
            pickle.dump(self.workload_generator.wl_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # 打开用于保存验证工作负载的文件，以二进制写入模式打开
        with open(f"{self.experiment_folder_path}/validation_workloads.pickle", "wb") as handle:
            # 使用pickle模块将验证工作负载序列化为二进制数据，并写入文件
            pickle.dump(self.workload_generator.wl_validation, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def finish(self):
        self.end_time = datetime.datetime.now()

        self.model.training = False
        self.model.env.norm_reward = False
        self.model.env.training = False

        self.test_fm = self.test_model(self.model)[0]
        self.vali_fm = self.validate_model(self.model)[0]

        if os.path.exists(f"{self.experiment_folder_path}/moving_average_model.zip"):
            self.moving_average_model = self.model.load(f"{self.experiment_folder_path}/moving_average_model")
            self.moving_average_model.training = False
            self.test_ma = self.test_model(self.moving_average_model)[0]
            self.vali_ma = self.validate_model(self.moving_average_model)[0]
            if len(self.multi_validation_wl) > 0:
                self.moving_average_model_mv = self.model.load(
                    f"{self.experiment_folder_path}/moving_average_model_mv.zip"
                )
                self.moving_average_model_mv.training = False
                self.test_ma_mv = self.test_model(self.moving_average_model_mv)[0]
                self.vali_ma_mv = self.validate_model(self.moving_average_model_mv)[0]
        else:
            print("file {} is not exist!".format(f"{self.experiment_folder_path}/moving_average_model.zip"))
            self.test_ma = self.test_model(self.model)[0]
            self.vali_ma = self.validate_model(self.model)[0]

        if os.path.exists(f"{self.experiment_folder_path}/moving_average_model_3.zip"):
            self.moving_average_model_3 = self.model.load(f"{self.experiment_folder_path}/moving_average_model_3")
            self.moving_average_model_3.training = False
            self.test_ma_3 = self.test_model(self.moving_average_model_3)[0]
            self.vali_ma_3 = self.validate_model(self.moving_average_model_3)[0]
            if len(self.multi_validation_wl) > 0:
                self.moving_average_model_3_mv = self.model.load(
                    f"{self.experiment_folder_path}/moving_average_model_3_mv.zip"
                )
                self.moving_average_model_3_mv.training = False
                self.test_ma_3_mv = self.test_model(self.moving_average_model_3_mv)[0]
                self.vali_ma_3_mv = self.validate_model(self.moving_average_model_3_mv)[0]
        else:
            print("file {} is not exist!".format(f"{self.experiment_folder_path}/moving_average_model_3.zip"))
            self.test_ma_3 = self.test_model(self.model)[0]
            self.vali_ma_3 = self.validate_model(self.model)[0]


        if os.path.exists(f"{self.experiment_folder_path}/best_mean_reward_model.zip"):
            self.best_mean_reward_model = self.model.load(f"{self.experiment_folder_path}/best_mean_reward_model")
            self.best_mean_reward_model.training = False
            self.test_bm = self.test_model(self.best_mean_reward_model)[0]
            self.vali_bm = self.validate_model(self.best_mean_reward_model)[0]
            if len(self.multi_validation_wl) > 0:
                self.best_mean_reward_model_mv = self.model.load(
                    f"{self.experiment_folder_path}/best_mean_reward_model_mv.zip"
                )
                self.best_mean_reward_model_mv.training = False
                self.test_bm_mv = self.test_model(self.best_mean_reward_model_mv)[0]
                self.vali_bm_mv = self.validate_model(self.best_mean_reward_model_mv)[0]
        else:
            print("file {} is not exist!".format(f"{self.experiment_folder_path}/best_mean_reward_model.zip"))
            self.test_bm = self.test_model(self.model)[0]
            self.vali_bm = self.validate_model(self.model)[0]


        self._write_report()

        logging.critical(
            (
                f"Finished training of ID {self.id}. Report can be found at "
                f"./{self.experiment_folder_path}/report_ID_{self.id}.txt"
            )
        )

    def _get_wl_budgets_from_model_perfs(self, perfs):
        wl_budgets = []
        for perf in perfs:
            assert perf["evaluated_workload"].budget == perf["available_budget"], "Budget mismatch!"
            wl_budgets.append(perf["evaluated_workload"].budget)
        return wl_budgets

    def start_learning(self):
        self.training_start_time = datetime.datetime.now()

    def set_model(self, model):
        self.model = model

    def finish_learning(self, training_env, moving_average_model_step, best_mean_model_step):
        self.training_end_time = datetime.datetime.now()

        self.moving_average_validation_model_at_step = moving_average_model_step
        self.best_mean_model_step = best_mean_model_step

        self.model.save(f"{self.experiment_folder_path}/final_model")
        training_env.save(f"{self.experiment_folder_path}/vec_normalize.pkl")

        self.evaluated_episodes = 0
        for number_of_resets in training_env.get_attr("number_of_resets"):
            self.evaluated_episodes += number_of_resets

        self.total_steps_taken = 0
        for total_number_of_steps in training_env.get_attr("total_number_of_steps"):
            self.total_steps_taken += total_number_of_steps

        self.cache_hits = 0
        self.cost_requests = 0
        self.costing_time = datetime.timedelta(0)
        for cache_info in training_env.env_method("get_cost_eval_cache_info"):
            self.cache_hits += cache_info[1]
            self.cost_requests += cache_info[0]
            self.costing_time += cache_info[2]
        self.costing_time /= self.config["parallel_environments"]

        self.cache_hit_ratio = self.cache_hits / self.cost_requests * 100

        if self.config["pickle_cost_estimation_caches"]:
            caches = []
            for cache in training_env.env_method("get_cost_eval_cache"):
                caches.append(cache)
            combined_caches = {}
            for cache in caches:
                combined_caches = {**combined_caches, **cache}
            with gzip.open(f"{self.experiment_folder_path}/caches.pickle.gzip", "wb") as handle:
                pickle.dump(combined_caches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _init_times(self):
        self.start_time = datetime.datetime.now()

        self.end_time = None
        self.training_start_time = None
        self.training_end_time = None

    def _create_experiment_folder(self):
        """
        创建实验文件夹，用于存储实验结果。

        此方法首先检查实验结果的根目录是否存在，如果不存在则抛出断言错误。
        然后，根据实验的 ID 和当前时间戳生成一个唯一的实验文件夹路径。
        接着，检查该文件夹是否已经存在，如果存在则抛出断言错误，以避免覆盖现有数据。
        最后，创建该实验文件夹。

        Raises:
            AssertionError: 如果实验结果的根目录不存在，或者实验文件夹已经存在。
        """
        # 检查实验结果的根目录是否存在，如果不存在则抛出断言错误
        assert os.path.isdir(
            self.EXPERIMENT_RESULT_PATH
        ), f"Folder for experiment results should exist at: ./{self.EXPERIMENT_RESULT_PATH}"

        # 根据实验的 ID 和当前时间戳生成一个唯一的实验文件夹路径
        self.experiment_folder_path = f"{self.EXPERIMENT_RESULT_PATH}/ID_{self.id}_timetamps_{int(time.time())}"
        # 检查该文件夹是否已经存在，如果存在则抛出断言错误，以避免覆盖现有数据
        assert os.path.isdir(self.experiment_folder_path) is False, (
            f"Experiment folder already exists at: ./{self.experiment_folder_path} - "
            "terminating here because we don't want to overwrite anything."
        )

        # 创建该实验文件夹
        os.mkdir(self.experiment_folder_path)


    def _write_report(self):
        with open(f"{self.experiment_folder_path}/report_ID_{self.id}.txt", "w") as f:
            f.write(f"##### Report for Experiment with ID: {self.id} #####\n")
            f.write(f"Description: {self.config['description']}\n")
            f.write("\n")

            f.write(f"Start:                         {self.start_time}\n")
            f.write(f"End:                           {self.start_time}\n")
            f.write(f"Duration:                      {self.end_time - self.start_time}\n")
            f.write("\n")
            f.write(f"Start Training:                {self.training_start_time}\n")
            f.write(f"End Training:                  {self.training_end_time}\n")
            f.write(f"Duration Training:             {self.training_end_time - self.training_start_time}\n")
            f.write(f"Index Creating Time:           {selection_args.index_create_duration}\n")
            f.write(f"Moving Average model at step:  {self.moving_average_validation_model_at_step}\n")
            f.write(f"Mean reward model at step:     {self.best_mean_model_step}\n")
            f.write(f"Git Hash:                      {subprocess.check_output(['git', 'rev-parse', 'HEAD'])}\n")
            f.write(f"Number of features:            {self.number_of_features}\n")
            f.write(f"Number of actions:             {self.number_of_actions}\n")
            f.write("\n")
            if self.config["workload"]["unknown_queries"] > 0:
                f.write(f"Unknown Query Classes {sorted(self.workload_generator.unknown_query_classes)}\n")
                f.write(f"Known Queries: {self.workload_generator.known_query_classes}\n")
                f.write("\n")
            probabilities = len(self.config["workload"]["validation_testing"]["unknown_query_probabilities"])
            for idx, unknown_query_probability in enumerate(
                self.config["workload"]["validation_testing"]["unknown_query_probabilities"]
            ):
                f.write(f"Unknown query probability: {unknown_query_probability}:\n")
                f.write("    Final mean performance test:\n")
                test_fm_perfs, self.performance_test_final_model, self.test_fm_details = self.test_fm[idx]
                vali_fm_perfs, self.performance_vali_final_model, self.vali_fm_details = self.vali_fm[idx]

                _, self.performance_test_moving_average_model, self.test_ma_details = self.test_ma[idx]
                _, self.performance_vali_moving_average_model, self.vali_ma_details = self.vali_ma[idx]
                _, self.performance_test_moving_average_model_3, self.test_ma_details_3 = self.test_ma_3[idx]
                _, self.performance_vali_moving_average_model_3, self.vali_ma_details_3 = self.vali_ma_3[idx]
                _, self.performance_test_best_mean_reward_model, self.test_bm_details = self.test_bm[idx]
                _, self.performance_vali_best_mean_reward_model, self.vali_bm_details = self.vali_bm[idx]

                if len(self.multi_validation_wl) > 0:
                    _, self.performance_test_moving_average_model_mv, self.test_ma_details_mv = self.test_ma_mv[idx]
                    _, self.performance_vali_moving_average_model_mv, self.vali_ma_details_mv = self.vali_ma_mv[idx]
                    _, self.performance_test_moving_average_model_3_mv, self.test_ma_details_3_mv = self.test_ma_3_mv[
                        idx
                    ]
                    _, self.performance_vali_moving_average_model_3_mv, self.vali_ma_details_3_mv = self.vali_ma_3_mv[
                        idx
                    ]
                    _, self.performance_test_best_mean_reward_model_mv, self.test_bm_details_mv = self.test_bm_mv[idx]
                    _, self.performance_vali_best_mean_reward_model_mv, self.vali_bm_details_mv = self.vali_bm_mv[idx]

                self.test_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(test_fm_perfs)
                self.vali_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(vali_fm_perfs)

                f.write(
                    (
                        "        Final model:               "
                        f"{self.performance_test_final_model:.2f} ({self.test_fm_details})\n"
                    )
                )
                f.write(
                    (
                        "        Moving Average model:      "
                        f"{self.performance_test_moving_average_model:.2f} ({self.test_ma_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average model (MV): "
                            f"{self.performance_test_moving_average_model_mv:.2f} ({self.test_ma_details_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Moving Average 3 model:    "
                        f"{self.performance_test_moving_average_model_3:.2f} ({self.test_ma_details_3})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average 3 mod (MV): "
                            f"{self.performance_test_moving_average_model_3_mv:.2f} ({self.test_ma_details_3_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Best mean reward model:    "
                        f"{self.performance_test_best_mean_reward_model:.2f} ({self.test_bm_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Best mean reward mod (MV): "
                            f"{self.performance_test_best_mean_reward_model_mv:.2f} ({self.test_bm_details_mv})\n"
                        )
                    )
                for key, value in self.comparison_performances["test"].items():
                    if len(value) < 1:
                        continue
                    f.write(f"        {key}:                    {np.mean(value):.2f} ({value})\n")
                f.write("\n")
                f.write(f"        Budgets:                   {self.test_fm_wl_budgets}\n")
                f.write("\n")
                f.write("    Final mean performance validation:\n")
                f.write(
                    (
                        "        Final model:               "
                        f"{self.performance_vali_final_model:.2f} ({self.vali_fm_details})\n"
                    )
                )
                f.write(
                    (
                        "        Moving Average model:      "
                        f"{self.performance_vali_moving_average_model:.2f} ({self.vali_ma_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average model (MV): "
                            f"{self.performance_vali_moving_average_model_mv:.2f} ({self.vali_ma_details_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Moving Average 3 model:    "
                        f"{self.performance_vali_moving_average_model_3:.2f} ({self.vali_ma_details_3})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average 3 mod (MV): "
                            f"{self.performance_vali_moving_average_model_3_mv:.2f} ({self.vali_ma_details_3_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Best mean reward model:    "
                        f"{self.performance_vali_best_mean_reward_model:.2f} ({self.vali_bm_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Best mean reward mod (MV): "
                            f"{self.performance_vali_best_mean_reward_model_mv:.2f} ({self.vali_bm_details_mv})\n"
                        )
                    )
                for key, value in self.comparison_performances["validation"].items():
                    if len(value) < 1:
                        continue
                    f.write(f"        {key}:                    {np.mean(value):.2f} ({value})\n")
                f.write("\n")
                f.write(f"        Budgets:                   {self.vali_fm_wl_budgets}\n")
                f.write("\n")
                f.write("\n")
            f.write("Overall Test:\n")

            def final_avg(values, probabilities):
                val = 0
                for res in values:
                    val += res[1]
                return val / probabilities

            f.write(("        Final model:               " f"{final_avg(self.test_fm, probabilities):.2f}\n"))
            f.write(("        Moving Average model:      " f"{final_avg(self.test_ma, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average model (MV): " f"{final_avg(self.test_ma_mv, probabilities):.2f}\n"))
            f.write(("        Moving Average 3 model:    " f"{final_avg(self.test_ma_3, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average 3 mod (MV): " f"{final_avg(self.test_ma_3_mv, probabilities):.2f}\n"))
            f.write(("        Best mean reward model:    " f"{final_avg(self.test_bm, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Best mean reward mod (MV): " f"{final_avg(self.test_bm_mv, probabilities):.2f}\n"))
            f.write(
                (
                    "        Extend:                    "
                    f"{np.mean(self.comparison_performances['test']['Extend']):.2f}\n"
                )
            )
            f.write(
                (
                    "        Extend_partition:          "
                    f"{np.mean(self.comparison_performances['test']['Extend_partition']):.2f}\n"
                )
            )
            f.write(
                (
                    "        Slalom:                    "
                    f"{np.mean(self.comparison_performances['test']['Slalom']):.2f}\n"
                )
            )
            f.write(
                (
                    "        DB2Adv:                    "
                    f"{np.mean(self.comparison_performances['test']['DB2Adv']):.2f}\n"
                )
            )
            f.write("\n")
            f.write("Overall Validation:\n")
            f.write(("        Final model:               " f"{final_avg(self.vali_fm, probabilities):.2f}\n"))
            f.write(("        Moving Average model:      " f"{final_avg(self.vali_ma, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average model (MV): " f"{final_avg(self.vali_ma_mv, probabilities):.2f}\n"))
            f.write(("        Moving Average 3 model:    " f"{final_avg(self.vali_ma_3, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average 3 mod (MV): " f"{final_avg(self.vali_ma_3_mv, probabilities):.2f}\n"))
            f.write(("        Best mean reward model:    " f"{final_avg(self.vali_bm, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Best mean reward mod (MV): " f"{final_avg(self.vali_bm_mv, probabilities):.2f}\n"))
            f.write(
                (
                    "        Extend:                    "
                    f"{np.mean(self.comparison_performances['validation']['Extend']):.2f}\n"
                )
            )
            f.write(
                (
                    "        Extend_partition:          "
                    f"{np.mean(self.comparison_performances['validation']['Extend_partition']):.2f}\n"
                )
            )
            f.write(
                (
                    "        Slalom:                    "
                    f"{np.mean(self.comparison_performances['validation']['Slalom']):.2f}\n"
                )
            )
            f.write(
                (
                    "        DB2Adv:                    "
                    f"{np.mean(self.comparison_performances['validation']['DB2Adv']):.2f}\n"
                )
            )
            f.write("\n")
            f.write("\n")
            f.write(f"Evaluated episodes:            {self.evaluated_episodes}\n")
            f.write(f"Total steps taken:             {self.total_steps_taken}\n")
            f.write(
                (
                    f"CostEval cache hit ratio:      "
                    f"{self.cache_hit_ratio:.2f} ({self.cache_hits} of {self.cost_requests})\n"
                )
            )
            training_time = self.training_end_time - self.training_start_time
            f.write(
                f"Cost eval time (% of total):   {self.costing_time} ({self.costing_time / training_time * 100:.2f}%)\n"
            )
            # f.write(f"Cost eval time:                {self.costing_time:.2f}\n")

            f.write("\n\n")
            f.write("Used configuration:\n")
            json.dump(self.config, f)
            f.write("\n\n")
            f.write("Evaluated test workloads:\n")
            for evaluated_workload in self.evaluated_workloads_strs[: (len(self.evaluated_workloads_strs) // 2)]:
                f.write(f"{evaluated_workload}\n")
            f.write("Evaluated validation workloads:\n")
            # fmt: off
            for evaluated_workload in self.evaluated_workloads_strs[(len(self.evaluated_workloads_strs) // 2) :]:  # noqa: E203, E501
                f.write(f"{evaluated_workload}\n")
            # fmt: on
            f.write("\n\n")

    def compare(self):
        # 检查配置文件中指定的比较算法列表是否为空，如果为空则直接返回
        if len(self.config["comparison_algorithms"]) < 1:
            return

        # 如果配置文件中指定了 "extend" 算法，则调用 _compare_extend 方法进行比较
        if "extend" in self.config["comparison_algorithms"]:
            self._compare_extend()
        if "extend_partition" in self.config["comparison_algorithms"]:
            self._compare_extend_partition()
        if "extend_partition_histogram" in self.config["comparison_algorithms"]:
            self._compare_extend_partition_histogram()
        # 如果配置文件中指定了 "slalom" 算法，则调用 _compare_slalom 方法进行比较
        if "slalom" in self.config["comparison_algorithms"]:
            self._compare_slalom()
        # 如果配置文件中指定了 "db2advis" 算法，则调用 _compare_db2advis 方法进行比较
        if "db2advis" in self.config["comparison_algorithms"]:
            self._compare_db2advis()
        # 遍历比较性能字典，打印每个比较的结果
        for key, comparison_performance in self.comparison_performances.items():
            print(f"Comparison for {key}:")
            # 遍历每个比较的性能指标，打印指标名称、平均值和具体值
            for key, value in comparison_performance.items():
                print(f"    {key}: {np.mean(value):.2f} ({value})")

        # 调用 _evaluate_comparison 方法对比较结果进行评估
        self._evaluate_comparison()


    def _evaluate_comparison(self):
        """
        评估比较结果，检查比较算法找到的索引是否包含不可索引的列。

        该方法遍历所有比较算法找到的索引，提取出所有索引涉及的列，
        并与可索引的单列集合进行比较。如果发现有不可索引的列，
        则记录严重错误信息，并抛出断言错误。

        Returns:
            None
        """
        # 遍历比较索引字典中的每个键值对
        for key, comparison_indexes in self.comparison_indexes.items():
            # 初始化一个空集合，用于存储从索引中提取的所有列
            columns_from_indexes = set()
            # 遍历当前比较算法找到的所有索引
            for index in comparison_indexes:
                # 遍历当前索引中的所有列
                for column in index.columns:
                    # 将当前列添加到 columns_from_indexes 集合中
                    columns_from_indexes |= set([column])

            # 找出 columns_from_indexes 集合中不在 self.single_column_flat_set 集合中的列，即不可索引的列
            impossible_index_columns = columns_from_indexes - self.single_column_flat_set
            # 记录严重错误信息，显示当前比较算法找到的索引中包含不可索引的列
            logging.critical(f"{key} finds indexes on these not indexable columns:\n    {impossible_index_columns}")

            # 断言不可索引的列数量为 0，如果不为 0 则抛出断言错误
            assert len(impossible_index_columns) == 0, "Found indexes on not indexable columns."

    def _compare_extend(self):
        self.evaluated_workloads = set()
        extend_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
        extend_connector.drop_indexes()
        extend_algorithm = ExtendAlgorithm(extend_connector)

        run_type = "test"
        for test_wl in self.workload_generator.wl_testing[0]:
            self.comparison_performances[run_type]["Extend"].append([])
            # self.evaluated_workloads.add(test_wl)

            parameters = {
                "budget_MB": test_wl.budget,
                "max_index_width": self.config["max_index_width"],
                "min_cost_improvement": 1.0003,
            }
            extend_algorithm.reset(parameters)
            indexes = extend_algorithm.calculate_best_indexes(test_wl)
            self.comparison_indexes["Extend"] |= frozenset(indexes)

            self.comparison_performances[run_type]["Extend"][-1].append(extend_algorithm.final_cost_proportion)

        run_type = "validation"
        for validation_wl in self.workload_generator.wl_validation[0]:
            self.comparison_performances[run_type]["Extend"].append([])

            parameters = {
                "budget_MB": validation_wl.budget,
                "max_index_width": self.config["max_index_width"],
                "min_cost_improvement": 1.0003,
            }
            extend_algorithm.reset(parameters)
            indexes = extend_algorithm.calculate_best_indexes(validation_wl)
            self.comparison_indexes["Extend"] |= frozenset(indexes)

            self.comparison_performances[run_type]["Extend"][-1].append(extend_algorithm.final_cost_proportion)

    def _compare_extend_partition(self):
        """
        该方法用于对比 Extend 算法在测试和验证工作负载上的性能。

        它会初始化一个 Postgres 数据库连接，使用 Extend 算法为每个测试和验证工作负载计算最佳索引，
        并记录这些索引和最终的成本比例。
        """
        # 初始化一个集合，用于存储已经评估过的工作负载
        self.evaluated_workloads = set()
        # 创建一个 Postgres 数据库连接器，用于与数据库交互，设置自动提交
        extend_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
        # 删除数据库中现有的所有索引
        extend_connector.drop_indexes()
        # 初始化 Extend 算法，传入数据库连接器
        extend_algorithm = ExtendAlgorithmPartition(extend_connector)

        # 定义运行类型为测试
        run_type = "test"
        # 遍历测试工作负载列表
        for index, test_wl in enumerate(self.workload_generator.wl_testing[0]):
            # 为当前运行类型和算法添加一个空列表，用于存储性能数据
            self.comparison_performances[run_type]["Extend_partition"].append([])
            # 将当前测试工作负载添加到已评估工作负载集合中
            # self.evaluated_workloads.add(test_wl)

            # 定义 Extend 算法的参数
            parameters = {
                # 设置预算，单位为 MB
                "budget_MB": test_wl.budget,
                # 设置最大索引宽度
                "max_index_width": self.config["max_index_width"],
                # 设置最小成本改进阈值
                "min_cost_improvement": 1.0003,
                "partition_num": self.config["partition_num"]
            }
            # 重置 Extend 算法，传入参数
            extend_algorithm.reset(parameters)
            # 计算当前测试工作负载的最佳索引
            indexes = extend_algorithm.calculate_best_indexes(test_wl)
            # 将计算得到的索引添加到比较索引集合中
            self.comparison_indexes["Extend_partition"] |= frozenset(indexes)

            # 记录当前测试工作负载的最终成本比例
            self.comparison_performances[run_type]["Extend_partition"][-1].append(extend_algorithm.final_cost_proportion)

        # 定义运行类型为验证
        run_type = "validation"
        # 遍历验证工作负载列表
        for index, validation_wl in enumerate(self.workload_generator.wl_validation[0]):
            # 为当前运行类型和算法添加一个空列表，用于存储性能数据
            self.comparison_performances[run_type]["Extend_partition"].append([])

            # 定义 Extend 算法的参数
            parameters = {
                # 设置预算，单位为 MB
                "budget_MB": validation_wl.budget,
                # 设置最大索引宽度
                "max_index_width": self.config["max_index_width"],
                # 设置最小成本改进阈值
                "min_cost_improvement": 1.0003,
                "partition_num": self.config["partition_num"]
            }
            # 重置 Extend 算法，传入参数
            extend_algorithm.reset(parameters)
            # 计算当前验证工作负载的最佳索引
            indexes = extend_algorithm.calculate_best_indexes(validation_wl)
            # 将计算得到的索引添加到比较索引集合中
            self.comparison_indexes["Extend_partition"] |= frozenset(indexes)

            # 记录当前验证工作负载的最终成本比例
            self.comparison_performances[run_type]["Extend_partition"][-1].append(extend_algorithm.final_cost_proportion)

    def _compare_extend_partition_histogram(self):
        """
        该方法用于对比 Extend 算法在测试和验证工作负载上的性能。

        它会初始化一个 Postgres 数据库连接，使用 Extend 算法为每个测试和验证工作负载计算最佳索引，
        并记录这些索引和最终的成本比例。
        """
        # 初始化一个集合，用于存储已经评估过的工作负载
        self.evaluated_workloads = set()
        # 创建一个 Postgres 数据库连接器，用于与数据库交互，设置自动提交
        extend_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
        # 删除数据库中现有的所有索引
        extend_connector.drop_indexes()
        # 初始化 Extend 算法，传入数据库连接器
        extend_algorithm = ExtendAlgorithmPartitionHistogram(extend_connector)

        # 定义运行类型为测试
        run_type = "test"
        # 遍历测试工作负载列表
        for index, test_wl in enumerate(self.workload_generator.wl_testing[0]):
            # 为当前运行类型和算法添加一个空列表，用于存储性能数据
            self.comparison_performances[run_type]["Extend_partition_histogram"].append([])
            # 将当前测试工作负载添加到已评估工作负载集合中
            # self.evaluated_workloads.add(test_wl)

            # 定义 Extend 算法的参数
            parameters = {
                # 设置预算，单位为 MB
                "budget_MB": test_wl.budget,
                # 设置最大索引宽度
                "max_index_width": self.config["max_index_width"],
                # 设置最小成本改进阈值
                "min_cost_improvement": 1.0003,
                "partition_num": self.config["partition_num"],
                "global_columns": self._make_global_columns(self.globally_indexable_columns[0]),
                "database_name": self.schema.database_name,
                "workload_size": self.config["workload"]["size"]
            }
            # 重置 Extend 算法，传入参数
            extend_algorithm.reset(parameters)
            # 计算当前测试工作负载的最佳索引
            indexes = extend_algorithm.calculate_best_indexes(test_wl)
            # 将计算得到的索引添加到比较索引集合中
            self.comparison_indexes["Extend_partition_histogram"] |= frozenset(indexes)

            # 记录当前测试工作负载的最终成本比例
            self.comparison_performances[run_type]["Extend_partition_histogram"][-1].append(extend_algorithm.final_cost_proportion)

        # 定义运行类型为验证
        run_type = "validation"
        # 遍历验证工作负载列表
        for index, validation_wl in enumerate(self.workload_generator.wl_validation[0]):
            # 为当前运行类型和算法添加一个空列表，用于存储性能数据
            self.comparison_performances[run_type]["Extend_partition_histogram"].append([])

            # 定义 Extend 算法的参数
            parameters = {
                # 设置预算，单位为 MB
                "budget_MB": validation_wl.budget,
                # 设置最大索引宽度
                "max_index_width": self.config["max_index_width"],
                # 设置最小成本改进阈值
                "min_cost_improvement": 1.0003,
                "partition_num": self.config["partition_num"],
                "global_columns": self._make_global_columns(self.globally_indexable_columns[0]),
                "database_name": self.schema.database_name,
                "workload_size": self.config["workload"]["size"]
            }
            # 重置 Extend 算法，传入参数
            extend_algorithm.reset(parameters)
            # 计算当前验证工作负载的最佳索引
            indexes = extend_algorithm.calculate_best_indexes(validation_wl)
            # 将计算得到的索引添加到比较索引集合中
            self.comparison_indexes["Extend_partition_histogram"] |= frozenset(indexes)

            # 记录当前验证工作负载的最终成本比例
            self.comparison_performances[run_type]["Extend_partition_histogram"][-1].append(extend_algorithm.final_cost_proportion)

    def _compare_slalom(self):
        self.evaluated_workloads = set()
        slalom_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
        slalom_connector.drop_indexes()
        slalom_algorithm = SlalomAlgorithm(slalom_connector)

        run_type = "test"
        for test_wl in self.workload_generator.wl_testing[0]:
            self.comparison_performances[run_type]["Slalom"].append([])

            parameters = {
                "budget_MB": test_wl.budget,
                "max_index_width": self.config["max_index_width"],
                # Add any other Slalom-specific parameters here
            }

            slalom_algorithm.reset(parameters)
            indexes = slalom_algorithm.calculate_best_indexes(test_wl)
            self.comparison_indexes["Slalom"] |= frozenset(indexes)
            self.comparison_performances[run_type]["Slalom"][-1].append(slalom_algorithm.final_cost_proportion)

        run_type = "validation"
        for validation_wl in self.workload_generator.wl_validation[0]:
            self.comparison_performances[run_type]["Slalom"].append([])

            parameters = {
                "budget_MB": validation_wl.budget,
                "max_index_width": self.config["max_index_width"],
                # Add any other Slalom-specific parameters here
            }

            slalom_algorithm.reset(parameters)
            indexes = slalom_algorithm.calculate_best_indexes(validation_wl)
            self.comparison_indexes["Slalom"] |= frozenset(indexes)
            self.comparison_performances[run_type]["Slalom"][-1].append(slalom_algorithm.final_cost_proportion)



    def _compare_db2advis(self):
        for model_performances_outer, run_type in [self.test_model(self.model), self.validate_model(self.model)]:
            for model_performances, _, _ in model_performances_outer:
                self.comparison_performances[run_type]["DB2Adv"].append([])
                for model_performance in model_performances:
                    parameters = {
                        "budget_MB": model_performance["available_budget"],
                        "max_index_width": self.config["max_index_width"],
                        "try_variations_seconds": 0,
                    }
                    db2advis_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
                    db2advis_connector.drop_indexes()
                    db2advis_algorithm = DB2AdvisAlgorithm(db2advis_connector, parameters)
                    indexes = db2advis_algorithm.calculate_best_indexes(model_performance["evaluated_workload"])
                    self.comparison_indexes["DB2Adv"] |= frozenset(indexes)

                    self.comparison_performances[run_type]["DB2Adv"][-1].append(
                        db2advis_algorithm.final_cost_proportion
                    )

                    self.evaluated_workloads_strs.append(f"{model_performance['evaluated_workload']}\n")

    # todo: code duplication with validate_model
    def test_model(self, model):
        model_performances = []
        for test_wl in self.workload_generator.wl_testing:
            test_env = self.DummyVecEnv([self.make_env(0, EnvironmentType.TESTING, test_wl)])
            test_env = self.VecNormalize(
                test_env, norm_obs=True, norm_reward=False, gamma=self.config["rl_algorithm"]["gamma"], training=False
            )

            if model != self.model:
                model.set_env(self.model.env)

            model_performance = self._evaluate_model(model, test_env, len(test_wl))
            # model_performance = self._evaluate_model(model, test_env, 1)
            model_performances.append(model_performance)

        return model_performances, "test"

    def validate_model(self, model):
        model_performances = []
        for validation_wl in self.workload_generator.wl_validation:
            validation_env = self.DummyVecEnv([self.make_env(0, EnvironmentType.VALIDATION, validation_wl)])
            validation_env = self.VecNormalize(
                validation_env,
                norm_obs=True,
                norm_reward=False,
                gamma=self.config["rl_algorithm"]["gamma"],
                training=False,
            )

            if model != self.model:
                model.set_env(self.model.env)

            model_performance = self._evaluate_model(model, validation_env, len(validation_wl))
            # model_performance = self._evaluate_model(model, validation_env, 1)
            model_performances.append(model_performance)

        return model_performances, "validation"

    def _evaluate_model(self, model, evaluation_env, n_eval_episodes):
        training_env = model.get_vec_normalize_env()
        self.sync_envs_normalization(training_env, evaluation_env)

        self.evaluate_policy(model, evaluation_env, n_eval_episodes)

        episode_performances = evaluation_env.get_attr("episode_performances")[0]
        perfs = []
        for perf in episode_performances:
            perfs.append(round(perf["achieved_cost"], 2))

        mean_performance = np.mean(perfs)
        print(f"Mean performance: {mean_performance:.2f} ({perfs})")

        return episode_performances, mean_performance, perfs

    def make_env(self, env_id, environment_type=EnvironmentType.TRAINING, workloads_in=None):
        def _init():
            # 修改成sb3
            action_manager_class = getattr(
                importlib.import_module("swirl.multagent_action_manager"), self.config["action_manager"]
            )
            action_manager = action_manager_class(
                indexable_column_combinations=self.globally_indexable_columns,
                action_storage_consumptions=self.action_storage_consumptions,
                sb_version=self.config["rl_algorithm"]["stable_baselines_version"],
                max_index_width=self.config["max_index_width"],
                reenable_indexes=self.config["reenable_indexes"],
                partition_num=self.config["partition_num"],
            )

            if self.number_of_actions is None:
                self.number_of_actions = action_manager.number_of_actions

            # 增加块输入
            observation_manager_config = {
                "number_of_query_classes": self.workload_generator.number_of_query_classes,
                "workload_embedder": self.workload_embedder if "workload_embedder" in self.config else None,
                # "workload_embedder": None,
                "workload_size": self.config["workload"]["size"],
                "partition_num": self.config["partition_num"],
                "histogram_feature": self.config["histogram_feature"],
                "global_columns": self._make_global_columns(self.globally_indexable_columns[0]),
                "database_name": self.schema.database_name
            }
            observation_manager_class = getattr(
                importlib.import_module("swirl.observation_manager"), self.config["observation_manager"]
            )
            observation_manager = observation_manager_class(
                action_manager.number_of_columns, observation_manager_config
            )   # number_of_columns = 26

            if self.number_of_features is None:
                self.number_of_features = observation_manager.number_of_features

            reward_calculator_class = getattr(
                importlib.import_module("swirl.reward_calculator"), self.config["reward_calculator"]
            )
            reward_calculator = reward_calculator_class()

            if environment_type == EnvironmentType.TRAINING:
                workloads = self.workload_generator.wl_training if workloads_in is None else workloads_in
            elif environment_type == EnvironmentType.TESTING:
                # Selecting the hardest workload by default
                workloads = self.workload_generator.wl_testing[-1] if workloads_in is None else workloads_in
            elif environment_type == EnvironmentType.VALIDATION:
                # Selecting the hardest workload by default
                workloads = self.workload_generator.wl_validation[-1] if workloads_in is None else workloads_in
            else:
                raise ValueError

            env = gym.make(
                f"DB-v{self.config['gym_version']}",
                environment_type=environment_type,
                config={
                    "database_name": self.schema.database_name,
                    "globally_indexable_columns": self.globally_indexable_columns_flat,
                    "workloads": workloads,
                    "random_seed": self.config["random_seed"] + env_id,
                    "max_steps_per_episode": self.config["max_steps_per_episode"],
                    "action_manager": action_manager,
                    "observation_manager": observation_manager,
                    "reward_calculator": reward_calculator,
                    "env_id": env_id,
                    "similar_workloads": self.config["workload"]["similar_workloads"],
                    "partition_num": self.config["partition_num"],
                    "workload_size": self.config["workload"]["size"],
                    "global_columns": self._make_global_columns(self.globally_indexable_columns[0]),
                },
            )
            return env

        self.set_random_seed(self.config["random_seed"])

        return _init
    
    def make_mult_env(self, env_id, environment_type=EnvironmentType.TRAINING, workloads_in=None):
        def _init():
            action_manager_class = getattr(
                importlib.import_module("swirl.multagent_action_manager"), self.config["action_manager"]
            )
            action_manager = action_manager_class(
                indexable_column_combinations=self.globally_indexable_columns,
                action_storage_consumptions=self.action_storage_consumptions,
                sb_version=self.config["rl_algorithm"]["stable_baselines_version"],
                max_index_width=self.config["max_index_width"],
                reenable_indexes=self.config["reenable_indexes"],
                partition_num = self.config["partition_num"],
            )

            if self.number_of_actions is None:
                self.number_of_actions = action_manager.number_of_actions

            observation_manager_config = {
                "number_of_query_classes": self.workload_generator.number_of_query_classes,
                # "workload_embedder": self.workload_embedder if "workload_embedder" in self.config else None,
                "workload_embedder": None,
                "workload_size": self.config["workload"]["size"],
            }
            observation_manager_class = getattr(
                importlib.import_module("swirl.observation_manager"), self.config["observation_manager"]
            )
            observation_manager = observation_manager_class(
                action_manager.number_of_columns, observation_manager_config
            )   # number_of_columns = 26

            if self.number_of_features is None:
                self.number_of_features = observation_manager.number_of_features

            reward_calculator_class = getattr(
                importlib.import_module("swirl.reward_calculator"), self.config["reward_calculator"]
            )
            reward_calculator = reward_calculator_class()

            if environment_type == EnvironmentType.TRAINING:
                workloads = self.workload_generator.wl_training if workloads_in is None else workloads_in
            elif environment_type == EnvironmentType.TESTING:
                # Selecting the hardest workload by default
                workloads = self.workload_generator.wl_testing[-1] if workloads_in is None else workloads_in
            elif environment_type == EnvironmentType.VALIDATION:
                # Selecting the hardest workload by default
                workloads = self.workload_generator.wl_validation[-1] if workloads_in is None else workloads_in
            else:
                raise ValueError
            env = eie.env(
                environment_type=environment_type,
                config={
                    "database_name": self.schema.database_name,
                    "globally_indexable_columns": self.globally_indexable_columns_flat,
                    "workloads": workloads,
                    "random_seed": self.config["random_seed"] + env_id,
                    "max_steps_per_episode": self.config["max_steps_per_episode"],
                    "action_manager": action_manager,
                    "observation_manager": observation_manager,
                    "reward_calculator": reward_calculator,
                    "env_id": env_id,
                    "similar_workloads": self.config["workload"]["similar_workloads"],
                    "partition_num":self.config["partition_num"]
                },
            )
            return env

        self.set_random_seed(self.config["random_seed"])

        return _init

    def _set_sb_version_specific_methods(self):
        """Set Stable Baselines helpers based on configured SB version.

        We prefer SB3 to avoid the TensorFlow dependency of SB2. If the config
        explicitly requests SB2, the caller must ensure tensorflow is installed.
        """

        sb_version = self.config["rl_algorithm"].get("stable_baselines_version", 3)

        if sb_version == 2:
            # SB2 path (requires tensorflow)
            from stable_baselines.common import set_global_seeds as set_global_seeds_sb2
            from stable_baselines.common.evaluation import evaluate_policy as evaluate_policy_sb2
            from stable_baselines.common.vec_env import (
                DummyVecEnv as DummyVecEnv_sb2,
                VecNormalize as VecNormalize_sb2,
                sync_envs_normalization as sync_envs_normalization_sb2,
            )

            self.set_random_seed = set_global_seeds_sb2
            self.evaluate_policy = evaluate_policy_sb2
            self.DummyVecEnv = DummyVecEnv_sb2
            self.VecNormalize = VecNormalize_sb2
            self.sync_envs_normalization = sync_envs_normalization_sb2
        elif sb_version == 3:
            # SB3 path (no TF dependency)
            from stable_baselines3.common.utils import set_random_seed as set_random_seed_sb3
            from sb3_contrib.common.maskable.evaluation import evaluate_policy
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization

            self.set_random_seed = set_random_seed_sb3
            self.evaluate_policy = evaluate_policy
            self.DummyVecEnv = DummyVecEnv
            self.VecNormalize = VecNormalize
            self.sync_envs_normalization = sync_envs_normalization
        else:
            raise ValueError("stable_baselines_version must be 2 or 3")

    def _make_global_columns(self, all_columns):
        indexed_columns = {}
        for column in all_columns:
            if "tpcds" in self.schema.database_name:
                table = column[0].table.name.split("_1")[0]
            elif "tpch" in self.schema.database_name:
                table = str(column[0]).split(" ")[1].split("_")[0]
            elif "ssb" in self.schema.database_name:
                table = str(column[0]).split(" ")[1].split("_")[0]
            attribute = str(column[0]).split(".")[-1]
            if table not in indexed_columns:
                indexed_columns[table] = set()
            indexed_columns[table].add(attribute)
        print("Global Columns: {}".format(indexed_columns))
        return indexed_columns
