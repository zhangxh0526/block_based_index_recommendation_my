import copy
import datetime
import gzip
import importlib
import logging
import os
import pickle
import random
import sys

sys.path.append("..")
import PettingZoo.custom_environment.env.index_environment as eie
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

import gym
import numpy as np

import time

from gym_db.common import EnvironmentType
from index_selection_evaluation.selection.algorithms.db2advis_algorithm import DB2AdvisAlgorithm
from index_selection_evaluation.selection.algorithms.extend_algorithm import ExtendAlgorithm
from index_selection_evaluation.selection.algorithms.slalom_algorithm import SlalomAlgorithm
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.param import *

import utils
from configuration_parser import ConfigurationParser
from schema import Schema
from workload_generator import WorkloadGenerator


class Experiment(object):
    def __init__(self, configuration_file):
        self._init_times()

        cp = ConfigurationParser(configuration_file)
        self.config = cp.config
        self._set_sb_version_specific_methods()

        self.id = self.config["id"]
        self.model = None

        self.rnd = random.Random()
        self.rnd.seed(self.config["random_seed"])

        self.comparison_performances = {
            "test": {"Extend": [], "DB2Adv": [], "Slalom": []},
            "validation": {"Extend": [], "DB2Adv": [], "Slalom": []},
        }
        self.comparison_rows = {
            "test": {"Extend": [], "DB2Adv": [], "Slalom": []},
            "validation": {"Extend": [], "DB2Adv": [], "Slalom": []},
        }
        self.comparison_indexes = {"Extend": set(), "DB2Adv": set(), "Slalom": set()}

        self.number_of_features = None
        self.number_of_actions = None
        self.evaluated_workloads_strs = []

        self.EXPERIMENT_RESULT_PATH = self.config["result_path"]
        # self._create_experiment_folder()

    def prepare(self, experiment_folder_path, violation_queries):
        self.experiment_folder_path = experiment_folder_path
        self.schema = Schema(
            self.config["workload"]["benchmark"],
            self.config["workload"]["scale_factor"],
            self.config["column_filters"],
            self.config["partition_num"]
        )

        self.workload_generator = WorkloadGenerator(
            self.config["workload"],
            workload_columns=self.schema.columns,
            random_seed=self.config["random_seed"],
            database_name=self.schema.database_name,
            experiment_id=self.id,
            filter_utilized_columns=self.config["filter_utilized_columns"],
            partition_num=self.config["partition_num"],
            violation_queries=violation_queries
        )
        self._assign_budgets_to_workloads()
        self._pickle_workloads()

        self.globally_indexable_columns = self.workload_generator.globally_indexable_columns

        # [[single column indexes], [2-column combinations], [3-column combinations]...]
        self.globally_indexable_columns = utils.create_column_permutation_indexes(
            self.globally_indexable_columns, self.config["max_index_width"]
        )

        self.single_column_flat_set = set(map(lambda x: x[0], self.globally_indexable_columns[0]))

        self.globally_indexable_columns_flat = [item for sublist in self.globally_indexable_columns for item in sublist]
        block_index = []
        for partition_id in range(self.config["partition_num"]):
            block_index.append([])
        for indexable_column_combination_flat in self.globally_indexable_columns_flat:
            partition_id = int(indexable_column_combination_flat[0].table.name.split("prt_p")[-1])
            cc = indexable_column_combination_flat
            block_index[partition_id].append(cc)
        for partition_id in range(self.config["partition_num"]):
            block_index[partition_id].sort()
        self.globally_indexable_columns_flat = [item for sublist in block_index for item in sublist]
        logging.info(f"Feeding {len(self.globally_indexable_columns_flat)} candidates into the environments.")

        self.action_storage_consumptions = []
        # 取消注释
        self.action_storage_consumptions = utils.predict_index_sizes(
            self.globally_indexable_columns_flat, self.schema.database_name, self.config["partition_num"]
        )

        if "workload_embedder" in self.config:
            workload_embedder_class = getattr(
                importlib.import_module("swirl.workload_embedder"), self.config["workload_embedder"]["type"]
            )
            workload_embedder_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
            self.workload_embedder = workload_embedder_class(
                self.schema.database_name,
                self.workload_generator.available_query_texts,
                self.config["workload_embedder"]["representation_size"],
                workload_embedder_connector,
                self.globally_indexable_columns,
                partition_num=self.config["partition_num"],
            )

        self.multi_validation_wl = []
        if len(self.workload_generator.wl_validation) > 1:
            for workloads in self.workload_generator.wl_validation:
                self.multi_validation_wl.extend(self.rnd.sample(workloads, min(7, len(workloads))))

    def _assign_budgets_to_workloads(self):
        for workload_list in self.workload_generator.wl_testing:
            for workload in workload_list:
                workload.budget = self.rnd.choice(self.config["budgets"]["validation_and_testing"])

        # assign different sizes to validate workloads
        result_workloads = []
        for workload_list in self.workload_generator.wl_validation:
            for workload in workload_list:
                _tmp_workloads = []
                for _budget in self.config["budgets"]["validation"]:
                    _workload = copy.deepcopy(workload)
                    _workload.budget = _budget
                    _tmp_workloads.append(_workload)
                result_workloads.extend(_tmp_workloads)
        self.workload_generator.wl_validation = [copy.deepcopy(result_workloads)]


    def _pickle_workloads(self):
        with open(f"{self.experiment_folder_path}/testing_workloads.pickle", "wb") as handle:
            pickle.dump(self.workload_generator.wl_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{self.experiment_folder_path}/validation_workloads.pickle", "wb") as handle:
            pickle.dump(self.workload_generator.wl_validation, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def finish(self):
        self.model.training = False

        print("Start violating final model ...... \n")
        # self.test_fm = self.test_model(self.model)[0]
        self.vali_fm = self.validate_model(self.model)[0]

        if os.path.exists(f"{self.experiment_folder_path}/moving_average_model.zip"):
            print("Start violating moving_average_model ...... \n")
            self.moving_average_model = self.model.load(f"{self.experiment_folder_path}/moving_average_model")
            self.moving_average_model.training = False
            # self.test_ma = self.test_model(self.moving_average_model)[0]
            self.vali_ma = self.validate_model(self.moving_average_model)[0]
            if len(self.multi_validation_wl) > 0:
                self.moving_average_model_mv = self.model.load(
                    f"{self.experiment_folder_path}/moving_average_model_mv.zip"
                )
                self.moving_average_model_mv.training = False
                self.test_ma_mv = self.test_model(self.moving_average_model_mv)[0]
                self.vali_ma_mv = self.validate_model(self.moving_average_model_mv)[0]
        else:
            print("file {} is not exist!\n".format(f"{self.experiment_folder_path}/moving_average_model.zip"))
            # self.test_ma = self.test_model(self.model)[0]
            self.vali_ma = self.validate_model(self.model)[0]

        if os.path.exists(f"{self.experiment_folder_path}/moving_average_model_3.zip"):
            print("Start violating moving_average_model_3 ...... \n")
            self.moving_average_model_3 = self.model.load(f"{self.experiment_folder_path}/moving_average_model_3")
            self.moving_average_model_3.training = False
            # self.test_ma_3 = self.test_model(self.moving_average_model_3)[0]
            self.vali_ma_3 = self.validate_model(self.moving_average_model_3)[0]
            if len(self.multi_validation_wl) > 0:
                self.moving_average_model_3_mv = self.model.load(
                    f"{self.experiment_folder_path}/moving_average_model_3_mv.zip"
                )
                self.moving_average_model_3_mv.training = False
                # self.test_ma_3_mv = self.test_model(self.moving_average_model_3_mv)[0]
                self.vali_ma_3_mv = self.validate_model(self.moving_average_model_3_mv)[0]
        else:
            print("file {} is not exist!\n".format(f"{self.experiment_folder_path}/moving_average_model_3.zip"))
            # self.test_ma_3 = self.test_model(self.model)[0]
            self.vali_ma_3 = self.validate_model(self.model)[0]

        if os.path.exists(f"{self.experiment_folder_path}/best_mean_reward_model.zip"):
            self.best_mean_reward_model = self.model.load(f"{self.experiment_folder_path}/best_mean_reward_model")
            self.best_mean_reward_model.training = False
            # self.test_bm = self.test_model(self.best_mean_reward_model)[0]
            self.vali_bm = self.validate_model(self.best_mean_reward_model)[0]
            if len(self.multi_validation_wl) > 0:
                self.best_mean_reward_model_mv = self.model.load(
                    f"{self.experiment_folder_path}/best_mean_reward_model_mv.zip"
                )
                self.best_mean_reward_model_mv.training = False
                # self.test_bm_mv = self.test_model(self.best_mean_reward_model_mv)[0]
                self.vali_bm_mv = self.validate_model(self.best_mean_reward_model_mv)[0]
        else:
            print("file {} is not exist!\n".format(f"{self.experiment_folder_path}/best_mean_reward_model.zip"))
            # self.test_bm = self.test_model(self.model)[0]
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

    def set_model(self):
        self.model = MaskablePPO.load(f"{self.experiment_folder_path}/final_model")

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


    def _write_report(self):
        with open(f"{self.experiment_folder_path}/report_ID_{self.id}.txt", "w") as f:
            f.write(f"##### Report for Experiment with ID: {self.id} #####\n")
            f.write(f"Description: {self.config['description']}\n")
            f.write("\n")

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
                vali_fm_perfs, self.performance_vali_final_model, self.vali_fm_details, self.row_vali_final_model = \
                self.vali_fm[idx]
                _, self.performance_vali_moving_average_model, self.vali_ma_details, self.row_vali_moving_average_model = \
                self.vali_ma[idx]
                _, self.performance_vali_moving_average_model_3, self.vali_ma_details_3, self.row_vali_moving_average_model_3 = \
                self.vali_ma_3[idx]
                _, self.performance_vali_best_mean_reward_model, self.vali_bm_details, self.row_vali_best_mean_reward_model = \
                self.vali_bm[idx]

                if len(self.multi_validation_wl) > 0:
                    _, self.performance_test_moving_average_model_mv, self.test_ma_details_mv, self.row_test_moving_average_model_mv = \
                    self.test_ma_mv[idx]
                    _, self.performance_vali_moving_average_model_mv, self.vali_ma_details_mv, self.row_vali_moving_average_model_mv = \
                    self.vali_ma_mv[idx]
                    _, self.performance_test_moving_average_model_3_mv, self.test_ma_details_3_mv, self.row_test_moving_average_model_3_mv = \
                    self.test_ma_3_mv[idx]
                    _, self.performance_vali_moving_average_model_3_mv, self.vali_ma_details_3_mv, self.row_vali_moving_average_model_3_mv = \
                    self.vali_ma_3_mv[idx]
                    _, self.performance_test_best_mean_reward_model_mv, self.test_bm_details_mv, self.row_test_best_mean_reward_model_mv = \
                    self.test_bm_mv[idx]
                    _, self.performance_vali_best_mean_reward_model_mv, self.vali_bm_details_mv, self.row_vali_best_mean_reward_model_mv = \
                    self.vali_bm_mv[idx]

                self.vali_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(vali_fm_perfs)


                f.write("    Final mean performance validation:\n")
                f.write(
                    (
                        "        Final model:               "
                        f"{self.performance_vali_final_model:.2f} ({self.vali_fm_details})"
                        f"\t\t\t{self.row_vali_final_model:.2f}\n"
                    )
                )
                f.write(
                    (
                        "        Moving Average model:      "
                        f"{self.performance_vali_moving_average_model:.2f} ({self.vali_ma_details})"
                        f"\t\t\t{self.row_vali_moving_average_model:.2f}\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average model (MV): "
                            f"{self.performance_vali_moving_average_model_mv:.2f} ({self.vali_ma_details_mv})"
                            f"\t\t\t{self.row_vali_moving_average_model_mv:.2f}\n"
                        )
                    )
                f.write(
                    (
                        "        Moving Average 3 model:    "
                        f"{self.performance_vali_moving_average_model_3:.2f} ({self.vali_ma_details_3})"
                        f"\t\t\t{self.row_vali_moving_average_model_3:.2f}\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average 3 mod (MV): "
                            f"{self.performance_vali_moving_average_model_3_mv:.2f} ({self.vali_ma_details_3_mv})"
                            f"\t\t\t{self.row_vali_moving_average_model_3_mv:.2f}\n"
                        )
                    )
                f.write(
                    (
                        "        Best mean reward model:    "
                        f"{self.performance_vali_best_mean_reward_model:.2f} ({self.vali_bm_details})"
                        f"\t\t\t{self.row_vali_best_mean_reward_model:.2f}\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Best mean reward mod (MV): "
                            f"{self.performance_vali_best_mean_reward_model_mv:.2f} ({self.vali_bm_details_mv})"
                            f"\t\t\t{self.row_vali_best_mean_reward_model_mv:.2f}\n"
                        )
                    )
                for key, value in self.comparison_performances["validation"].items():
                    if len(value) < 1:
                        continue
                    f.write(f"        {key}:                    {np.mean(value[idx]):.2f} ({value[idx]})")
                    value = self.comparison_rows["validation"][key]
                    f.write(f"\t\t\t{np.mean(value[idx]):.2f} ({value[idx]})\n")
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

            def final_avg_row(values, probabilities):
                row = 0
                for res in values:
                    row += res[3]
                return row / probabilities

            f.write(("        Final model:               " f"{final_avg(self.test_fm, probabilities):.2f}"))
            f.write(("\t\t\t"f"{final_avg_row(self.test_fm, probabilities):.2f}\n"))
            f.write(("        Moving Average model:      " f"{final_avg(self.test_ma, probabilities):.2f}"))
            f.write(("\t\t\t"f"{final_avg_row(self.test_ma, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average model (MV): " f"{final_avg(self.test_ma_mv, probabilities):.2f}"))
                f.write(("\t\t\t"f"{final_avg_row(self.test_ma_mv, probabilities):.2f}\n"))
            f.write(("        Moving Average 3 model:    " f"{final_avg(self.test_ma_3, probabilities):.2f}"))
            f.write(("\t\t\t"f"{final_avg_row(self.test_ma_3, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average 3 mod (MV): " f"{final_avg(self.test_ma_3_mv, probabilities):.2f}"))
                f.write(("\t\t\t"f"{final_avg_row(self.test_ma_3_mv, probabilities):.2f}\n"))
            f.write(("        Best mean reward model:    " f"{final_avg(self.test_bm, probabilities):.2f}"))
            f.write(("\t\t\t"f"{final_avg_row(self.test_bm, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Best mean reward mod (MV): " f"{final_avg(self.test_bm_mv, probabilities):.2f}"))
                f.write(("\t\t\t"f"{final_avg_row(self.test_bm_mv, probabilities):.2f}\n"))
            f.write(
                (
                    "        Extend:                    "
                    f"{np.mean(self.comparison_performances['test']['Extend']):.2f}"
                    f"\t\t\t{np.mean(self.comparison_rows['test']['Extend']):.2f}\n"
                )
            )
            f.write(
                (
                    "        Slalom:                    "
                    f"{np.mean(self.comparison_performances['test']['Slalom']):.2f}"
                    f"\t\t\t{np.mean(self.comparison_rows['test']['Slalom']):.2f}\n"
                )
            )
            f.write("\n")
            f.write("Overall Validation:\n")
            f.write(("        Final model:               " f"{final_avg(self.vali_fm, probabilities):.2f}"))
            f.write(("\t\t\t"f"{final_avg_row(self.vali_fm, probabilities):.2f}\n"))
            f.write(("        Moving Average model:      " f"{final_avg(self.vali_ma, probabilities):.2f}"))
            f.write(("\t\t\t"f"{final_avg_row(self.vali_ma, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average model (MV): " f"{final_avg(self.vali_ma_mv, probabilities):.2f}"))
                f.write(("\t\t\t"f"{final_avg_row(self.vali_ma_mv, probabilities):.2f}\n"))
            f.write(("        Moving Average 3 model:    " f"{final_avg(self.vali_ma_3, probabilities):.2f}"))
            f.write(("\t\t\t"f"{final_avg_row(self.vali_ma_3, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Moving Average 3 mod (MV): " f"{final_avg(self.vali_ma_3_mv, probabilities):.2f}"))
                f.write(("\t\t\t"f"{final_avg_row(self.vali_ma_3_mv, probabilities):.2f}\n"))
            f.write(("        Best mean reward model:    " f"{final_avg(self.vali_bm, probabilities):.2f}"))
            f.write(("\t\t\t"f"{final_avg_row(self.vali_bm, probabilities):.2f}\n"))
            if len(self.multi_validation_wl) > 0:
                f.write(("        Best mean reward mod (MV): " f"{final_avg(self.vali_bm_mv, probabilities):.2f}"))
                f.write(("\t\t\t"f"{final_avg_row(self.vali_bm_mv, probabilities):.2f}\n"))
            f.write(
                (
                    "        Extend:                    "
                    f"{np.mean(self.comparison_performances['validation']['Extend']):.2f}"
                    f"\t\t\t{np.mean(self.comparison_rows['validation']['Extend']):.2f}\n"
                )
            )
            f.write(
                (
                    "        Slalom:                    "
                    f"{np.mean(self.comparison_performances['validation']['Slalom']):.2f}"
                    f"\t\t\t{np.mean(self.comparison_rows['validation']['Slalom']):.2f}\n"
                )
            )

    def compare(self):
        if len(self.config["comparison_algorithms"]) < 1:
            return

        if "extend" in self.config["comparison_algorithms"]:
            self._compare_extend()
        if "slalom" in self.config["comparison_algorithms"]:
            self._compare_slalom()
        for key, comparison_performance in self.comparison_performances.items():
            print(f"Comparison for {key}:")
            for key, value in comparison_performance.items():
                print(f"    {key}: {np.mean(value):.2f} ({value})")

        self._evaluate_comparison()

    def _evaluate_comparison(self):
        for key, comparison_indexes in self.comparison_indexes.items():
            columns_from_indexes = set()
            for index in comparison_indexes:
                for column in index.columns:
                    columns_from_indexes |= set([column])

            impossible_index_columns = columns_from_indexes - self.single_column_flat_set
            logging.critical(f"{key} finds indexes on these not indexable columns:\n    {impossible_index_columns}")

            assert len(impossible_index_columns) == 0, "Found indexes on not indexable columns."

    def _compare_extend(self):
        self.evaluated_workloads = set()
        extend_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
        extend_connector.drop_indexes()
        extend_algorithm = ExtendAlgorithm(extend_connector)

        run_type = "test"
        for test_wl in self.workload_generator.wl_testing[0]:
            self.comparison_performances[run_type]["Extend"].append([])
            self.comparison_rows[run_type]["Extend"].append([])
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
            self.comparison_rows[run_type]["Extend"].append([])

            parameters = {
                "budget_MB": validation_wl.budget,
                "max_index_width": self.config["max_index_width"],
                "min_cost_improvement": 1.0003,
            }
            extend_algorithm.reset(parameters)
            indexes = extend_algorithm.calculate_best_indexes(validation_wl)
            self.comparison_indexes["Extend"] |= frozenset(indexes)

            self.comparison_performances[run_type]["Extend"][-1].append(extend_algorithm.final_cost_proportion)

    def _compare_slalom(self):
        self.evaluated_workloads = set()
        slalom_connector = PostgresDatabaseConnector(self.schema.database_name, autocommit=True)
        slalom_connector.drop_indexes()
        slalom_algorithm = SlalomAlgorithm(slalom_connector)

        run_type = "test"
        for test_wl in self.workload_generator.wl_testing[0]:
            self.comparison_performances[run_type]["Slalom"].append([])
            self.comparison_rows[run_type]["Slalom"].append([])

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
            self.comparison_rows[run_type]["Slalom"].append([])

            parameters = {
                "budget_MB": validation_wl.budget,
                "max_index_width": self.config["max_index_width"],
                # Add any other Slalom-specific parameters here
            }

            slalom_algorithm.reset(parameters)
            indexes = slalom_algorithm.calculate_best_indexes(validation_wl)
            self.comparison_indexes["Slalom"] |= frozenset(indexes)
            self.comparison_performances[run_type]["Slalom"][-1].append(slalom_algorithm.final_cost_proportion)

    # todo: code duplication with validate_model
    def test_model(self, model):
        model_performances = []
        for test_wl in self.workload_generator.wl_testing:
            test_env = self.DummyVecEnv([self.make_env(0, EnvironmentType.TESTING, test_wl)])
            test_env = self.VecNormalize(
                test_env, norm_obs=True, norm_reward=False, gamma=self.config["rl_algorithm"]["gamma"], training=False
            )

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

            # if model != self.model:
            #     model.set_env(self.model.env)

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
        rows = []
        for perf in episode_performances:
            perfs.append(round(perf["achieved_cost"], 2))
            rows.append(perf["rows"])

        mean_performance = np.mean(perfs)
        mean_rows = np.mean(rows)
        print(f"Mean performance: {mean_performance:.2f} ({perfs})")
        print(f"Mean rows: {mean_rows:.2f} ({rows})")

        return episode_performances, mean_performance, perfs, mean_rows

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
            )  # number_of_columns = 26

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
                partition_num=self.config["partition_num"],
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
            )  # number_of_columns = 26

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
                    "partition_num": self.config["partition_num"]
                },
            )
            return env

        self.set_random_seed(self.config["random_seed"])

        return _init

    def _set_sb_version_specific_methods(self):
        # if self.config["rl_algorithm"]["stable_baselines_version"] == 2:
        from stable_baselines.common import set_global_seeds as set_global_seeds_sb2
        from sb3_contrib.common.maskable.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
        # from stable_baselines.common.evaluation import evaluate_policy as evaluate_policy_sb2
        # from stable_baselines.common.vec_env import DummyVecEnv as DummyVecEnv_sb2
        # from stable_baselines.common.vec_env import VecNormalize as VecNormalize_sb2
        # from stable_baselines.common.vec_env import sync_envs_normalization as sync_envs_normalization_sb2

        self.set_random_seed = set_global_seeds_sb2
        self.evaluate_policy = evaluate_policy
        self.DummyVecEnv = DummyVecEnv
        self.VecNormalize = VecNormalize
        self.sync_envs_normalization = sync_envs_normalization

    def _make_global_columns(self, all_columns):
        indexed_columns = {}
        for column in all_columns:
            table = str(column[0]).split(" ")[1].split("_")[0]
            attribute = str(column[0]).split(".")[-1]
            if table not in indexed_columns:
                indexed_columns[table] = set()
            indexed_columns[table].add(attribute)
        return indexed_columns