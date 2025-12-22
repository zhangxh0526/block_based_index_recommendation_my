import logging
import random

import numpy as np
from gym import spaces
from histgrams_manager import HistogramsManager

from utils import *

VERY_HIGH_BUDGET = 100_000_000_000
from index_selection_evaluation.selection.utils import b_to_mb


class ObservationManager(object):
    def __init__(self, number_of_actions):
        self.number_of_actions = number_of_actions

    def _init_episode(self, state_fix_for_episode):
        self.episode_budget = state_fix_for_episode["budget"]
        if self.episode_budget is None:
            self.episode_budget = VERY_HIGH_BUDGET

        self.initial_cost = state_fix_for_episode["initial_cost"]

    def init_episode(self, state_fix_for_episode):
        raise NotImplementedError

    def get_observation(self, environment_state):
        raise NotImplementedError

    def get_observation_space(self):
        observation_space = spaces.Box(
            low=self._create_low_boundaries(), high=self._create_high_boundaries(), shape=self._create_shape(), dtype=np.float32
        )

        logging.info(f"Creating ObservationSpace with {self.number_of_features} features.")

        return observation_space

    def _create_shape(self):
        return (self.number_of_features,)

    def _create_low_boundaries(self):
        low = [-np.inf for feature in range(self.number_of_features)]

        return np.array(low)

    def _create_high_boundaries(self):
        high = [np.inf for feature in range(self.number_of_features)]

        return np.array(high)


class EmbeddingObservationManager(ObservationManager):
    def __init__(self, number_of_actions, config):
        ObservationManager.__init__(self, number_of_actions)

        self.workload_embedder = config["workload_embedder"]
        self.representation_size = self.workload_embedder.representation_size
        self.workload_size = config["workload_size"]
        self.partition_num = config["partition_num"]

        self.number_of_features = (
            self.number_of_actions  # Indicates for each action whether it was taken or not
            + (
                self.representation_size * self.workload_size
            )  # embedding representation for every query in the workload
            + self.workload_size  # The frequencies for every query in the workloads
            + 1  # The episode's budget
            + 1  # The current storage consumption
            + 1  # The initial workload cost
            + 1  # The current workload cost
        )

    def _init_episode(self, state_fix_for_episode):
        episode_workload = state_fix_for_episode["workload"]
        self.frequencies = np.array(EmbeddingObservationManager._get_frequencies_from_workload(episode_workload))

        super()._init_episode(state_fix_for_episode)

    def init_episode(self, state_fix_for_episode):
        raise NotImplementedError

    def get_observation(self, environment_state):
        if self.UPDATE_EMBEDDING_PER_OBSERVATION:
            workload_embedding = np.array(self.workload_embedder.get_embeddings(environment_state["plans_per_query"]))
        else:
            # In this case the workload embedding is not updated with every step but also not set during init
            if self.workload_embedding is None:
                self.workload_embedding = np.array(
                    self.workload_embedder.get_embeddings(environment_state["plans_per_query"])
                )

            workload_embedding = self.workload_embedding

        observation = np.array(environment_state["action_status"])
        observation = np.append(observation, workload_embedding)
        observation = np.append(observation, self.frequencies)
        observation = np.append(observation, self.episode_budget)
        observation = np.append(observation, environment_state["current_storage_consumption"])
        observation = np.append(observation, self.initial_cost)
        observation = np.append(observation, environment_state["current_cost"])

        return observation

    @staticmethod
    def _get_frequencies_from_workload(workload):
        frequencies = []
        for query in workload.queries:
            frequencies.append(query.frequency)
        return frequencies



# Todo: Rename. Single/multi-column is not handled by the ObservationManager anymore.
# All managers are capable of handling single and multi-attribute indexes now.
class SingleColumnIndexPlanEmbeddingObservationManagerWithCost(EmbeddingObservationManager):
    def __init__(self, number_of_actions, config):
        super().__init__(number_of_actions, config)

        self.UPDATE_EMBEDDING_PER_OBSERVATION = True

        # This overwrites EmbeddingObservationManager's features
        # self.number_of_features = (
        #     self.number_of_actions  # Indicates for each action whether it was taken or not
        #     + (
        #         self.representation_size * self.workload_size * self.partition_num
        #     )  # embedding representation for every query in the workload
        #     + self.workload_size * self.partition_num  # The costs for every query in the workload
        #     + self.workload_size * self.partition_num # The frequencies for every query in the workloads
        #     + 1  # The episode's budget
        #     + 1  # The current storage consumption
        #     + 1  # The initial workload cost
        #     + 1  # The current workload cost
        # )
        self.number_of_features = (
                self.number_of_actions  # Indicates for each action whether it was taken or not
                + (
                        self.representation_size * self.workload_size
                )  # embedding representation for every query in the workload
                + self.workload_size  # The costs for every query in the workload
                + self.workload_size  # The frequencies for every query in the workloads
                + 1  # The episode's budget
                + 1  # The current storage consumption
                + 1  # The initial workload cost
                + 1  # The current workload cost
        )

    def init_episode(self, state_fix_for_episode):
        super()._init_episode(state_fix_for_episode)

    def trans_block_unaware_cut(self, list_data):
        if type(list_data) == 'numpy.ndarray':
            list_data = list_data.to_list()

        result = []
        for _ in range(int(len(list_data)/self.partition_num)):
            result.append(list_data[_*self.partition_num])
        return result

    def trans_block_unaware_sum(self, list_data):
        result = []
        tmp_value = []
        for _ in (list_data):
            if len(tmp_value) < self.partition_num:
                tmp_value.append(_)
            else:
                result.append(sum(tmp_value))
                tmp_value.clear()
                tmp_value.append(_)
        result.append(sum(tmp_value))
        return result


    # This overwrite EmbeddingObservationManager.get_observation() because further features are added
    def get_observation(self, environment_state):
        workload_embedding = np.array(self.workload_embedder.get_embeddings(self.trans_block_unaware_cut(environment_state["plans_per_query"])))
        observation = np.array(environment_state["action_status"])
        observation = np.append(observation, workload_embedding)
        observation = np.append(observation, self.trans_block_unaware_sum(environment_state["costs_per_query"]))
        observation = np.append(observation, self.trans_block_unaware_cut(self.frequencies))
        observation = np.append(observation, self.episode_budget)
        observation = np.append(observation, environment_state["current_storage_consumption"])
        observation = np.append(observation, self.initial_cost)
        observation = np.append(observation, environment_state["current_cost"])

        return observation


class BISLearnerObservationManager(EmbeddingObservationManager):
    def __init__(self, number_of_actions, config):
        super().__init__(number_of_actions, config)

        self.histogram_feature = config["histogram_feature"]

        self.global_columns = config["global_columns"]


        # fake records


        # self.histogram_manager = HistogramsManager(
        #     # self._fake_records(self.global_columns),
        #     indexed_columns=config["global_columns"],
        #     database_name=config["database_name"],
        #     query_limit=self.workload_size
        # )

        self.UPDATE_EMBEDDING_PER_OBSERVATION = True

        # This overwrites EmbeddingObservationManager's features

        self.state_features = (
                self.number_of_actions  # Indicates for each action whether it was taken or not
                + 1  # The episode's budget
                + 1  # The current storage consumption
                + 1  # The initial workload cost
                + 1  # The current workload cost
        )

        self.workload_features = (
            (
                self.representation_size * self.workload_size * self.partition_num
            )  # embedding representation for every query in the workload
            + self.workload_size * self.partition_num  # The costs for every query in the workload
            + self.workload_size * self.partition_num  # The frequencies for every query in the workloads
        )

        self.histgram_features = self.workload_size * self.histogram_feature * self.columns_per_histogram(config["global_columns"])

    def columns_per_histogram(self, indexed_columns):
        column_num = 0
        for table in indexed_columns:
            for column in indexed_columns[table]:
                column_num += 1
        return column_num

    def init_episode(self, state_fix_for_episode):
        self.current_workloads = state_fix_for_episode["workload"]
        super()._init_episode(state_fix_for_episode)

    # This overwrite EmbeddingObservationManager.get_observation() because further features are added
    # def get_observation(self, environment_state):
    #     workload_embedding = np.array(self.workload_embedder.get_embeddings(environment_state["plans_per_query"]))
    #
    #     state_observation = np.array(environment_state["action_status"])
    #     state_observation = np.append(state_observation, self.episode_budget)
    #     state_observation = np.append(state_observation, environment_state["current_storage_consumption"])
    #     state_observation = np.append(state_observation, self.initial_cost)
    #     state_observation = np.append(state_observation, environment_state["current_cost"])
    #
    #     workload_observation = np.array(workload_embedding)
    #     workload_observation = np.append(workload_observation, environment_state["costs_per_query"])
    #     workload_observation = np.append(workload_observation, self.frequencies)
    #
    #
    #     # todo 验证功能，待实现
    #
    #     # block_based_workload_observation = \
    #     #     self.histogram_manager.block_based_workload_histgrams(environment_state["workload_inf"])
    #
    #
    #     block_based_workload_observation = \
    #         self.histogram_manager.block_based_workload_histgrams(self.handle_workload())
    #
    #     # block_based_workload_observation = \
    #     #     self.histogram_manager.block_based_workload_histgrams(self._fake_workload())
    #
    #     observation = {
    #         "state_inf": state_observation,
    #         "workload_inf": [workload_observation],
    #         "block_based_workload_inf": block_based_workload_observation}
    #
    #     return observation

    def get_observation_histogramed(self, environment_state, histogram_manager):
        workload_embedding = np.array(self.workload_embedder.get_embeddings(environment_state["plans_per_query"]))

        state_observation = np.array(environment_state["action_status"])
        state_observation = np.append(state_observation, self.episode_budget)
        state_observation = np.append(state_observation, b_to_mb(environment_state["current_storage_consumption"]))
        state_observation = np.append(state_observation, self.initial_cost)
        state_observation = np.append(state_observation, environment_state["current_cost"])

        workload_observation = np.array(workload_embedding)
        workload_observation = np.append(workload_observation, environment_state["costs_per_query"])
        workload_observation = np.append(workload_observation, self.frequencies)



        # todo 验证功能，待实现

        # block_based_workload_observation = \
        #     self.histogram_manager.block_based_workload_histgrams(environment_state["workload_inf"])


        block_based_workload_observation = \
            histogram_manager.block_based_workload_histgrams(self.handle_workload())


        observation = {
            "state_inf": state_observation,
            "workload_inf": [workload_observation],
            "block_based_workload_inf": block_based_workload_observation}

        return observation


    def get_histgramed_observation_space(self, histogram_manager):
        observation_space = spaces.Dict(
            {"state_inf": spaces.Box(low=-np.inf, high=np.inf,
                                                   shape=(self.state_features,), dtype=np.float32),
             "workload_inf": spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(1, self.workload_features), dtype=np.float32),
             "block_based_workload_inf": spaces.Box(low=-np.inf, high=np.inf,
                                                            shape=(histogram_manager.histogram_num(), self.histgram_features),
                                                            dtype=np.float32),
            })


        return observation_space

    def _fake_records(self, indexed_records):
        records = {}
        for table in indexed_records:
            records[table] = {}
            for partition in range(self.partition_num):
                records[table][str(partition)] = {}
                for column in indexed_records[table]:
                    records[table][str(partition)][column] = [_ for _ in range(50000)]
        return records


    def handle_workload(self):
        workloads = []
        for query in self.current_workloads.queries:
            workloads.append(query.text)
        workloads = predicate_splitting(workloads)

        # eliminate results
        _workloads = []
        for _ in workloads:
            if _ not in _workloads:
                _workloads.append(_)
        return _workloads
