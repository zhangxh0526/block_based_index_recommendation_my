import copy
import logging
import math

import numpy as np
from gym import spaces

from index_selection_evaluation.selection.utils import b_to_mb

from cmath import inf
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

FORBIDDEN_ACTION_SB3 = 0
ALLOWED_ACTION_SB3 = 1

FORBIDDEN_ACTION_SB2 = 0
ALLOWED_ACTION_SB2 = 1


class ActionManager(object):
    def __init__(self, sb_version, max_index_width, partition_num=None):
        self.valid_actions = None
        self._remaining_valid_actions = None
        self.number_of_actions = None
        self.current_action_status = None

        self.test_variable = None
        self.partition_num = partition_num
        assert partition_num is not None, "specify partition_num"
        self.sb_version = sb_version
        self.MAX_INDEX_WIDTH = max_index_width

        if self.sb_version == 2:
            self.FORBIDDEN_ACTION = FORBIDDEN_ACTION_SB2
            self.ALLOWED_ACTION = ALLOWED_ACTION_SB2
        else:
            self.FORBIDDEN_ACTION = FORBIDDEN_ACTION_SB3
            self.ALLOWED_ACTION = ALLOWED_ACTION_SB3

    def get_action_space(self):
        return spaces.Discrete(self.number_of_actions)

    def get_initial_valid_actions(self, workload, budget):
        # 0 for actions not taken yet, 1 for single column index present, 0.5 for two-column index present,
        # 0.33 for three-column index present, ...
        self.current_action_status = [0 for action in range(self.number_of_columns)]

        self.valid_actions = [self.FORBIDDEN_ACTION for action in range(self.number_of_actions)]
        self._remaining_valid_actions = []

        self._valid_actions_based_on_workload(workload)
        self._valid_actions_based_on_budget(budget, current_storage_consumption=0)

        self.current_combinations = set()

        return np.array(self.valid_actions)

    def update_valid_actions(self, last_action, budget, current_storage_consumption):
        assert self.indexable_column_combinations_flat[last_action] not in self.current_combinations

        actions_index_width = len(self.indexable_column_combinations_flat[last_action])
        if actions_index_width == 1:
            self.current_action_status[last_action] += 1
        else:
            combination_to_be_extended = self.indexable_column_combinations_flat[last_action][:-1]
            assert combination_to_be_extended in self.current_combinations

            status_value = 1 / actions_index_width

            last_action_back_column = self.indexable_column_combinations_flat[last_action][-1]
            last_action_back_columns_idx = self.column_to_idx[last_action_back_column]
            self.current_action_status[last_action_back_columns_idx] += status_value

            self.current_combinations.remove(combination_to_be_extended)

        self.current_combinations.add(self.indexable_column_combinations_flat[last_action])

        self.valid_actions[last_action] = self.FORBIDDEN_ACTION
        self._remaining_valid_actions.remove(last_action)

        self._valid_actions_based_on_last_action(last_action)
        self._valid_actions_based_on_budget(budget, current_storage_consumption)

        is_valid_action_left = len(self._remaining_valid_actions) > 0

        return np.array(self.valid_actions), is_valid_action_left

    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        if budget is None:
            return
        else:
            new_remaining_actions = []
            for action_idx in self._remaining_valid_actions:
                if b_to_mb(current_storage_consumption + self.action_storage_consumptions[action_idx]) > budget:
                    self.valid_actions[action_idx] = self.FORBIDDEN_ACTION
                else:
                    new_remaining_actions.append(action_idx)

            self._remaining_valid_actions = new_remaining_actions

    def _valid_actions_based_on_workload(self, workload):
        raise NotImplementedError

    def _valid_actions_based_on_last_action(self, last_action):
        raise NotImplementedError


class BlockUnAwareActionManager(ActionManager):
    def __init__(
            self, indexable_column_combinations, action_storage_consumptions, sb_version, max_index_width,
            reenable_indexes, partition_num=None
    ):
        ActionManager.__init__(self, sb_version, max_index_width=max_index_width, partition_num=partition_num)

        self.indexable_column_combinations_flat = indexable_column_combinations
        self.partition_num = partition_num
        assert self.partition_num is not None, "In Partition envs, you need specify the number of block"
        # This is the same as the Expdriment's object globally_indexable_columns_flat attribute
        self.trans_to_partition_unaware_indexes(action_storage_consumptions)

        self.per_index_storage = {}
        for _ in self.indexable_column_combinations_flat:
            self.per_index_storage[_] = action_storage_consumptions[self.indexable_column_combinations_flat.index(_)]

        self.REENABLE_INDEXES = reenable_indexes

        self.candidate_dependent_map = {}
        for indexable_column_combination in self.index_names:
            if len(indexable_column_combination) > max_index_width - 1:
                continue
            self.candidate_dependent_map[indexable_column_combination] = []

        for column_combination_idx, indexable_column_combination in enumerate(self.index_names):
            if len(indexable_column_combination) < 2:
                continue
            dependent_of = indexable_column_combination[:-1]
            self.candidate_dependent_map[dependent_of].append(column_combination_idx)
        pass

    def trans_to_partition_unaware_indexes(self, action_storage_consumptions):
        # self.indexable_column_combinations_flat = [
        #     item for sublist in self.indexable_column_combinations for item in sublist
        # ]  # all columns including single-columns,2 col, 3cols
        self.available_indexes = {}
        self.action_storage_consumptions = {}
        self.indexable_columns = list()
        self.index_names = list()
        num_of_columns = 0
        for index in self.indexable_column_combinations_flat:
            index_name = tuple([_.name for _ in index])
            # if len(index_name) == 1 and index_name[0] == "lo_commitdate":
            #     if index_name not in self.available_indexes:
            #         self.index_names.append(index_name)
            #         self.available_indexes[index_name] = []
            #         self.action_storage_consumptions[index_name] = 0
            #         if len(index_name) == 1:
            #             num_of_columns += 1
            #             self.indexable_columns.append(index_name)
            #     self.available_indexes[index_name].append(index)
            #     self.action_storage_consumptions[index_name] += action_storage_consumptions[
            #         self.indexable_column_combinations_flat.index(index)]
            #     temp = action_storage_consumptions[
            #         self.indexable_column_combinations_flat.index(index)]
            #     print(f"{index} -------  {temp}")
            #
            # if len(index_name) == 2 and index_name[0] == "lo_commitdate" and index_name[1] == "lo_quantity":
            #     if index_name not in self.available_indexes:
            #         self.index_names.append(index_name)
            #         self.available_indexes[index_name] = []
            #         self.action_storage_consumptions[index_name] = 0
            #         if len(index_name) == 1:
            #             num_of_columns += 1
            #             self.indexable_columns.append(index_name)
            #     self.available_indexes[index_name].append(index)
            #     self.action_storage_consumptions[index_name] += action_storage_consumptions[
            #         self.indexable_column_combinations_flat.index(index)]
            #     temp = action_storage_consumptions[
            #         self.indexable_column_combinations_flat.index(index)]
            #     print(f"{index} -------  {temp}")

            if index_name not in self.available_indexes:
                self.index_names.append(index_name)
                self.available_indexes[index_name] = []
                self.action_storage_consumptions[index_name] = 0
                if len(index_name) == 1:
                    num_of_columns += 1
                    self.indexable_columns.append(index_name)
            self.available_indexes[index_name].append(index)
            self.action_storage_consumptions[index_name] += action_storage_consumptions[self.indexable_column_combinations_flat.index(index)]

        self.number_of_actions = len(self.available_indexes)
        self.number_of_columns = num_of_columns


    def get_action_space(self):
        # 改动 return index_id
        # print("Action Dim: {}".format([self.partition_num, self.number_of_index_per_partition]))
        # return spaces.MultiDiscrete([self.partition_num, self.number_of_index_per_partition])
        return spaces.Discrete(self.number_of_actions)

    def get_initial_valid_actions(self, workload, budget):
        # 0 for actions not taken yet, 1 for single column index present, 0.5 for two-column index present,
        # 0.33 for three-column index present, ...
        self.current_action_status = [0 for action in range(self.number_of_columns)]

        self.valid_actions = [self.FORBIDDEN_ACTION for action in range(self.number_of_actions)]
        self._remaining_valid_actions = []

        self._valid_actions_based_on_workload(workload)
        self._valid_actions_based_on_budget(budget, current_storage_consumption=0)

        self.current_combinations = set()

        return np.array(self.valid_actions)

    def get_global_action(self, index_action):
        return self.available_indexes[self.index_names[index_action]]

    def update_valid_actions(self, last_action, budget, current_storage_consumption):

        last_action = last_action

        column_name = self.index_names[last_action]

        assert self.index_names[last_action] not in self.current_combinations

        # 如果action是[blockid, indid]，将其转为action

        actions_index_width = len(self.available_indexes[self.index_names[last_action]][0])
        if actions_index_width == 1:
            self.current_action_status[self.indexable_columns.index(column_name)] += 1

        else:
            combination_to_be_extended = self.index_names[last_action][:-1]

            status_value = 1 / actions_index_width

            self.current_action_status[self.indexable_columns.index(tuple([self.index_names[last_action][-1]]))] += status_value

            self.current_combinations.remove(combination_to_be_extended)

        self.current_combinations.add(self.index_names[last_action])

        self.valid_actions[last_action] = self.FORBIDDEN_ACTION
        self._remaining_valid_actions.remove(last_action)

        self._valid_actions_based_on_last_action(last_action)
        self._valid_actions_based_on_budget(budget, current_storage_consumption)

        is_valid_action_left = len(self._remaining_valid_actions) > 0

        # print("Is valid action:{}, present number of available actions is {}".format(is_valid_action_left, sum(self.valid_actions)))
        return np.array(self.valid_actions), is_valid_action_left

    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        if budget is None:
            return
        else:
            new_remaining_actions = []
            for action_idx in self._remaining_valid_actions:
                #print(b_to_mb(self.action_storage_consumptions[self.index_names[action_idx]]))
                if b_to_mb(current_storage_consumption + self.action_storage_consumptions[self.index_names[action_idx]]) > budget:
                    self.valid_actions[action_idx] = self.FORBIDDEN_ACTION
                else:
                    new_remaining_actions.append(action_idx)

            self._remaining_valid_actions = new_remaining_actions

    def _valid_actions_based_on_last_action(self, last_action):
        last_combination = self.index_names[last_action]
        last_combination_length = len(last_combination)

        if last_combination_length != self.MAX_INDEX_WIDTH:
            for column_combination_idx in self.candidate_dependent_map[last_combination]:
                indexable_column_combination = self.index_names[column_combination_idx]
                possible_extended_column = tuple([indexable_column_combination[-1]])

                if possible_extended_column not in self.wl_indexable_columns:
                    continue
                if indexable_column_combination in self.current_combinations:
                    continue

                self._remaining_valid_actions.append(column_combination_idx)
                self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION

        # Disable now (after the last action) invalid combinations
        for column_combination_idx in copy.copy(self._remaining_valid_actions):
            indexable_column_combination = self.index_names[column_combination_idx]
            indexable_column_combination_length = len(indexable_column_combination)
            if indexable_column_combination_length == 1:
                continue

            if indexable_column_combination_length != last_combination_length:
                continue

            if last_combination[:-1] != indexable_column_combination[:-1]:
                continue

            if column_combination_idx in self._remaining_valid_actions:
                self._remaining_valid_actions.remove(column_combination_idx)
            self.valid_actions[column_combination_idx] = self.FORBIDDEN_ACTION


    def _valid_actions_based_on_workload(self, workload):
        indexable_columns = workload.indexable_columns(return_sorted=False)
        _indexable_columns = set()
        for _ in indexable_columns:
            _indexable_columns.add(tuple([_.name]))
        indexable_columns = copy.deepcopy(_indexable_columns)
        # temp = copy.deepcopy(self.indexable_columns)
        # self.indexable_columns = [_[0] for _ in temp]
        indexable_columns = indexable_columns & frozenset(self.indexable_columns)
        # self.indexable_columns = temp
        # remove not in where

        self.wl_indexable_columns = indexable_columns

        for _ in self.index_names:
            if len(_) == 1 and _ in indexable_columns:
                self.valid_actions[self.index_names.index(_)] = self.ALLOWED_ACTION
                self._remaining_valid_actions.append(self.index_names.index(_))


        assert np.count_nonzero(np.array(self.valid_actions) == self.ALLOWED_ACTION) == len(
            indexable_columns
        ), "Valid actions mismatch indexable columns"

