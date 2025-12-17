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


class DRLindaActionManager(ActionManager):
    def __init__(
        self, indexable_column_combinations, action_storage_consumptions, sb_version, max_index_width, reenable_indexes
    ):
        ActionManager.__init__(self, sb_version, max_index_width=max_index_width)

        self.indexable_column_combinations = indexable_column_combinations
        # This is the same as the Expdriment's object globally_indexable_columns_flat attribute
        self.indexable_column_combinations_flat = [
            item for sublist in self.indexable_column_combinations for item in sublist
        ]
        self.number_of_actions = len(self.indexable_column_combinations_flat)
        self.number_of_columns = len(self.indexable_column_combinations[0])

        self.action_storage_consumptions = action_storage_consumptions

        self.indexable_columns = list(
            map(lambda one_column_combination: one_column_combination[0], self.indexable_column_combinations[0])
        )

        self.column_to_idx = {}
        for idx, column in enumerate(self.indexable_column_combinations[0]):
            c = column[0]
            self.column_to_idx[c] = idx

    def get_action_space(self):
        return spaces.Discrete(self.number_of_actions)

    def get_initial_valid_actions(self, workload, budget):
        # 0 for actions not taken yet, 1 for single column index present
        self.current_action_status = [0 for action in range(self.number_of_columns)]

        self.valid_actions = [self.ALLOWED_ACTION for action in range(self.number_of_actions)]
        self._remaining_valid_actions = list(range(self.number_of_columns))

        self.current_combinations = set()

        return np.array(self.valid_actions)

    def update_valid_actions(self, last_action, budget, current_storage_consumption):
        assert self.indexable_column_combinations_flat[last_action] not in self.current_combinations

        # actions_index_width = len(self.indexable_column_combinations_flat[last_action])
        # if actions_index_width == 1:
        self.current_action_status[last_action] = 1

        self.current_combinations.add(self.indexable_column_combinations_flat[last_action])

        self.valid_actions[last_action] = self.FORBIDDEN_ACTION
        self._remaining_valid_actions.remove(last_action)

        is_valid_action_left = len(self._remaining_valid_actions) > 0

        return np.array(self.valid_actions), is_valid_action_left

    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        pass

    def _valid_actions_based_on_workload(self, workload):
        pass

    def _valid_actions_based_on_last_action(self, last_action):
        pass

class BlockAwaredMultiColumnIndexActionManager(ActionManager):
    def __init__(
        self, indexable_column_combinations, action_storage_consumptions, sb_version, max_index_width, reenable_indexes,partition_num=None
    ):
        ActionManager.__init__(self, sb_version, max_index_width=max_index_width, partition_num=partition_num)

        self.indexable_column_combinations = indexable_column_combinations
        self.partition_num = partition_num
        assert self.partition_num is not None, "In Partition envs, you need specify the number of block"
        # This is the same as the Expdriment's object globally_indexable_columns_flat attribute
        self.indexable_column_combinations_flat = [
            item for sublist in self.indexable_column_combinations for item in sublist
        ]   # all columns including single-columns,2 col, 3cols
        self.number_of_actions = len(self.indexable_column_combinations_flat)
        self.number_of_columns = len(self.indexable_column_combinations[0])
        self.action_storage_consumptions = action_storage_consumptions
        self.number_of_index_per_partition = math.ceil(len(self.indexable_column_combinations_flat)/self.partition_num)
        self.number_of_columns_per_partition = math.ceil(
            len(self.indexable_column_combinations[0]) / self.partition_num)
        self.block_and_index_to_column_combination = {} # map (blockid, index_id) to column_combination
        self.column_combination_to_block_and_index = {} # map column_combination to (blockid, index_id)
        self.id_to_block_and_index = {}
        self.block_and_index_to_id = {}
        indexable_column_combinations_flat_partbyblock = []
        for partition_id in range(self.partition_num):
            indexable_column_combinations_flat_partbyblock.append([])
        for indexable_column_combination_flat in self.indexable_column_combinations_flat:
            partition_id = int(indexable_column_combination_flat[0].table.name.split("prt_p")[-1])
            cc = str(indexable_column_combination_flat)
            indexable_column_combinations_flat_partbyblock[partition_id].append(indexable_column_combination_flat)
        for partition_id in range(self.partition_num):
            indexable_column_combinations_flat_partbyblock[partition_id].sort()
            for idx, cc in enumerate(indexable_column_combinations_flat_partbyblock[partition_id]):
                cc = str(cc)
                self.block_and_index_to_column_combination[partition_id,idx] = cc
                self.column_combination_to_block_and_index[cc] = [partition_id,idx]

        # sort self.indexable_column_combinations_flat:[block_0,block_1,....]
        self.indexable_column_combinations_flat = [item for sublist in indexable_column_combinations_flat_partbyblock for item in sublist]
        self.indexable_columns = list(
            map(lambda one_column_combination: one_column_combination[0], self.indexable_column_combinations[0])
        )

        self.REENABLE_INDEXES = reenable_indexes
        # cc to idx
        self.column_combination_to_idx = {}
        for idx, column_combination in enumerate(self.indexable_column_combinations_flat):
            cc = str(column_combination)
            self.column_combination_to_idx[cc] = idx
        # the sequence of self.indexable_column_combinations is different with indexable_column_combinations_flat
        self.column_to_idx = {}
        for idx, column in enumerate(self.indexable_column_combinations[0]):
            c = column[0]
            c_to_idx = self.column_combination_to_idx[str(column)]
            self.column_to_idx[c] = c_to_idx
        # column -> position in current_action_status
        self.column_to_action_status_idx = {}
        for index in self.indexable_column_combinations_flat:
            if len(index) == 1:
                self.column_to_action_status_idx[index[0]] = len(self.column_to_action_status_idx)

        self.candidate_dependent_map = {}
        for indexable_column_combination in self.indexable_column_combinations_flat:
            if len(indexable_column_combination) > max_index_width - 1:
                continue
            self.candidate_dependent_map[indexable_column_combination] = []

        for column_combination_idx, indexable_column_combination in enumerate(self.indexable_column_combinations_flat):
            if len(indexable_column_combination) < 2:
                continue
            dependent_of = indexable_column_combination[:-1]
            self.candidate_dependent_map[dependent_of].append(column_combination_idx)

        for idx, column_combination in enumerate(self.indexable_column_combinations_flat):
            cc = str(column_combination)
            [block_id, index_id] = self.column_combination_to_block_and_index[cc]
            self.id_to_block_and_index[idx] = [block_id, index_id]
            self.block_and_index_to_id[block_id, index_id] = idx

    def get_action_space(self):
        # 改动 return index_id
        # print("Action Dim: {}".format([self.partition_num, self.number_of_index_per_partition]))
        # return spaces.MultiDiscrete([self.partition_num, self.number_of_index_per_partition])
        return spaces.Discrete(self.partition_num * self.number_of_index_per_partition)


    def get_initial_valid_actions(self, workload, budget):
        # 0 for actions not taken yet, 1 for single column index present, 0.5 for two-column index present,
        # 0.33 for three-column index present, ...
        self.current_action_status = [0 for action in range(self.number_of_columns)]

        self.valid_actions = [self.FORBIDDEN_ACTION for action in range(self.number_of_actions)]
        self._remaining_valid_actions = []

        self._valid_actions_based_on_workload(workload)
        self._valid_actions_based_on_budget(budget, current_storage_consumption=0)

        self.current_combinations = set()
        self.valid_blockidx_indexidx = [self.id_to_block_and_index[valid_action] for valid_action in self.valid_actions]

        return np.array(self.valid_actions)
    
    def get_global_action(self, block_id, index_action):
        return self.block_and_index_to_id[block_id, index_action]

    def update_valid_actions(self, last_action, budget, current_storage_consumption):

        last_action = last_action

        assert self.indexable_column_combinations_flat[last_action] not in self.current_combinations

        # 如果action是[blockid, indid]，将其转为action

        actions_index_width = len(self.indexable_column_combinations_flat[last_action])
        if actions_index_width == 1:
            last_action_single_column = self.indexable_column_combinations_flat[last_action][0]
            last_action_single_column_idx = self.column_to_action_status_idx[last_action_single_column]
            self.current_action_status[last_action_single_column_idx] += 1

        else:
            combination_to_be_extended = self.indexable_column_combinations_flat[last_action][:-1]
            assert combination_to_be_extended in self.current_combinations

            status_value = 1 / actions_index_width

            last_action_back_column = self.indexable_column_combinations_flat[last_action][-1]
            last_action_back_columns_idx = self.column_to_action_status_idx[last_action_back_column]
            self.current_action_status[last_action_back_columns_idx] += status_value

            self.current_combinations.remove(combination_to_be_extended)

        self.current_combinations.add(self.indexable_column_combinations_flat[last_action])

        self.valid_actions[last_action] = self.FORBIDDEN_ACTION
        self._remaining_valid_actions.remove(last_action)
        self._remaining_valid_actions_without_blockmask = copy.deepcopy(self._remaining_valid_actions)

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
                if b_to_mb(current_storage_consumption + self.action_storage_consumptions[action_idx]) > budget:
                    self.valid_actions[action_idx] = self.FORBIDDEN_ACTION
                else:
                    new_remaining_actions.append(action_idx)

            self._remaining_valid_actions = new_remaining_actions

    def _valid_actions_based_on_last_action(self, last_action):
        last_combination = self.indexable_column_combinations_flat[last_action]
        last_combination_length = len(last_combination)

        if last_combination_length != self.MAX_INDEX_WIDTH:
            for column_combination_idx in self.candidate_dependent_map[last_combination]:
                indexable_column_combination = self.indexable_column_combinations_flat[column_combination_idx]
                possible_extended_column = indexable_column_combination[-1]

                if possible_extended_column not in self.wl_indexable_columns:
                    continue
                if indexable_column_combination in self.current_combinations:
                    continue

                self._remaining_valid_actions.append(column_combination_idx)
                self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION

        # Disable now (after the last action) invalid combinations
        for column_combination_idx in copy.copy(self._remaining_valid_actions):
            indexable_column_combination = self.indexable_column_combinations_flat[column_combination_idx]
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

        if self.REENABLE_INDEXES and last_combination_length > 1:
            last_combination_without_extension = last_combination[:-1]

            if len(last_combination_without_extension) > 1:
                # The presence of last_combination_without_extension's parent is a precondition
                last_combination_without_extension_parent = last_combination_without_extension[:-1]
                if last_combination_without_extension_parent not in self.current_combinations:
                    return

            column_combination_idx = self.column_combination_to_idx[str(last_combination_without_extension)]
            self._remaining_valid_actions.append(column_combination_idx)
            self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION

            logging.debug(f"REENABLE_INDEXES: {last_combination_without_extension} after {last_combination}")

    def _valid_actions_based_on_workload(self, workload):
        workload_columns = self.handle_workload(workload)
        indexable_columns = workload.indexable_columns(return_sorted=False)
        indexable_columns = indexable_columns & frozenset(self.indexable_columns)

        # remove not in where
        _indexable_columns = set()
        for _col in indexable_columns:
            if _col.name in workload_columns:
                _indexable_columns.add(_col)

        indexable_columns = copy.deepcopy(_indexable_columns)

        self.wl_indexable_columns = indexable_columns

        for indexable_column in indexable_columns:
            # only single column indexes
            for column_combination_idx, indexable_column_combination in enumerate(
                self.indexable_column_combinations[0]
            ):
                if indexable_column == indexable_column_combination[0]:
                    cc = str(indexable_column_combination)
                    idx = self.column_combination_to_idx[cc]
                    self.valid_actions[idx] = self.ALLOWED_ACTION
                    self._remaining_valid_actions.append(idx)

        assert np.count_nonzero(np.array(self.valid_actions) == self.ALLOWED_ACTION) == len(
            indexable_columns
        ), "Valid actions mismatch indexable columns"

    # def _valid_actions_based_on_workload(self, workload):
    #     columns = self.handle_workload(workload)
        # for col in self.indexable_column_combinations_flat


    def handle_workload(self, current_workloads):
        workloads = []
        for query in current_workloads.queries:
            workloads.append(query.text)
        indexable_columns = self.where_columns(workloads)
        return indexable_columns

    def where_columns(self, workload):
        columns = set()

        for query in workload:
            query_inf = {}
            where_clauses = re.findall(r'where(.*?);', query, flags=re.IGNORECASE)
            for where_clause in where_clauses:

                conditions = where_clause.split('and')
                column = None
                for condition in conditions:
                    lower_bound = None
                    upper_bound = None

                    if 'select' in condition:
                        continue

                    # 处理下界
                    if '>' in condition or 'between' in condition:
                        if '>=' in condition:
                            column, lower_bound = condition.split('>=')
                        elif '>' in condition:
                            column, lower_bound = condition.split('>')
                        else:
                            column, lower_bound = condition.split('between')
                        column = column.strip().split()[-1]
                        columns.add(column)


                    # 处理上界
                    elif '<' in condition or condition.strip()[0].isdigit():
                        if '<=' in condition:
                            column, upper_bound = condition.split('<=')
                        elif '<' in condition:
                            column, upper_bound = condition.split('<')
                        else:
                            upper_bound = condition.strip()
                        column = column.strip().split()[-1]
                        columns.add(column)


                    elif '=' in condition:
                        column, lower_bound = condition.split('=')
                        column = column.strip().split()[-1]
                        columns.add(column)


        return columns