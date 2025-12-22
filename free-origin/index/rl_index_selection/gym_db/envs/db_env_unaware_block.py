import collections
import copy
import logging
import random
import gym
import time
from gym_db.common import EnvironmentType
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.utils import b_to_mb
from swirl.histgrams_manager import HistogramsManager


class DBEnvMaskUnaware(gym.Env):
    def __init__(self, environment_type=EnvironmentType.TRAINING, config=None):
        super(DBEnvMaskUnaware, self).__init__()

        self.rnd = random.Random()
        self.rnd.seed(config["random_seed"])
        self.env_id = config["env_id"]
        self.environment_type = environment_type
        self.config = config

        self.number_of_resets = 0
        self.total_number_of_steps = 0

        self.connector = PostgresDatabaseConnector(config["database_name"], autocommit=True)
        self.connector.drop_indexes()

        self.cost_evaluation = CostEvaluation(self.connector)

        # if environment_type == EnvironmentType.TESTING or environment_type == EnvironmentType.VALIDATION:
        #     self.cost_evaluation = CostEvaluation(self.connector, "actual_runtimes")
        # else:
        #     self.cost_evaluation = CostEvaluation(self.connector)

        self.globally_indexable_columns = config["globally_indexable_columns"]
        # In certain cases, workloads are consumed: therefore, we need copy
        self.workloads = copy.copy(config["workloads"])
        self.current_workload_idx = 0
        self.similar_workloads = config["similar_workloads"]
        self.max_steps_per_episode = config["max_steps_per_episode"]
        self.partition_num = config["partition_num"]

        self.action_manager = config["action_manager"]
        self.action_manager.test_variable = self.env_id
        self.action_space = self.action_manager.get_action_space()


        self.observation_manager = config["observation_manager"]
        # self.observation_space = self.observation_manager.get_observation_space(self.histogram_manager)
        self.observation_space = self.observation_manager.get_observation_space()

        self.reward_calculator = config["reward_calculator"]

        self.cost_evaluation_time = 0
        self.total_step_time = 0

        self._init_modifiable_state()



        # set testing environment


        if self.environment_type != environment_type.TRAINING:
            self.episode_performances = collections.deque(maxlen=len(config["workloads"]))

        # self.histogram_manager = HistogramsManager(
        #     # self._fake_records(self.global_columns),
        #     indexed_columns=config["global_columns"],
        #     database_name=config["database_name"],
        #     query_limit=self.workload_size
        # )

    def reset(self):
        self.number_of_resets += 1
        self.total_number_of_steps += self.steps_taken
        self.total_step_time = 0
        self.cost_evaluation_time = 0

        initial_observation = self._init_modifiable_state()

        return initial_observation

    def _step_asserts(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        assert (
                self.valid_total_actions[self.global_index] == self.action_manager.ALLOWED_ACTION
        ), f"Agent has chosen invalid action: {self.global_index}"
        assert (
                Index(self.globally_indexable_columns[self.global_index]) not in self.current_indexes
        ), f"{Index(self.globally_indexable_columns[self.global_index])} already in self.current_indexes"

    def step(self, action):
        _step_start_time = time.time()
        #block_id, index_action = action

        # # fake:
        # action = 185

        #self.global_index = self.action_manager.get_global_action(block_id, index_action)
        self.global_index = action

        self._step_asserts(action)
        self.steps_taken += 1
        old_index_size = 0

        new_indexes = [Index(_) for _ in self.action_manager.get_global_action(action)]
        for _ in range(len(new_indexes)):
            new_indexes[_].estimated_size = self.action_manager.per_index_storage[self.action_manager.get_global_action(action)[_]]


            self.current_indexes.add(_)

        old_index_size = 0
        if not new_indexes[0].is_single_column():
            for _ in new_indexes:
                parent_index = Index(_.columns[:-1])

                for index in self.current_indexes:
                    if index == parent_index:
                        old_index_size += index.estimated_size

                self.current_indexes.remove(parent_index)

                assert old_index_size > 0, "Parent index size must have been found if not single column index."

        environment_state = self._update_return_env_state(
            init=False, new_index=new_indexes, old_index_size=old_index_size
        )

        current_observation = self.observation_manager.get_observation(environment_state)

        self.valid_total_actions, is_valid_action_left = self.action_manager.update_valid_actions(
            action, self.current_budget, self.current_storage_consumption
        )

        #self.valid_actions = self._split_list(self.valid_total_actions, self.partition_num)
        episode_done = self.steps_taken >= self.max_steps_per_episode or not is_valid_action_left

        reward = self.reward_calculator.calculate_reward(environment_state)

        self.total_step_time += round(float(time.time() - _step_start_time), 2)


        if episode_done and self.environment_type != EnvironmentType.TRAINING:
            self._report_episode_performance(environment_state)
            self.current_workload_idx += 1
            # print(f"Indexes: {len(self.current_indexes)}")

        return current_observation, reward, episode_done, {}

    def action_masks(self):
        #print("Action Mask Dim:{}".format([len(self.valid_total_actions), len([action == 1 for action in self.valid_total_actions])]))
        return [action == 1 for action in self.valid_total_actions]

    def _report_episode_performance(self, environment_state):

        print("Total step time: {}, pure step time:{}".format(self.total_step_time, self.total_step_time - self.cost_evaluation_time))

        achieved_cost, start_cost, final_cost = self._calculate_final_cost_proportion(self.current_workload, self.current_indexes)

        episode_performance = {
            "achieved_cost": achieved_cost,
            "memory_consumption": self.current_storage_consumption,
            "available_budget": self.current_budget,
            "evaluated_workload": self.current_workload,
            "indexes": self.current_indexes,
        }

        output = (
            f"Evaluated Workload ({self.environment_type}): {self.current_workload}\n    "
            f"Initial cost: {start_cost:,.2f}, now: {final_cost:,.2f} "
            f"({episode_performance['achieved_cost']:.2f}). Reward: {self.reward_calculator.accumulated_reward}.\n    "
            f"Size: {b_to_mb(self.current_storage_consumption):.2f} with {len(self.current_indexes)} indexes:\n    "
            f"{self.current_indexes}\n    "
        )
        logging.warning(output)

        self.episode_performances.append(episode_performance)

    def _init_modifiable_state(self):
        self.current_indexes = set()
        self.connector.drop_indexes()
        self.cost_evaluation.reset()
        self.steps_taken = 0
        self.current_storage_consumption = 0
        self.reward_calculator.reset()

        if len(self.workloads) == 0:
            self.workloads = copy.copy(self.config["workloads"])

        if self.environment_type == EnvironmentType.TRAINING:
            if self.similar_workloads:
                # 200 is an arbitrary value
                self.current_workload = self.workloads.pop(0 + self.env_id * 200)
            else:
                self.current_workload = self.rnd.choice(self.workloads)
        else:
            self.current_workload = self.workloads[self.current_workload_idx % len(self.workloads)]

        self.current_budget = self.current_workload.budget
        self.previous_cost = None

        self.valid_total_actions = self.action_manager.get_initial_valid_actions(self.current_workload,
                                                                                 self.current_budget)
        self.valid_actions = self._split_list(self.valid_total_actions, self.partition_num)
        environment_state = self._update_return_env_state(init=True)

        state_fix_for_episode = {
            "budget": self.current_budget,
            "workload": self.current_workload,
            "initial_cost": self.initial_costs,
        }
        self.observation_manager.init_episode(state_fix_for_episode)

        initial_observation = self.observation_manager.get_observation_histogramed(environment_state, self.histogram_manager)

        return initial_observation

    def _update_return_env_state(self, init, new_index=None, old_index_size=None):
        start_time = time.time()
        total_costs, plans_per_query, costs_per_query = self.cost_evaluation.calculate_cost_and_plans(
            self.current_workload, self.current_indexes, store_size=True
        )
        self.cost_evaluation_time += round(float(time.time() - start_time), 2)

        if not init:
            self.previous_cost = self.current_costs
            self.previous_storage_consumption = self.current_storage_consumption

        self.current_costs = total_costs

        if init:
            self.initial_costs = total_costs

        new_index_size = None

        add_index_size = sum([_.estimated_size for _ in new_index])

        if new_index is not None:
            self.current_storage_consumption += add_index_size
            self.current_storage_consumption -= old_index_size

            # This assumes that old_index_size is not None if new_index is not None
            assert add_index_size >= old_index_size

            new_index_size = add_index_size - old_index_size
            if new_index_size == 0:
                new_index_size = 1

            if self.current_budget:
                assert b_to_mb(self.current_storage_consumption) <= self.current_budget, (
                    "Storage consumption exceeds budget: "
                    f"{b_to_mb(self.current_storage_consumption)} "
                    f" > {self.current_budget}"
                )

        environment_state = {
            "action_status": self.action_manager.current_action_status,
            "current_storage_consumption": self.current_storage_consumption,
            "current_cost": self.current_costs,
            "previous_cost": self.previous_cost,
            "initial_cost": self.initial_costs,
            "new_index_size": new_index_size,
            "plans_per_query": plans_per_query,
            "costs_per_query": costs_per_query,
        }

        return environment_state

    def get_cost_eval_cache_info(self):
        return self.cost_evaluation.cost_requests, self.cost_evaluation.cache_hits, self.cost_evaluation.costing_time

    def get_cost_eval_cache(self):
        return self.cost_evaluation.cache

    # BEGIN OF NOT IMPLEMENTED ##########
    def render(self, mode="human"):
        print("render() was called")
        pass

    def close(self):
        print("close() was called")

    def _split_list(self, lst, n):
        """
        将列表 lst 分割成 n 个等份的列表，并返回一个包含这些等份列表的列表
        """
        # 计算每个等份列表的长度
        length = len(lst) // n
        # 切分列表，得到 n 个等份列表
        equal_parts = [lst[i:i + length] for i in range(0, len(lst), length)]
        # 如果最后一个等份列表的长度不足 length，则将其添加到前一个等份列表中
        if len(equal_parts[-1]) < length:
            equal_parts[-2] += equal_parts[-1]
            equal_parts.pop()
        # 返回等份列表的列表
        return equal_parts

    def _calculate_final_cost_proportion(self, workload, indexes):

        print("Start testing index ...")

        #self.cost_evaluation.cost_estimation = "actual_runtimes"

        start_cost = self.cost_evaluation.calculate_cost(workload, [])
        final_cost = self.cost_evaluation.calculate_cost(workload, indexes)

        index_combination_size = 0
        for index in indexes:
            index_combination_size += index.estimated_size


        self.cost_evaluation.cost_estimation = "whatif"

        print("Finish testing index ...")

        return round(final_cost / start_cost * 100, 2), start_cost, final_cost

    # END OF NOT IMPLEMENTED ##########
