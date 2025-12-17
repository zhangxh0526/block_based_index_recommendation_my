import numpy as np
import copy
from collections import OrderedDict
from complex_version.param import *


class Environment(object):
    def __init__(self, block_num, index_num):

        # isolated random number generator
        self.np_random = np.random.RandomState()

        self.init_cost = self.workload_cost(None)

        self.last_workload_cost = self.init_cost
        self.last_storage_cost = 0

        self.step_num = 0

        # block_num, index_num
        self.index_config = [[0] * len(index_num) for _ in range(len(block_num))]

    def observe(self):



        return self.workload_inputs, self.block_based_workloads, \
               self.block_inputs, self.candidate_inputs, \
               self.space_inputs, \
               self.block_valid_mask, self.index_valid_mask

    #[index_id, block_id]
    def step(self, new_index):

        self.step_num += 1

        # reward
        reward = self.get_reward(new_index)

        # is done
        done = False
        # step limit
        if self.step_num >= args.step_threshold:
            done = True
        # no more valid action
        if new_index is None:
            done = True

        return self.observe(), reward, done

    def seed(self, seed):
        self.np_random.seed(seed)


    def get_reward(self, new_index):
        # add index
        assert self.index_config[new_index[0], new_index[1]] == 0
        self.index_config[new_index[0], new_index[1]] = 1
        # reward calculate
        reward = (self.last_workload_cost - self.workload_cost()) / self.init_cost
        reward /= (self.storage_cost() - self.last_storage_cost)
        # record last workload&storage cost
        self.last_workload_cost = self.workload_cost()
        self.last_storage_cost = self.storage_cost()
        return reward

    def storage_cost(self):
        return 0

    def workload_cost(self):
        return 0
