import copy

from environment import *

# a fake environment with 2 queries and 3 blocks
# query: 2 column-> 4 index candidates
# block: 2 column (a, b)
# block1: [0-5, 0-100], block2: [5-10, 100-200], block3: [10-15, 200-300]
# query1: select * from where 5>a>0
# query2: select * from where b=50
# storage: b构建索引开销为2， a为1
# workload: query1 无索引：5 / 有索引 1       ｜｜  query2 无索引:100 / 有索引 2
# index candidate: a/b
class fakeEnv(Environment):
    def __init__(self, block_num, index_num):
        Environment.__init__(self, block_num, index_num)

    def observe(self):
        # vocabulary = {between:1, =:2, >:3, <:4}
        # [column, operation, start, end]
        self.workload_inputs = [[1, 1, 0, 5], [2, 2, 50, 0]]
        # selectivity, bitmap
        self.block_based_workloads = [[[1, 1, 1],
                                      [0, 0, 0],
                                      [0, 0, 0]],
                                      [[0.1, 0, 1],
                                      [0, 0, 0],
                                      [0, 0, 0]]]
        #[min, max, average]
        self.block_inputs = [[0, 5, 2.5, 0, 100, 50],
                             [5, 10, 7.5, 100, 200, 150],
                             [10, 15, 12.5, 200, 300, 250]]
        #
        self.candidate_inputs = copy.deepcopy(self.index_config)
        #
        self.space_inputs = [3, self.last_storage_cost]
        #no mask
        self.block_valid_mask = [0, 0, 0]
        self.index_valid_mask = [[0, 0],
                                 [0, 0],
                                 [0, 0]]


        return self.workload_inputs, self.block_based_workloads, \
               self.block_inputs, self.candidate_inputs, \
               self.space_inputs, \
               self.block_valid_mask, self.index_valid_mask

    # new_index [index_id, block_id]
    def storage_cost(self):
        cost = 0
        for i in range(3):
            if self.index_config[i][0] == 1:
                cost += 1
            if self.index_config[i][1] == 1:
                cost += 2
        return cost

    #
    def workload_cost(self):
        cost = 0
        # query1
        if self.index_config[0][0] == 1:
            cost += 1
        else:
            cost += 5
        # query2
        if self.index_config[0][1] == 1:
            cost += 2
        else:
            cost += 100

        return cost