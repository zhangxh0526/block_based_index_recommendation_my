import logging
import random
import math
import time
import decimal

from ..utils import b_to_mb, mb_to_b

from ..index import Index
from ..selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
from ..utils import b_to_mb, mb_to_b

from ..query_generator import QueryGenerator
from ..workload import Workload

DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "min_cost_improvement": 1.003,
}

# This algorithm is a reimplementation of the Salalom
class SlalomAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS
        )
        self.workload = None
        self.statistic = {}

    def reset(self, parameters):
        self.did_run = False
        self.parameters = parameters
        # Store default values for missing parameters
        self.database_connector.drop_indexes()
        self.cost_evaluation.what_if.drop_all_simulated_indexes()
        if "cost_estimation" in self.parameters:
            estimation = self.parameters["cost_estimation"]
            self.cost_evaluation.cost_estimation = estimation
        self.budget = (self.parameters["budget_MB"])
        self.max_index_width = self.parameters["max_index_width"]
        self.workload = None
        self.cost_evaluation_time = 0

    def _calculate_best_indexes(self, workload, total_wordload=None):
        self.start_time = time.time()
        self.cost_evaluation_time = 0
        logging.info("Calculating best indexes Slalom")
        self.workload = workload
        self.total_workload = total_wordload
        columns = self.workload.indexable_columns()
        single_attribute_index_candidates = self.workload.potential_indexes()
        # current indexs
        index_combination = []
        index_combination_size = 0
        self.LRU = LRUCache(self.budget)
        # calculate C_i,C_f,C_b
        for candidate in single_attribute_index_candidates:
            # Slalom only refer to single-col index, so we can extract column as follow
            column = candidate.columns[0]
            self.statistic[column] = {}
            self.statistic[column]['index'] = candidate
            # self.cost_evaluation.calculate_cost_slalom(self.workload, index_combination, store_size=True)
            start_time = time.time()
            c_f = self.cost_evaluation.calculate_cost_slalom(
                self.workload, index_combination, store_size=False
                )
            c_i = self.cost_evaluation.calculate_cost_slalom(
                self.workload, index_combination+[candidate], store_size=False
            )
            c_b = self.cost_evaluation.get_index_build_cost(candidate)*1000
            self.cost_evaluation_time += round(float(time.time() - start_time), 2)
            if abs(c_f-c_i) < 0.01:
                if c_f < c_i:
                    c_i += 0.01
                else:
                    c_f += 0.01
            self.statistic[column]['c_f'] = c_f
            self.statistic[column]['c_i'] = c_i
            self.statistic[column]['c_b'] = c_b
            self.statistic[column]['threshold'] = c_b / (c_f - c_i)
            self.statistic[column]['access_freq'] = 0
            self.statistic[column]['build'] = False
        print('----')
        self.current_size = 0
        # self.LRU = LRUCache(self.budget)
        index_combination = self._dynamic_build_index()

        total_execution_time = round(float(time.time() - self.start_time), 2)
        print("Slalom takes {} time to find indexes, and pure time is {}\n".format(total_execution_time,
                                                                                   total_execution_time - self.cost_evaluation_time))
        return index_combination

    def _dynamic_build_index(self):
        index_comination = []
        p_start = 0.5
        seed = time.time()
        random.seed(seed)
        random.shuffle(self.workload.queries)
        for query in self.workload.queries:
            for column in query.columns:
                self.statistic[column]['access_freq'] += 1
                self.statistic[column]['build_p'] = p = self._calculate_build_p(column, p_start)
                if p < 0:
                    p = random.random() * pow(0.5, self.statistic[column]['access_freq'])
                if not self.statistic[column]['build']:
                    if self._random_unit(p):
                        # print('在第', candidate_row_group.access_freq[column_id], '以概率 ', p, 'build tree')
                        # calculate the index size:
                        del_cols = self.LRU.put(column, self.statistic[column]['index'].estimated_size)
                        index_comination.append(self.statistic[column]['index'])
                        self.statistic[column]['build'] = True
                        if len(del_cols) >0 :
                            for col in del_cols:
                                self.statistic[col]['build'] = False
                                index_comination.remove(self.statistic[col]['index'])
                else:
                    # has established
                    self.LRU.get(column)
        return index_comination



    
    def _calculate_build_p(self, column, p_start):
        a = decimal.Decimal(random.random())
        c_f = decimal.Decimal(self.statistic[column]['c_f'])
        c_i = decimal.Decimal(self.statistic[column]['c_i'])
        c_b = decimal.Decimal(self.statistic[column]['c_b'])
        if self.statistic[column]['access_freq'] ==1 :
            p = p_start
        else:
            if self.statistic[column]['access_freq'] <= math.ceil(self.statistic[column]['threshold']):
                p = pow((c_b + c_f - c_i) / c_b, self.statistic[column]['access_freq'] - 2) * (
                            (c_f - c_i) / c_b) * (
                                decimal.Decimal(p_start) + a * c_f / (c_f - c_i))
            else:
                 p = pow((c_b + c_f - c_i) / c_b, self.statistic[column]['access_freq'] - 2) * (
                            -(c_f - c_i) / c_b) * (
                                decimal.Decimal(1 - p_start) + a * c_i / (c_f - c_i))
        p = min(1, p)
        return p
    
    def _random_unit(self,p):
        assert 0 <= p <= 1, "概率P的值应该处在[0,1]之间！"
        if p == 0:  # 概率为0，直接返回False
            return False
        if p == 1:  # 概率为1，直接返回True
            return True
        p_digits = len(str(p).split(".")[1])
        interval_begin = 1
        interval__end = pow(10, p_digits)
        R = random.randint(interval_begin, interval__end)
        if float(R) / interval__end < p:
            return True
        else:
            return False
        

# 输入
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
        
    def get(self, key):
        if key in self.cache:
            # 更新最近访问顺序
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        else:
            return -1
        
    def put(self, key, value):
        if key in self.cache:
            # 更新值和最近访问顺序
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
            return []
        else:
            # 插入新的值和最近访问顺序
            self.cache[key] = value
            self.order.append(key)
            # 移除最早的访问项
            del_keys = []
            if sum([self.cache[key] for key in self.order]) > mb_to_b(self.capacity):
                pop_size = 0
                while sum([self.cache[key] for key in self.order]) > mb_to_b(self.capacity):
                    del_key = self.order.pop(0)
                    del self.cache[del_key]
                    del_keys.append(del_key)
            return del_keys

