import logging
import time
from copy import deepcopy
import functools

from ..index import Index
from ..selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
from ..utils import b_to_mb, mb_to_b
from ..workload import Workload

# 沿用默认参数
DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "min_cost_improvement": 1.003,
    "max_runtime_minutes": 60 * 24,
}


def indexsort(x, y):
    # 简单的确定性排序，方便复现
    if x.columns[0].table.name > y.columns[0].table.name:
        return -1
    return 1


class ExtendAlgorithmGlobalIndependent(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS
        )
        self.workload = None
        self.max_runtime_minutes = 60 * 24
        self.cost_evaluation_time = 0
        print(">>> [MODE] Global Independent Extend (The Ground Truth Finder) Loaded! <<<")

    def reset(self, parameters):
        self.did_run = False
        self.parameters = parameters
        self.database_connector.drop_indexes()
        self.cost_evaluation.what_if.drop_all_simulated_indexes()
        if "cost_estimation" in self.parameters:
            self.cost_evaluation.cost_estimation = self.parameters["cost_estimation"]
        self.budget = self.parameters["budget_MB"]
        self.max_index_width = self.parameters["max_index_width"]
        self.workload = None
        self.min_cost_improvement = self.parameters["min_cost_improvement"]
        self.cost_evaluation_time = 0

    def _should_stop_due_to_time(self):
        current_time = time.time()
        consumed_time = current_time - self.start_time
        if consumed_time > self.max_runtime_minutes * 60:
            return True
        return False

    def _group_candidates_by_table_and_column(self, candidates):
        """
        【关键区别】
        这里不再把不同分区的同名列合并。
        而是严格按照 (Table, Column) 进行分组。
        P0.colA 和 P1.colA 会变成两个独立的 Candidate。
        """
        grouped_candidates = []
        mapping = {}

        for index in candidates:
            key = f"{index.table().name}.{index.columns[0].name}"

            if key not in mapping:
                mapping[key] = []
            mapping[key].append(index)

        for key in mapping:
            # 用字符串表示排序，保证确定性
            mapping[key].sort(key=lambda x: str(x))
            grouped_candidates.append(mapping[key])

        return grouped_candidates

    def _calculate_best_indexes(self, workload):
        self.cost_evaluation.what_if.drop_all_simulated_indexes()
        self.cost_evaluation_time = 0
        self.workload = workload

        logging.info("Starting Global Independent Search...")

        all_candidates = workload.potential_indexes()

        candidates_pool = self._group_candidates_by_table_and_column(all_candidates)
        extension_pool = deepcopy(candidates_pool)

        index_combination = []
        index_combination_size = 0
        best = {"combination": [], "benefit_to_size_ratio": 0, "cost": None}

        self.start_time = time.time()

        current_cost = self.cost_evaluation.calculate_cost(
            workload, index_combination, store_size=False
        )
        self.initial_cost = current_cost

        while True:
            valid_candidates = self._get_candidates_within_budget(
                index_combination_size, candidates_pool, self.budget
            )

            for candidate_group in valid_candidates:
                for candidate in candidate_group:
                    if candidate not in index_combination:
                        self._evaluate_combination(
                            index_combination + [candidate],
                            best,
                            current_cost,
                            self.budget,
                        )
                    if self._should_stop_due_to_time():
                        return self._finalize_result(index_combination)

            for attribute_group in extension_pool:
                for attribute in attribute_group:
                    self._attach_to_indexes(
                        index_combination,
                        attribute,
                        best,
                        current_cost,
                        self.budget,
                    )
                    if self._should_stop_due_to_time():
                        return self._finalize_result(index_combination)

            if best["benefit_to_size_ratio"] <= 0:
                logging.info("No further improvement found.")
                break

            index_combination = best["combination"]
            index_combination_size = self._total_index_size(index_combination)
            current_cost = best["cost"]
            best["benefit_to_size_ratio"] = 0

            logging.info(
                f"Selected new configuration. Cost: {current_cost:.2f}. "
                f"Storage: {index_combination_size:.2f}/{self.budget:.2f} MB"
            )

        return self._finalize_result(index_combination)

    def _finalize_result(self, index_combination):
        """
        最后一步：把选出来的索引详情打印出来，供你分析。
        """
        print("\n" + "=" * 50)
        print(">>> GLOBAL OPTIMAL SOLUTION (INDEPENDENT) <<<")
        print("=" * 50)

        sorted_indexes = sorted(index_combination, key=lambda x: x.table().name)

        total_size = 0
        partition_stats = {}

        for idx in sorted_indexes:
            t_name = idx.table().name
            cols = ",".join([c.name for c in idx.columns])
            size = idx.estimated_size
            total_size += size

            print(f"[Table: {t_name}] Index: ({cols}) | Size: {size:.2f} MB")

            if t_name not in partition_stats:
                partition_stats[t_name] = 0
            partition_stats[t_name] += size

        print("-" * 50)
        print(f"Total Indexes: {len(index_combination)}")
        print(f"Total Size: {total_size:.2f} MB / {self.budget:.2f} MB")
        print("-" * 50)
        print(">>> Budget Distribution Analysis <<<")
        for t_name, size in sorted(partition_stats.items()):
            ratio = (size / self.budget) * 100
            print(f"{t_name}: {size:.2f} MB ({ratio:.1f}%)")
        print("=" * 50 + "\n")

        return index_combination

    # === 以下是辅助函数，逻辑基本复用，但去掉了分区的 budget 限制 ===

    def _attach_to_indexes(self, index_combination, attribute, best, current_cost, budget_limit):
        for position, index in enumerate(index_combination):
            if len(index.columns) >= self.max_index_width:
                continue
            # 关键：appendable_by 会检查表名是否一致，所以不会出现把 p0 的列加到 p1 的索引上
            if index.appendable_by(attribute):
                new_index = Index(index.columns + attribute.columns)
                if new_index in index_combination:
                    continue
                new_combination = index_combination.copy()
                del new_combination[position]
                new_combination.append(new_index)

                self._evaluate_combination(
                    new_combination, best, current_cost, budget_limit, self._total_index_size(index_combination)
                )

    def _get_candidates_within_budget(self, current_size, candidates_pool, budget_limit):
        valid = []
        for group in candidates_pool:
            for cand in group:
                # 确保 size 有值；缺失时调用成本评估去估算，避免将 size=0 误判为免费
                if cand.estimated_size is None:
                    self.cost_evaluation.estimate_size(cand)
                if cand.estimated_size is None:
                    continue  # 跳过无法估算大小的候选

                if current_size + cand.estimated_size <= mb_to_b(budget_limit):
                    valid.append([cand])  # 保持结构一致性
        return valid

    def _evaluate_combination(self, index_combination, best, current_cost, budget_limit, old_index_size=0):
        # 这里的 index_combination 已经是扁平的 list of Index 对象
        start_time = time.time()
        cost = self.cost_evaluation.calculate_cost(self.workload, index_combination, store_size=False)
        self.cost_evaluation_time += round(float(time.time() - start_time), 2)

        if (cost * self.min_cost_improvement) >= current_cost:
            return

        benefit = current_cost - cost
        new_size = self._total_index_size(index_combination)
        size_diff = new_size - old_index_size

        if size_diff == 0 and new_size != 0:
            ratio = best["benefit_to_size_ratio"]
        else:
            ratio = benefit / size_diff

        if ratio >= best["benefit_to_size_ratio"] and new_size <= mb_to_b(budget_limit):
            best["combination"] = index_combination
            best["benefit_to_size_ratio"] = ratio
            best["cost"] = cost

    def _total_index_size(self, indexes):
        total = 0
        for idx in indexes:
            if idx.estimated_size is None:
                self.cost_evaluation.estimate_size(idx)
            total += idx.estimated_size
        return total
