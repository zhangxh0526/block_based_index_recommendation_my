import logging
import time
from copy import deepcopy

from ..index import Index
from ..selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
from ..utils import mb_to_b

# 全局贪婪（不分区、不打包）默认参数
DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "min_cost_improvement": 1.003,
    "max_runtime_minutes": 60 * 24,
}


class ExtendAlgorithmGlobal(SelectionAlgorithm):
    """全局视野、单一预算的贪婪基线。"""

    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS
        )
        self.workload = None
        self.max_runtime_minutes = 60 * 24
        self.cost_evaluation_time = 0
        print(">>> [MODE] Global Greedy Algorithm (True Global Optimal Baseline) Loaded! <<<")

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
            logging.debug(f"Stopping due to time constraints. Time: {consumed_time/60:.2f} min.")
            return True
        return False

    def _calculate_best_indexes(self, workload):
        """核心：全局贪婪，不分区、不打包。"""
        self.cost_evaluation.what_if.drop_all_simulated_indexes()
        self.cost_evaluation_time = 0
        logging.info("Calculating best indexes (Global Greedy)...")
        self.workload = workload

        # 单列表候选（跨所有分区），不打包
        single_attribute_index_candidates = workload.potential_indexes()
        extension_attribute_candidates = deepcopy(single_attribute_index_candidates)

        self.start_time = time.time()

        index_combination = []
        index_combination_size = 0
        best = {"combination": [], "benefit_to_size_ratio": 0, "cost": None}

        # 初始全局成本（无索引）
        start_time = time.time()
        current_cost = self.cost_evaluation.calculate_cost(
            self.workload, index_combination, store_size=False
        )
        self.cost_evaluation_time += round(float(time.time() - start_time), 2)
        self.initial_cost = current_cost

        while True:
            # 能放进剩余全局预算的候选
            valid_candidates = self._get_candidates_within_budget(
                index_combination_size, single_attribute_index_candidates
            )

            # 评估新增单列索引
            for candidate in valid_candidates:
                if candidate not in index_combination:
                    self._evaluate_combination(
                        index_combination + [candidate],
                        best,
                        current_cost,
                    )
                if self._should_stop_due_to_time():
                    return self._finalize_results(index_combination)

            # 评估扩展索引（同表追加列）
            for attribute in extension_attribute_candidates:
                self._attach_to_indexes(
                    index_combination,
                    attribute,
                    best,
                    current_cost,
                )
                if self._should_stop_due_to_time():
                    return self._finalize_results(index_combination)

            if best["benefit_to_size_ratio"] <= 0:
                break

            index_combination = best["combination"]
            index_combination_size = self._total_index_size(index_combination)

            logging.info(
                f"Add Index. Global Cost: {best['cost']:.2f}. "
                f"Storage: {index_combination_size/1024/1024:.2f} / {self.budget:.2f} MB"
            )

            best["benefit_to_size_ratio"] = 0
            current_cost = best["cost"]

        return self._finalize_results(index_combination)

    def _finalize_results(self, index_combination):
        total_execution_time = round(float(time.time() - self.start_time), 2)

        print("\n" + "=" * 50)
        print(f">>> GLOBAL GREEDY RESULTS (Time: {total_execution_time}s) <<<")
        print("=" * 50)

        partition_usage = {}
        total_size = 0

        for idx in index_combination:
            t_name = idx.table().name
            if idx.estimated_size is None:
                self.cost_evaluation.estimate_size(idx)
            size = idx.estimated_size
            total_size += size

            print(f"SELECTED: {idx} | Size: {size/1024/1024:.2f} MB")

            partition_usage[t_name] = partition_usage.get(t_name, 0) + size

        print("-" * 50)
        print(f"Total Used Budget: {total_size/1024/1024:.2f} MB / {self.budget:.2f} MB")
        print("Budget Distribution by Partition:")
        for t_name, usage in sorted(partition_usage.items()):
            print(f"  {t_name}: {usage/1024/1024:.2f} MB ({(usage/mb_to_b(self.budget))*100:.1f}%)")
        print("=" * 50 + "\n")

        return index_combination

    def _attach_to_indexes(self, index_combination, attribute, best, current_cost):
        # attribute 是单列索引
        assert attribute.is_single_column() is True, "Attach called with multi column index"

        for position, index in enumerate(index_combination):
            if len(index.columns) >= self.max_index_width:
                continue
            if index.appendable_by(attribute):  # 同表
                new_index = Index(index.columns + attribute.columns)
                if new_index in index_combination:
                    continue

                new_combination = index_combination.copy()
                del new_combination[position]
                new_combination.append(new_index)

                self._evaluate_combination(
                    new_combination,
                    best,
                    current_cost,
                )

    def _get_candidates_within_budget(self, index_combination_size, candidates):
        new_candidates = []
        for candidate in candidates:
            if candidate.estimated_size is None:
                self.cost_evaluation.estimate_size(candidate)
            if candidate.estimated_size is None:
                continue
            if candidate.estimated_size + index_combination_size <= mb_to_b(self.budget):
                new_candidates.append(candidate)
        return new_candidates

    def _evaluate_combination(self, index_combination, best, current_cost):
        start_time = time.time()
        cost = self.cost_evaluation.calculate_cost(
            self.workload, index_combination, store_size=False
        )
        self.cost_evaluation_time += round(float(time.time() - start_time), 2)

        if (cost * self.min_cost_improvement) >= current_cost:
            return

        benefit = current_cost - cost

        # 计算总大小（需确保已估算）
        start_time = time.time()
        total_size = self._total_index_size(index_combination)
        self.cost_evaluation_time += round(float(time.time() - start_time), 2)

        # 使用 Benefit/Size 作为性价比，避免只看 benefit
        ratio = benefit / (total_size if total_size != 0 else 1)

        if ratio > best["benefit_to_size_ratio"] and total_size <= mb_to_b(self.budget):
            best["combination"] = index_combination
            best["benefit_to_size_ratio"] = ratio
            best["cost"] = cost

    def _total_index_size(self, indexes):
        total_size = 0
        for idx in indexes:
            if idx.estimated_size is None:
                self.cost_evaluation.estimate_size(idx)
            if idx.estimated_size is None:
                continue
            total_size += idx.estimated_size
        return total_size
