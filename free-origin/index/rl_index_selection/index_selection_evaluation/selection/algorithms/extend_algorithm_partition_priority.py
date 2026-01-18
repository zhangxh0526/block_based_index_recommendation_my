import logging
import time
from copy import deepcopy
import functools

from ..index import Index
from ..selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
from ..utils import b_to_mb, mb_to_b
from ..workload import Workload

# 贪心 Extend 的分区版 + OCW/优先级分配启发式
DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "min_cost_improvement": 1.003,
    "max_runtime_minutes": 60 * 24,
    # 预算分配参数
    "min_useful_budget_MB": 150.0,  # 单分区“起步价”
    "max_partition_share": 0.45,    # 单分区最大占比
}


def indexsort(x, y):
    if x.columns[0].table.name > y.columns[0].table.name:
        return -1
    return 1


class ExtendAlgorithmPartitionPriority(SelectionAlgorithm):
    """Extend 的分区版，使用 OCW + 优先级填充分配预算。"""

    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(self, database_connector, parameters, DEFAULT_PARAMETERS)
        self.workload = None
        self.max_runtime_minutes = 60 * 24
        self.cost_evaluation_time = 0

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
        self.partition_num = self.parameters["partition_num"]
        self.min_useful_budget = float(self.parameters.get("min_useful_budget_MB", 150.0))
        self.max_partition_share = float(self.parameters.get("max_partition_share", 0.45))
        self.active_partition_budget_mb = self.budget
        self.partition_scores = None

    def _should_stop_due_to_time(self):
        current_time = time.time()
        consumed_time = current_time - self.start_time
        if consumed_time > self.max_runtime_minutes * 60:
            logging.debug(
                "Stopping because of timing constraints. " f"Time: {consumed_time / 60:.2f} minutes."
            )
            return True
        return False

    def _group_partition_index(self, single_attribute_index_candidates):
        index_candidates_with_partition = [[] for _ in range(self.partition_num)]
        for index in single_attribute_index_candidates:
            index_candidates_with_partition[int(index.table().name.split("_1_prt_p")[-1])].append(index)
        for _ in index_candidates_with_partition:
            self._total_index_size(_)
            _.sort(key=functools.cmp_to_key(indexsort))
        return index_candidates_with_partition

    @staticmethod
    def _extract_plan_rows(plan):
        if not isinstance(plan, dict):
            return 0
        return plan.get("Plan Rows", 0)

    @staticmethod
    def _extract_plan_cost(plan):
        if not isinstance(plan, dict):
            return 0
        return plan.get("Total Cost", 0)

    def _partition_tables(self, grouped_candidates):
        partition_tables = [set() for _ in range(self.partition_num)]
        for partition_id, candidate_list in enumerate(grouped_candidates):
            for index in candidate_list:
                partition_tables[partition_id].add(index.table().name)
        return partition_tables

    def _partition_total_rows(self, partition_tables):
        totals = []
        for tables in partition_tables:
            total_rows = 0
            for table_name in tables:
                try:
                    rows = self.database_connector.exec_fetch(
                        f"SELECT COALESCE(reltuples, 0)::bigint FROM pg_class WHERE relname = '{table_name}'",
                        one=True,
                    )
                    rows = rows[0] if rows else 0
                except Exception as exc:
                    logging.warning("pg_class reltuples failed for %s: %s", table_name, exc)
                    rows = 0
                total_rows += max(int(rows), 0)
            totals.append(max(total_rows, 1))
        return totals

    def _compute_partition_scores(self, workload, partition_tables, total_rows_per_partition):
        scores = [0.0 for _ in range(self.partition_num)]
        for partition_id in range(self.partition_num):
            queries_partition = workload.queries[partition_id::self.partition_num]
            part_total_rows = total_rows_per_partition[partition_id]
            for query in queries_partition:
                try:
                    plan = self.database_connector.get_plan(query)
                    plan_rows = self._extract_plan_rows(plan)
                    plan_cost = self._extract_plan_cost(plan)
                except Exception as exc:
                    logging.warning("Plan extraction failed for Q%s: %s", query.nr, exc)
                    plan_rows = 0
                    plan_cost = 0

                selectivity_factor = 1.0 - (plan_rows / max(part_total_rows, 1))
                score = plan_cost * selectivity_factor * getattr(query, "frequency", 1)
                scores[partition_id] += max(score, 0)
        return scores

    def _priority_budget_distribution(self, scores):
        total_budget = float(self.budget)
        num_parts = self.partition_num
        min_useful = self.min_useful_budget
        max_share = self.max_partition_share * total_budget

        # 无分数时均分
        score_sum = sum(scores)
        if score_sum == 0:
            return [round(total_budget / num_parts, 2) for _ in range(num_parts)]

        ranked = sorted([(scores[i], i) for i in range(num_parts)], key=lambda x: x[0], reverse=True)
        budgets = [0.0 for _ in range(num_parts)]
        remaining = total_budget

        # 小预算直接全给最痛分区
        if total_budget <= 2 * min_useful:
            budgets[ranked[0][1]] = round(total_budget, 2)
            return budgets

        # 1) 保底：按优先级给有分数的分区 min_useful
        for score, pid in ranked:
            if score <= 0 or remaining <= 0:
                continue
            alloc = min(min_useful, remaining)
            budgets[pid] += alloc
            remaining -= alloc

        if remaining <= 0:
            return [round(b, 2) for b in budgets]

        # 2) 增量：剩余预算按分数比例分配
        for i in range(num_parts):
            extra = remaining * (scores[i] / score_sum)
            budgets[i] += extra

        # 3) Max share 约束，溢出再分配
        overflow = 0.0
        for i in range(num_parts):
            if budgets[i] > max_share:
                overflow += budgets[i] - max_share
                budgets[i] = max_share

        if overflow > 0:
            # 分给未达上限的分区，按分数再分
            available = [i for i in range(num_parts) if budgets[i] < max_share and scores[i] > 0]
            weight_sum = sum(scores[i] for i in available) or len(available)
            for i in available:
                share = overflow * (scores[i] / weight_sum)
                budgets[i] += share

        return [round(b, 2) for b in budgets]

    def _compute_partition_budgets(self, single_attribute_index_candidates, workload):
        partition_tables = self._partition_tables(single_attribute_index_candidates)
        total_rows_per_partition = self._partition_total_rows(partition_tables)
        scores = self._compute_partition_scores(workload, partition_tables, total_rows_per_partition)
        budgets_mb = self._priority_budget_distribution(scores)
        self.partition_scores = scores

        logging.info("Partition total rows: %s", total_rows_per_partition)
        logging.info("Partition scores (OCW): %s", scores)
        logging.info("Partition budgets (MB): %s", budgets_mb)

        total_budget = sum(budgets_mb)
        self.partition_budget = [b / total_budget if total_budget else 0 for b in budgets_mb]
        return budgets_mb

    def _calculate_best_indexes(self, workload):
        self.cost_evaluation.what_if.drop_all_simulated_indexes()
        self.cost_evaluation_time = 0
        logging.info("Calculating best indexes Extend_partition_priority")
        self.workload = workload

        single_attribute_index_candidates = workload.potential_indexes()
        single_attribute_index_candidates = self._group_partition_index(single_attribute_index_candidates)
        extension_attribute_candidates = deepcopy(single_attribute_index_candidates)

        workload_partition_budget = self._compute_partition_budgets(
            single_attribute_index_candidates,
            workload,
        )

        self.start_time = time.time()
        self.workload_partition_list = []
        self.index_combination_list = []
        final_indexes = []

        for i in range(self.partition_num):
            queries_partition = workload.queries[i::self.partition_num]
            workload_partition = Workload(queries_partition)
            workload_partition.budget = workload_partition_budget[i]
            self.workload_partition_list.append(workload_partition)

            self.active_partition_budget_mb = workload_partition.budget

            if workload_partition.budget <= 0:
                logging.info("Partition %s skipped (0 budget)", i)
                continue

            index_combination = []
            index_combination_size = 0
            best = {"combination": [], "benefit_to_size_ratio": 0, "cost": None}

            start_time = time.time()
            current_cost = self.cost_evaluation.calculate_cost(
                workload_partition, index_combination, store_size=False
            )
            self.cost_evaluation_time += round(float(time.time() - start_time), 2)
            self.initial_cost = current_cost

            while True:
                single_attribute_index_candidates[i] = self._get_candidates_within_budget(
                    index_combination_size, single_attribute_index_candidates[i]
                )

                for candidate in single_attribute_index_candidates[i]:
                    if candidate not in index_combination:
                        self._evaluate_combination(
                            index_combination + [candidate],
                            best,
                            current_cost,
                            workload=workload_partition,
                        )
                    if self._should_stop_due_to_time():
                        return final_indexes

                for attribute in extension_attribute_candidates[i]:
                    self._attach_to_indexes(
                        index_combination,
                        attribute,
                        best,
                        current_cost,
                        workload=workload_partition,
                    )
                    if self._should_stop_due_to_time():
                        return final_indexes

                if best["benefit_to_size_ratio"] <= 0:
                    break

                index_combination = best["combination"]
                index_combination_size = self._total_index_size(index_combination)
                logging.info(
                    "Add index P%s. Cost savings: %s, initial %s. Storage: %.2f MB",
                    i,
                    f"{(1 - best['cost'] / current_cost) * 100:.3f}",
                    f"{(1 - best['cost'] / self.initial_cost) * 100:.3f}",
                    b_to_mb(index_combination_size),
                )
                best["benefit_to_size_ratio"] = 0
                current_cost = best["cost"]

            self.index_combination_list.append(index_combination)
            final_indexes.extend(index_combination)

        total_execution_time = round(float(time.time() - self.start_time), 2)
        print(
            "Extend_partition_priority takes {} time to find indexes, pure time {}\n".format(
                total_execution_time, total_execution_time - self.cost_evaluation_time
            )
        )
        return final_indexes

    def _attach_to_indexes(self, index_combination, attribute, best, current_cost, workload=None):
        assert attribute.is_single_column() is True, "Attach to indexes called with multi column index"
        for position, index in enumerate(index_combination):
            if len(index.columns) >= self.max_index_width:
                continue
            if index.appendable_by(attribute):
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
                    self._total_index_size(index_combination),
                    workload=workload,
                )

    def _get_candidates_within_budget(self, index_combination_size, candidates):
        new_candidates = []
        for candidate in candidates:
            if candidate.estimated_size is None:
                candidate.estimated_size = 0
            if candidate.estimated_size + index_combination_size <= mb_to_b(self.active_partition_budget_mb):
                new_candidates.append(candidate)
        return new_candidates

    def _evaluate_combination(self, index_combination, best, current_cost, index_combination_size=0, workload=None):
        start_time = time.time()
        cost = self.cost_evaluation.calculate_cost(
            workload if workload else self.workload,
            index_combination,
            store_size=False,
        )
        self.cost_evaluation_time += round(float(time.time() - start_time), 2)

        if (cost * self.min_cost_improvement) >= current_cost:
            return

        start_time = time.time()
        total_size = self._total_index_size(index_combination)
        self.cost_evaluation_time += round(float(time.time() - start_time), 2)
        benefit = current_cost - cost
        ratio = benefit / (total_size if total_size != 0 else 1)

        if ratio > best["benefit_to_size_ratio"] and total_size <= mb_to_b(self.active_partition_budget_mb):
            best["combination"] = index_combination
            best["benefit_to_size_ratio"] = ratio
            best["cost"] = cost

    def _total_index_size(self, indexes):
        total_size = 0
        for index in indexes:
            if index.estimated_size is None:
                self.cost_evaluation.estimate_size(index)
            if index.estimated_size is None:
                continue
            total_size += index.estimated_size
        return total_size
