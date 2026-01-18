import logging
import math
import time
from copy import deepcopy
import functools

from ..index import Index
from ..selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
from ..utils import b_to_mb, mb_to_b
from ..workload import Workload

# 分区版 Extend，使用“模拟退火式范围衰减”+“全局锦标赛单赢家”策略
DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "min_cost_improvement": 1.003,
    "max_runtime_minutes": 60 * 24,
    # 退温控制
    "anneal_alpha": 3.0,
}


def indexsort(x, y):
    if x.columns[0].table.name > y.columns[0].table.name:
        return -1
    return 1


class ExtendAlgorithmPartitionSA(SelectionAlgorithm):
    """基于退火的分区全局锦标赛：每轮只录取全局 ROI 最高的一个索引。"""

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
        self.anneal_alpha = float(self.parameters.get("anneal_alpha", 3.0))

    def _should_stop_due_to_time(self):
        current_time = time.time()
        consumed_time = current_time - self.start_time
        if consumed_time > self.max_runtime_minutes * 60:
            logging.debug(
                "Stopping because of timing constraints. " f"Time: {consumed_time / 60:.2f} minutes."
            )
            return True
        return False

    # 退温函数：根据剩余预算比例决定活跃分区数
    def _calculate_active_k(self, remaining_ratio):
        n = self.partition_num
        if remaining_ratio <= 0.1:
            return 1
        # 归一化指数曲线：ratio=1 -> k=n, ratio->0 -> k->1
        k = math.ceil(
            n * (math.exp(self.anneal_alpha * remaining_ratio) - 1) / (math.exp(self.anneal_alpha) - 1)
        )
        return max(1, min(n, int(k)))

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

    def _evaluate_combination(self, workload, current_cost, index_combination, best):
        start_time = time.time()
        cost = self.cost_evaluation.calculate_cost(
            workload,
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

        if ratio > best["benefit_to_size_ratio"]:
            best.update({
                "combination": index_combination,
                "benefit_to_size_ratio": ratio,
                "cost": cost,
                "benefit": benefit,
                "size": total_size,
            })

    def _best_candidate_for_partition(
        self,
        partition_id,
        workload_partition,
        index_combination,
        single_candidates,
        extension_candidates,
        remaining_budget_b,
        current_cost,
    ):
        best = {"benefit_to_size_ratio": 0}
        current_size = self._total_index_size(index_combination)

        # 单列候选
        valid_single = self._get_candidates_within_budget(current_size, single_candidates, remaining_budget_b)
        for candidate in valid_single:
            if candidate in index_combination:
                continue
            self._evaluate_combination(
                workload_partition,
                current_cost,
                index_combination + [candidate],
                best,
            )

        # 扩展候选
        for attribute in extension_candidates:
            if attribute.is_single_column() is False:
                continue
            for position, index in enumerate(index_combination):
                if len(index.columns) >= self.max_index_width:
                    continue
                if not index.appendable_by(attribute):
                    continue
                new_index = Index(index.columns + attribute.columns)
                if new_index in index_combination:
                    continue
                new_combination = index_combination.copy()
                del new_combination[position]
                new_combination.append(new_index)
                self._evaluate_combination(
                    workload_partition,
                    current_cost,
                    new_combination,
                    best,
                )

        # 尺寸检查
        if best.get("size") and (best["size"] - current_size) > remaining_budget_b:
            return None

        return best if best.get("benefit_to_size_ratio", 0) > 0 else None

    def _calculate_best_indexes(self, workload):
        self.cost_evaluation.what_if.drop_all_simulated_indexes()
        self.cost_evaluation_time = 0
        logging.info("Calculating best indexes Extend_partition_SA (Batch + GapFilling)")
        self.workload = workload

        single_attribute_index_candidates = workload.potential_indexes()
        single_attribute_index_candidates = self._group_partition_index(single_attribute_index_candidates)
        extension_attribute_candidates = deepcopy(single_attribute_index_candidates)

        partition_tables = self._partition_tables(single_attribute_index_candidates)
        total_rows_per_partition = self._partition_total_rows(partition_tables)
        partition_scores = self._compute_partition_scores(workload, partition_tables, total_rows_per_partition)

        # 初始化每分区的工作负载与成本
        partition_workloads = []
        current_costs = []
        for pid in range(self.partition_num):
            queries_partition = workload.queries[pid::self.partition_num]
            wl = Workload(queries_partition)
            wl.budget = self.budget
            partition_workloads.append(wl)
            start_time = time.time()
            cost = self.cost_evaluation.calculate_cost(wl, [], store_size=False)
            self.cost_evaluation_time += round(float(time.time() - start_time), 2)
            current_costs.append(cost)

        index_combinations = [[] for _ in range(self.partition_num)]
        candidate_cache = [None for _ in range(self.partition_num)]

        remaining_budget_mb = float(self.budget)
        remaining_budget_b = mb_to_b(remaining_budget_mb)
        log_top_n = int(self.parameters.get("log_top_n", 0) or 0)

        self.start_time = time.time()

        while remaining_budget_b > 0:
            ratio = remaining_budget_mb / self.budget if self.budget else 0
            k = self._calculate_active_k(ratio)
            if log_top_n > 0:
                logging.info("Active K=%s (remaining %.2f MB)", k, remaining_budget_mb)

            ranked_pids_scores = sorted(
                enumerate(partition_scores), key=lambda x: x[1], reverse=True
            )
            active_pids = [pid for pid, _ in ranked_pids_scores[:k]]
            if log_top_n > 0:
                logging.info("Active partitions: %s", active_pids)

            # 阶段 A：收集活跃分区候选
            round_candidates = []
            for pid in active_pids:
                if candidate_cache[pid] is None:
                    best = self._best_candidate_for_partition(
                        pid,
                        partition_workloads[pid],
                        index_combinations[pid],
                        single_attribute_index_candidates[pid],
                        extension_attribute_candidates[pid],
                        remaining_budget_b,
                        current_costs[pid],
                    )
                    candidate_cache[pid] = best
                if candidate_cache[pid]:
                    round_candidates.append((pid, candidate_cache[pid]))

            # 阶段 B：兜底填缝（活跃分区无货/太贵时扩展到全局）
            need_fallback = False
            if not round_candidates:
                need_fallback = True
            else:
                all_too_expensive = True
                for _, cand in round_candidates:
                    current_size = self._total_index_size(index_combinations[_])
                    if (cand["size"] - current_size) <= remaining_budget_b:
                        all_too_expensive = False
                        break
                if all_too_expensive:
                    need_fallback = True

            if need_fallback:
                if log_top_n > 0:
                    fallback_reasons = []
                    if not round_candidates:
                        fallback_reasons.append("no_candidates")
                    else:
                        fallback_reasons.append("all_too_expensive")
                    logging.info(
                        "Active K=%s exhausted/expensive (%s). Falling back to GLOBAL SEARCH.",
                        k,
                        ",".join(fallback_reasons),
                    )
                else:
                    logging.info("Active K=%s exhausted/expensive. Falling back to GLOBAL SEARCH.", k)
                non_active_pids = [p for p in range(self.partition_num) if p not in active_pids]
                for pid in non_active_pids:
                    if candidate_cache[pid] is None:
                        best = self._best_candidate_for_partition(
                            pid,
                            partition_workloads[pid],
                            index_combinations[pid],
                            single_attribute_index_candidates[pid],
                            extension_attribute_candidates[pid],
                            remaining_budget_b,
                            current_costs[pid],
                        )
                        candidate_cache[pid] = best
                    if candidate_cache[pid]:
                        round_candidates.append((pid, candidate_cache[pid]))

            if not round_candidates:
                logging.info("No more indexes found in any partition.")
                break

            if log_top_n > 0:
                ranked_preview = []
                for pid, cand in round_candidates:
                    current_size = self._total_index_size(index_combinations[pid])
                    delta_size = cand["size"] - current_size
                    ranked_preview.append(
                        {
                            "pid": pid,
                            "ratio": cand["benefit_to_size_ratio"],
                            "size_mb": b_to_mb(cand["size"]),
                            "delta_mb": b_to_mb(delta_size),
                        }
                    )
                ranked_preview.sort(key=lambda x: x["ratio"], reverse=True)
                logging.info(
                    "Top-%s candidates (remain %.2f MB): %s",
                    log_top_n,
                    remaining_budget_mb,
                    ranked_preview[:log_top_n],
                )

            # 阶段 C：批量录取（每轮尽可能多买）
            round_candidates.sort(key=lambda x: x[1]["benefit_to_size_ratio"], reverse=True)

            something_bought = False
            for pid, candidate in round_candidates:
                current_size = self._total_index_size(index_combinations[pid])
                delta_size = candidate["size"] - current_size
                if delta_size <= remaining_budget_b:
                    index_combinations[pid] = candidate["combination"]
                    remaining_budget_b -= delta_size
                    remaining_budget_mb = b_to_mb(remaining_budget_b)

                    current_costs[pid] = candidate["cost"]
                    partition_scores[pid] = max(
                        0, partition_scores[pid] - (candidate.get("benefit", 0) * 0.8)
                    )
                    candidate_cache[pid] = None
                    something_bought = True

                    logging.info(
                        "Selected P%s, ratio %.4f, size %.2f MB (+%.2f MB), budget left %.2f MB",
                        pid,
                        candidate["benefit_to_size_ratio"],
                        b_to_mb(candidate["size"]),
                        b_to_mb(delta_size),
                        remaining_budget_mb,
                    )

                if remaining_budget_b <= 0:
                    break

            if not something_bought:
                logging.info("Budget insufficient for any remaining candidates.")
                break

            if self._should_stop_due_to_time():
                break

        total_execution_time = round(float(time.time() - self.start_time), 2)
        print(f"Extend_partition_SA finished. Time: {total_execution_time}s")

        # 扁平化结果
        final_indexes = []
        for combo in index_combinations:
            final_indexes.extend(combo)
        return final_indexes

    def _get_candidates_within_budget(self, index_combination_size, candidates, budget_b=None):
        new_candidates = []
        limit = budget_b if budget_b is not None else mb_to_b(self.budget)
        for candidate in candidates:
            if candidate.estimated_size is None:
                candidate.estimated_size = 0
            if candidate.estimated_size + index_combination_size <= limit:
                new_candidates.append(candidate)
        return new_candidates

    def _total_index_size(self, indexes):
        total_size = 0
        for index in indexes:
            if index.estimated_size is None:
                self.cost_evaluation.estimate_size(index)
            if index.estimated_size is None:
                continue
            total_size += index.estimated_size
        return total_size
