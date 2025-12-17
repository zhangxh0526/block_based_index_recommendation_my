import logging
import copy
import re
import os
from utils import *

import gensim
from sklearn.decomposition import PCA

from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.index import Index
from index_selection_evaluation.selection.workload import Query

from boo import BagOfOperators


class WorkloadEmbedder(object):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns=None, partition_num=0, retrieve_plans=False):
        """
        初始化 WorkloadEmbedder 类的实例。

        :param database_name: 数据库的名称。
        :param query_texts: 查询文本列表。
        :param representation_size: 表示的大小。
        :param database_connector: 数据库连接器。
        :param columns: 列信息，默认为 None。
        :param partition_num: 分区数量，默认为 0。
        :param retrieve_plans: 是否获取查询计划，默认为 False。
        """
        # 定义停用词列表，用于过滤查询文本中的常见 SQL 关键字
        self.STOPTOKENS = [
            "as",
            "and",
            "or",
            "min",
            "max",
            "avg",
            "join",
            "on",
            "substr",
            "between",
            "count",
            "sum",
            "case",
            "then",
            "when",
            "end",
            "else",
            "select",
            "from",
            "where",
            "by",
            "cast",
            "in",
        ]
        # 并行模拟的索引数量
        self.INDEXES_SIMULATED_IN_PARALLEL = 1000
        # 存储数据库名称
        self.database_name = database_name
        # 存储分区数量
        self.partition_num = partition_num
        # 确保分区数量大于 0
        assert self.partition_num > 0, "you need specify partition_num"
        # 存储按分区划分的查询文本
        self.query_texts = {}
        # 深拷贝查询文本，避免修改原始数据
        new_query_texts = copy.deepcopy(query_texts)
        # 遍历每个分区
        for partition_id in range(self.partition_num):
            self.query_texts[partition_id] = []
            # 遍历每个查询文本列表
            for new_query_text_list in new_query_texts:
                new_query = []
                # 遍历每个查询文本
                for new_query_text in new_query_text_list:
                    # 对查询文本进行分区处理
                    new_query_text = self.query_text_parition(new_query_text, partition_id)
                    # print(new_query_text)
                    new_query.append(new_query_text)
                # 将处理后的查询文本添加到对应分区的列表中
                self.query_texts[partition_id].append(new_query)

        # 存储表示的大小
        self.representation_size = representation_size
        # 存储数据库连接器
        self.database_connector = database_connector
        # 初始化查询计划为空
        self.plans = None
        # 对列组合进行分区分类
        self.columns = classfy_colCombines_by_partition(columns)

        # 如果需要获取查询计划
        if retrieve_plans:
            # 创建成本评估对象
            cost_evaluation = CostEvaluation(self.database_connector)
            print("Obtain plans for each query ...")
            # 初始化查询计划列表，分别存储无索引和有索引的计划
            self.plans = ([], [])
            # 遍历每个分区
            for partition in self.query_texts:
                # 遍历每个查询类
                for query_idx, query_texts_per_query_class in enumerate(self.query_texts[partition]):
                    # 遍历每个查询文本
                    for _ in query_texts_per_query_class:
                        query_text = _
                        # print(query_text)
                        # 创建查询对象
                        query = Query(query_idx, query_text)
                        # 获取查询计划
                        plan = self.database_connector.get_plan(query)
                        # 将无索引的查询计划添加到列表中
                        self.plans[0].append(plan)
                    print("Success obtain plan from query {}".format(query_idx))

            # 首先尝试从文件中读取计划
            plans_file = "../experiments/back_up_plans/{}-{}".format(self.database_name, self.partition_num)
            # 存储从文件中读取的计划
            plan_bak = {}
            # 如果计划备份文件存在
            if os.path.exists(plans_file):
                # 以读写模式打开文件
                plan_f = open(plans_file, "r+", encoding="utf-8")
                # 逐行读取文件
                for line in plan_f:
                    # 解析计划名称和计划内容
                    plan_name = str(line.strip().split("|||||")[0])
                    plan = str(line.strip().split("|||||")[1])
                    # 将计划存储到字典中
                    plan_bak[str(plan_name)] = eval(plan)
                # 关闭文件
                plan_f.close()

                # 遍历每个列组合
                for n, n_column_combinations in enumerate(self.columns):
                    # 遍历每个分区
                    for partition in n_column_combinations.keys():
                        print("Present column leve is {}, partition is {}".format(n, partition))
                        #print("Indexes to be tested are {}".format(n_column_combinations[partition]))
                        #print("Queries to be tested are {}".format(self.query_texts[partition]))
                        # 对列组合进行排序
                        partitioned_n_column_combinations = sort_columns(n_column_combinations[partition])
                        logging.critical(f"Creating all indexes of width {n + 1}.")

                        created_indexes = 0
                        # 遍历所有列组合
                        while created_indexes < len(partitioned_n_column_combinations):
                            # 存储潜在索引的名称
                            potential_indexes_names = []
                            # 每次并行模拟一定数量的索引
                            for i in range(self.INDEXES_SIMULATED_IN_PARALLEL):
                                potential_indexes_names.append(str(partitioned_n_column_combinations[created_indexes]))
                                created_indexes += 1
                                # 如果已经处理完所有列组合，跳出循环
                                if created_indexes == len(partitioned_n_column_combinations):
                                    break

                            # 对潜在索引名称进行排序
                            potential_indexes_names.sort()
                            # 遍历每个查询类
                            for query_idx, query_texts_per_query_class in enumerate(self.query_texts[partition]):
                                #query_texts = self._obtain_query_class(query_texts_per_query_class, query_idx)
                                # 遍历每个查询文本
                                for _ in query_texts_per_query_class:
                                    query_text = _
                                    # 从备份中获取有索引的查询计划
                                    self.plans[1].append(plan_bak["{}-{}".format(str(potential_indexes_names), str(query_text))])

                            logging.critical(f"Finished checking {created_indexes} indexes of width {n + 1}.")

            else:
                # 如果计划备份文件不存在，创建文件
                os.mknod(plans_file)

                # 以读写模式打开文件
                plan_f = open(plans_file, "w+", encoding="utf-8")

                # 遍历每个列组合
                for n, n_column_combinations in enumerate(self.columns):
                    # 遍历每个分区
                    for partition in n_column_combinations.keys():
                        # 对列组合进行排序
                        partitioned_n_column_combinations = sort_columns(n_column_combinations[partition])
                        logging.critical(f"Creating all indexes of width {n+1}.")

                        print("Start Creating {} indexes ...".format(len(partitioned_n_column_combinations)))

                        created_indexes = 0
                        # 遍历所有列组合
                        while created_indexes < len(partitioned_n_column_combinations):
                            # 存储潜在索引对象
                            potential_indexes = []
                            # 存储潜在索引的名称
                            potential_indexes_names = []
                            # 每次并行模拟一定数量的索引
                            for i in range(self.INDEXES_SIMULATED_IN_PARALLEL):
                                # 创建潜在索引对象
                                potential_index = Index(partitioned_n_column_combinations[created_indexes])
                                # 改动
                                cost_evaluation.what_if.simulate_index(potential_index, True)
                                # 在数据库中创建索引
                                # database_connector.create_index(potential_index)

                                print("Successfully create {}th index {}".
                                      format(created_indexes, partitioned_n_column_combinations[created_indexes]))

                                # 将潜在索引对象添加到列表中
                                potential_indexes.append(potential_index)
                                # 将潜在索引名称添加到列表中
                                potential_indexes_names.append(str(partitioned_n_column_combinations[created_indexes]))
                                created_indexes += 1
                                # 如果已经处理完所有列组合，跳出循环
                                if created_indexes == len(partitioned_n_column_combinations):
                                    break

                            # 对潜在索引名称进行排序
                            potential_indexes_names.sort()
                            # 遍历每个查询类
                            for query_idx, query_texts_per_query_class in enumerate(self.query_texts[partition]):
                                #query_texts = self._obtain_query_class(query_texts_per_query_class, query_idx)
                                # 遍历每个查询文本
                                for _ in query_texts_per_query_class:
                                    query_text = _
                                    # 创建查询对象
                                    query = Query(query_idx, query_text)
                                    # 获取查询计划
                                    plan = self.database_connector.get_plan(query)
                                    # 将计划信息写入文件
                                    plan_f.write("{}-{}|||||{}\n".format(str(potential_indexes_names), str(query_text), str(plan)))
                                    # 将有索引的查询计划添加到列表中
                                    self.plans[1].append(plan)

                            # 遍历每个潜在索引对象
                            for potential_index in potential_indexes:
                                # 改动
                                cost_evaluation.what_if.drop_simulated_index(potential_index)
                                # 在数据库中删除索引
                                # database_connector.drop_index(potential_index)

                            logging.critical(f"Finished checking {created_indexes} indexes of width {n+1}.")

                # 关闭文件
                plan_f.close()

        # 将数据库连接器置为 None，避免后续使用
        self.database_connector = None





    def _obtain_query_class(self, query_texts, query_idx):
        """
        根据查询索引获取查询类。

        当查询索引在 0 到 14 之间时，对查询文本进行去重处理，去重依据是 WHERE 子句中的列集合；
        当查询索引不在此范围内时，直接返回查询文本列表的第一个元素。

        :param query_texts: 查询文本列表
        :param query_idx: 查询索引
        :return: 处理后的查询文本列表
        """
        # 检查查询索引是否在 0 到 14 之间
        if query_idx >= 0 and query_idx <= 14:
            # 初始化结果查询列表
            result_queries = []
            # 初始化一个集合，用于存储 WHERE 子句中的列集合
            tmp_types = set()
            # 遍历查询文本列表
            for _ in query_texts:
                # 调用 where_columns 方法获取查询文本中 WHERE 子句的列集合，并将其转换为不可变集合
                columns = frozenset(self.where_columns(_))
                # 检查该列集合是否已经存在于 tmp_types 集合中
                if columns not in tmp_types:
                    # 如果不存在，则将该列集合添加到 tmp_types 集合中
                    tmp_types.add(columns)
                    # 并将该查询文本添加到结果查询列表中
                    result_queries.append(_)
            # 返回处理后的查询文本列表
            return result_queries
        else:
            # 当查询索引不在 0 到 14 之间时，直接返回查询文本列表的第一个元素
            return [query_texts[0]]


    def where_columns(self, query):
        """
        从 SQL 查询语句中提取 WHERE 子句里涉及的列名。

        :param query: 输入的 SQL 查询语句
        :return: 包含 WHERE 子句中涉及的列名的集合
        """
        # 初始化一个空集合，用于存储提取的列名
        columns = set()
        # 使用正则表达式查找查询语句中所有的 WHERE 子句
        where_clauses = re.findall(r'where(.*?);', query, flags=re.IGNORECASE)
        # 遍历每个 WHERE 子句
        for where_clause in where_clauses:
            # 将 WHERE 子句按 'and' 分割成多个条件
            conditions = where_clause.split('and')
            # 初始化列名变量
            column = None
            # 遍历每个条件
            for condition in conditions:
                # 如果条件中包含 'select'，则跳过该条件
                if 'select' in condition:
                    continue

                # 处理下界条件
                if '>' in condition or 'between' in condition:
                    if '>=' in condition:
                        # 按 '>=' 分割条件，获取列名和下界值
                        column, lower_bound = condition.split('>=')
                    elif '>' in condition:
                        # 按 '>' 分割条件，获取列名和下界值
                        column, lower_bound = condition.split('>')
                    else:
                        # 按 'between' 分割条件，获取列名和下界值
                        column, lower_bound = condition.split('between')
                    # 去除列名前后的空格，并取最后一个单词作为列名
                    column = column.strip().split()[-1]
                    # 将列名添加到集合中
                    columns.add(column)

                # 处理上界条件
                elif '<' in condition or condition.strip()[0].isdigit():
                    if '<=' in condition:
                        # 按 '<=' 分割条件，获取列名和上界值
                        column, upper_bound = condition.split('<=')
                    elif '<' in condition:
                        # 按 '<' 分割条件，获取列名和上界值
                        column, upper_bound = condition.split('<')
                    else:
                        # 若条件以数字开头，直接获取上界值
                        upper_bound = condition.strip()
                    # 去除列名前后的空格，并取最后一个单词作为列名
                    column = column.strip().split()[-1]
                    # 将列名添加到集合中
                    columns.add(column)

                # 处理等于条件
                elif '=' in condition:
                    # 按 '=' 分割条件，获取列名和值
                    column, lower_bound = condition.split('=')
                    # 去除列名前后的空格，并取最后一个单词作为列名
                    column = column.strip().split()[-1]
                    # 将列名添加到集合中
                    columns.add(column)

        # 返回包含所有列名的集合
        return columns




    def get_embeddings(self, workload):
        raise NotImplementedError


    def _partitioned_table_map(self, partition_id):
        if "tpch" in self.database_name:
            return {
                'customer': f"customer_1_prt_p{partition_id}",
                'lineitem': f"lineitem_1_prt_p{partition_id}",
                'nation': f"nation_1_prt_p{partition_id}",
                'orders': f"orders_1_prt_p{partition_id}",
                'part': f"part_1_prt_p{partition_id}",
                'partsupp': f"partsupp_1_prt_p{partition_id}",
                'region': f"region_1_prt_p{partition_id}",
                'supplier': f"supplier_1_prt_p{partition_id}"
        }
        elif "ssb" in self.database_name:
            return {
                'lineorder': f"lineorder_1_prt_p{partition_id}",
                'supplier': f"supplier_1_prt_p{partition_id}",
                'part': f"part_1_prt_p{partition_id}",
                'customer': f"customer_1_prt_p{partition_id}",
                'date': f"dim_date_1_prt_p{partition_id}"
            }
        elif "tpcds" in self.database_name:
            return {
                'dbgen_version': f"dbgen_version_1_prt_p{partition_id}",
                'customer_address': f"customer_address_1_prt_p{partition_id}",
                'customer_demographics': f"customer_demographics_1_prt_p{partition_id}",
                'date_dim': f"date_dim_1_prt_p{partition_id}",
                'warehouse': f"warehouse_1_prt_p{partition_id}",
                'ship_mode': f"ship_mode_1_prt_p{partition_id}",
                'time_dim': f"time_dim_1_prt_p{partition_id}",
                'reason': f"reason_1_prt_p{partition_id}",
                'income_band': f"income_band_1_prt_p{partition_id}",
                'item': f"item_1_prt_p{partition_id}",
                'store': f"store_1_prt_p{partition_id}",
                'call_center': f"call_center_1_prt_p{partition_id}",
                'customer': f"customer_1_prt_p{partition_id}",
                'web_site': f"web_site_1_prt_p{partition_id}",
                'store_returns': f"store_returns_1_prt_p{partition_id}",
                'household_demographics': f"household_demographics_1_prt_p{partition_id}",
                'web_page': f"web_page_1_prt_p{partition_id}",
                'promotion': f"promotion_1_prt_p{partition_id}",
                'catalog_page': f"catalog_page_1_prt_p{partition_id}",
                'inventory': f"inventory_1_prt_p{partition_id}",
                'catalog_returns': f"catalog_returns_1_prt_p{partition_id}",
                'web_returns': f"web_returns_1_prt_p{partition_id}",
                'web_sales': f"web_sales_1_prt_p{partition_id}",
                'catalog_sales': f"catalog_sales_1_prt_p{partition_id}",
                'store_sales': f"store_sales_1_prt_p{partition_id}"
            }
        else:
            return None
    
    def query_text_parition(self, query_text, partition_id):
        """
        对查询文本进行分区处理，将查询中的表名和别名替换为分区表名。

        :param query_text: 原始查询文本。
        :param partition_id: 分区的 ID。
        :return: 处理后的查询文本。
        """
        # 深拷贝查询文本，避免修改原始数据
        new_text = copy.deepcopy(query_text)
        # 获取分区表映射
        table_alias_map = self._partitioned_table_map(partition_id)
        # 确保表映射不为空，表名应为 tpcds 或 tpch
        assert table_alias_map is not None, "table should be tpcds or tpch"
        # 将查询中的表名和别名替换为"原表_1"格式
        for table, alias in table_alias_map.items():
            # 编译正则表达式，匹配独立的表名
            table_pattern = re.compile(rf'(?<!\w){table}(?!\w)')
            # 使用正则表达式将查询文本中的表名替换为分区表名
            new_text = table_pattern.sub(f'{alias}', new_text)

            # 编译正则表达式，匹配独立的别名
            alias_pattern = re.compile(rf'(?<!\w){alias}(?!\w)')
            # 使用正则表达式将查询文本中的别名替换为分区表名
            new_text = alias_pattern.sub(f'{alias}', new_text)
        return new_text


class SQLWorkloadEmbedder(WorkloadEmbedder):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns):
        WorkloadEmbedder.__init__(self, database_name, query_texts, representation_size, database_connector)

        tagged_queries = []

        for query_idx, query_texts_per_query_class in enumerate(query_texts):
            query_text = query_texts_per_query_class[0]
            tokens = gensim.utils.simple_preprocess(query_text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            tagged_queries.append(gensim.models.doc2vec.TaggedDocument(tokens, [query_idx]))

        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.representation_size, min_count=2, epochs=500)
        self.model.build_vocab(tagged_queries)
        logger = logging.getLogger("gensim.models.base_any2vec")
        logger.setLevel(logging.CRITICAL)
        self.model.train(tagged_queries, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        logger.setLevel(logging.INFO)

    def get_embeddings(self, workload):
        embeddings = []

        for query in workload.queries:
            tokens = gensim.utils.simple_preprocess(query.text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            vector = self.model.infer_vector(tokens)

            embeddings.append(vector)

        return embeddings


class SQLWorkloadLSI(WorkloadEmbedder):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns):
        WorkloadEmbedder.__init__(self, database_name, query_texts, representation_size, database_connector)

        self.processed_queries = []

        for query_idx, query_texts_per_query_class in enumerate(query_texts):
            query_text = query_texts_per_query_class[0]
            tokens = gensim.utils.simple_preprocess(query_text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            self.processed_queries.append(tokens)

        self.dictionary = gensim.corpora.Dictionary(self.processed_queries)
        self.bow_corpus = [self.dictionary.doc2bow(query) for query in self.processed_queries]
        self.lsi_bow = gensim.models.LsiModel(
            self.bow_corpus, id2word=self.dictionary, num_topics=self.representation_size
        )

    def get_embeddings(self, workload):
        embeddings = []

        for query in workload.queries:
            tokens = gensim.utils.simple_preprocess(query.text, max_len=50)
            tokens = [token for token in tokens if token not in self.STOPTOKENS]
            bow = self.dictionary.doc2bow(tokens)
            result = self.lsi_bow[bow]
            result = [x[1] for x in result]

            embeddings.append(result)

        return embeddings


class PlanEmbedder(WorkloadEmbedder):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns, partition_num=0, without_indexes=False):
        """
        初始化 PlanEmbedder 类的实例。

        :param database_name: 数据库的名称。
        :param query_texts: 查询文本列表。
        :param representation_size: 表示的大小。
        :param database_connector: 数据库连接器。
        :param columns: 列信息。
        :param partition_num: 分区数量，默认为 0。
        :param without_indexes: 是否不考虑索引，默认为 False。
        """
        # 调用父类 WorkloadEmbedder 的构造函数，设置 retrieve_plans 为 True 以获取查询计划
        WorkloadEmbedder.__init__(
            self, database_name, query_texts, representation_size, database_connector, columns, partition_num, retrieve_plans=True
        )

        # 用于缓存查询计划的嵌入向量，避免重复计算
        self.plan_embedding_cache = {}

        # 存储所有相关操作符的列表
        self.relevant_operators = []
        # 存储不考虑索引的相关操作符的列表
        self.relevant_operators_wo_indexes = []
        # 存储考虑索引的相关操作符的列表
        self.relevant_operators_with_indexes = []

        # 用于从查询计划中提取操作符的 BagOfOperators 类的实例
        self.boo_creator = BagOfOperators()

        # 遍历没有索引的查询计划列表，提取操作符并添加到相关列表中
        for plan in self.plans[0]:
            # 从查询计划中提取操作符
            boo = self.boo_creator.boo_from_plan(plan)
            # 将操作符添加到所有相关操作符列表中
            self.relevant_operators.append(boo)
            # 将操作符添加到不考虑索引的相关操作符列表中
            self.relevant_operators_wo_indexes.append(boo)

        # 如果考虑索引
        if without_indexes is False:
            # 遍历有索引的查询计划列表，提取操作符并添加到相关列表中
            for plan in self.plans[1]:
                # 从查询计划中提取操作符
                boo = self.boo_creator.boo_from_plan(plan)
                # 将操作符添加到所有相关操作符列表中
                self.relevant_operators.append(boo)
                # 将操作符添加到考虑索引的相关操作符列表中
                self.relevant_operators_with_indexes.append(boo)

        # 删除查询计划，避免后续昂贵的复制操作
        # Deleting the plans to avoid costly copying later.
        self.plans = None

        # 使用 gensim 库的 Dictionary 类创建一个字典，用于将操作符映射到唯一的整数 ID
        self.dictionary = gensim.corpora.Dictionary(self.relevant_operators)
        # 记录字典中的条目数量
        logging.warning(f"Dictionary has {len(self.dictionary)} entries.")
        # 将每个查询的操作符转换为词袋表示
        self.bow_corpus = [self.dictionary.doc2bow(query) for query in self.relevant_operators]

        # 调用 _create_model 方法来创建嵌入模型
        self._create_model()

        # 删除词袋语料库，避免后续昂贵的复制操作
        # Deleting the bow_corpus to avoid costly copying later.
        self.bow_corpus = None

    def _create_model(self):
        """
        创建嵌入模型的抽象方法，需要在子类中实现。
        """
        raise NotImplementedError

    def _infer(self, bow, boo):
        """
        从词袋表示和操作符列表中推断嵌入向量的抽象方法，需要在子类中实现。

        :param bow: 词袋表示。
        :param boo: 操作符列表。
        """
        raise NotImplementedError

    def get_embeddings(self, plans):
        """
        获取查询计划的嵌入向量。

        :param plans: 查询计划列表。
        :return: 嵌入向量列表。
        """
        # 存储嵌入向量的列表
        embeddings = []

        # 遍历每个查询计划
        for plan in plans:
            # 生成缓存键
            cache_key = str(plan)
            # 检查缓存中是否存在该查询计划的嵌入向量
            if cache_key not in self.plan_embedding_cache:
                # 从查询计划中提取操作符
                boo = self.boo_creator.boo_from_plan(plan)
                # 将操作符转换为词袋表示
                bow = self.dictionary.doc2bow(boo)
                # if len(bow) == 0:
                #     a = 1

                # 调用 _infer 方法推断嵌入向量
                vector = self._infer(bow, boo)

                # 将嵌入向量缓存起来
                self.plan_embedding_cache[cache_key] = vector
            else:
                # 从缓存中获取嵌入向量
                vector = self.plan_embedding_cache[cache_key]

            # 将嵌入向量添加到结果列表中
            embeddings.append(vector)

        # 返回嵌入向量列表
        return embeddings


class PlanEmbedderPCA(PlanEmbedder):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns):
        PlanEmbedder.__init__(self, database_name, query_texts, representation_size, database_connector, columns)

    def _to_full_corpus(self, corpus):
        new_corpus = []
        for bow in corpus:
            new_bow = [0 for i in range(len(self.dictionary))]
            for elem in bow:
                index, value = elem
                new_bow[index] = value
            new_corpus.append(new_bow)

        return new_corpus

    def _create_model(self):
        new_corpus = self._to_full_corpus(self.bow_corpus)

        self.pca = PCA(n_components=self.representation_size)
        self.pca.fit(new_corpus)

        assert (
            sum(self.pca.explained_variance_ratio_) > 0.8
        ), f"Explained variance must be larger than 80% (is {sum(self.pca.explained_variance_ratio_)})"

    def _infer(self, bow, boo):
        new_bow = self._to_full_corpus([bow])

        return self.pca.transform(new_bow)


class PlanEmbedderDoc2Vec(PlanEmbedder):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns, without_indexes=False):
        self.without_indexes = without_indexes

        PlanEmbedder.__init__(self, database_name, query_texts, representation_size, database_connector, columns, without_indexes)

    def _create_model(self):
        tagged_plans = []
        for plan_idx, operators in enumerate(self.relevant_operators):
            tagged_plans.append(gensim.models.doc2vec.TaggedDocument(operators, [plan_idx]))

        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.representation_size, min_count=2, epochs=1000)
        self.model.build_vocab(tagged_plans)
        logger = logging.getLogger("gensim.models.base_any2vec")
        logger.setLevel(logging.CRITICAL)
        self.model.train(tagged_plans, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        logger.setLevel(logging.INFO)

    def _infer(self, bow, boo):
        vector = self.model.infer_vector(boo)

        return vector


class PlanEmbedderDoc2VecWithoutIndexes(PlanEmbedderDoc2Vec):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns):
        PlanEmbedderDoc2Vec.__init__(
            self, database_name, query_texts, representation_size, database_connector, columns, without_indexes=True
        )


class PlanEmbedderBOW(PlanEmbedder):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns, partition_num=0):
        PlanEmbedder.__init__(self, database_name, query_texts, representation_size, database_connector, columns, partition_num)

    def _create_model(self):
        assert self.representation_size == len(self.dictionary), f"{self.representation_size} == {len(self.dictionary)}"

    def _to_full_bow(self, bow):
        new_bow = [0 for i in range(len(self.dictionary))]
        for elem in bow:
            index, value = elem
            new_bow[index] = value

        return new_bow

    def _infer(self, bow, boo):
        return self._to_full_bow(bow)


class PlanEmbedderLSIBOW(PlanEmbedder):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns, partition_num=0, without_indexes=False):
        """
        初始化 PlanEmbedderLSIBOW 类的实例。

        :param database_name: 数据库的名称。
        :param query_texts: 查询文本列表。
        :param representation_size: 表示的大小。
        :param database_connector: 数据库连接器。
        :param columns: 列信息。
        :param partition_num: 分区数量，默认为 0。
        :param without_indexes: 是否不考虑索引，默认为 False。
        """
        # 调用父类 PlanEmbedder 的构造函数进行初始化
        PlanEmbedder.__init__(self, database_name, query_texts, representation_size, database_connector, columns, partition_num, without_indexes)

    def _create_model(self):
        """
        创建 LSI（Latent Semantic Indexing）模型，用于将查询计划的词袋表示转换为低维向量表示。
        """
        # 使用 gensim 库创建 LSI 模型，输入为词袋语料库、字典和主题数量（即表示大小）
        self.lsi_bow = gensim.models.LsiModel(
            self.bow_corpus, id2word=self.dictionary, num_topics=self.representation_size
        )

        # 断言检查 LSI 模型的主题数量是否与表示大小一致，不一致则抛出异常
        assert (
            len(self.lsi_bow.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.lsi_bow.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, boo):
        """
        根据输入的词袋表示和操作符列表推断嵌入向量。

        :param bow: 词袋表示。
        :param boo: 操作符列表。
        :return: 推断出的嵌入向量。
        """
        # 使用 LSI 模型对输入的词袋表示进行转换
        result = self.lsi_bow[bow]

        # 如果转换结果的长度等于表示大小，直接提取向量值
        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            # 否则，初始化一个全零向量，并将转换结果中的值填充到对应位置
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value

        # 断言检查向量的长度是否与表示大小一致
        assert len(vector) == self.representation_size

        # 返回推断出的嵌入向量
        return vector


class PlanEmbedderLSIBOWWithoutIndexes(PlanEmbedderLSIBOW):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns):
        PlanEmbedderLSIBOW.__init__(
            self, database_name, query_texts, representation_size, database_connector, columns, without_indexes=True
        )


class PlanEmbedderLSITFIDF(PlanEmbedder):
    def __init__(self, database_name, query_texts, representation_size, database_connector, columns):
        PlanEmbedder.__init__(self, database_name, query_texts, representation_size, database_connector, columns)

    def _create_model(self):
        self.tfidf = gensim.models.TfidfModel(self.bow_corpus, normalize=True)
        self.corpus_tfidf = self.tfidf[self.bow_corpus]
        self.lsi_tfidf = gensim.models.LsiModel(
            self.corpus_tfidf, id2word=self.dictionary, num_topics=self.representation_size
        )

        assert (
            len(self.lsi_tfidf.get_topics()) == self.representation_size
        ), f"Topic-representation_size mismatch: {len(self.lsi_tfidf.get_topics())} vs {self.representation_size}"

    def _infer(self, bow, boo):
        result = self.lsi_tfidf[self.tfidf[bow]]

        if len(result) == self.representation_size:
            vector = [x[1] for x in result]
        else:
            vector = [0] * self.representation_size
            for topic, value in result:
                vector[topic] = value
        assert len(vector) == self.representation_size

        return vector
