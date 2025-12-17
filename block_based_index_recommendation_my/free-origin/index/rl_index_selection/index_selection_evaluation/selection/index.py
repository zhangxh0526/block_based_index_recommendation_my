from functools import total_ordering


@total_ordering
class Index:
    """
    表示数据库索引的类。索引由一系列列组成，可用于数据库的查询优化。

    Attributes:
        columns (tuple): 组成索引的列的元组。
        estimated_size (int, optional): 索引的估计大小。默认为 None。
        hypopg_name (str, optional): HypoPG 扩展中索引的名称。默认为 None。
    """
    def __init__(self, columns, estimated_size=None):
        """
        初始化 Index 类的实例。

        Args:
            columns (list): 组成索引的列的列表。
            estimated_size (int, optional): 索引的估计大小。默认为 None。

        Raises:
            ValueError: 如果 columns 列表为空。
        """
        if len(columns) == 0:
            raise ValueError("Index needs at least 1 column")
        # 将列列表转换为元组，确保索引列的不可变性
        self.columns = tuple(columns)
        # 当 `store_size=True` (whatif) 时存储 HypoPG 估计的大小
        self.estimated_size = estimated_size
        # HypoPG 中索引的名称，初始化为 None
        self.hypopg_name = None

    # 用于对索引进行排序
    def __lt__(self, other):
        """
        实现小于比较运算符，用于对索引进行排序。

        Args:
            other (Index): 另一个 Index 实例。

        Returns:
            bool: 如果当前索引小于另一个索引，则返回 True，否则返回 False。
        """
        if len(self.columns) != len(other.columns):
            # 先比较列的数量，数量少的索引较小
            return len(self.columns) < len(other.columns)
        # 列数量相同，按列的字典序比较
        return self.columns < other.columns

    def __repr__(self):
        """
        返回索引的字符串表示形式。

        Returns:
            str: 索引的字符串表示，格式为 "I(列名1,列名2,...)"。
        """
        # 将列转换为字符串并使用逗号连接
        columns_string = ",".join(map(str, self.columns))
        return f"I({columns_string})"

    def __eq__(self, other):
        """
        实现相等比较运算符，用于判断两个索引是否相等。

        Args:
            other (Index): 另一个 Index 实例。

        Returns:
            bool: 如果两个索引相等，则返回 True，否则返回 False。
        """
        if not isinstance(other, Index):
            # 如果 other 不是 Index 类的实例，返回 False
            return False
        # 比较两个索引的列是否相同
        return self.columns == other.columns

    def __hash__(self):
        """
        返回索引的哈希值，用于在集合和字典中使用。

        Returns:
            int: 索引的哈希值，基于列的元组计算。
        """
        return hash(self.columns)

    def _column_names(self):
        """
        获取索引列的名称列表。

        Returns:
            list: 包含索引列名称的列表。
        """
        return [x.name for x in self.columns]

    def is_single_column(self):
        """
        判断索引是否为单列索引。

        Returns:
            bool: 如果索引只包含一个列，则返回 True，否则返回 False。
        """
        return True if len(self.columns) == 1 else False

    def table(self):
        """
        获取索引所属的表。

        Returns:
            str: 索引所属表的名称。

        Raises:
            AssertionError: 如果索引的第一列的表名是 None。
        """
        assert (
            self.columns[0].table is not None
        ), "Table should not be None to avoid false positive comparisons."
        return self.columns[0].table

    def index_idx(self):
        """
        生成索引的唯一标识符。

        Returns:
            str: 索引的唯一标识符，格式为 "表名_列名1_列名2_..._idx"。
        """
        # 将列名用下划线连接
        columns = "_".join(self._column_names())
        # 原注释的返回方式，可根据实际情况选择
        # return f"{self.table()}_{columns}_idx"
        # 从表名中提取最后一部分，然后与列名组合成索引名
        return f"{str(self.table()).split('_')[-1]}_{columns}_idx"

    def joined_column_names(self):
        """
        获取索引列名的字符串表示，列名之间用逗号连接。

        Returns:
            str: 索引列名的字符串表示。
        """
        return ",".join(self._column_names())

    def appendable_by(self, other):
        """
        判断当前索引是否可以被另一个索引追加。

        Args:
            other (Index): 另一个 Index 实例。

        Returns:
            bool: 如果可以追加，则返回 True，否则返回 False。
        """
        if not isinstance(other, Index):
            # 如果 other 不是 Index 类的实例，返回 False
            return False
        if self.table() != other.table():
            # 如果两个索引不在同一个表上，返回 False
            return False
        if not other.is_single_column():
            # 如果 other 不是单列索引，返回 False
            return False
        if other.columns[0] in self.columns:
            # 如果 other 的列已经在当前索引中，返回 False
            return False
        return True

    def subsumes(self, other):
        """
        判断当前索引是否包含另一个索引。

        Args:
            other (Index): 另一个 Index 实例。

        Returns:
            bool: 如果当前索引包含另一个索引，则返回 True，否则返回 False。
        """
        if not isinstance(other, Index):
            # 如果 other 不是 Index 类的实例，返回 False
            return False
        # 比较当前索引的前缀是否与 other 的列相同
        return self.columns[: len(other.columns)] == other.columns

    def prefixes(self):
        """
        考虑 I(K;S)。对于 K 的任何前缀 K'（如果 S 不为空，包括 K' = K），
        可以得到一个索引 I_P = (K';Ø)。
        返回按宽度递减排序的索引前缀列表。

        Returns:
            list: 包含索引前缀的列表。
        """
        index_prefixes = []
        # 从最大前缀宽度开始递减，生成所有可能的前缀
        for prefix_width in range(len(self.columns) - 1, 0, -1):
            index_prefixes.append(Index(self.columns[:prefix_width]))
        return index_prefixes



# The following methods implement the index transformation rules presented by
# Bruno and Chaudhuri their 2005 paper Automatic Physical Database Tuning:
# A Relaxation-based Approach.
#   The "removal" transformation is not implemented, because it does not directly work on
#     index objects, but more on an index configuration.
#   The "promotion to clustered" transformation is not implemented, because clustered
#     indexes are currently not chosen by selection algorithms
#   The "prefixing" is implemented as method of the Index class
#
# The authors define an index I as a sequence of key columns K and a set of suffix
# columns S: I = (K;S). If the database system does not support suffix columns, only
# key columns are considered.

# A merged index is the best index that can answer all requests that either previous
# index did. Merging I_1(K_1;S_1) and I_2(K_2;S_2) results in
# I_1_2 = (K1;(S_1 ∪ K_2 ∪ S_2) - K_1).
# If K_1 is a prefix of K_2, I_1_2 = (K2; (S_1 ∪ S_2) - K_2)).
# Returns the merged index.
def index_merge(index_1, index_2):
    """
    合并两个索引对象，生成一个新的索引对象，该索引能满足原两个索引的所有查询需求。

    :param index_1: 第一个索引对象
    :param index_2: 第二个索引对象
    :return: 合并后的索引对象
    """
    # 确保两个索引属于同一个表
    assert index_1.table() == index_2.table()
    # 初始化合并后的列列表，先添加 index_1 的列
    merged_columns = list(index_1.columns)
    # 遍历 index_2 的列
    for column in index_2.columns:
        # 如果该列不在 index_1 中
        if column not in index_1.columns:
            # 将该列添加到合并后的列列表中
            merged_columns.append(column)
    # 使用合并后的列列表创建一个新的索引对象并返回
    return Index(merged_columns)


# Splitting two indexes produces a common index I_C and at most two additional
# residual indexes I_R1 and I_R2. Splitting I_1(K_1;S_1) and I_2(K_2;S_2):
# I_C = (K_C;S_C) with K_C = K_1 ∩ K_2 and S_C = S_1 ∩ S_2 where K_C cannot be empty.
# Split is undefined if K_1 and K_2 have no common columns. If K_1 and K_C are different:
# I_R_1 = (K_1 - K_C, I_1 - I_C) and if K_2 and K_C are different
# I_R_2 = (K_2 - K_C, I_2 - I_C).
# Returns None if K_1 and K_2 have no common columns or a set: {I_C, I_R_1, I_R_2} where
# both I_R_1 are I_R_2 optional.
def index_split(index_1, index_2):
    """
    拆分两个索引对象，生成一个公共索引和最多两个剩余索引。

    :param index_1: 第一个索引对象
    :param index_2: 第二个索引对象
    :return: 如果两个索引没有公共列，返回 None；否则返回一个集合，包含公共索引和可能的剩余索引
    """
    # 确保两个索引属于同一个表
    assert index_1.table() == index_2.table()
    # 用于存储公共列的列表
    common_columns = []
    # 用于存储 index_1 中除去公共列后的剩余列的列表
    index_1_residual_columns = []
    # 遍历 index_1 的列
    for column in index_1.columns:
        # 如果该列也在 index_2 中
        if column in index_2.columns:
            # 将该列添加到公共列列表中
            common_columns.append(column)
        else:
            # 否则将该列添加到 index_1 的剩余列列表中
            index_1_residual_columns.append(column)
    # 如果没有公共列
    if len(common_columns) == 0:
        # 返回 None
        return None
    # 初始化结果集合，添加公共索引
    result = {Index(common_columns)}
    # 如果 index_1 有剩余列
    if len(index_1_residual_columns) > 0:
        # 将剩余列组成的索引添加到结果集合中
        result.add(Index(index_1_residual_columns))
    # 用于存储 index_2 中除去公共列后的剩余列的列表
    index_2_residual_columns = []
    # 遍历 index_2 的列
    for column in index_2.columns:
        # 如果该列不在 index_1 中
        if column not in index_1.columns:
            # 将该列添加到 index_2 的剩余列列表中
            index_2_residual_columns.append(column)
    # 如果 index_2 有剩余列
    if len(index_2_residual_columns) > 0:
        # 将剩余列组成的索引添加到结果集合中
        result.add(Index(index_2_residual_columns))
    # 返回结果集合
    return result
