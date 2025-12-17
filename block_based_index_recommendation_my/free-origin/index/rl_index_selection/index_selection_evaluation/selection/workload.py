from .index import Index


class Workload:
    """
    表示一个数据库工作负载，包含一组查询。

    Attributes:
        queries (list): 包含 Query 对象的列表，代表工作负载中的查询集合。
        budget (float): 工作负载的预算，初始值为 None。
        description (str): 工作负载的描述信息，默认为空字符串。
    """
    def __init__(self, queries, description=""):
        """
        初始化 Workload 类的实例。

        Args:
            queries (list): 包含 Query 对象的列表，代表工作负载中的查询集合。
            description (str, optional): 工作负载的描述信息，默认为空字符串。
        """
        # 存储传入的查询列表
        self.queries = queries
        # 初始化工作负载的预算为 None
        self.budget = None
        # 存储工作负载的描述信息
        self.description = description

    def indexable_columns(self, return_sorted=True):
        """
        获取工作负载中所有查询的可索引列。

        Args:
            return_sorted (bool, optional): 指示是否对结果进行排序，默认为 True。

        Returns:
            set or list: 如果 return_sorted 为 False，返回包含可索引列的集合；否则返回排序后的列列表。
        """
        # 初始化一个空集合，用于存储可索引列
        indexable_columns = set()
        # 遍历工作负载中的每个查询
        for query in self.queries:
            # 将查询的列添加到可索引列集合中
            indexable_columns |= set(query.columns)
        # 如果不需要排序
        if not return_sorted:
            # 直接返回可索引列集合
            return indexable_columns
        # 对可索引列集合进行排序并返回
        return sorted(list(indexable_columns))

    def potential_indexes(self):
        """
        生成工作负载中所有可能的单列表索引。

        Returns:
            list: 包含 Index 对象的列表，代表所有可能的单列表索引。
        """
        # 调用 indexable_columns 方法获取可索引列
        # 为每个可索引列创建一个 Index 对象
        # 对生成的索引列表进行排序并返回
        return sorted([Index([c]) for c in self.indexable_columns()])

    def __repr__(self):
        """
        返回 Workload 实例的字符串表示形式，方便调试和打印。

        Returns:
            str: 包含查询编号、频率、工作负载描述和预算的字符串。
        """
        # 初始化一个空列表，用于存储查询的编号
        ids = []
        # 初始化一个空列表，用于存储查询的频率
        fr = []
        # 遍历工作负载中的每个查询
        for query in self.queries:
            # 将查询的编号添加到 ids 列表中
            ids.append(query.nr)
            # 将查询的频率添加到 fr 列表中
            fr.append(query.frequency)
        # 返回包含查询编号、频率、工作负载描述和预算的字符串
        return f"Query IDs: {ids} with {fr}. {self.description} Budget: {self.budget}"



class Column:
    """
    表示数据库表中的列。

    Attributes:
        name (str): 列的名称，存储为小写。
        table (Table): 列所属的表，初始值为 None。
        global_column_id (int): 全局列 ID，初始值为 None。
        length (int): 列的长度，初始值为 None。
        distinct_values (int): 列中不同值的数量，初始值为 None。
        is_padding_column (bool): 指示该列是否为填充列，默认为 False。
        width (int): 列的宽度，初始值为 None。
    """
    def __init__(self, name):
        """
        初始化 Column 类的实例。

        Args:
            name (str): 列的名称。
        """
        # 将列名转换为小写并存储
        self.name = name.lower()
        # 列所属的表，初始化为 None
        self.table = None
        # 全局列 ID，初始化为 None
        self.global_column_id = None
        # 列的长度，初始化为 None
        self.length = None
        # 列中不同值的数量，初始化为 None
        self.distinct_values = None
        # 指示该列是否为填充列，默认为 False
        self.is_padding_column = False
        # 列的宽度，初始化为 None
        self.width = None

    def __lt__(self, other):
        """
        实现小于比较运算符，用于对列进行排序。

        Args:
            other (Column): 另一个 Column 实例。

        Returns:
            bool: 如果当前列的名称按字典序小于另一个列的名称，则返回 True，否则返回 False。
        """
        return self.name < other.name

    def __repr__(self):
        """
        返回列的字符串表示形式。

        Returns:
            str: 列的字符串表示，格式为 "C <表名>.<列名>"。
        """
        return f"C {self.table}.{self.name}"

    # 我们不能在这里检查 self.table == other.table，因为 Table.__eq__()
    # 内部会检查 Column.__eq__。这会导致无限递归。
    def __eq__(self, other):
        """
        实现相等比较运算符，用于判断两个列是否相等。

        Args:
            other (Column): 另一个 Column 实例。

        Returns:
            bool: 如果两个列所属表的名称和列名都相同，则返回 True，否则返回 False。
        """
        # 检查 other 是否为 Column 类的实例
        if not isinstance(other, Column):
            return False

        # 确保在比较列时，两个列所属的表对象不为 None
        assert (
            self.table is not None and other.table is not None
        ), "Table objects should not be None for Column.__eq__()"

        # 比较两个列所属表的名称和列名是否相同
        return self.table.name == other.table.name and self.name == other.name

    def __hash__(self):
        """
        返回列的哈希值，用于在集合和字典中使用。

        Returns:
            int: 列的哈希值，基于列名和所属表的名称计算。
        """
        return hash((self.name, self.table.name))



class Table:
    """
    表示数据库中的表。

    Attributes:
        name (str): 表的名称，转换为小写存储。
        columns (list): 表中包含的列的列表。
    """
    def __init__(self, name):
        """
        初始化 Table 类的实例。

        Args:
            name (str): 表的名称。
        """
        # 将表名转换为小写并存储
        self.name = name.lower()
        # 初始化列列表为空
        self.columns = []

    def add_column(self, column):
        """
        向表中添加一个列。

        Args:
            column (Column): 要添加的列对象。
        """
        # 设置列所属的表为当前表
        column.table = self
        # 将列添加到表的列列表中
        self.columns.append(column)

    def add_columns(self, columns):
        """
        向表中添加多个列。

        Args:
            columns (list): 要添加的列对象列表。
        """
        # 遍历列列表
        for column in columns:
            # 调用 add_column 方法添加每一列
            self.add_column(column)

    def __repr__(self):
        """
        返回表的字符串表示形式。

        Returns:
            str: 表的名称。
        """
        return self.name

    def __eq__(self, other):
        """
        实现相等比较运算符，用于判断两个表是否相等。

        Args:
            other (Table): 另一个 Table 实例。

        Returns:
            bool: 如果两个表的名称和列都相同，则返回 True，否则返回 False。
        """
        # 检查 other 是否为 Table 类的实例
        if not isinstance(other, Table):
            return False

        # 比较两个表的名称和列是否相同
        return self.name == other.name and tuple(self.columns) == tuple(other.columns)

    def __hash__(self):
        """
        返回表的哈希值，用于在集合和字典中使用。

        Returns:
            int: 表的哈希值，基于表名和列的元组计算。
        """
        return hash((self.name, tuple(self.columns)))



class Query:
    """
    表示数据库查询的类。每个查询包含一个唯一的 ID、查询文本、可索引的列和查询频率。

    Attributes:
        nr (int): 查询的唯一标识符。
        text (str): 查询的文本内容。
        frequency (int): 查询的执行频率，默认为 1。
        columns (list): 可用于索引的列的列表。
    """
    def __init__(self, query_id, query_text, columns=None, frequency=1):
        """
        初始化 Query 类的实例。

        Args:
            query_id (int): 查询的唯一标识符。
            query_text (str): 查询的文本内容。
            columns (list, optional): 可用于索引的列的列表。默认为 None。
            frequency (int, optional): 查询的执行频率。默认为 1。
        """
        # 查询的唯一编号
        self.nr = query_id
        # 查询的文本内容
        self.text = query_text
        # 查询的执行频率
        self.frequency = frequency

        # 可索引的列
        if columns is None:
            # 如果未提供列，则初始化为空列表
            self.columns = []
        else:
            # 否则使用提供的列列表
            self.columns = columns

    def __repr__(self):
        """
        返回查询的字符串表示形式。

        Returns:
            str: 查询的字符串表示，格式为 "Q<查询编号>"。
        """
        return f"Q{self.nr}"

    def __eq__(self, other):
        """
        实现相等比较运算符，用于判断两个查询是否相等。

        Args:
            other (Query): 另一个 Query 实例。

        Returns:
            bool: 如果两个查询的编号相同，则返回 True，否则返回 False。
        """
        if not isinstance(other, Query):
            # 如果 other 不是 Query 类的实例，返回 False
            return False

        # 比较两个查询的编号是否相同
        return self.nr == other.nr

    def __hash__(self):
        """
        返回查询的哈希值，用于在集合和字典中使用。

        Returns:
            int: 查询的哈希值，基于查询编号计算。
        """
        return hash(self.nr)

