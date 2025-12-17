# according to paper harma V, Dyreson C. Indexer++ workload-aware online index tuning with transformers and reinforcement
# learning[C]//Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing. 2022: 372-380.

import random

# columns:{column:[min, max]}
#Randomly extract a distinct value :
# Randomly select operator [>, <, =, =>, <=, > <] 7:
# Randomly select predicates [0=3, >A]
def generate_custom_workloads(query_num, max_column_num, columns, table_name):
    queries = []
    operators = [">", "<", "=>", "<=", "><"]
    relations = ["and", "or"]
    column_names = columns.keys()
    for _ in range(query_num):
        column_num = random.randint(1, max_column_num)
        query = "select * from {} where".format(table_name)
        for col in range(column_num):
            column = random.choice(column_names)
            operator = random.choice(operators)
            if operator != "><":
                value = random.randrange(columns[column][0], columns[column][-1])
            else:
                start = random.randrange(columns[column][0], columns[column][-1])
                end = random.randrange(start, columns[column][-1])
            relation = random.choice(relations)
            if col != 0:
                query += relation
            if operator != "><":
                query += "{} {} {}".format(column, operator, value)
            else:
                query += "{} BETWEEN {} AND {}".format(column, start, end)
        queries.append(query)
    return  queries


