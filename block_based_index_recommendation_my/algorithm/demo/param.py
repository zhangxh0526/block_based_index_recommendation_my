import argparse

parser = argparse.ArgumentParser(description='DAG_ML')

# -- Model --
parser.add_argument('--block_feature_dim', type=int, default=2,
                    help='feature dim of blocks')
parser.add_argument('--query_feature_dim', type=int, default=2,
                    help='feature dim of origin queries')
# [[start_pos, end_pos], [start_pos, end_pos], [start_pos, end_pos] ...]
parser.add_argument('--indexed_attributes', type=int, default=3,
                    help='number of attributes needs to be indexed')
parser.add_argument('--attention_dim', type=int, default=5,
                    help='dim of query, key and value for attention layer')
parser.add_argument('--block_type', type=int, default=2,
                    help='number of blocks after aggregation')
parser.add_argument('--workload_size', type=int, default=2,
                    help='number of blocks after aggregation')

# input feature extract
parser.add_argument('--block_inf_out', type=int, default=8,
                    help='dim of extracted block feature')
parser.add_argument('--workload_inf_out', type=int, default=8,
                    help='dim of extracted workload feature')
parser.add_argument('--block_based_workload_inf_out', type=int, default=16,
                    help='dim of extracted block_based workload feature')




args = parser.parse_args()
