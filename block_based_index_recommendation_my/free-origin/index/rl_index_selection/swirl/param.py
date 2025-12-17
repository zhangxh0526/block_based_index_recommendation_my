import argparse

parser = argparse.ArgumentParser(description='DAG_ML')

# -- Model --
parser.add_argument('--attention_dim', type=int, default=128,
                    help='dim of query, key and value for attention layer')
parser.add_argument('--state_dim', type=int, default=64,
                    help='dim of state')
parser.add_argument('--workload_dim', type=int, default=512,
                    help='dim of workload')





def get_args(argv=None):
    """Parse args only when explicitly requested to avoid side effects on import."""
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(get_args())
