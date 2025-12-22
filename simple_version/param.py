import argparse

parser = argparse.ArgumentParser(description='Block_Index')

# -- FeaturesExtractor --
parser.add_argument('--attention_input', type=int, default=64,
                    help='output of dense layer for workload')

parser.add_argument('--attention_input', type=int, default=64,
                    help='output of dense layer for block')

parser.add_argument('--candidate_linear_1', type=int, default=128,
                    help='output of dense layer for candidate')

parser.add_argument('--space_linear_1', type=int, default=2,
                    help='output of dense layer for space')

## -- FeaturesExtractor --
parser.add_argument('--drop_pro', type=float, default=0.2,
                    help='drop out probability for attention layer')

parser.add_argument('--heads', type=int, default=1,
                    help='number of heads for multi-head attention layer')

