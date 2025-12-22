import torch as th
import torch.nn as nn
from gym import spaces
from algorithm.demo.param import *

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Block_based_workload_embedding(nn.Module):
    def __init__(self):
        super(Block_based_workload_embedding, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=args.attention_dim, num_heads=1, dropout=0)


    def forward(self, input):
        query = nn.Linear(input[0].shape[-1], args.attention_dim)(input[0])
        key = nn.Linear(input[1].shape[-1], args.attention_dim)(input[1])
        value = nn.Linear(input[1].shape[-1], args.attention_dim)(input[1])
        att_o, att_o_w = self.attention(query, key, value)
        return att_o



class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0

        query_num = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            # [block_num, block_dim]
            if key == "block_state_inf":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[1], args.block_inf_out), nn.Flatten())
                total_concat_size += subspace.shape[0] * args.block_inf_out
            # [query_type, query_dim]
            # [[frequency, a, b, c...], []...]
            elif key == "workload_inf":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[1], args.workload_inf_out), nn.Flatten())
                total_concat_size += subspace.shape[1] * args.workload_inf_out
            elif key == "block_based_workload_inf":
                extractors[key] = nn.Sequential(Block_based_workload_embedding(), nn.Flatten())
                total_concat_size += args.workload_size * args.attention_dim
            elif key == "position":
                extractors[key] = nn.Sequential(nn.Flatten())
                total_concat_size += 2

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "block_based_workload_inf":
                encoded_tensor_list.append(extractor([observations["workload_inf"], observations[key]]))
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)