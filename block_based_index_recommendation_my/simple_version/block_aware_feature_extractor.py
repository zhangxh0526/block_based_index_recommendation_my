import gym
import torch as th
from torch import nn
from gym import spaces
from param import *

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BlockAwareFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "workload":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[1], parser.attention_input)
                total_concat_size += parser.attention_input * subspace.shape[0]
            elif key == "block":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[2], parser.attention_input)
                total_concat_size += parser.attention_input * subspace.shape[0] * subspace.shape[1]
                # attention layer query * output
                total_concat_size += subspace.shape[0] * parser.attention_input * parser.heads
            elif key == "candidate":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0], parser.candidate_linear_1))
                total_concat_size += parser.candidate_linear_1
            elif key == "space":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], parser.space_linear_1)
                total_concat_size += parser.space_linear_1



        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:

        attention = nn.MultiheadAttention(embed_dim=parser.attention_input * parser.heads, num_heads=parser.heads, dropout=parser.drop_pro)

        encoded_tensor_list = []

        attention_dict = {}

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "workload":
                attention_dict["query"] = th.reshape(observations[key], shape=[observations[key].shape[0], 1, observations[key].shape[1]])
                encoded_tensor_list.append(nn.Flatten(extractor(observations[key])))
            elif key == "block":
                attention_dict["key"] = observations[key]
                attention_dict["value"] = observations[key]
                encoded_tensor_list.append(nn.Flatten(extractor(observations[key])))
            else:
                encoded_tensor_list.append(extractor(observations[key]))

        encoded_tensor_list.append(nn.Sequential(attention(query=attention_dict["query"],
                                             key=attention_dict["key"],
                                             value=attention_dict["value"],
                                             attn_output_weights=False,
                                             batch_first=True),
                                                 nn.Flatten()))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


